"""Physical estimators shared across before/after DEMs on identical support."""

import numpy as np
from scipy import ndimage as ndi


def plane_maps(z, valid, dx, dy, scale):
    nx, ny = max(3, int(round(scale / dx)) | 1), max(3, int(round(scale / dy)) | 1)
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    z = np.where(valid, z, 0).astype(np.float64)
    w = valid.astype(float)
    count = ndi.uniform_filter(w, size=(ny,nx), mode='constant')
    den = np.maximum(count,1e-12)
    def moment(a, xp=0, yp=0):
        return ndi.correlate1d(ndi.correlate1d(a,x**xp/nx,axis=1,mode='constant'),
                               y**yp/ny,axis=0,mode='constant')/den
    mx,my,mz = moment(w,1),moment(w,0,1),moment(z)
    cxx,cyy,cxy = moment(w,2)-mx*mx,moment(w,0,2)-my*my,moment(w,1,1)-mx*my
    cxz,cyz = moment(z,1)-mx*mz,moment(z,0,1)-my*mz
    det = cxx*cyy-cxy*cxy
    safe_det = np.where(det>1e-9,det,1)
    bx,by = (cxz*cyy-cyz*cxy)/safe_det,(cyz*cxx-cxz*cxy)/safe_det
    slope = np.degrees(np.arctan(np.hypot(bx, by)))
    roughness = np.sqrt(np.maximum(0, moment(z*z)-mz*mz-bx*cxz-by*cyz))
    support = valid & (count>=.6) & (det>1e-6)
    return slope, roughness, support, [nx*dx, ny*dy]


def errors(pred, reference, valid, dx, dy):
    residual = np.asarray(pred-reference, dtype=float)
    e = residual[valid]
    if len(e) == 0:
        return dict(n=0, rmse=None, mae=None, bias=None, spatial_blocks=[])
    block_rows, block_cols = max(1, round(1000/dy)), max(1, round(1000/dx))
    blocks = []
    for r in range(0, e.shape[0] if residual.ndim == 1 else residual.shape[0], block_rows):
        for c in range(0, residual.shape[1], block_cols):
            m = valid[r:r+block_rows, c:c+block_cols]
            if m.sum() >= 128:
                q = residual[r:r+block_rows,c:c+block_cols][m]
                blocks.append(dict(row=r,column=c,n=int(m.sum()),rmse=float(np.sqrt(np.mean(q*q)))))
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(abs(e))),
                bias=float(np.mean(e)), spatial_blocks=blocks)


def profile(z, mask, axis, index):
    block = z[:,index-2:index+3] if axis == 'column' else z[index-2:index+3,:].T
    good = mask[:,index-2:index+3] if axis == 'column' else mask[index-2:index+3,:].T
    counts = good.sum(axis=1)
    values = np.sum(np.where(good,block,0),axis=1) / np.maximum(counts,1)
    valid = counts >= 3
    return values, valid


def smooth_profile(values, valid, spacing, sigma_m=60):
    half = max(3,int(np.ceil(3*sigma_m/spacing)))
    x = np.arange(-half,half+1)*spacing/sigma_m
    weights=np.exp(-x*x/2)
    w=valid.astype(float);v=np.where(valid,values,0)
    m=[ndi.correlate1d(w,weights*x**k,mode='constant') for k in range(5)]
    rhs=np.stack([ndi.correlate1d(v,weights*x**k,mode='constant') for k in range(3)],axis=-1)
    matrix=np.stack([m[0],m[1],m[2],m[1],m[2],m[3],m[2],m[3],m[4]],axis=-1).reshape(-1,3,3)
    count=ndi.uniform_filter1d(w,len(x),mode='constant')
    left=ndi.correlate1d(w,(x<0).astype(float),mode='constant')
    right=ndi.correlate1d(w,(x>0).astype(float),mode='constant')
    safe=(count>=.6)&(left>=3)&(right>=3)&(np.linalg.det(matrix)>1e-8)
    matrix[~safe]=np.eye(3);rhs[~safe]=0
    coefficients=np.linalg.solve(matrix,rhs[...,None])[...,0]
    return coefficients[:,0],safe


def freeze_features(ref, mask, region, dx, dy, protocol):
    result = []
    if region == 'fresh_crater_noachis':
        cfg = protocol['rim_profiles']
        for col in cfg['columns']:
            z, valid = profile(ref, mask, 'column', col)
            smooth, safe = smooth_profile(z,valid,dy)
            rr = np.arange(504)
            rim = safe & (rr>=cfg['rim_search_rows'][0]) & (rr<cfg['rim_search_rows'][1])
            outside = valid & (rr>=cfg['outside_rows'][0]) & (rr<cfg['outside_rows'][1])
            floor = valid & (rr>=cfg['floor_rows'][0]) & (rr<cfg['floor_rows'][1])
            usable = rim.sum()>=3 and outside.sum()>=5 and floor.sum()>=20
            crest = int(np.flatnonzero(rim)[np.argmax(smooth[rim])]) if rim.any() else None
            result.append(dict(kind='rim',axis='column',index=col,usable=bool(usable),
                               rim_indices=np.flatnonzero(rim).tolist(),outside_indices=np.flatnonzero(outside).tolist(),
                               floor_indices=np.flatnonzero(floor).tolist(),reference_position_index=crest,
                               valid_indices=np.flatnonzero(valid).tolist(),spacing_m=dy,
                               reference_crest_minus_outside_m=float(smooth[crest]-np.median(z[outside])) if usable else None,
                               reference_crest_minus_floor_m=float(smooth[crest]-np.median(z[floor])) if usable else None))
    else:
        cfg = protocol['break_profiles']
        for row in cfg['rows']:
            z,valid = profile(ref,mask,'row',row)
            smooth,safe = smooth_profile(z,valid,dx,cfg['smoothing_sigma_m'])
            ids = np.arange(504)
            safe_derivative = ndi.minimum_filter1d(safe.astype(np.uint8),5,mode='constant') > 0
            domain = safe_derivative & (ids>=cfg['search_columns'][0]) & (ids<cfg['search_columns'][1])
            labels,n = ndi.label(domain)
            sizes = np.bincount(labels)[1:]
            if len(sizes):
                longest = int(np.argmax(sizes)+1)
                domain = labels==longest
            curvature = abs(np.gradient(np.gradient(smooth,dx),dx))
            usable = domain.sum()*dx >= cfg['minimum_contiguous_m']
            peak = int(np.flatnonzero(domain)[np.argmax(curvature[domain])]) if domain.any() else None
            result.append(dict(kind='slope_break',axis='row',index=row,usable=bool(usable),
                               search_indices=np.flatnonzero(domain).tolist(),reference_position_index=peak,
                               valid_indices=np.flatnonzero(valid).tolist(),spacing_m=dx,
                               reference_curvature_per_m=float(curvature[peak]) if usable else None))
    return result


def evaluate_features(z,mask,features):
    rows=[]
    for f in features:
        row=dict(kind=f['kind'],axis=f['axis'],index=f['index'],usable=f['usable'])
        values,valid=profile(z,mask,f['axis'],f['index'])
        assert np.flatnonzero(valid).tolist()==f['valid_indices']
        smooth,_=smooth_profile(values,valid,f['spacing_m'])
        row['profile_values_m']=values.tolist()
        if f['usable']:
            if f['kind']=='rim':
                ids=np.array(f['rim_indices'])
                peak=int(ids[np.argmax(smooth[ids])])
                outside=float(smooth[peak]-np.median(values[f['outside_indices']]))
                floor=float(smooth[peak]-np.median(values[f['floor_indices']]))
                row.update(crest_minus_outside_m=outside,crest_minus_floor_m=floor,
                           outside_relief_error_m=outside-f['reference_crest_minus_outside_m'],
                           floor_relief_error_m=floor-f['reference_crest_minus_floor_m'])
            else:
                ids=np.array(f['search_indices'])
                curve=abs(np.gradient(np.gradient(smooth,f['spacing_m']),f['spacing_m']))
                peak=int(ids[np.argmax(curve[ids])])
                row['curvature_per_m']=float(curve[peak])
            row.update(position_index=peak,position_error_m=(peak-f['reference_position_index'])*f['spacing_m'])
        rows.append(row)
    return rows


def self_test():
    rr,cc=np.indices((100,100));z=40+.12*cc*15+.07*rr*20
    s,r,v,_=plane_maps(z,np.ones_like(z,dtype=bool),15,20,100)
    assert np.max(abs(s[v]-np.degrees(np.arctan(np.hypot(.12,.07))))) < 1e-9
    assert np.max(r[v]) < 2e-5
    s2,r2,v2,_=plane_maps(z+1000,np.ones_like(z,dtype=bool),15,20,100)
    assert np.max(abs(s2[v2]-s[v])) < 1e-8
    assert np.max(r2[v2]) < 1e-4
    missing=np.ones_like(z,dtype=bool);missing[::3,::3]=False
    sm,rm,vm,_=plane_maps(z,missing,15,20,100)
    assert np.max(abs(sm[vm]-np.degrees(np.arctan(np.hypot(.12,.07))))) < 1e-8
    assert np.max(rm[vm]) < 1e-4
    return dict(state='passed',purpose='estimator unit test only; not experimental data')
