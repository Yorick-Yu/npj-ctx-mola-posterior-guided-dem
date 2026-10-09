"""Frozen inference and physical-variable evaluation for reviewer 2 comment 7."""

import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from runtime import base, make_variant, sha, write_json
import run_frozen_external_v36 as original
from physical import plane_maps, errors, freeze_features, evaluate_features, self_test

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())


def read(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:np.asarray(data[k]) for k in data.files}


def save(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=read(path)
        if old.keys()!=arrays.keys() or not all(np.array_equal(old[k],v,equal_nan=True) for k,v in arrays.items()):
            raise RuntimeError(f'Refusing to replace a different frozen output: {path}')
        return
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays)
    temp.replace(path)


def preflight():
    prepared=json.loads((ROOT/'prepared.json').read_text())
    assert prepared['protocol_sha256']==sha(ROOT/'protocol.json')
    for name,digest in prepared['files'].items():
        assert sha(ROOT/name)==digest,name
    test=self_test()
    feature_records=[]
    for region in P['regions']:
        d=ROOT/'cases'/region
        a=read(d/'reference.npz');i=read(d/'inference.npz')
        meta=json.loads((d/'prepared.json').read_text())
        assert len(i['row'])==20 and not set(i['orbit']) & set(a['heldout_orbit'])
        assert not any(k in i for k in ['dem','heldout_elevation','reference'])
        features=freeze_features(a['dem'],a['strict'],region,meta['dx_m'],meta['dy_m'],P)
        count=sum(f['usable'] for f in features)
        payload=dict(features=features,usable=count,reference_sha256=sha(d/'reference.npz'),
                     protocol_sha256=sha(ROOT/'protocol.json'),frozen_before_prediction=True)
        path=d/'features.json'
        if path.exists():
            assert json.loads(path.read_text())==payload
        else:
            write_json(path,payload)
        feature_records.append(dict(region=region,usable_profiles=count,sha256=sha(path)))
        print('FEATURE_PREFLIGHT',region,count,flush=True)
        if count<2:
            raise RuntimeError(f'{region}: only {count} valid fixed profiles; resolve quality before prediction')
    torch,_,_=base.torch_components()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==P['gpu_uuid']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    x=torch.ones(10,device='cuda');assert float(x.sum())==10
    write_json(ROOT/'preflight.json',dict(state='passed',features=feature_records,unit_test=test,
               torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),
               gpu_uuid=P['gpu_uuid'],protocol_sha256=sha(ROOT/'protocol.json'),completed=time.time()))


def infer():
    assert json.loads((ROOT/'preflight.json').read_text())['state']=='passed'
    torch,_,_=base.torch_components()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0')
    models=json.loads((ROOT/'models/manifest.json').read_text())
    for region in P['regions']:
        directory=ROOT/'cases'/region;out=ROOT/'predictions'/region
        receipt=out/'receipt.json'
        if receipt.exists():
            for r in json.loads(receipt.read_text())['records']:
                assert sha(ROOT/r['path'])==r['sha256']
            continue
        meta=json.loads((directory/'prepared.json').read_text())
        assert sha(directory/'inference.npz')==meta['inference_sha256']
        a=read(directory/'inference.npz')
        assert set(a)=={'z0','image','valid','row','column','elevation','orbit'}
        case=base.LoadedCase(region,region,'r2c7-prospective',a)
        context=np.arange(20,dtype=np.int64)
        gp=base.build_gp_fields(case,context,lengthscale_px=original.PRIOR_LENGTHSCALE_PX,
                               noise_m=original.PRIOR_NOISE_M)
        no=(a['z0'].copy(),np.full(a['z0'].shape,base.HEIGHT_SCALE_M,np.float32),np.full(a['z0'].shape,400,np.float32))
        records=[]
        def record(method,seed,k,stage,field,checkpoint=None):
            assert np.isfinite(field).all()
            path=out/f'{method}-{seed}-k{k}-{stage}.npz'
            save(path,mean=np.asarray(field,dtype=np.float32))
            records.append(dict(method=method,seed=seed,k=k,stage=stage,path=str(path.relative_to(ROOT)),
                                sha256=sha(path),checkpoint_sha256=checkpoint))
        record('coarse_CTX',0,0,'final',a['z0'])
        shift=np.median(a['elevation']-base.sample_raster(a['z0'],a['row'],a['column']))
        record('median_offset',0,20,'final',a['z0']+shift)
        record('GP',0,20,'final',gp[0])
        for m in models:
            cp=ROOT/m['path'];assert sha(cp)==m['sha256']
            state=torch.load(cp,map_location='cpu',weights_only=False)
            if m['method']=='original_frozen_v36':
                arch=state['architecture']
                model=base.make_model(int(arch['base_channels']),int(arch['transport_steps']),float(arch['minimum_mobility']))
            else:
                assert state['variant']=='plain_unet'
                model=make_variant('plain_unet')
            model.load_state_dict(state['model'],strict=True)
            model=model.to(device).eval()
            base.DISABLE_TERMINAL=False
            k0=base.predict_from_fields(model,case,*no,np.empty(0,dtype=np.int64),device)
            base.DISABLE_TERMINAL=True
            raw=base.predict_from_fields(model,case,*gp,context,device)
            base.DISABLE_TERMINAL=False
            terminal=base.predict_from_fields(model,case,*gp,context,device)
            low,_=original.v34.project_lowpass_shape(terminal,k0,a['valid'],20.0)
            original.v35.LOCAL_NOISE_M=0.0
            final,_=original.v35.impose_local_laser_consistency(low,context,a)
            for k,stage,field in [(0,'final',k0),(20,'raw',raw),(20,'terminal',terminal),(20,'final',final)]:
                record(m['method'],m['seed'],k,stage,field,m['sha256'])
            print('INFERRED',region,m['method'],m['seed'],flush=True)
            del model,state;torch.cuda.empty_cache()
        write_json(receipt,dict(state='frozen',records=records,inference_sha256=sha(directory/'inference.npz'),
                     reference_opened=False,heldout_opened=False,protocol_sha256=sha(ROOT/'protocol.json')))


def score():
    allrows=[];allprofiles=[]
    for region in P['regions']:
        d=ROOT/'cases'/region;out=ROOT/'results'/region
        out.mkdir(parents=True,exist_ok=True)
        a=read(d/'reference.npz');inp=read(d/'inference.npz')
        meta=json.loads((d/'prepared.json').read_text());dx,dy=meta['dx_m'],meta['dy_m']
        features=json.loads((d/'features.json').read_text())['features']
        receipts=json.loads((ROOT/'predictions'/region/'receipt.json').read_text())['records']
        items=[(dict(method='native_CTX',seed=0,k=0,stage='final'),a['native_ctx'])]
        for r in receipts:
            assert sha(ROOT/r['path'])==r['sha256']
            if r['stage']=='final':items.append((r,read(ROOT/r['path'])['mean']))
        refs={(mask,scale):plane_maps(a['dem'],a[mask],dx,dy,scale)
              for mask in ['strict','geometry'] for scale in P['physical_scales_m']}
        refprof=evaluate_features(a['dem'],a['strict'],features)
        write_json(out/'reference-profiles.json',refprof)
        for r,z in items:
            identity=dict(region=region,method=r['method'],seed=r['seed'],k=r['k'])
            export_geotiff(z,meta,out/f"{r['method']}-{r['seed']}-k{r['k']}.tif",inp['valid'])
            held=base.sample_raster(z,a['heldout_row'],a['heldout_column'])-a['heldout_elevation']
            for mask in ['strict','geometry']:
                for scale in P['physical_scales_m']:
                    slope,rough,valid,support=plane_maps(z,a[mask],dx,dy,scale)
                    rs,rr,rv,_=refs[mask,scale]
                    assert np.array_equal(valid,rv)
                    es,er=errors(slope,rs,valid,dx,dy),errors(rough,rr,valid,dx,dy)
                    height=errors(z,a['dem'],a[mask],dx,dy)
                    row=dict(**identity,mask=mask,scale_m=scale,actual_window_m=support,
                             physical_pixels=int(valid.sum()),slope_rmse_deg=es['rmse'],slope_mae_deg=es['mae'],
                             roughness_rmse_m=er['rmse'],roughness_mae_m=er['mae'],
                             reference_mean_slope_deg=float(np.mean(rs[valid])),
                             predicted_mean_slope_deg=float(np.mean(slope[valid])),
                             reference_mean_roughness_m=float(np.mean(rr[valid])),
                             predicted_mean_roughness_m=float(np.mean(rough[valid])),
                             hirise_height_rmse_m=height['rmse'],
                             heldout_mola_rmse_m=float(np.sqrt(np.mean(held*held))),heldout_points=len(held),
                             slope_blocks=es['spatial_blocks'],roughness_blocks=er['spatial_blocks'])
                    allrows.append(row)
                    save(out/f"{r['method']}-{r['seed']}-k{r['k']}-{mask}-{scale}m-maps.npz",
                         slope=slope.astype(np.float32),roughness=rough.astype(np.float32),valid=valid,
                         reference_slope=rs.astype(np.float32),reference_roughness=rr.astype(np.float32))
            profiles=evaluate_features(z,a['strict'],features)
            for entry in profiles:allprofiles.append(dict(**identity,**entry))
            print('SCORED',region,r['method'],r['seed'],r['k'],flush=True)
        write_json(out/'stage-done.json',dict(state='scored',time=time.time()))
    write_json(ROOT/'results/physical-metrics.json',allrows)
    write_json(ROOT/'results/physical-profiles.json',allprofiles)
    flat=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in allrows]
    with (ROOT/'results/physical-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    summaries=[]
    for region in P['regions']:
        for mask in ['strict','geometry']:
            for scale in P['physical_scales_m']:
                rows=[r for r in allrows if r['region']==region and r['mask']==mask and r['scale_m']==scale]
                for method,k in sorted({(r['method'],r['k']) for r in rows}):
                    group=[r for r in rows if (r['method'],r['k'])==(method,k)]
                    s=dict(region=region,mask=mask,scale_m=scale,method=method,k=k,seeds=len(group))
                    for key in ['slope_rmse_deg','roughness_rmse_m','hirise_height_rmse_m','heldout_mola_rmse_m']:
                        values=[r[key] for r in group];s[key]=float(np.mean(values));s[key+'_seed_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
                    summaries.append(s)
    write_json(ROOT/'results/summary.json',summaries)


def export_geotiff(z,meta,path,valid):
    import rasterio
    from affine import Affine
    if path.exists():return
    with rasterio.open(path,'w',driver='GTiff',width=504,height=504,count=1,dtype='float32',
                       crs=meta['crs'],transform=Affine(*meta['transform']),nodata=-9999,compress='deflate') as ds:
        ds.write(np.where(valid,z,-9999).astype(np.float32),1)
        ds.update_tags(vertical_datum='Mars areoid as published; no target-fitted registration',units='m')


def audit():
    rows=json.loads((ROOT/'results/physical-metrics.json').read_text())
    assert len(rows)==128,(len(rows),'Expected 2 scenes x 16 fields x 2 masks x 2 scales')
    for r in rows:
        assert all(np.isfinite(r[k]) for k in ['slope_rmse_deg','roughness_rmse_m','hirise_height_rmse_m'])
    checks={}
    for region in P['regions']:
        d=ROOT/'cases'/region;a=read(d/'reference.npz');i=read(d/'inference.npz')
        meta=json.loads((d/'prepared.json').read_text());assert sha(d/'reference.npz')==meta['reference_sha256']
        assert sha(d/'inference.npz')==meta['inference_sha256']
        assert not set(i['orbit']) & set(a['heldout_orbit'])
        receipt=json.loads((ROOT/'predictions'/region/'receipt.json').read_text())
        assert len(receipt['records'])==27
        for r in receipt['records']:
            assert sha(ROOT/r['path'])==r['sha256']
        # Independent scalar least-squares fits cross-check the fast map implementation.
        dx,dy=meta['dx_m'],meta['dy_m'];scale=100
        s,q,v,sizes=plane_maps(a['dem'],a['strict'],dx,dy,scale)
        yy,xx=np.where(v);take=np.linspace(0,len(yy)-1,min(20,len(yy))).astype(int)
        nx,ny=round(sizes[0]/dx),round(sizes[1]/dy);diff=[]
        for idx in take:
            y,x=int(yy[idx]),int(xx[idx]);r0,r1=max(0,y-ny//2),min(504,y+ny//2+1);c0,c1=max(0,x-nx//2),min(504,x+nx//2+1)
            ry,cx=np.indices((r1-r0,c1-c0));m=a['strict'][r0:r1,c0:c1]
            design=np.stack([np.ones(m.sum()),((cx+c0-x)*dx)[m],((ry+r0-y)*dy)[m]],1)
            target=a['dem'][r0:r1,c0:c1][m].astype(float)
            coef=np.linalg.lstsq(design,target,rcond=None)[0]
            slope=np.degrees(np.arctan(np.hypot(coef[1],coef[2])))
            rough=float(np.sqrt(np.mean((target-design@coef)**2)))
            diff.append([abs(slope-s[y,x]),abs(rough-q[y,x])])
        maximum=np.max(diff,axis=0);assert maximum[0]<1e-6 and maximum[1]<1e-3,maximum
        checks[region]=dict(independent_plane_max_error=maximum.tolist(),prediction_count=27,orbit_disjoint=True)
    manifest={str(p.relative_to(ROOT)):sha(p) for folder in ['results','predictions','cases','models']
              for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
    write_json(ROOT/'download-manifest.json',manifest)
    write_json(ROOT/'audit.json',dict(state='passed',checks=checks,metric_rows=len(rows),
               protocol_sha256=sha(ROOT/'protocol.json'),download_manifest_sha256=sha(ROOT/'download-manifest.json'),
               scientific_outcome='Report measured improvements AND regressions; no superiority gate.',
               visual_review_required=True,completed=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','infer','score','audit'])
    args=parser.parse_args();globals()[args.stage]()
