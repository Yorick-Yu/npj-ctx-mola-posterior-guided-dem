"""Prospective, target-isolated cross-product tests using frozen audited models."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'npj-r2c3-ablation-20260923'
sys.path[:0] = [str(ROOT/'deps'), str(OLD)]
import argparse
import csv
import hashlib
import json
import re
import time
import numpy as np
from scipy import ndimage
from runtime import base, make_variant, sha, write_json
import run_frozen_external_v36 as original
from paper2dem.detail_metrics import masked_gaussian_smooth, detail_field

PROTOCOL = json.loads((ROOT/'protocol.json').read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def download(url, path):
    import requests
    if path.exists():
        return
    response = requests.get(url, timeout=(30,180))
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def nearest_fill(data, valid):
    if not valid.any():
        raise ValueError('No raster support')
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(data[tuple(indices)], dtype=np.float32)


def raster_source(url):
    path=ROOT/'source-cache'/(hashlib.sha256(url.encode()).hexdigest()[:20]+'.tif')
    if path.exists():
        receipt=json.loads(path.with_suffix('.json').read_text())
        assert receipt['url']==url and receipt['sha256']==sha(path)
        return path
    return url


def warp(url, crs, transform, method):
    import rasterio
    from rasterio.vrt import WarpedVRT
    def read(source):
        with rasterio.open(source) as ds:
            with WarpedVRT(ds, crs=crs, transform=transform, width=504, height=504,
                           resampling=method, nodata=np.nan, dtype='float32') as vrt:
                a = vrt.read(1, masked=True).filled(np.nan)
            a = a * ds.scales[0] + ds.offsets[0]
            info = {'url':url,'crs_wkt':ds.crs.to_wkt(),'shape':[ds.height,ds.width],
                    'transform':list(ds.transform)[:6], 'nodata':str(ds.nodata),
                    'tags':ds.tags(), 'units':ds.units, 'scales':ds.scales, 'offsets':ds.offsets}
        return a,info
    try:
        a,info=read(raster_source(url))
    except rasterio.errors.RasterioIOError:
        import requests
        cache=ROOT/'source-cache';cache.mkdir(exist_ok=True)
        destination=cache/(hashlib.sha256(url.encode()).hexdigest()[:20]+'.tif')
        if not destination.exists():
            print('FULL_DOWNLOAD_FALLBACK',url,flush=True)
            if sum(p.stat().st_size for p in cache.glob('*'))>10*1024**3:
                raise RuntimeError('Task-local raster cache exceeded 10GB')
            temporary=destination.with_suffix('.partial')
            with requests.get(url,stream=True,timeout=(30,180),headers={'Cache-Control':'no-cache'}) as response:
                response.raise_for_status()
                if response.status_code!=200:raise ValueError('Full-object download returned partial response')
                total=0
                with temporary.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):
                        total+=len(chunk)
                        if total>2*1024**3:raise ValueError('Source raster exceeds 2GB download cap')
                        f.write(chunk)
                if response.headers.get('Content-Length'):
                    assert total==int(response.headers['Content-Length'])
                temporary.replace(destination)
                write_json(destination.with_suffix('.json'),dict(url=url,sha256=sha(destination),bytes=total,headers=dict(response.headers)))
        a,info=read(destination)
        info['downloaded_source_sha256']=sha(destination)
    good = np.isfinite(a) & (np.abs(a)<1e6)
    return np.where(good,a,np.nan).astype(np.float32),good,info


def prepare(region):
    import rasterio
    from rasterio.enums import Resampling
    from affine import Affine
    from pyproj import CRS, Transformer
    import requests
    directory = ROOT/'cases'/region
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'ready.json').exists():
        print('PREP_ALREADY_READY',region,flush=True)
        return
    candidates = json.loads((ROOT/'candidates.json').read_text())['candidates']
    record = next(r for r in candidates if r['region']==region)
    h = record['hirise']
    west,south,east,north = h['bbox']
    lon,lat = (west+east)/2,(south+north)/2
    mars = CRS.from_proj4('+proj=longlat +R=3396190 +no_defs')
    h_orbits = set(re.findall(r'\d{6}',h['id']))
    ctx = sorted(record['ctx_candidates'],key=lambda c:(bool(h_orbits & set(re.findall(r'\d{6}',c['id']))),c['id']))
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.TIF',
               GDAL_HTTP_MULTIRANGE='SERIAL',GDAL_HTTP_MULTIPLEX='NO',GDAL_HTTP_VERSION='1.1',
               GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',
               GDAL_HTTP_TIMEOUT='120',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_RETRY_DELAY='3',
               GDAL_NUM_THREADS='2',GDAL_CACHEMAX=256)
    errors=[]
    with rasterio.Env(**env):
        for c in ctx[:8]:
            try:
                assets=c['assets']
                dem_url=next(assets[k]['href'] for k in ('geoid_adjusted_dem','dtm') if k in assets)
                print('CTX_READ',region,c['id'],flush=True)
                with rasterio.open(raster_source(dem_url)) as ds:
                    crs=ds.crs
                    x,y=Transformer.from_crs(mars,crs,always_xy=True).transform(lon,lat)
                transform=Affine(20,0,x-5040,0,-20,y+5040)
                native,valid,ctx_info=warp(dem_url,crs,transform,Resampling.average)
                image,iv,image_info=warp(assets['orthoimage']['href'],crs,transform,Resampling.average)
                valid &= iv
                if valid.mean()<0.98:
                    raise ValueError(f'Input valid fraction {valid.mean():.4f} < .98')
                break
            except Exception as exc:
                errors.append({'ctx_id':c['id'],'error':str(exc)})
                write_json(directory/'coverage-attempts.json',{'attempts':errors})
        else:
            raise ValueError(f'No valid CTX input: {errors}')
        print('HIRISE_READ',region,h['id'],flush=True)
        reference,rv,ref_info=warp(h['assets']['dtm']['href'],crs,transform,Resampling.average)
    common=rv & valid
    _,support=masked_gaussian_smooth(reference,common,sigma_pixels=16)
    if common.sum()<25000 or support.sum()<10000:
        raise ValueError(f'HiRISE support too small: {common.sum()}, smoothed {support.sum()}')
    native_fill=nearest_fill(native,valid)
    coarse=ndimage.zoom(native_fill.reshape(21,24,21,24).mean(axis=(1,3)),24,order=3).astype(np.float32)
    image_fill=nearest_fill(image,valid)
    for name,item in [('hirise',h),('ctx',c)]:
        for key in ('provenance','qa_metrics','ortho_label_0'):
            if key in item['assets']:
                try:
                    download(item['assets'][key]['href'],directory/f'{name}-{key}.txt')
                except Exception as exc:
                    errors.append({'metadata':key,'error':str(exc)})
    # Query independent point records from ODE, never synthesize heights from either DTM.
    inverse=Transformer.from_crs(crs,mars,always_xy=True)
    xs=[transform.c,transform.c+10080,transform.c+10080,transform.c]
    ys=[transform.f,transform.f,transform.f-10080,transform.f-10080]
    lons,lats=inverse.transform(xs,ys)
    lons=np.asarray(lons)%360
    box=[float(min(lons)),float(min(lats)),float(max(lons)),float(max(lats))]
    params=dict(query='molapedr',results='v',minlat=box[1],maxlat=box[3],westernlon=box[0],easternlon=box[2],output='JSON')
    print('MOLA_QUERY',region,box,flush=True)
    response=requests.get('https://oderest.rsl.wustl.edu/livegds',params=params,timeout=(30,300))
    response.raise_for_status()
    payload=response.json()
    write_json(directory/'mola-query.json',{'url':response.url,'response':payload})
    files=payload['GDSResults']['ResultFiles']['ResultFile']
    if isinstance(files,dict):files=[files]
    for suffix,name in [('_pts_csv.csv','mola.csv'),('_pts_csv.lbl','mola.lbl')]:
        url=next(f['URL'] for f in files if f['URL'].endswith(suffix))
        download(url,directory/name)
    with (directory/'mola.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f,skipinitialspace=True)]
    def column(key,dtype=float):return np.asarray([r[key] for r in rows],dtype=dtype)
    mx,my=Transformer.from_crs(mars,crs,always_xy=True).transform(column('LONG_EAST'),column('LAT_NORTH'))
    cols=(np.asarray(mx)-transform.c)/20-0.5
    rr=(np.asarray(my)-transform.f)/(-20)-0.5
    elev=column('TOPOGRAPHY');orbit=column('ORBIT',np.int64)
    keep=np.isfinite(rr)&np.isfinite(cols)&np.isfinite(elev)&(column('A',int)==0)&(rr>=1)&(rr<502)&(cols>=1)&(cols<502)
    keep &= base.sample_raster(valid.astype(float),np.clip(rr,0,503),np.clip(cols,0,503))>.999
    rr,cols,elev,orbit=[a[keep] for a in (rr,cols,elev,orbit)]
    unique=np.unique(orbit)
    assim=np.isin(orbit,unique[::2]);test=np.isin(orbit,unique[1::2])
    if assim.sum()<20 or test.sum()<20:
        raise ValueError(f'Insufficient orbit-separated MOLA points {assim.sum()}/{test.sum()}')
    ci=np.flatnonzero(assim)
    xy=np.stack([rr[ci],cols[ci]],1)
    order=[int(np.argmin(np.sum((xy-251.5)**2,axis=1)))]
    distance=np.full(len(xy),np.inf)
    while len(order)<20:
        distance=np.minimum(distance,np.sum((xy-xy[order[-1]])**2,axis=1));distance[order]=-1
        order.append(int(np.argmax(distance)))
    chosen=ci[order]
    assert not(set(orbit[chosen]) & set(orbit[test]))
    save_npz(directory/'inference.npz',z0=coarse,image=image_fill,valid=valid,
             row=rr[chosen].astype(np.float32),column=cols[chosen].astype(np.float32),elevation=elev[chosen].astype(np.float32),orbit=orbit[chosen])
    save_npz(directory/'reference.npz',dem=reference,valid=common,native_ctx=native_fill,
             heldout_row=rr[test],heldout_column=cols[test],heldout_elevation=elev[test],heldout_orbit=orbit[test])
    provenance=dict(region=region,longitude=lon%360,latitude=lat,hirise=h,ctx=c,
            terrain_description=h['properties'].get('description'), input_ctx=ctx_info,input_image=image_info,reference_hirise=ref_info,
            output_transform=list(transform)[:6],output_crs=crs.to_wkt(),mola_query_bbox=box,
            assimilation_count=int(assim.sum()),heldout_count=int(test.sum()),assimilation_orbits=unique[::2].tolist(),heldout_orbits=unique[1::2].tolist(),
            selected_orbits=orbit[chosen].tolist(),selected_original_filtered_indices=chosen.tolist(),
            valid_reference_pixels=int(common.sum()),lowpass_support_pixels=int(support.sum()),
            input_valid_fraction=float(valid.mean()),shared_acquisition_orbits=sorted(h_orbits & set(re.findall(r'\d{6}',c['id']))),
            datum='Published areoid-referenced HiRISE and geoid-adjusted/controlled USGS CTX; no target-fitted registration. Manual provenance audit still required.',
            independence='Cross-product stereo reference, not proven MOLA-independent control',metadata_errors=errors,
            protocol_sha256=sha(ROOT/'protocol.json'))
    write_json(directory/'provenance.json',provenance)
    write_json(directory/'ready.json',dict(state='prepared_pending_audit',region=region,
            files={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='ready.json'},
            reference_used_for_input=False,heldout_used_for_inference=False))
    print('PREPARED',region,flush=True)


def infer(region):
    directory=ROOT/'cases'/region
    receipt=ROOT/'predictions'/region/'receipt.json'
    if receipt.exists():
        print('INFER_ALREADY_DONE',region,flush=True);return
    ready=json.loads((directory/'ready.json').read_text())
    assert sha(directory/'inference.npz')==ready['files']['inference.npz']
    a=read_npz(directory/'inference.npz')
    assert 'dem' not in a and not any(k.startswith('heldout') for k in a)
    case=base.LoadedCase(region,region,'prospective-external',a)
    context=np.arange(20,dtype=np.int64)
    gp=base.build_gp_fields(case,context,lengthscale_px=original.PRIOR_LENGTHSCALE_PX,noise_m=original.PRIOR_NOISE_M)
    shape=a['z0'].shape
    no=(a['z0'].copy(),np.full(shape,base.HEIGHT_SCALE_M,np.float32),np.full(shape,400,np.float32))
    records=[]
    def save(method,seed,budget,field,stage='final',checkpoint=None):
        assert np.isfinite(field).all()
        path=receipt.parent/f'{method}-{seed}-k{budget}-{stage}.npz'
        save_npz(path,mean=field.astype(np.float32))
        records.append(dict(method=method,seed=seed,budget=budget,stage=stage,path=str(path.relative_to(ROOT)),sha256=sha(path),checkpoint_sha256=checkpoint))
    save('coarse_CTX',0,0,a['z0'])
    shift=np.median(a['elevation']-base.sample_raster(a['z0'],a['row'],a['column']))
    save('median_offset',0,20,a['z0']+shift)
    save('GP',0,20,gp[0])
    torch,_,_=base.torch_components()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    base.DISABLE_TERMINAL=False
    original.v35.LOCAL_NOISE_M=0.0
    for variant in PROTOCOL['models']:
        for seed in PROTOCOL['seeds']:
            cp=OLD/'training'/variant/f'seed-{seed}/stage2/final.pt'
            manifest=json.loads((cp.parent/'training-manifest.json').read_text())
            assert manifest['state']=='complete' and sha(cp)==manifest['final_checkpoint_sha256']
            model=make_variant(variant).cuda().eval()
            checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
            assert checkpoint['variant']==variant
            model.load_state_dict(checkpoint['model'],strict=True)
            base.DISABLE_TERMINAL=False
            k0=base.predict_from_fields(model,case,*no,np.empty(0,dtype=np.int64),torch.device('cuda:0'))
            base.DISABLE_TERMINAL=True
            raw=base.predict_from_fields(model,case,*gp,context,torch.device('cuda:0'))
            base.DISABLE_TERMINAL=False
            k20=base.predict_from_fields(model,case,*gp,context,torch.device('cuda:0'))
            low,_=original.v34.project_lowpass_shape(k20,k0,a['valid'],20.0)
            final,_=original.v35.impose_local_laser_consistency(low,context,a)
            save(variant,seed,0,k0,checkpoint=sha(cp))
            save(variant,seed,20,raw,'raw',sha(cp))
            save(variant,seed,20,k20,'terminal',sha(cp))
            save(variant,seed,20,final,checkpoint=sha(cp))
            print('INFER',region,variant,seed,flush=True)
            del model,checkpoint;torch.cuda.empty_cache()
    write_json(receipt,dict(state='predictions_frozen',region=region,records=records,
               inference_sha256=sha(directory/'inference.npz'),protocol_sha256=sha(ROOT/'protocol.json'),
               reference_file_opened=False,heldout_heights_opened=False,created_unix=time.time()))


def rmse(a):return float(np.sqrt(np.mean(np.asarray(a,dtype=float)**2))) if len(a) else None


def score(region):
    directory=ROOT/'cases'/region
    receipt=ROOT/'predictions'/region/'receipt.json'
    meta=json.loads(receipt.read_text())
    a=read_npz(directory/'reference.npz');inp=read_npz(directory/'inference.npz')
    ready=json.loads((directory/'ready.json').read_text())
    assert sha(directory/'reference.npz')==ready['files']['reference.npz']
    assert not set(inp['orbit']) & set(a['heldout_orbit'])
    valid=a['valid'];reference=a['dem']
    rows=[];fields={}
    for r in meta['records']:
        path=ROOT/r['path'];assert sha(path)==r['sha256']
        fields[(r['method'],r['seed'],r['budget'],r['stage'])]=read_npz(path)['mean']
    fields[('native_CTX_20m',0,0,'reference_input_diagnostic')]=a['native_ctx']
    for (method,seed,budget,stage),field in fields.items():
        error=field[valid]-reference[valid];bias=float(np.median(error))
        metrics=dict(absolute_hirise_rmse_m=rmse(error),median_datum_bias_m=bias,bias_removed_rmse_m=rmse(error-bias))
        detail=[]
        for scale in (40,80,160):
            p,s=detail_field(field,valid,pixel_size_m=20,scale_m=scale)
            q,t=detail_field(reference,valid,pixel_size_m=20,scale_m=scale)
            e=rmse((p-q)[s&t]);metrics[f'detail_rmse_{scale}m']=e;detail.append(e)
        metrics['mean_detail_rmse_m']=float(np.mean(detail))
        pl,ps=masked_gaussian_smooth(field,valid,sigma_pixels=16)
        rl,rs=masked_gaussian_smooth(reference,valid,sigma_pixels=16)
        metrics['lowpass_rmse_m']=rmse((pl-rl)[ps&rs])
        anchor=fields.get((method,seed,0,'final'),inp['z0'])
        al,_=masked_gaussian_smooth(anchor,valid,sigma_pixels=16)
        metrics['lowpass_change_vs_k0_m']=rmse((pl-al)[ps&rs])
        he=base.sample_raster(field,a['heldout_row'],a['heldout_column'])-a['heldout_elevation']
        intersect=base.sample_raster(valid.astype(float),a['heldout_row'],a['heldout_column'])>.999
        metrics['heldout_mola_rmse_m']=rmse(he)
        metrics['heldout_mola_hirise_intersection_rmse_m']=rmse(he[intersect])
        rows.append(dict(region=region,method=method,seed=seed,budget=budget,stage=stage,
                    heldout_points=len(he),heldout_intersection_points=int(intersect.sum()),**metrics))
    write_json(ROOT/'results'/f'{region}.json',dict(state='scored_pending_independent_audit',rows=rows,
                receipt_sha256=sha(receipt),protocol_sha256=sha(ROOT/'protocol.json'),independent_regions=1))
    print('SCORED',region,len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','score']);parser.add_argument('region')
    args=parser.parse_args();globals()[args.stage](args.region)
