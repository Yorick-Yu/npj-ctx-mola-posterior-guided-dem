"""Prospective preparation; never loads a new-site model prediction."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent
SITE = ROOT.parent / 'npj_r2c7_site_selection_20260925'
ARCHIVE = ROOT.parents[1] / '实验数据归档/npj_服务器完整备份_20260922/mars-npj-real-v1'
ABLATION = ROOT.parent / 'npj_r2c3_ablation_20260923'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fill(z, valid):
    indices = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(z[tuple(indices)], dtype=np.float32)


def quality_rgb(product, window, out):
    path = out / 'fom-rgb-5m.npz'
    url = product['assets']['fom']['href'].replace('s3-us-west-2.amazonaws.com', 's3.us-west-2.amazonaws.com')
    if path.exists():
        return np.load(path)['rgb']
    t = Affine(*window['transform']) * Affine.scale(.25, .25)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='90',
                      GDAL_HTTP_MAX_RETRY='1', GDAL_HTTP_MULTIRANGE='SERIAL',
                      GDAL_HTTP_MULTIPLEX='NO', GDAL_CACHEMAX=128):
        with rasterio.open(url) as ds:
            assert ds.count == 3
            with WarpedVRT(ds, crs=window['crs'], transform=t, width=2016, height=2016,
                           resampling=Resampling.nearest, nodata=0) as vrt:
                rgb = vrt.read([1, 2, 3])
    np.savez_compressed(path, rgb=rgb)
    write(out / 'fom-source.json', dict(url=url, resampling='nearest', spacing_map_m=5,
                                      classification='RGB green dominance, not calibrated numerical confidence',
                                      sha256=sha(path)))
    return rgb


def main():
    p = json.loads((ROOT / 'protocol.json').read_text())
    if (ROOT / 'prepared.json').exists():
        old = json.loads((ROOT / 'prepared.json').read_text())
        assert old['protocol_sha256'] == sha(ROOT / 'protocol.json')
        for name, digest in old['files'].items():
            assert sha(ROOT / name) == digest, name
        print('ALREADY_PREPARED', flush=True)
        return
    freeze = json.loads((ARCHIVE / 'runs/sparse-detail-v36-exact-laser-projection/frozen-candidate-evaluator.json').read_text())
    for rec in freeze['code']:
        src = ARCHIVE / rec['path'].split('mars-npj-real-v1/', 1)[1]
        assert sha(src) == rec['sha256'], src
    for source in (ARCHIVE / 'code').rglob('*'):
        if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
            continue
        destination = ROOT / 'project/code' / source.relative_to(ARCHIVE / 'code')
        if destination.exists():
            assert sha(destination) == sha(source), destination
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    shutil.copy2(ABLATION / 'runtime.py', ROOT / 'runtime.py')
    checkpoints = []
    for rec in freeze['checkpoints']:
        src = ARCHIVE / rec['path'].split('mars-npj-real-v1/', 1)[1]
        assert sha(src) == rec['sha256']
        dst = ROOT / 'models' / f"original-{rec['seed']}.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        checkpoints.append(dict(method='original_frozen_v36', seed=rec['seed'], path=str(dst.relative_to(ROOT)),
                                sha256=sha(dst), source=rec['path']))
    for seed in p['seeds']:
        src = ABLATION / f'remote_snapshot/training/plain_unet/seed-{seed}/stage2/final.pt'
        manifest = json.loads((src.parent / 'training-manifest.json').read_text())
        assert manifest['state'] == 'complete' and sha(src) == manifest['final_checkpoint_sha256']
        dst = ROOT / 'models' / f'plain-unet-{seed}.pt'
        shutil.copy2(src, dst)
        shutil.copy2(src.parent / 'training-manifest.json', dst.with_suffix('.manifest.json'))
        checkpoints.append(dict(method='r2c3_plain_unet', seed=seed, path=str(dst.relative_to(ROOT)),
                                sha256=sha(dst), source=str(src)))
    write(ROOT / 'models/manifest.json', checkpoints)
    for region in p['regions']:
        print('PREPARE', region, flush=True)
        src = SITE / 'cases' / region
        out = ROOT / 'cases' / region
        out.mkdir(parents=True, exist_ok=True)
        available = json.loads((src / 'availability.json').read_text())
        assert available['criteria_passed']
        assert sha(src / 'availability_inputs.npz') == available['files']['availability_inputs.npz']
        for name in ['availability.json', 'quality-audit.json', 'prospective-window.json', 'ctx-stac.json',
                     'hirise-stac.json', 'mola.csv', 'mola.lbl', 'mola-query-url.json']:
            shutil.copy2(src / name, out / name)
        a = np.load(src / 'availability_inputs.npz')
        window = json.loads((out / 'prospective-window.json').read_text())
        h = json.loads((out / 'hirise-stac.json').read_text())
        rgb = quality_rgb(h, window, out).astype(np.int16)
        r, g, b = rgb
        good = (g >= r + 20) & (g >= b + 20) & (g >= 40)
        fraction = good.reshape(504, 4, 504, 4).mean(axis=(1, 3))
        image = a['ctx_image']
        shadow_cut = float(np.nanpercentile(image[a['input_valid']], 2))
        geometry = a['reference_valid'] & a['input_valid']
        strict = geometry & (fraction >= .9) & (image > shadow_cut)
        dx, dy = available['approximate_ground_spacing_m']
        counts = {}
        for scale in p['physical_scales_m']:
            nx = max(3, int(round(scale / dx)) | 1)
            ny = max(3, int(round(scale / dy)) | 1)
            counts[str(scale)] = int((strict & (ndi.uniform_filter(strict.astype(float), size=(ny,nx), mode='constant')>=.6)).sum())
        ci = np.flatnonzero(a['input_pool'])
        xy = np.stack([a['mola_row'][ci], a['mola_column'][ci]], axis=1)
        selected = [int(np.argmin(np.sum((xy - 251.5) ** 2, axis=1)))]
        distance = np.full(len(xy), np.inf)
        while len(selected) < 20:
            distance = np.minimum(distance, np.sum((xy - xy[selected[-1]]) ** 2, axis=1))
            distance[selected] = -1
            selected.append(int(np.argmax(distance)))
        ids = ci[selected]
        native = fill(a['ctx_dem'], a['input_valid'])
        coarse = ndi.zoom(native.reshape(21,24,21,24).mean(axis=(1,3)), 24, order=3).astype(np.float32)
        assert coarse.shape == (504,504)
        np.savez_compressed(out / 'inference.npz', z0=coarse, image=fill(image, a['input_valid']),
                            valid=a['input_valid'], row=a['mola_row'][ids].astype(np.float32),
                            column=a['mola_column'][ids].astype(np.float32),
                            elevation=a['mola_elevation'][ids].astype(np.float32), orbit=a['mola_orbit'][ids])
        test = a['heldout']
        assert not set(a['mola_orbit'][ids]) & set(a['mola_orbit'][test])
        np.savez_compressed(out / 'reference.npz', dem=a['hirise_dem'], native_ctx=native,
                            geometry=geometry, strict=strict, fom_green_fraction=fraction,
                            heldout_row=a['mola_row'][test], heldout_column=a['mola_column'][test],
                            heldout_elevation=a['mola_elevation'][test], heldout_orbit=a['mola_orbit'][test])
        metadata = dict(region=region, dx_m=dx, dy_m=dy, center=available['center'],
                        shadow_cut=shadow_cut, geometry_pixels=int(geometry.sum()),
                        strict_pixels=int(strict.sum()), complete_window_pixels=counts,
                        selected_pool_ids=ids.tolist(), selected_orbits=np.unique(a['mola_orbit'][ids]).tolist(),
                        heldout_orbits=np.unique(a['mola_orbit'][test]).tolist(),
                        transform=window['transform'], crs=window['crs'],
                        source_sha256=sha(src / 'availability_inputs.npz'),
                        inference_sha256=sha(out / 'inference.npz'), reference_sha256=sha(out / 'reference.npz'))
        write(out / 'prepared.json', metadata)
        print('QUALITY_SUPPORT', region, metadata['strict_pixels'], counts, flush=True)
        if counts['100'] < 10000:
            raise RuntimeError(f'{region}: strict physical quality support <10000; no model inference authorized')
    shutil.copy2(SITE / 'selection-audit.json', ROOT / 'site-selection-audit.json')
    files = {str(path.relative_to(ROOT)): sha(path) for directory in ['models','cases','project']
             for path in sorted((ROOT / directory).rglob('*')) if path.is_file()}
    files['runtime.py'] = sha(ROOT / 'runtime.py')
    write(ROOT / 'prepared.json', dict(state='prepared', protocol_sha256=sha(ROOT/'protocol.json'),
                                      files=files, reference_input_separated=True))


if __name__ == '__main__':
    main()
