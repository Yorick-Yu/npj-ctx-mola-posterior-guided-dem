#!/usr/bin/env python3
"""Target-isolated inference, then frozen six-tile scoring; never tune here."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
from runtime import ROOT, PROJECT, VARIANTS, SEEDS, base, make_variant, sha, write_json
import numpy as np
import run_frozen_external_v36 as original

RUN = PROJECT / 'runs/sparse-detail-v36-exact-laser-projection'
PRED = ROOT / 'predictions'
RESULT = ROOT / 'results'


def validate_training():
    records = []
    for seed in SEEDS:
        for stage in (1, 2):
            streams = set()
            for variant in VARIANTS:
                directory = ROOT / 'training' / variant / f'seed-{seed}' / f'stage{stage}'
                manifest = json.loads((directory / 'training-manifest.json').read_text())
                assert manifest['state'] == 'complete'
                assert manifest['variant'] == variant and manifest['seed'] == seed
                assert manifest['completed_updates'] == (2000 if stage == 1 else 1000)
                assert manifest['protocol_sha256'] == sha(ROOT / 'protocol.json')
                assert manifest['final_checkpoint_sha256'] == sha(directory / 'final.pt')
                assert manifest['validation_targets_read'] is False
                assert manifest['external_targets_read'] is False
                streams.add(manifest['sample_stream_sha256'])
                records.append(manifest)
            assert len(streams) == 1, f'Training sample sequence differs: seed {seed} stage {stage}'
    return records


def terminal_correction(raw, case, context):
    rows = case.arrays['row'][context]
    cols = case.arrays['column'][context]
    remaining = case.arrays['elevation'][context] - base.sample_raster(raw, rows, cols)
    gr, gc = np.indices(raw.shape, dtype=np.float32)
    correction, _ = base.gp_predict(rows, cols, remaining, gr.ravel(), gc.ravel(),
                                    lengthscale_px=base.TERMINAL_LENGTHSCALE_PX, noise_m=8.0)
    return raw + correction.reshape(raw.shape).astype(np.float32)


def infer():
    validate_training()
    if (PRED / 'receipt.json').exists():
        raise FileExistsError('Completed inference exists; verify it instead of rerunning')
    torch, _, _ = base.torch_components()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cases = original.load_inference_cases(PROJECT / 'data/sparse-detail-external-inference-v36')
    plans = original.load_plans(RUN / 'external-selector-plans')
    assert len(cases) == 6
    inputs = []
    for tile_id in sorted(cases):
        isolated = cases[tile_id]
        assert not {'dem', 'evaluation', 'elevation'} & set(isolated.arrays)
        current = original.model_case(isolated)
        shape = current.arrays['z0'].shape
        empty = np.empty(0, dtype=np.int64)
        fields0 = (current.arrays['z0'].astype(np.float32).copy(),
                   np.full(shape, base.HEIGHT_SCALE_M, dtype=np.float32),
                   np.full(shape, 400.0, dtype=np.float32))
        inputs.append((tile_id, 'shared-k000', current, empty, empty, fields0))
        for selector in original.DETAIL_SELECTORS:
            global_context = plans[tile_id][selector][:20]
            context = original.remap(isolated, global_context)
            fields = base.build_gp_fields(current, context,
                lengthscale_px=original.PRIOR_LENGTHSCALE_PX, noise_m=original.PRIOR_NOISE_M)
            inputs.append((tile_id, selector, current, context, global_context, fields))
    records = []
    parity_checked = False
    for seed in SEEDS:
        for variant in VARIANTS:
            checkpoint_path = ROOT / 'training' / variant / f'seed-{seed}/stage2/final.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert checkpoint['variant'] == variant
            model = make_variant(variant).cuda().eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            checkpoint_sha = sha(checkpoint_path)
            for tile_id, selector, case, context, global_context, fields in inputs:
                base.DISABLE_TERMINAL = True
                raw = base.predict_from_fields(model, case, *fields, context, torch.device('cuda:0')).astype(np.float32)
                terminal = terminal_correction(raw, case, context).astype(np.float32)
                if variant == 'full' and selector != 'shared-k000' and not parity_checked:
                    base.DISABLE_TERMINAL = False
                    expected = base.predict_from_fields(model, case, *fields, context, torch.device('cuda:0'))
                    np.testing.assert_array_equal(terminal, expected)
                    parity_checked = True
                assert np.isfinite(raw).all() and np.isfinite(terminal).all()
                path = PRED / variant / f'seed-{seed}' / selector / f'{tile_id}.npz'
                if path.exists():
                    raise FileExistsError(f'Refusing to replace prediction {path}')
                original.atomic_npz(path, raw=raw, terminal=terminal,
                                    context_global_indices=global_context)
                records.append(dict(variant=variant, seed=seed, tile_id=tile_id,
                    selector=selector, path=str(path.relative_to(ROOT)), sha256=sha(path),
                    checkpoint_sha256=checkpoint_sha))
                write_json(ROOT / 'inference-progress.json', dict(completed=len(records), total=576,
                           variant=variant, seed=seed, tile=tile_id, updated_unix=time.time()))
                print('INFER', len(records), '/576', variant, seed, tile_id, selector, flush=True)
            del model, checkpoint
            torch.cuda.empty_cache()
    assert len(records) == 576 and parity_checked
    write_json(PRED / 'receipt.json', dict(state='complete', predictions=records,
               count=576, terminal_parity_verified=True, reference_dem_read=False,
               heldout_mola_read=False, protocol_sha256=sha(ROOT / 'protocol.json')))


def collapse(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), **{m: float(np.mean([r[m] for r in group])) for m in metrics}}
            for key, group in sorted(groups.items())]


def score():
    validate_training()
    if (RESULT / 'summary.json').exists():
        raise FileExistsError('Results exist; do not overwrite them')
    receipt = json.loads((PRED / 'receipt.json').read_text())
    assert receipt['count'] == 576 and receipt['protocol_sha256'] == sha(ROOT / 'protocol.json')
    records = {(r['variant'], r['seed'], r['tile_id'], r['selector']): r for r in receipt['predictions']}
    cases, manifest = original.load_full_cases(PROJECT / 'data/sparse-detail-external-cases-v36')
    calibration = original.DetailThresholdCalibration.from_dict(json.loads(
        (PROJECT / 'runs/sparse-detail-v30-formal/calibration/detail-thresholds.json').read_text()))
    original.v35.LOCAL_NOISE_M = 0.0
    rows = []

    def load(key):
        record = records[key]
        path = ROOT / record['path']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as archive:
            return {k: np.asarray(archive[k]) for k in archive.files}

    def add(variant, seed, tile, selector, stage, field, context):
        case = cases[tile]
        metrics = original.diagnostic.evaluate_prediction(field, case, calibration)
        assert all(np.isfinite(v) for v in metrics.values())
        arrays = case['arrays']
        context_rmse = None
        if len(context):
            residual = base.sample_raster(field, arrays['row'][context], arrays['column'][context]) - arrays['elevation'][context]
            context_rmse = float(np.sqrt(np.mean(residual ** 2)))
        rows.append(dict(variant=variant, seed=seed, tile_id=tile, selector=selector,
                         family='random' if selector.startswith('random-') else selector,
                         stage=stage, context_rmse_m=context_rmse, **metrics))

    for seed in SEEDS:
        for tile in sorted(cases):
            for variant in VARIANTS:
                k0 = load((variant, seed, tile, 'shared-k000'))['terminal'].astype(np.float64)
                add(variant, seed, tile, 'shared-k000', 'k0', k0, [])
                for selector in original.DETAIL_SELECTORS:
                    data = load((variant, seed, tile, selector))
                    raw, terminal = data['raw'].astype(np.float64), data['terminal'].astype(np.float64)
                    context = data['context_global_indices'].astype(np.int64)
                    arrays = cases[tile]['arrays']
                    assert len(context) == 20 and np.all(arrays['assimilation'][context])
                    assert not np.any(arrays['evaluation'][context])
                    low, _ = original.v34.project_lowpass_shape(terminal, k0, arrays['valid'], 20.0)
                    final, _ = original.v35.impose_local_laser_consistency(low, context, arrays)
                    for stage, field in [('raw', raw), ('terminal', terminal), ('projected', final)]:
                        add(variant, seed, tile, selector, stage, field, context)
                    if variant == 'full':
                        add('no_projection', seed, tile, selector, 'final', terminal, context)
                        low_without, _ = original.v34.project_lowpass_shape(raw, k0, arrays['valid'], 20.0)
                        without, _ = original.v35.impose_local_laser_consistency(low_without, context, arrays)
                        add('no_terminal', seed, tile, selector, 'before_projection', raw, context)
                        add('no_terminal', seed, tile, selector, 'final', without, context)
                    print('SCORE', variant, seed, tile, selector, flush=True)
    metrics = list(original.METRICS)
    final_rows = [r for r in rows if r['stage'] == ('projected' if r['variant'] in VARIANTS else 'final')]
    assert len(final_rows) == 6 * 3 * 6 * 7
    aggregate_metrics = metrics + ['context_rmse_m']
    family = collapse(final_rows, ['variant', 'seed', 'tile_id', 'family'], aggregate_metrics)
    tile = collapse(family, ['variant', 'tile_id'], aggregate_metrics)
    summary = collapse(tile, ['variant'], aggregate_metrics)
    full = {r['tile_id']: r for r in tile if r['variant'] == 'full'}
    paired = []
    for variant in list(VARIANTS[1:]) + ['no_terminal', 'no_projection']:
        part = sorted([r for r in tile if r['variant'] == variant], key=lambda r: r['tile_id'])
        for metric in aggregate_metrics:
            differences = np.asarray([r[metric] - full[r['tile_id']][metric] for r in part])
            paired.append(dict(variant=variant, metric=metric,
                ablation_minus_full=float(differences.mean()),
                bootstrap95=original.diagnostic.bootstrap(differences),
                full_better_tiles=int(np.sum(differences > 0)), independent_tiles=6))
    RESULT.mkdir(exist_ok=True)
    stage_family = collapse(rows, ['variant', 'stage', 'seed', 'tile_id', 'family'], metrics)
    stage_tile = collapse(stage_family, ['variant', 'stage', 'tile_id'], metrics)
    stage_summary = collapse(stage_tile, ['variant', 'stage'], metrics)
    write_json(RESULT / 'raw-metrics.json', {'rows': rows})
    write_json(RESULT / 'stage-summary.json', {'rows': stage_summary})
    write_json(RESULT / 'per-family.json', {'rows': family})
    write_json(RESULT / 'per-tile.json', {'rows': tile})
    write_json(RESULT / 'summary.json', dict(state='complete_pending_independent_audit',
               final_summary=summary, paired_differences=paired, tile_count=6,
               raw_row_count=len(rows), final_row_count=len(final_rows),
               protocol_sha256=sha(ROOT / 'protocol.json'),
               prediction_receipt_sha256=sha(PRED / 'receipt.json'),
               observation_operator='point-bilinear, NOT finite-area integration',
               caveat='Ablation on reused archived external tiles, fixed a priori settings; no new cross-source DEM evaluation.'))
    with (RESULT / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('infer', 'score'))
    args = parser.parse_args()
    infer() if args.command == 'infer' else score()
