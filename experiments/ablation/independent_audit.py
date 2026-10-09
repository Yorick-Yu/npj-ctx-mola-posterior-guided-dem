#!/usr/bin/env python3
"""Post-run integrity, isolation, intervention and aggregation checks; CPU only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
sys.path.insert(0, str(root))
import runtime
import numpy as np
import run_frozen_external_v36 as original
import run_sparse_detail_v30 as v30


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def progress(message):
    print(message, flush=True)


protocol_hash = sha(root / 'protocol.json')
assert protocol_hash == '0fa2c1e523119ba8a60d62d78751890c4803afcf23d468b6bd4ba8e71b21b372'
protocol = read(root / 'protocol.json')
variants = protocol['training_variants']
seeds = protocol['seeds']
inputs = read(root / 'input_manifest.json')
assert len(inputs['files']) == 854
for item in inputs['files']:
    path = root / 'project' / item['path']
    assert path.stat().st_size == item['size']
    assert sha(path) == item['sha256'], str(path)
progress('INPUT_HASHES 854/854')

torch, nn, _ = runtime.base.torch_components()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
stages = []
streams = defaultdict(set)
loss_records = 0
checkpoints = {}
for variant, seed, stage in itertools.product(variants, seeds, (1, 2)):
    folder = root / 'training' / variant / f'seed-{seed}' / f'stage{stage}'
    manifest = read(folder / 'training-manifest.json')
    current = read(folder / 'progress.json')
    expected_steps = 2000 if stage == 1 else 1000
    assert manifest['state'] == 'complete'
    assert manifest['variant'] == variant and manifest['seed'] == seed
    assert manifest['stage'] == stage and not manifest['smoke_only']
    assert manifest['completed_updates'] == expected_steps == current['update']
    assert manifest['training_tile_count'] == 163
    assert not manifest['validation_targets_read'] and not manifest['external_targets_read']
    assert manifest['protocol_sha256'] == protocol_hash
    assert manifest['adapter_sha256'] == sha(root / 'runtime.py')
    assert manifest['driver_sha256'] == sha(root / 'train.py')
    assert manifest['final_checkpoint_sha256'] == sha(folder / 'final.pt')
    if stage == 2:
        assert manifest['initial_checkpoint_sha256'] == sha(folder.parent / 'stage1/final.pt')
    checkpoint = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['update'] == expected_steps
    assert checkpoint['variant'] == variant and checkpoint['seed'] == seed
    assert checkpoint['sample_stream_sha256'] == manifest['sample_stream_sha256']
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    del checkpoint
    streams[(seed, stage)].add(manifest['sample_stream_sha256'])
    log = (root / 'logs' / f'{variant}-seed-{seed}-stage{stage}.log').read_text()
    assert 'Traceback (most recent call last)' not in log
    observed_updates = []
    losses = []
    for line in log.splitlines():
        if not line.startswith('{'):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'loss' not in entry:
            continue
        assert all(math.isfinite(value) for value in entry.values() if isinstance(value, (int, float)))
        observed_updates.append(entry['update'])
        losses.append(entry['loss'])
    assert observed_updates[-1] == expected_steps
    assert observed_updates == sorted(set(observed_updates))
    loss_records += len(losses)
    stages.append(dict(variant=variant, seed=seed, stage=stage, updates=expected_steps,
                       final_loss=current['loss'], min_logged_loss=min(losses),
                       max_logged_loss=max(losses), checkpoint_sha256=manifest['final_checkpoint_sha256']))
    checkpoints[(variant, seed, stage)] = manifest['final_checkpoint_sha256']
assert len(stages) == 24 and all(len(values) == 1 for values in streams.values())
progress('TRAINING 24/24 stages; matched sample streams; finite checkpoints and logs')

# Repeat switch checks on CPU without claiming these are evaluation experiments.
torch.manual_seed(919)
reference = runtime.ORIGINAL_FACTORY()
initial = {key: value.clone() for key, value in reference.state_dict().items()}
condition = torch.randn(2, 10, 32, 32)
condition[:, 3] = torch.rand_like(condition[:, 3]) * 4
unchanged = condition.clone()
for variant in variants:
    torch.manual_seed(919)
    model = runtime.make_variant(variant)
    assert all(torch.equal(initial[key], value) for key, value in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
    captured = []
    hook = model.velocity.register_forward_pre_hook(lambda _, value: captured.append(value[0].detach().clone()))
    output, kinetic = model(condition)
    (output.square().mean() + .005 * kinetic).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(captured) == (1 if variant == 'plain_unet' else 3)
    expected = condition.clone()
    if variant in ('no_uncertainty', 'plain_unet'):
        expected[:, 3] = 0
    for actual in captured:
        assert torch.equal(actual[:, :10], expected)
    if variant == 'full':
        wanted = reference(condition)
        torch.testing.assert_close(output, wanted[0], rtol=0, atol=0)
        torch.testing.assert_close(kinetic, wanted[1], rtol=0, atol=0)
    hook.remove()
    class ConstantVelocity(nn.Module):
        def forward(self, value):
            return torch.ones_like(value[:, :1])
    model.velocity = ConstantVelocity()
    actual, _ = model(condition)
    wanted = .2 + .8 * torch.clamp(condition[:, 3:4] / 2, 0, 1)
    if variant in ('no_gate', 'plain_unet'):
        wanted = torch.ones_like(wanted)
    torch.testing.assert_close(actual, wanted, atol=2e-7, rtol=1e-6)
    assert torch.equal(condition, unchanged)
del model, reference
progress('SWITCHES CPU independent checks passed')

project = root / 'project'
training = v30.load_training_cases(project, runtime.paths()['training_index'])
train_plans = v30.load_plans(runtime.paths()['plans'], {'train'})
assert len(training) == 163 and {c.tile_id for c in training} == set(train_plans)
for case in training:
    assert not np.any(case.arrays['assimilation'] & case.arrays['evaluation'])
isolated = original.load_inference_cases(project / 'data/sparse-detail-external-inference-v36')
cases, _ = original.load_full_cases(project / 'data/sparse-detail-external-cases-v36')
plans = original.load_plans(project / 'runs/sparse-detail-v36-exact-laser-projection/external-selector-plans')
assert len(cases) == 6 and set(cases) == set(isolated) == set(plans)
assert not set(cases) & set(train_plans)
tiles = sorted(cases)
selectors = original.DETAIL_SELECTORS
for tile, case in isolated.items():
    assert not {'dem', 'evaluation', 'elevation'} & set(case.arrays)
    arrays = cases[tile]['arrays']
    assert float(cases[tile]['pixel_size_m']) == 20.0
    indices = case.arrays['candidate_global_index']
    assert np.all(arrays['assimilation'][indices]) and not np.any(arrays['evaluation'][indices])
    np.testing.assert_array_equal(case.arrays['candidate_elevation'], arrays['elevation'][indices])
    assert not np.any(arrays['assimilation'] & arrays['evaluation'])
progress('ISOLATION 163 training / 6 external; withheld samples absent from inference input')

receipt = read(root / 'predictions/receipt.json')
assert receipt['count'] == 576 and receipt['protocol_sha256'] == protocol_hash
assert not receipt['reference_dem_read'] and not receipt['heldout_mola_read']
records = {(r['variant'], r['seed'], r['tile_id'], r['selector']): r for r in receipt['predictions']}
expected_keys = set(itertools.product(variants, seeds, tiles, ('shared-k000', *selectors)))
assert len(records) == len(receipt['predictions']) == 576 and set(records) == expected_keys
for key, record in records.items():
    variant, seed, tile, selector = key
    path = root / record['path']
    assert sha(path) == record['sha256']
    assert record['checkpoint_sha256'] == checkpoints[(variant, seed, 2)]
    with np.load(path, allow_pickle=False) as data:
        assert data['raw'].shape == data['terminal'].shape == cases[tile]['arrays']['z0'].shape
        assert np.isfinite(data['raw']).all() and np.isfinite(data['terminal']).all()
        context = data['context_global_indices']
        expected = [] if selector == 'shared-k000' else plans[tile][selector][:20]
        np.testing.assert_array_equal(context, expected)
        if selector == 'shared-k000':
            np.testing.assert_array_equal(data['raw'], data['terminal'])
progress('PREDICTIONS 576/576 hashes, shapes, contexts and finite values passed')

results = root / 'results'
summary = read(results / 'summary.json')
rows = read(results / 'raw-metrics.json')['rows']
assert len(rows) == summary['raw_row_count'] == 1962
assert summary['protocol_sha256'] == protocol_hash
assert summary['prediction_receipt_sha256'] == sha(root / 'predictions/receipt.json')
metrics = list(original.METRICS)
all_metrics = metrics + ['context_rmse_m']
indexed = {(r['variant'], r['seed'], r['tile_id'], r['selector'], r['stage']): r for r in rows}
assert len(indexed) == len(rows)
for row in rows:
    assert all(np.isfinite(row[m]) for m in metrics)
    assert row['family'] == ('random' if row['selector'].startswith('random-') else row['selector'])
final = [r for r in rows if r['stage'] == ('projected' if r['variant'] in variants else 'final')]
assert len(final) == summary['final_row_count'] == 756
all_variants = variants + protocol['inference_only_variants']
families = sorted({r['family'] for r in final})
assert len(families) == 5
family_values = defaultdict(list)
for row in final:
    family_values[(row['variant'], row['seed'], row['tile_id'], row['family'])].append(row)
assert set(family_values) == set(itertools.product(all_variants, seeds, tiles, families))
recomputed_family = {}
for key, members in family_values.items():
    assert len(members) == (3 if key[-1] == 'random' else 1)
    recomputed_family[key] = {m: float(np.mean([r[m] for r in members])) for m in all_metrics}
recomputed_tile = {}
for variant, tile in itertools.product(all_variants, tiles):
    recomputed_tile[(variant, tile)] = {m: float(np.mean([
        recomputed_family[(variant, seed, tile, family)][m] for seed in seeds for family in families
    ])) for m in all_metrics}
for row in read(results / 'per-family.json')['rows']:
    expected = recomputed_family[tuple(row[k] for k in ('variant', 'seed', 'tile_id', 'family'))]
    for metric in all_metrics:
        close(row[metric], expected[metric])
for row in read(results / 'per-tile.json')['rows']:
    for metric in all_metrics:
        close(row[metric], recomputed_tile[(row['variant'], row['tile_id'])][metric])
for row in summary['final_summary']:
    for metric in all_metrics:
        close(row[metric], np.mean([recomputed_tile[(row['variant'], tile)][metric] for tile in tiles]))
for row in summary['paired_differences']:
    differences = np.array([recomputed_tile[(row['variant'], tile)][row['metric']] -
                            recomputed_tile[('full', tile)][row['metric']] for tile in tiles])
    close(row['ablation_minus_full'], differences.mean())
    indices = np.random.default_rng(20260811).integers(0, 6, size=(20000, 6))
    close(row['bootstrap95'], np.quantile(differences[indices].mean(axis=1), [.025, .975]))
    assert row['independent_tiles'] == 6 and row['full_better_tiles'] == int(np.sum(differences > 0))
stage_groups = defaultdict(list)
for row in rows:
    stage_groups[(row['variant'], row['stage'], row['seed'], row['tile_id'], row['family'])].append(row)
for reported in read(results / 'stage-summary.json')['rows']:
    for metric in metrics:
        selected = [np.mean([r[metric] for r in group]) for key, group in stage_groups.items()
                    if key[:2] == (reported['variant'], reported['stage'])]
        close(reported[metric], np.mean(selected))
progress('STATISTICS all aggregate levels and 25 paired tile-bootstrap intervals reproduced')

# Recompute metric samples across all six tiles, all trained variants and all seeds.
calibration = original.DetailThresholdCalibration.from_dict(read(
    project / 'runs/sparse-detail-v30-formal/calibration/detail-thresholds.json'))
original.v35.LOCAL_NOISE_M = 0.0
samples = set(itertools.product(variants, seeds, [tiles[0]], ['terrain-faopt']))
samples.update(('full', seeds[0], tile, 'uniform-farthest') for tile in tiles)
checked_metric_rows = 0
for variant, seed, tile, selector in sorted(samples):
    with np.load(root / records[(variant, seed, tile, selector)]['path']) as data:
        raw = data['raw'].astype(np.float64)
        terminal = data['terminal'].astype(np.float64)
        context = data['context_global_indices'].astype(np.int64)
    with np.load(root / records[(variant, seed, tile, 'shared-k000')]['path']) as data:
        k0 = data['terminal'].astype(np.float64)
    arrays = cases[tile]['arrays']
    low, _ = original.v34.project_lowpass_shape(terminal, k0, arrays['valid'], cases[tile]['pixel_size_m'])
    projected, _ = original.v35.impose_local_laser_consistency(low, context, arrays)
    fields = [(variant, 'raw', raw), (variant, 'terminal', terminal), (variant, 'projected', projected)]
    if variant == 'full':
        low, _ = original.v34.project_lowpass_shape(raw, k0, arrays['valid'], cases[tile]['pixel_size_m'])
        without, _ = original.v35.impose_local_laser_consistency(low, context, arrays)
        fields.extend([('no_projection', 'final', terminal), ('no_terminal', 'before_projection', raw),
                       ('no_terminal', 'final', without)])
    for label, stage, field in fields:
        expected = indexed[(label, seed, tile, selector, stage)]
        actual = original.diagnostic.evaluate_prediction(field, cases[tile], calibration)
        residual = runtime.base.sample_raster(field, arrays['row'][context], arrays['column'][context]) - arrays['elevation'][context]
        actual['context_rmse_m'] = float(np.sqrt(np.mean(residual ** 2)))
        for metric in all_metrics:
            close(actual[metric], expected[metric])
        checked_metric_rows += 1
progress(f'METRIC_RECOMPUTATION {checked_metric_rows} rows across {len(samples)} prediction cases')

payload = dict(state='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
    protocol_sha256=protocol_hash, frozen_input_count=854, training_models=12,
    training_stages=24, training_updates_total=36000, finite_loss_records=loss_records,
    checked_predictions=576, checked_raw_rows=1962, checked_final_rows=756,
    independently_recomputed_metric_rows=checked_metric_rows,
    independently_recomputed_prediction_cases=len(samples),
    independent_tile_bootstrap_units=6, matched_sample_streams=True,
    cpu_switch_checks_passed=True, matched_initial_weights=True,
    training_external_tiles_disjoint=True, withheld_observations_absent_from_inference=True,
    checkpoint_chain_and_hashes_passed=True, aggregate_recalculation_passed=True,
    stages=stages, result_file_hashes={p.name: sha(p) for p in sorted(results.iterdir()) if p.is_file()},
    script_hashes={name: sha(root / name) for name in ('runtime.py', 'train.py', 'evaluate.py', 'verify.py')},
    caveats=[
        'Historical point observation only; finite-area footprint question remains unresolved.',
        'Six reused external tiles; not a fresh cross-source validation.',
        'Reference-aligned image-detail metrics and absolute heldout-laser metrics differ.',
        'Full model does not outperform every ablation; no result-based retuning performed.',
        'Metric recomputation sampled 18 prediction cases; all stored metric rows and aggregates audited.'
    ])
runtime.write_json(root / 'independent-audit.json', payload)
progress(json.dumps({key: value for key, value in payload.items() if key != 'stages'}, ensure_ascii=False))
