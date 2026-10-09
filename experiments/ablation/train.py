#!/usr/bin/env python3
"""Run the archived 2000-update stage followed by 1000-update fine-tuning."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from runtime import ROOT, VARIANTS, base, make_variant, paths, sha, write_json
import run_sparse_detail_v30 as v30
import run_sparse_detail_v31 as v31
import run_sparse_detail_v32 as v32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch, _, _ = base.torch_components()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = ROOT / ('smoke' if args.smoke else 'training') / args.variant / f'seed-{args.seed}' / f'stage{args.stage}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    receipt = ROOT / 'verification.json'
    if json.loads(receipt.read_text())['state'] != 'passed':
        raise RuntimeError('Preflight verification has not passed')
    base.make_model = lambda base_channels=32, steps=3, minimum_mobility=.2: make_variant(
        args.variant, base_channels, steps, minimum_mobility)
    metadata = dict(variant=args.variant, stage=args.stage,
                    protocol_sha256=sha(ROOT / 'protocol.json'),
                    adapter_sha256=sha(ROOT / 'runtime.py'),
                    driver_sha256=sha(Path(__file__)), smoke_only=args.smoke)
    sample_stream = hashlib.sha256()
    original_crop = v30.crop_sample

    def crop(case, fields, crop_size, rng):
        sample_stream.update(json.dumps([case.tile_id, rng.bit_generator.state,
                                         fields[3].tolist()], sort_keys=True).encode())
        return original_crop(case, fields, crop_size, rng)

    v30.crop_sample = crop
    original_atomic = v30.atomic_json

    def atomic(path, payload):
        if output in Path(path).parents:
            payload = {**payload, **metadata, 'pid': os.getpid(), 'updated_unix': time.time(),
                       'sample_stream_sha256': sample_stream.hexdigest()}
        return original_atomic(path, payload)

    v30.atomic_json = atomic
    original_save = torch.save

    def save(obj, path, *a, **kw):
        if isinstance(obj, dict) and 'model' in obj:
            obj = {**obj, **metadata, 'sample_stream_sha256': sample_stream.hexdigest()}
            if not all(torch.isfinite(value).all() for value in obj['model'].values()):
                raise FloatingPointError('Nonfinite checkpoint parameter')
        return original_save(obj, path, *a, **kw)

    torch.save = save
    common = dict(**paths(), output=output, device='cuda:0', seed=args.seed,
                  updates=2 if args.smoke else (2000 if args.stage == 1 else 1000),
                  batch_size=2, crop_size=320, base_channels=32, transport_steps=3,
                  minimum_mobility=.20, learning_rate=2e-4 if args.stage == 1 else 5e-5,
                  field_cache_size=64, log_interval=1 if args.smoke else 25,
                  snapshot_update=[1] if args.smoke else [250, 500, 750, 1000],
                  field_equivalence_receipt=receipt)
    if args.stage == 1:
        result = v31.command_train(argparse.Namespace(**common))
    else:
        initial = output.parent / 'stage1/final.pt'
        checkpoint = torch.load(initial, map_location='cpu', weights_only=False)
        for key in ('variant', 'protocol_sha256', 'smoke_only'):
            if checkpoint[key] != metadata[key]:
                raise ValueError(f'Initial checkpoint mismatch: {key}')
        if checkpoint['update'] != (2 if args.smoke else 2000):
            raise ValueError('Initial checkpoint did not complete stage1')
        del checkpoint
        result = v32.command_train(argparse.Namespace(**common, initial_checkpoint=initial,
                                                      lowpass_weight=2.0, detail_weight=0.5))
    if result != 0:
        raise RuntimeError(f'Training returned {result}')


if __name__ == '__main__':
    main()
