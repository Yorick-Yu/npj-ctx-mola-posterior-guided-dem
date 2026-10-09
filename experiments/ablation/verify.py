#!/usr/bin/env python3
"""Verify immutable inputs, model interventions, and GPU GP arithmetic."""
import json
import platform
import sys
import time
from runtime import ROOT, VARIANTS, ORIGINAL_FACTORY, base, make_variant, paths, sha, write_json
import numpy as np
import scipy
import run_sparse_detail_v30 as v30
import run_sparse_detail_v31 as v31
from paper2dem.sparse_detail_runtime import PRIOR_LENGTHSCALE_PX, PRIOR_NOISE_M


def main():
    started = time.time()
    manifest = json.loads((ROOT / 'input_manifest.json').read_text())
    for record in manifest['files']:
        path = paths()['project_root'] / record['path']
        if sha(path) != record['sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
    print('INPUT_HASHES_PASSED', len(manifest['files']), flush=True)
    torch, nn, _ = base.torch_components()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    assert np.array_equal(torch.from_numpy(np.ones(3)).cuda().cpu().numpy(), np.ones(3))
    torch.manual_seed(919)
    reference = ORIGINAL_FACTORY().cuda()
    state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    condition = torch.randn(2, 10, 32, 32, device='cuda')
    condition[:, 3] = torch.rand_like(condition[:, 3]) * 4
    condition_copy = condition.clone()
    tested = []
    for variant in VARIANTS:
        torch.manual_seed(919)
        model = make_variant(variant).cuda()
        assert all(torch.equal(value.cpu(), state[key]) for key, value in model.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
        captured = []
        hook = model.velocity.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach().clone()))
        output, kinetic = model(condition)
        assert torch.isfinite(output).all() and torch.isfinite(kinetic)
        (output.square().mean() + .005 * kinetic).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert len(captured) == (1 if variant == 'plain_unet' else 3)
        for actual in captured:
            expected = condition.clone()
            if variant in ('no_uncertainty', 'plain_unet'):
                expected[:, 3] = 0
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
        measured, _ = model(condition)
        expected_gate = .2 + .8 * torch.clamp(condition[:, 3:4] / 2, 0, 1)
        expected = expected_gate if variant in ('full', 'no_uncertainty') else torch.ones_like(expected_gate)
        torch.testing.assert_close(measured, expected, atol=2e-7, rtol=1e-6)
        assert torch.equal(condition, condition_copy)
        tested.append(variant)
        print('VARIANT_PASSED', variant, flush=True)
        del model
    del reference
    torch.cuda.empty_cache()
    runtime_cases = v30.load_training_cases(paths()['project_root'], paths()['training_index'])
    plans = v30.load_plans(paths()['plans'], {'train'})
    assert len(runtime_cases) == 163
    assert set(plans) == {c.tile_id for c in runtime_cases}
    external_manifest = json.loads((paths()['project_root'] / 'data/sparse-detail-external-inference-v36/cases-manifest.json').read_text())
    external_ids = {r['tile_id'] for r in external_manifest['cases']}
    assert len(external_ids) == 6 and not external_ids & set(plans)
    for case in runtime_cases:
        assert not np.any(case.arrays['assimilation'] & case.arrays['evaluation'])
    comparisons = []
    for index in (0, 81, 162):
        case = runtime_cases[index]
        for selector in ('uniform-farthest', 'terrain-faopt'):
            for budget in (5, 20, 80):
                context = plans[case.tile_id][selector][:budget]
                cpu = base.build_gp_fields(case, context, PRIOR_LENGTHSCALE_PX, PRIOR_NOISE_M)
                gpu = v31.torch_gp_fields(case, context, device=torch.device('cuda:0'))
                differences = [float(np.max(np.abs(a - b))) for a, b in zip(cpu, gpu)]
                assert differences[0] <= 1e-3 and differences[1] <= 1e-4 and differences[2] == 0, differences
                comparisons.append(dict(tile=case.tile_id, selector=selector, budget=budget,
                                        max_abs_mean_m=differences[0], max_abs_std_m=differences[1],
                                        max_abs_distance_px=differences[2]))
        print('GP_PASSED', case.tile_id, flush=True)
    write_json(ROOT / 'verification.json', dict(
        state='passed', checked_files=len(manifest['files']), checked_training_cases=163,
        variants=tested, full_exact_parity=True, independent_interventions=True,
        matched_initial_weights=True, finite_forward_and_gradients=True,
        training_external_tiles_disjoint=True, training_orbit_masks_disjoint=True,
        gp_comparisons=comparisons, gp_tolerance_mean_m=.001, gp_tolerance_std_m=.0001,
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                         scipy=scipy.__version__, gpu=torch.cuda.get_device_name(0),
                         platform=platform.platform()),
        input_manifest_sha256=sha(ROOT / 'input_manifest.json'),
        runtime_sha256=sha(ROOT / 'runtime.py'), protocol_sha256=sha(ROOT / 'protocol.json'),
        elapsed_seconds=time.time() - started))
    print('VERIFICATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
