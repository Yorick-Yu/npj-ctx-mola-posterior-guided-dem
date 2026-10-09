#!/usr/bin/env python3
"""Variant-only adapters around the archived deterministic implementation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'
sys.path[:0] = [str(PROJECT / 'code/scripts'), str(PROJECT / 'code/src')]
import npj_real_pipeline_v3 as base

ORIGINAL_FACTORY = base.make_model
VARIANTS = ('full', 'no_uncertainty', 'no_gate', 'plain_unet')
SEEDS = (20260808, 20260809, 20260810)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def make_variant(variant, base_channels=32, steps=3, minimum_mobility=0.20):
    if variant not in VARIANTS:
        raise ValueError(variant)
    original = ORIGINAL_FACTORY(base_channels, steps, minimum_mobility)
    if variant == 'full':
        return original
    torch, nn, _ = base.torch_components()

    class AblatedRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity = original.velocity
            self.steps = steps

        def forward(self, condition):
            state = torch.zeros_like(condition[:, :1])
            # Gate and network conditioning are ablated independently.
            mobility = minimum_mobility + (1.0 - minimum_mobility) * torch.clamp(
                condition[:, 3:4] / 2.0, 0.0, 1.0)
            network_condition = condition
            if variant in ('no_uncertainty', 'plain_unet'):
                network_condition = condition.clone()
                network_condition[:, 3:4] = 0
            if variant == 'plain_unet':
                output = self.velocity(torch.cat((network_condition, state,
                                                 torch.full_like(state, 0.5)), dim=1))
                return output, torch.mean(output ** 2)
            if variant == 'no_gate':
                mobility = torch.ones_like(mobility)
            kinetic = torch.zeros((), device=condition.device)
            for step in range(self.steps):
                time_channel = torch.full_like(state, (step + 0.5) / self.steps)
                velocity = self.velocity(torch.cat((network_condition, state, time_channel), dim=1))
                state = state + mobility * velocity / self.steps
                kinetic = kinetic + torch.mean(velocity ** 2) / self.steps
            return state, kinetic

    return AblatedRefiner()


def paths():
    return dict(project_root=PROJECT,
                training_index=PROJECT / 'runs/sparse-detail-v30-preparation-20260810-v2/expanded-training-index.json',
                plans=PROJECT / 'runs/sparse-detail-v30-formal/selector-plans')
