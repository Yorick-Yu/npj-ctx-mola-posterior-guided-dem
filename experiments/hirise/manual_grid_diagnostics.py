"""Estimate grid correspondence from old/new prediction pairs, without references."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "local-audit-deps"))
import json
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

SEEDS = (20260808, 20260809, 20260810)
INCOMING = ROOT / "incoming_dem_20260924"
OUT = INCOMING / "manual_alignment"


def load_pairs(region):
    pairs = []
    for seed in SEEDS:
        with np.load(INCOMING / "predictions" / f"{region.title()}_{seed}.npz", allow_pickle=False) as z:
            new = z["dem"].astype(float)
        with np.load(ROOT / "remote_snapshot" / "predictions" / region / f"full-{seed}-k20-final.npz", allow_pickle=False) as z:
            old = z["mean"].astype(float)
        pairs.append((old, new))
    return pairs


def pair_metrics(old, new, params, stride=1):
    # Model: old_row = sr*new_row + br; old_col = sc*new_col + bc.
    rr, cc = np.mgrid[12:489:stride, 12:489:stride]
    sr, sc, br, bc = params
    sample = map_coordinates(old, [sr * rr + br, sc * cc + bc], order=1, mode="constant", cval=np.nan)
    target = new[rr, cc]
    valid = np.isfinite(target) & np.isfinite(sample)
    error = target[valid] - sample[valid]
    offset = np.median(error)
    centered = error - offset
    return dict(raw_rmse_m=float(np.sqrt(np.mean(error ** 2))), median_difference_m=float(offset),
                centered_rmse_m=float(np.sqrt(np.mean(centered ** 2))),
                correlation=float(np.corrcoef(sample[valid], target[valid])[0, 1]),
                robust_shape_score=float(np.mean(np.minimum(centered ** 2, 100.0))),
                sample_count=int(valid.sum()))


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    for region in ("candor", "hellespontus"):
        pairs = load_pairs(region)
        candidates = [(f"crop_r{r}_c{c}", [1., 1., float(r), float(c)]) for r in range(4) for c in range(4)]
        candidates += [("full_center_extent", [503/500, 503/500, 0., 0.]),
                       ("full_pixel_extent", [504/501, 504/501, (504/501-1)/2, (504/501-1)/2])]
        ranked = []
        for name, params in candidates:
            values = [pair_metrics(old, new, params, stride=3) for old, new in pairs]
            ranked.append(dict(name=name, params=params, mean_shape_score=float(np.mean([v["robust_shape_score"] for v in values])), per_seed=values))
        ranked.sort(key=lambda r: r["mean_shape_score"])
        def fit(subset, start):
            def loss(params):
                return float(np.mean([pair_metrics(old, new, params, stride=6)["robust_shape_score"] for old, new in subset]))
            opt = minimize(loss, start, method="Powell", bounds=[(.995, 1.015), (.995, 1.015), (-4., 5.), (-4., 5.)],
                           options={"maxiter":100, "xtol":1e-7, "ftol":1e-8})
            return dict(params=opt.x.tolist(), loss=float(opt.fun), success=bool(opt.success))
        shared = fit(pairs, ranked[0]["params"])
        individual = [dict(seed=seed, **fit([pair], shared["params"])) for seed, pair in zip(SEEDS, pairs)]
        scores = [dict(seed=seed, **pair_metrics(old, new, shared["params"])) for seed, (old, new) in zip(SEEDS, pairs)]
        results[region] = dict(fixed_candidates=ranked, estimated_shared_mapping=shared,
                               per_seed_estimates=individual, full_interior_scores=scores)
        print(json.dumps(dict(region=region, top_fixed=[(r["name"], r["mean_shape_score"]) for r in ranked[:4]],
                              shared=shared, individual=individual, scores=scores), ensure_ascii=False), flush=True)
    report = dict(status="estimated_not_original_georeferencing", mapping="old_index = scale * incoming_index + offset",
                  fit_data="Archived full K20 predictions and incoming DEMs only; no HiRISE or MOLA reference used",
                  loss="Per-seed median-centered height residual, squared and capped at 100 m^2; equal seed mean",
                  vertical_changes_applied=False, reference_or_target_metric_used_for_fit=False,
                  caution="Correspondence with old predictions cannot establish how incoming data were generated.", regions=results)
    (OUT / "grid-diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
