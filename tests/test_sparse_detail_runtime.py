from pathlib import Path

import numpy as np

from paper2dem.sparse_detail_runtime import (
    DETAIL_SELECTORS,
    SparseDetailCase,
    apply_terrain_prior,
    build_selector_orders,
    fit_terrain_prior,
)


def synthetic_case(tile_id: str = "tile") -> SparseDetailCase:
    row_grid, column_grid = np.indices((504, 504), dtype=np.float32)
    z0 = 0.02 * row_grid + 0.03 * column_grid
    dem = z0 + np.sin(row_grid / 13.0) * np.cos(column_grid / 17.0)
    image = dem + 0.1 * np.sin(column_grid / 5.0)
    row = np.linspace(10, 493, 90, dtype=np.float32)
    column = np.mod(np.arange(90) * 47, 480).astype(np.float32) + 12
    elevation = dem[np.rint(row).astype(int), np.rint(column).astype(int)]
    assimilation = np.zeros(90, dtype=bool)
    assimilation[:82] = True
    evaluation = ~assimilation
    return SparseDetailCase(
        tile_id,
        "region",
        "train",
        Path(f"{tile_id}.npz"),
        {
            "image": image.astype(np.float32),
            "z0": z0.astype(np.float32),
            "dem": dem.astype(np.float32),
            "valid": np.ones((504, 504), dtype=bool),
            "row": row,
            "column": column,
            "elevation": elevation.astype(np.float32),
            "orbit": np.arange(90, dtype=np.int64),
            "assimilation": assimilation,
            "evaluation": evaluation,
        },
    )


def test_prior_is_finite_and_selector_orders_are_nested() -> None:
    case = synthetic_case()
    prior = fit_terrain_prior([case], samples_per_tile=128)
    field = apply_terrain_prior(case, prior)
    assert field.shape == (504, 504)
    assert np.all(np.isfinite(field))
    orders = build_selector_orders(case, prior, maximum_budget=12)
    for selector in DETAIL_SELECTORS:
        order = orders[selector]
        assert order.shape == (12,)
        assert len(set(order.tolist())) == 12
        assert np.all(case.arrays["assimilation"][order])


def test_prior_records_no_validation_or_external_targets() -> None:
    prior = fit_terrain_prior([synthetic_case()], samples_per_tile=64)
    assert prior["state"] == "frozen_from_development_training_only"
    assert prior["validation_or_external_targets_used"] is False
