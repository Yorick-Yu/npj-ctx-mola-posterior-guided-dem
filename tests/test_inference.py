from __future__ import annotations

import numpy as np

from paper2dem.inference import (
    OnlineMoments,
    infer_path_ensemble,
    infer_path_ensemble_batched,
    infer_paths,
    patch_slices,
    stitch_patches,
)


def test_formal_patch_counts() -> None:
    assert len(patch_slices((512, 512))) == 9
    assert len(patch_slices((2048, 2048))) == 121


def test_stitch_constant_patches_is_exact() -> None:
    layout = patch_slices((512, 512))
    patches = [np.full((256, 256), 7.0) for _ in layout]
    result = stitch_patches(patches, layout, (512, 512))
    np.testing.assert_allclose(result, 7.0)


def test_dc_runs_after_one_full_tile_stitch_per_path() -> None:
    calls: list[tuple[int, int]] = []

    def predictor(location, path_index):
        return np.full((256, 256), float(path_index))

    def dc(tile):
        calls.append(tile.shape)
        return tile + 1

    mean, variance = infer_paths(
        tile_shape=(512, 512), path_count=3, patch_predictor=predictor, terminal_dc=dc
    )
    assert calls == [(512, 512)] * 3
    np.testing.assert_allclose(mean, 2.0)
    np.testing.assert_allclose(variance, 1.0)


def test_online_welford() -> None:
    moments = OnlineMoments()
    for value in (1.0, 2.0, 3.0):
        moments.update(np.asarray([value]))
    np.testing.assert_allclose(moments.mean, [2.0])
    np.testing.assert_allclose(moments.variance, [1.0])


def test_retained_path_stack_is_ordered_and_matches_ensemble() -> None:
    result = infer_path_ensemble(
        tile_shape=(256, 256),
        path_count=4,
        patch_predictor=lambda _location, index: np.full(
            (256, 256), index + 1, dtype=np.float32
        ),
        terminal_dc=lambda tile: tile,
        retain_path_stack=True,
    )
    assert result.path_stack is not None
    assert result.path_stack.shape == (4, 256, 256)
    np.testing.assert_allclose(result.path_stack[:, 0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(result.mean, 2.5)
    np.testing.assert_allclose(result.variance, 5.0 / 3.0)


def test_batched_paths_preserve_global_order_and_full_tile_dc() -> None:
    dc_values: list[float] = []

    def predictor(_location, indices):
        return np.stack(
            [np.full((256, 256), index, dtype=np.float32) for index in indices]
        )

    def dc(tile):
        dc_values.append(float(tile[0, 0]))
        return tile + 10

    result = infer_path_ensemble_batched(
        tile_shape=(256, 256),
        path_count=5,
        path_batch_size=2,
        patch_predictor=predictor,
        terminal_dc=dc,
        retain_path_stack=True,
    )
    assert dc_values == [0, 1, 2, 3, 4]
    assert result.path_stack is not None
    np.testing.assert_array_equal(result.path_stack[:, 0, 0], [10, 11, 12, 13, 14])
    np.testing.assert_allclose(result.mean, 12)
    np.testing.assert_allclose(result.variance, 2.5)
