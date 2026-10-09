"""Sparse physical laser-footprint operators on affine DEM grids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.stats import qmc


FloatArray = NDArray[np.float64]


class FootprintSupportError(ValueError):
    """Raised when a footprint does not have the requested grid support."""


@dataclass(frozen=True)
class FootprintBuildResult:
    """Sparse operator plus the input-row provenance after support checks."""

    operator: sparse.csr_matrix
    kept_indices: NDArray[np.int64]
    rejected_indices: NDArray[np.int64]
    row_sums: FloatArray


def _affine_coefficients(transform: object) -> tuple[float, float, float, float, float, float]:
    if all(hasattr(transform, name) for name in ("a", "b", "c", "d", "e", "f")):
        values = tuple(float(getattr(transform, name)) for name in ("a", "b", "c", "d", "e", "f"))
    else:
        array = np.asarray(transform, dtype=np.float64)
        if array.shape == (6,):
            values = tuple(float(value) for value in array)
        elif array.shape == (2, 3) or array.shape == (3, 3):
            values = (
                float(array[0, 0]),
                float(array[0, 1]),
                float(array[0, 2]),
                float(array[1, 0]),
                float(array[1, 1]),
                float(array[1, 2]),
            )
        else:
            raise ValueError("transform must be an Affine-like object, six coefficients, or 2x3/3x3")
    if not np.all(np.isfinite(values)):
        raise ValueError("transform coefficients must be finite")
    determinant = values[0] * values[4] - values[1] * values[3]
    if abs(determinant) <= np.finfo(np.float64).eps:
        raise ValueError("transform is singular")
    return values


def world_to_pixel(points_xy: ArrayLike, transform: object) -> FloatArray:
    """Convert world ``(x, y)`` points to fractional ``(row, col)`` indices."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xy must contain only finite values")
    a, b, c, d, e, f = _affine_coefficients(transform)
    linear = np.asarray(((a, b), (d, e)), dtype=np.float64)
    shifted = points - np.asarray((c, f), dtype=np.float64)
    # Raster transforms locate pixel corners.  Paper 1's immutable numerical
    # contract represents grid samples at pixel centres, hence the half-pixel
    # shift in both forward and inverse mappings.
    col_row = np.linalg.solve(linear, shifted.T).T - 0.5
    return np.column_stack((col_row[:, 1], col_row[:, 0]))


def disk_quadrature_offsets(radius_m: float, quadrature_points: int = 32) -> FloatArray:
    """Return Paper 1's deterministic equal-area Sobol disk quadrature."""

    if not np.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("radius_m must be finite and non-negative")
    if not isinstance(quadrature_points, (int, np.integer)) or quadrature_points <= 0:
        raise ValueError("quadrature_points must be a positive integer")
    if radius_m == 0.0:
        return np.zeros((1, 2), dtype=np.float64)
    exponent = int(math.ceil(math.log2(quadrature_points)))
    unit = qmc.Sobol(d=2, scramble=False).random_base2(exponent)[:quadrature_points]
    radii = float(radius_m) * np.sqrt(unit[:, 0])
    angles = 2.0 * np.pi * unit[:, 1]
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def _bilinear_contributions(
    row: float,
    col: float,
    grid_shape: tuple[int, int],
    valid_mask: NDArray[np.bool_] | None,
) -> list[tuple[int, float]] | None:
    rows, cols = grid_shape
    # Match Paper 1's ``coordinate_inside_mask`` and
    # ``map_coordinates(..., mode='nearest')`` conventions.  Grid samples are
    # at integer centre indices and the outer pixel areas extend half a pixel.
    rounded_row = int(np.floor(row + 0.5))
    rounded_col = int(np.floor(col + 0.5))
    if rounded_row < 0 or rounded_row >= rows:
        return None
    if rounded_col < 0 or rounded_col >= cols:
        return None
    if valid_mask is not None and not bool(valid_mask[rounded_row, rounded_col]):
        return None
    tolerance = 64.0 * np.finfo(np.float64).eps
    row = float(np.clip(row, 0.0, rows - 1.0))
    col = float(np.clip(col, 0.0, cols - 1.0))
    row0 = int(np.floor(row))
    col0 = int(np.floor(col))
    row1 = min(row0 + 1, rows - 1)
    col1 = min(col0 + 1, cols - 1)
    dr = row - row0
    dc = col - col0
    entries = (
        (row0, col0, (1.0 - dr) * (1.0 - dc)),
        (row0, col1, (1.0 - dr) * dc),
        (row1, col0, dr * (1.0 - dc)),
        (row1, col1, dr * dc),
    )
    contributions: list[tuple[int, float]] = []
    for pixel_row, pixel_col, weight in entries:
        if weight <= tolerance:
            continue
        contributions.append((pixel_row * cols + pixel_col, float(weight)))
    return contributions


def build_footprint_operator_with_metadata(
    centers_xy: ArrayLike,
    *,
    grid_shape: Sequence[int],
    transform: object,
    diameter_m: float | ArrayLike,
    quadrature_points: int = 32,
    valid_mask: ArrayLike | None = None,
    require_full_support: bool = True,
    drop_invalid: bool = False,
) -> FootprintBuildResult:
    """Build a normalized CSR footprint-average operator.

    World-coordinate disk quadrature is transformed to the target grid, then
    bilinearly interpolated.  If ``require_full_support`` is false, quadrature
    samples outside the valid grid are omitted and the remaining row is
    renormalized.  A footprint with no remaining support is invalid.
    """

    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain (rows, cols)")
    shape = (int(grid_shape[0]), int(grid_shape[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("grid_shape entries must be positive")
    centers = np.asarray(centers_xy, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers[None, :]
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_xy must have shape (K, 2)")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers_xy must contain only finite values")

    diameters = np.asarray(diameter_m, dtype=np.float64)
    try:
        diameters = np.broadcast_to(diameters, (centers.shape[0],))
    except ValueError as exc:
        raise ValueError("diameter_m must be scalar or have one value per footprint") from exc
    if not np.all(np.isfinite(diameters)) or np.any(diameters < 0.0):
        raise ValueError("diameters must be finite and non-negative")

    mask: NDArray[np.bool_] | None
    if valid_mask is None:
        mask = None
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("valid_mask must have grid_shape")

    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    kept: list[int] = []
    rejected: list[int] = []
    for footprint_index, (center, diameter) in enumerate(zip(centers, diameters, strict=True)):
        offsets = disk_quadrature_offsets(float(diameter) / 2.0, quadrature_points)
        sample_pixels = world_to_pixel(center[None, :] + offsets, transform)
        accumulated: dict[int, float] = {}
        invalid_sample = False
        for row, col in sample_pixels:
            contributions = _bilinear_contributions(float(row), float(col), shape, mask)
            if contributions is None:
                invalid_sample = True
                if require_full_support:
                    break
                continue
            for flat_index, weight in contributions:
                accumulated[flat_index] = accumulated.get(flat_index, 0.0) + weight

        if (require_full_support and invalid_sample) or not accumulated:
            rejected.append(footprint_index)
            if not drop_invalid:
                reason = "does not have full support" if invalid_sample else "has no valid support"
                raise FootprintSupportError(f"footprint {footprint_index} {reason}")
            continue

        total_weight = float(sum(accumulated.values()))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise FootprintSupportError(f"footprint {footprint_index} has non-positive support")
        for flat_index in sorted(accumulated):
            indices.append(flat_index)
            data.append(accumulated[flat_index] / total_weight)
        kept.append(footprint_index)
        indptr.append(len(indices))

    operator = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(kept), shape[0] * shape[1]),
        dtype=np.float64,
    )
    operator.sum_duplicates()
    operator.sort_indices()
    row_sums = np.asarray(operator.sum(axis=1), dtype=np.float64).reshape(-1)
    if row_sums.size and not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("internal error: footprint rows are not normalized")
    return FootprintBuildResult(
        operator=operator,
        kept_indices=np.asarray(kept, dtype=np.int64),
        rejected_indices=np.asarray(rejected, dtype=np.int64),
        row_sums=row_sums,
    )


def build_footprint_operator(
    centers_xy: ArrayLike,
    *,
    grid_shape: Sequence[int],
    transform: object,
    diameter_m: float | ArrayLike,
    quadrature_points: int = 32,
    valid_mask: ArrayLike | None = None,
    require_full_support: bool = True,
) -> sparse.csr_matrix:
    """Build a CSR operator, raising if any requested footprint is invalid."""

    return build_footprint_operator_with_metadata(
        centers_xy,
        grid_shape=grid_shape,
        transform=transform,
        diameter_m=diameter_m,
        quadrature_points=quadrature_points,
        valid_mask=valid_mask,
        require_full_support=require_full_support,
        drop_invalid=False,
    ).operator


def project_footprints(operator: sparse.spmatrix, dem: ArrayLike) -> FloatArray:
    """Project a flattened or two-dimensional DEM through a footprint operator."""

    matrix = sparse.csr_matrix(operator, dtype=np.float64)
    values = np.asarray(dem, dtype=np.float64).reshape(-1)
    if matrix.shape[1] != values.size:
        raise ValueError("operator column count does not match DEM size")
    if not np.all(np.isfinite(values)):
        raise ValueError("dem must contain only finite values")
    return np.asarray(matrix @ values, dtype=np.float64).reshape(-1)
