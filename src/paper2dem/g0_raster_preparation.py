from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any
import math
import os
import re
import subprocess
import tempfile

import numpy as np
from scipy import ndimage

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)


RASTER_PREPARATION_SCHEMA = "paper2.g0.ctx-raster-preparation/v2"
MOLA_COARSE_CUBE_SCHEMA = "paper2.g0.mola-coarse-cube/v1"
COARSE_DEM_LINEAGE_SCHEMA = "paper2.g0.mola-coarse-dem-lineage/v1"
COARSE_DEM_ALGORITHM = "mola-megt-reproject-to-problem-grid-v1"
COREGISTRATION_LINEAGE_SCHEMA = "paper2.g0.ctx-coregistration-lineage/v2"
COREGISTRATION_CONTROLS_SCHEMA = "paper2.g0.ctx-coregistration-controls/v2"
COREGISTRATION_ALGORITHM = (
    "ctx-coreg-aux-global-translation-phase-correlation-grid-v3"
)
COREGISTRATION_SOURCE = "coreg_aux"
COREGISTRATION_MODE = "global_translation_then_residual_audit"
COREGISTRATION_TRANSFORM_FAMILY = "translation_2d_only"
MIN_COREGISTRATION_ERROR_COUNT = 4
FROZEN_ISIS_VERSION = "10.0.0"

RasterReader = Callable[[Path], "RasterBand"]
RasterReprojector = Callable[["RasterBand", "ProblemGrid", str], tuple[np.ndarray, np.ndarray]]
CommandRunner = Callable[[tuple[str, ...], Path], Mapping[str, Any]]


@dataclass(frozen=True)
class ProblemGrid:
    product_id: str
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]
    crs_wkt: str
    contract: Mapping[str, Any]

    def validate(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError("problem grid shape must contain two positive dimensions")
        if len(self.transform) != 6 or not np.all(np.isfinite(self.transform)):
            raise ValueError("problem grid transform must contain six finite coefficients")
        determinant = self.transform[0] * self.transform[4] - self.transform[1] * self.transform[3]
        if abs(determinant) <= np.finfo(np.float64).eps:
            raise ValueError("problem grid transform is singular")
        if not self.crs_wkt.strip():
            raise ValueError("problem grid CRS cannot be empty")


@dataclass(frozen=True)
class RasterBand:
    values: np.ndarray
    valid_mask: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs_wkt: str
    driver: str

    def validate(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_mask)
        if values.ndim != 2 or values.size == 0 or mask.shape != values.shape:
            raise ValueError("raster band and validity mask must share one non-empty 2-D shape")
        if mask.dtype != np.bool_:
            raise ValueError("raster validity mask must be boolean")
        if len(self.transform) != 6 or not np.all(np.isfinite(self.transform)):
            raise ValueError("source raster transform must contain six finite coefficients")
        if not self.crs_wkt.strip():
            raise ValueError("source raster lacks a usable CRS")


@dataclass(frozen=True)
class CoregistrationConfig:
    control_grid_rows: int = 4
    control_grid_columns: int = 4
    maximum_shift_pixels: int = 8
    highpass_sigma_pixels: float = 1.5
    minimum_valid_fraction: float = 0.70
    minimum_texture_standard_deviation: float = 1.0e-6
    minimum_peak_ratio: float = 1.02
    outlier_mad_multiplier: float = 4.0

    def validate(self) -> None:
        if self.control_grid_rows < 2 or self.control_grid_columns < 2:
            raise ValueError("coregistration requires at least a 2x2 independent control grid")
        if self.maximum_shift_pixels < 1:
            raise ValueError("maximum_shift_pixels must be positive")
        if self.highpass_sigma_pixels <= 0:
            raise ValueError("highpass_sigma_pixels must be positive")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must be in (0, 1]")
        if self.minimum_texture_standard_deviation <= 0:
            raise ValueError("minimum texture threshold must be positive")
        if self.minimum_peak_ratio <= 1:
            raise ValueError("minimum_peak_ratio must exceed one")
        if self.outlier_mad_multiplier <= 0:
            raise ValueError("outlier_mad_multiplier must be positive")


FROZEN_COREGISTRATION_CONFIG = CoregistrationConfig()


def _safe_identifier(value: str, *, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return value


def _relative_record(path: Path, *, paper2_root: Path) -> dict[str, Any]:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"required raster-preparation artifact is missing/empty: {resolved}")
    return {
        "relative_path": resolved.relative_to(paper2_root).as_posix(),
        "sha256": file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _array_bytes(values: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, np.asarray(values), allow_pickle=False)
    return stream.getvalue()


def _write_array(path: Path, values: np.ndarray, *, paper2_root: Path) -> Path:
    return write_immutable_bytes(path, _array_bytes(values), paper2_root=paper2_root)


def _grid_shape_from_archive(archive: Any) -> tuple[int, int]:
    declared: list[tuple[str, tuple[int, int]]] = []
    if "grid_shape" in archive.files:
        raw = np.asarray(archive["grid_shape"], dtype=np.int64).reshape(-1)
        if raw.shape != (2,) or np.any(raw < 1):
            raise ValueError("problem grid_shape must contain two positive integers")
        declared.append(("grid_shape", (int(raw[0]), int(raw[1]))))
    for name in ("coarse_dem", "Z0", "z0", "valid_mask", "image"):
        if name not in archive.files:
            continue
        values = np.asarray(archive[name])
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError(f"problem {name} must be a non-empty 2-D grid")
        declared.append((name, (int(values.shape[0]), int(values.shape[1]))))
    if not declared:
        raise ValueError(
            "problem does not expose a grid shape; add grid_shape or a 2-D coarse/valid array"
        )
    shapes = {shape for _, shape in declared}
    if len(shapes) != 1:
        raise ValueError(f"problem grid fields disagree on shape: {declared}")
    return next(iter(shapes))


def load_problem_grid(path: str | Path, *, product_id: str) -> ProblemGrid:
    problem = Path(path)
    if problem.suffix.lower() != ".npz":
        raise ValueError("problem grid contract must be a pickle-free NPZ")
    required = {
        "tile_id",
        "transform",
        "crs_wkt",
        "candidates",
        "candidate_ids",
        "candidate_track_ids",
    }
    with np.load(problem, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"problem is missing required grid/geometry fields: {missing}")
        forbidden = [
            name
            for name in archive.files
            if any(
                marker in name.lower()
                for marker in ("reference", "ground_truth", "truth_path", "target_dem")
            )
        ]
        if forbidden:
            raise ValueError(f"problem contains forbidden truth fields: {forbidden}")
        tile_id = str(archive["tile_id"].item())
        if tile_id != product_id:
            raise ValueError(f"problem tile_id differs from product_id: {tile_id}")
        shape = _grid_shape_from_archive(archive)
        transform_array = np.asarray(archive["transform"], dtype=np.float64).reshape(-1)
        if transform_array.shape != (6,):
            raise ValueError("problem transform must contain six coefficients")
        transform = tuple(float(value) for value in transform_array)
        crs_wkt = str(archive["crs_wkt"].item())
        candidates = np.asarray(archive["candidates"], dtype=np.float64)
        candidate_ids = np.asarray(archive["candidate_ids"]).astype(str)
        track_ids = np.asarray(archive["candidate_track_ids"]).astype(str)
        if candidates.ndim != 2 or candidates.shape[1] != 2 or not np.all(
            np.isfinite(candidates)
        ):
            raise ValueError("problem candidates must be a finite (n, 2) array")
        if candidate_ids.shape != (candidates.shape[0],) or track_ids.shape != candidate_ids.shape:
            raise ValueError("problem candidate identities do not align with geometry")
        if len(set(candidate_ids.tolist())) != len(candidate_ids):
            raise ValueError("problem candidate IDs must be unique")
    geometry = [
        {
            "candidate_id": str(candidate_ids[index]),
            "track_id": str(track_ids[index]),
            "x_hex": float(candidates[index, 0]).hex(),
            "y_hex": float(candidates[index, 1]).hex(),
        }
        for index in range(candidates.shape[0])
    ]
    contract = {
        "tile_id": tile_id,
        "grid_shape": list(shape),
        "transform_hex": [float(value).hex() for value in transform],
        "crs_wkt_sha256": canonical_sha256(crs_wkt),
        "candidate_count": int(candidates.shape[0]),
        "candidate_geometry_sha256": canonical_sha256(geometry),
    }
    grid = ProblemGrid(
        product_id=tile_id,
        shape=shape,
        transform=transform,
        crs_wkt=crs_wkt,
        contract=contract,
    )
    grid.validate()
    return grid


def _default_command_runner(
    argv: tuple[str, ...],
    cwd: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=None if environment is None else dict(environment),
        check=False,
        text=True,
        capture_output=True,
        timeout=2 * 60 * 60,
    )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _isis_runtime_environment(*, paper2_root: Path, isis_prefix: Path) -> dict[str, str]:
    """Activate the frozen Paper2 ISIS prefix without relying on the caller shell."""
    environment = dict(os.environ)
    existing_path = environment.get("PATH", "")
    existing_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment.update(
        {
            "ISISROOT": str(isis_prefix),
            "CONDA_PREFIX": str(isis_prefix),
            "ISISDATA": str(paper2_root / ".cache/isis-data"),
            "CONDA_PKGS_DIRS": str(paper2_root / ".cache/conda"),
            "XDG_CACHE_HOME": str(paper2_root / ".cache"),
            "PATH": f"{isis_prefix / 'bin'}:{existing_path}",
            "LD_LIBRARY_PATH": (
                f"{isis_prefix / 'lib'}:{existing_library_path}"
                if existing_library_path
                else str(isis_prefix / "lib")
            ),
        }
    )
    return environment


def _resolve_command_runner(
    command_runner: CommandRunner,
    *,
    paper2_root: Path,
    isis_prefix: Path,
) -> CommandRunner:
    if command_runner is not _default_command_runner:
        return command_runner
    return partial(
        _default_command_runner,
        environment=_isis_runtime_environment(
            paper2_root=paper2_root,
            isis_prefix=isis_prefix,
        ),
    )


def _run_command(
    argv: Sequence[str], *, cwd: Path, runner: CommandRunner
) -> dict[str, Any]:
    frozen = tuple(str(value) for value in argv)
    raw = dict(runner(frozen, cwd))
    record = {
        "argv": list(frozen),
        "returncode": int(raw.get("returncode", -1)),
        "stdout": str(raw.get("stdout", ""))[-4000:],
        "stderr": str(raw.get("stderr", ""))[-4000:],
    }
    if record["returncode"] != 0:
        raise RuntimeError(
            f"raster preparation command failed: {Path(frozen[0]).name}; "
            f"stderr={record['stderr'][-1000:]!r}"
        )
    return record


def _verify_content_hash(payload: Mapping[str, Any], *, field: str = "content_sha256") -> None:
    declared = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if declared != canonical_sha256(unhashed):
        raise ValueError(f"manifest {field} does not reproduce")


def _verify_record(record: Mapping[str, Any], path: Path, *, paper2_root: Path) -> None:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if (
        not resolved.is_file()
        or record.get("relative_path") != resolved.relative_to(paper2_root).as_posix()
        or record.get("sha256") != file_sha256(resolved)
        or int(record.get("byte_count", -1)) != resolved.stat().st_size
    ):
        raise ValueError(f"manifest artifact differs from local bytes: {resolved}")


def _mola_source_product(
    source_lock: Mapping[str, Any], *, product_id: str
) -> Mapping[str, Any]:
    matches = [
        dict(raw)
        for raw in source_lock.get("products", [])
        if raw.get("product_id") == product_id
    ]
    if len(matches) != 1:
        raise ValueError(f"MOLA source lock lacks exactly one {product_id} product")
    product = matches[0]
    if product.get("purpose") != "terrain_score":
        raise ValueError("formal coarse DEM must use the MEGT areoid-topography product")
    return product


def verify_mola_coarse_cube_manifest(
    manifest_path: str | Path,
    *,
    source_lock_path: str | Path,
    product_id: str,
    paper2_root: str | Path,
) -> Path:
    root = Path(paper2_root).resolve()
    manifest_file = guard_paper2_path(manifest_path, paper2_root=root)
    source_lock_file = guard_paper2_path(source_lock_path, paper2_root=root)
    payload = read_json(manifest_file)
    source_lock = read_json(source_lock_file)
    if not isinstance(payload, dict) or payload.get("schema") != MOLA_COARSE_CUBE_SCHEMA:
        raise ValueError("unsupported MOLA coarse-cube manifest")
    if payload.get("state") != "complete" or payload.get("product_id") != product_id:
        raise ValueError("MOLA coarse-cube manifest is not complete for the requested product")
    if payload.get("isis_version") != FROZEN_ISIS_VERSION:
        raise ValueError("MOLA coarse cube was not created with frozen ISIS 10.0.0")
    if payload.get("reference_dem_accessed") is not False:
        raise ValueError("MOLA coarse-cube manifest accessed a forbidden reference DEM")
    _verify_content_hash(payload)
    product = _mola_source_product(source_lock, product_id=product_id)
    expected_source = {
        "product_id": product_id,
        "image_sha256": product.get("image_sha256"),
        "label_sha256": product.get("label_sha256"),
    }
    if payload.get("mola_source") != expected_source:
        raise ValueError("MOLA coarse-cube source identity/hash differs")
    artifacts = dict(payload.get("artifacts", {}))
    required = {"mola_source_lock", "source_image", "source_label", "coarse_cube"}
    if set(artifacts) != required:
        raise ValueError("MOLA coarse-cube artifact set differs")
    source_image = guard_paper2_path(str(product["image_relative_path"]), paper2_root=root)
    source_label = guard_paper2_path(str(product["label_relative_path"]), paper2_root=root)
    _verify_record(artifacts["mola_source_lock"], source_lock_file, paper2_root=root)
    _verify_record(artifacts["source_image"], source_image, paper2_root=root)
    _verify_record(artifacts["source_label"], source_label, paper2_root=root)
    cube_record = dict(artifacts["coarse_cube"])
    cube = guard_paper2_path(str(cube_record.get("relative_path", "")), paper2_root=root)
    _verify_record(cube_record, cube, paper2_root=root)
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) < 2:
        raise ValueError("MOLA coarse-cube manifest lacks ISIS command evidence")
    names = [Path(str(dict(raw).get("argv", [""])[0])).name for raw in commands]
    if names[0] != "isisversion" or "pds2isis" not in names:
        raise ValueError("MOLA coarse-cube manifest lacks isisversion/pds2isis evidence")
    if any(int(dict(raw).get("returncode", -1)) != 0 for raw in commands):
        raise ValueError("MOLA coarse-cube manifest contains a failed command")
    return cube


def ensure_mola_coarse_cube(
    *,
    source_lock_path: str | Path,
    product_id: str,
    coarse_cube: str | Path,
    manifest_path: str | Path,
    isis_prefix: str | Path,
    paper2_root: str | Path,
    command_runner: CommandRunner = _default_command_runner,
) -> tuple[Path, Path]:
    """Create or verify an ISIS cube derived only from locked MOLA MEGT bytes."""
    root = Path(paper2_root).resolve()
    source_lock_file = guard_paper2_path(source_lock_path, paper2_root=root)
    cube = guard_paper2_path(coarse_cube, paper2_root=root)
    manifest = guard_paper2_path(manifest_path, paper2_root=root)
    prefix = guard_paper2_path(isis_prefix, paper2_root=root)
    runtime_command_runner = _resolve_command_runner(
        command_runner,
        paper2_root=root,
        isis_prefix=prefix,
    )
    if manifest.is_file():
        verified = verify_mola_coarse_cube_manifest(
            manifest,
            source_lock_path=source_lock_file,
            product_id=product_id,
            paper2_root=root,
        )
        if verified != cube:
            raise ValueError("requested MOLA coarse cube differs from its frozen manifest")
        return cube, manifest
    if cube.exists():
        raise FileExistsError("unmanifested MOLA coarse cube is untrusted and cannot be reused")
    source_lock = read_json(source_lock_file)
    product = _mola_source_product(source_lock, product_id=product_id)
    source_image = guard_paper2_path(str(product["image_relative_path"]), paper2_root=root)
    source_label = guard_paper2_path(str(product["label_relative_path"]), paper2_root=root)
    for kind, path in (("image", source_image), ("label", source_label)):
        if not path.is_file() or product.get(f"{kind}_sha256") != file_sha256(path):
            raise ValueError(f"locked MOLA {kind} bytes are missing or differ")
    isisversion = prefix / "bin/isisversion"
    pds2isis = prefix / "bin/pds2isis"
    cube.parent.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    commands.append(
        _run_command((str(isisversion),), cwd=cube.parent, runner=runtime_command_runner)
    )
    versions = set(
        re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", commands[0].get("stdout", ""))
    )
    if FROZEN_ISIS_VERSION not in versions:
        raise ValueError("runtime ISIS version does not include frozen 10.0.0")
    with tempfile.TemporaryDirectory(prefix=".mola-megt-cube-", dir=cube.parent) as attempt_name:
        temporary_cube = Path(attempt_name) / cube.name
        commands.append(
            _run_command(
                (
                    str(pds2isis),
                    f"from={source_label}",
                    f"to={temporary_cube}",
                ),
                cwd=Path(attempt_name),
                runner=runtime_command_runner,
            )
        )
        if not temporary_cube.is_file() or temporary_cube.stat().st_size <= 0:
            raise RuntimeError("pds2isis did not produce a non-empty MOLA coarse cube")
        temporary_cube.replace(cube)
    payload: dict[str, Any] = {
        "schema": MOLA_COARSE_CUBE_SCHEMA,
        "state": "complete",
        "product_id": product_id,
        "isis_version": FROZEN_ISIS_VERSION,
        "reference_dem_accessed": False,
        "mola_source": {
            "product_id": product_id,
            "image_sha256": product["image_sha256"],
            "label_sha256": product["label_sha256"],
        },
        "artifacts": {
            "mola_source_lock": _relative_record(source_lock_file, paper2_root=root),
            "source_image": _relative_record(source_image, paper2_root=root),
            "source_label": _relative_record(source_label, paper2_root=root),
            "coarse_cube": _relative_record(cube, paper2_root=root),
        },
        "commands": commands,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(manifest, payload, paper2_root=root)
    verify_mola_coarse_cube_manifest(
        manifest,
        source_lock_path=source_lock_file,
        product_id=product_id,
        paper2_root=root,
    )
    return cube, manifest


def _export_geotiff(
    source: Path,
    output: Path,
    *,
    gdal_translate: Path,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to reuse an unmanifested raster export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = _run_command(
        (
            str(gdal_translate),
            "-of",
            "GTiff",
            "-ot",
            "Float32",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=DEFLATE",
            str(source),
            str(output),
        ),
        cwd=output.parent,
        runner=command_runner,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"gdal_translate did not produce a non-empty GeoTIFF: {output}")
    return record


def _ephemeral_export_record(path: Path, *, paper2_root: Path) -> dict[str, Any]:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError("ephemeral raster export is missing/empty")
    sidecars = [
        candidate
        for candidate in sorted(resolved.parent.glob(f"{resolved.name}.*"))
        if candidate.is_file()
    ]
    return {
        "relative_path": resolved.relative_to(paper2_root).as_posix(),
        "sha256": file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
        "removed_after_verified_read": True,
        "sidecars": [
            {
                "relative_path": sidecar.relative_to(paper2_root).as_posix(),
                "sha256": file_sha256(sidecar),
                "byte_count": sidecar.stat().st_size,
            }
            for sidecar in sidecars
        ],
    }


def _remove_ephemeral_export(
    record: Mapping[str, Any], *, paper2_root: Path
) -> None:
    paths = [
        guard_paper2_path(str(record["relative_path"]), paper2_root=paper2_root),
        *[
            guard_paper2_path(
                str(dict(sidecar)["relative_path"]), paper2_root=paper2_root
            )
            for sidecar in record.get("sidecars", [])
        ],
    ]
    for path in paths:
        if not path.is_file():
            raise ValueError("ephemeral raster export changed before audited removal")
        path.unlink()
    if any(path.exists() for path in paths):
        raise RuntimeError("ephemeral raster export removal did not close")


def _write_or_resume_completed_lineage(
    path: Path,
    payload: Mapping[str, Any],
    *,
    paper2_root: Path,
) -> dict[str, Any]:
    """Retain verified partial-run lineage when only ephemeral evidence changed."""
    if not path.exists():
        write_canonical_json(path, payload, paper2_root=paper2_root)
        return dict(payload)
    existing = read_json(path)
    if not isinstance(existing, dict):
        raise ValueError("partial raster lineage must be one JSON object")
    _verify_content_hash(existing)
    stable_existing = dict(existing)
    stable_candidate = dict(payload)
    for record in (stable_existing, stable_candidate):
        record.pop("content_sha256", None)
        record.pop("commands", None)
        record.pop("ephemeral_exports", None)
    if stable_existing != stable_candidate:
        raise FileExistsError(
            f"refusing to replace divergent partial raster lineage: {path}"
        )
    existing_commands = existing.get("commands")
    candidate_commands = payload.get("commands")
    if not (
        isinstance(existing_commands, list)
        and isinstance(candidate_commands, list)
        and len(existing_commands) == len(candidate_commands)
    ):
        raise ValueError("partial raster lineage command evidence differs")
    for old_raw, new_raw in zip(existing_commands, candidate_commands, strict=True):
        old = dict(old_raw)
        new = dict(new_raw)
        if old.get("argv") != new.get("argv") or int(old.get("returncode", -1)) != 0:
            raise ValueError("partial raster lineage command identity differs")
    existing_exports = existing.get("ephemeral_exports")
    candidate_exports = payload.get("ephemeral_exports")
    if not (
        isinstance(existing_exports, dict)
        and isinstance(candidate_exports, dict)
        and set(existing_exports) == set(candidate_exports)
    ):
        raise ValueError("partial raster lineage ephemeral export set differs")
    for name, old_raw in existing_exports.items():
        old = dict(old_raw)
        new = dict(candidate_exports[name])
        old_path = guard_paper2_path(
            str(old.get("relative_path", "")), paper2_root=paper2_root
        )
        if not (
            old.get("relative_path") == new.get("relative_path")
            and old.get("removed_after_verified_read") is True
            and int(old.get("byte_count", -1)) > 0
            and re.fullmatch(r"[0-9a-f]{64}", str(old.get("sha256", "")))
            and not old_path.exists()
            and isinstance(old.get("sidecars"), list)
        ):
            raise ValueError(f"partial raster lineage export differs: {name}")
        for raw_sidecar in old["sidecars"]:
            sidecar = dict(raw_sidecar)
            sidecar_path = guard_paper2_path(
                str(sidecar.get("relative_path", "")), paper2_root=paper2_root
            )
            if not (
                int(sidecar.get("byte_count", -1)) > 0
                and re.fullmatch(r"[0-9a-f]{64}", str(sidecar.get("sha256", "")))
                and not sidecar_path.exists()
            ):
                raise ValueError(
                    f"partial raster lineage export sidecar differs: {name}"
                )
    return dict(existing)


def _rasterio_reader(path: Path) -> RasterBand:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised only without geo extra
        raise RuntimeError("formal raster preparation requires the paper2 geo extra") from exc
    with rasterio.open(path) as dataset:
        if dataset.count < 1:
            raise ValueError(f"raster contains no bands: {path}")
        values = dataset.read(1).astype(np.float64, copy=False)
        mask = dataset.read_masks(1) > 0
        mask &= np.isfinite(values)
        crs_wkt = dataset.crs.to_wkt() if dataset.crs is not None else ""
        transform = tuple(float(value) for value in dataset.transform[:6])
        band = RasterBand(
            values=values,
            valid_mask=mask.astype(bool, copy=False),
            transform=transform,
            crs_wkt=crs_wkt,
            driver=str(dataset.driver),
        )
    band.validate()
    return band


def _rasterio_reprojector(
    source: RasterBand, target: ProblemGrid, resampling: str
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover - exercised only without geo extra
        raise RuntimeError("formal raster preparation requires the paper2 geo extra") from exc
    source.validate()
    target.validate()
    modes = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest}
    if resampling not in modes:
        raise ValueError(f"unsupported frozen resampling mode: {resampling}")
    source_values = np.where(source.valid_mask, source.values, np.nan).astype(np.float64)
    destination = np.full(target.shape, np.nan, dtype=np.float64)
    reproject(
        source=source_values,
        destination=destination,
        src_transform=Affine(*source.transform),
        src_crs=CRS.from_wkt(source.crs_wkt),
        src_nodata=np.nan,
        dst_transform=Affine(*target.transform),
        dst_crs=CRS.from_wkt(target.crs_wkt),
        dst_nodata=np.nan,
        resampling=modes[resampling],
        num_threads=1,
        init_dest_nodata=True,
    )
    source_mask = source.valid_mask.astype(np.uint8)
    destination_mask = np.zeros(target.shape, dtype=np.uint8)
    reproject(
        source=source_mask,
        destination=destination_mask,
        src_transform=Affine(*source.transform),
        src_crs=CRS.from_wkt(source.crs_wkt),
        src_nodata=0,
        dst_transform=Affine(*target.transform),
        dst_crs=CRS.from_wkt(target.crs_wkt),
        dst_nodata=0,
        resampling=Resampling.nearest,
        num_threads=1,
        init_dest_nodata=True,
    )
    valid = (destination_mask > 0) & np.isfinite(destination)
    destination[~valid] = np.nan
    return destination, valid


def _registration_feature(values: np.ndarray, valid: np.ndarray, *, sigma: float) -> np.ndarray:
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float64)
    fill = float(np.median(values[valid]))
    dense = np.where(valid, values, fill).astype(np.float64, copy=False)
    highpass = dense - ndimage.gaussian_filter(dense, sigma=sigma, mode="nearest")
    row_gradient = ndimage.sobel(highpass, axis=0, mode="nearest")
    column_gradient = ndimage.sobel(highpass, axis=1, mode="nearest")
    return np.hypot(row_gradient, column_gradient)


def _quadratic_peak_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) <= np.finfo(np.float64).eps:
        return 0.0
    value = 0.5 * (left - right) / denominator
    return float(np.clip(value, -0.5, 0.5))


def phase_correlation_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    maximum_shift_pixels: int,
    highpass_sigma_pixels: float,
    minimum_valid_fraction: float,
    minimum_texture_standard_deviation: float,
) -> dict[str, float]:
    """Measure the translation to apply to ``moving``; never accepts supplied errors."""
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    moving_mask = np.asarray(moving_mask, dtype=bool)
    if (
        reference.shape != moving.shape
        or reference_mask.shape != reference.shape
        or moving_mask.shape != reference.shape
        or reference.ndim != 2
    ):
        raise ValueError("phase-correlation inputs must share one 2-D shape")
    common = reference_mask & moving_mask & np.isfinite(reference) & np.isfinite(moving)
    valid_fraction = float(np.mean(common))
    if valid_fraction < minimum_valid_fraction:
        raise ValueError("insufficient common valid support for phase correlation")
    reference_feature = _registration_feature(
        reference, common, sigma=highpass_sigma_pixels
    )
    moving_feature = _registration_feature(moving, common, sigma=highpass_sigma_pixels)
    reference_texture = float(np.std(reference_feature[common]))
    moving_texture = float(np.std(moving_feature[common]))
    if min(reference_texture, moving_texture) < minimum_texture_standard_deviation:
        raise ValueError("insufficient independent image texture for phase correlation")
    reference_center = float(np.mean(reference_feature[common]))
    moving_center = float(np.mean(moving_feature[common]))
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    ref = np.where(common, reference_feature - reference_center, 0.0) * window
    mov = np.where(common, moving_feature - moving_center, 0.0) * window
    cross_power = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, np.finfo(np.float64).eps)
    correlation = np.abs(np.fft.fftshift(np.fft.ifft2(cross_power)))
    center_row, center_column = (value // 2 for value in correlation.shape)
    radius = min(
        maximum_shift_pixels,
        center_row - 1,
        center_column - 1,
        correlation.shape[0] - center_row - 2,
        correlation.shape[1] - center_column - 2,
    )
    if radius < 1:
        raise ValueError("control window is too small for the frozen maximum shift")
    rows = slice(center_row - radius, center_row + radius + 1)
    columns = slice(center_column - radius, center_column + radius + 1)
    search = correlation[rows, columns]
    local_peak = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_row = center_row - radius + int(local_peak[0])
    peak_column = center_column - radius + int(local_peak[1])
    if abs(peak_row - center_row) == radius or abs(peak_column - center_column) == radius:
        raise ValueError("phase-correlation peak lies on the maximum-shift search boundary")
    peak = float(correlation[peak_row, peak_column])
    suppressed = search.copy()
    local_row, local_column = int(local_peak[0]), int(local_peak[1])
    suppressed[
        max(0, local_row - 1) : min(suppressed.shape[0], local_row + 2),
        max(0, local_column - 1) : min(suppressed.shape[1], local_column + 2),
    ] = -np.inf
    second = float(np.max(suppressed)) if np.any(np.isfinite(suppressed)) else 0.0
    peak_ratio = peak / max(second, np.finfo(np.float64).eps)
    row_offset = _quadratic_peak_offset(
        float(correlation[peak_row - 1, peak_column]),
        peak,
        float(correlation[peak_row + 1, peak_column]),
    )
    column_offset = _quadratic_peak_offset(
        float(correlation[peak_row, peak_column - 1]),
        peak,
        float(correlation[peak_row, peak_column + 1]),
    )
    return {
        "shift_row_pixels": float(peak_row - center_row + row_offset),
        "shift_column_pixels": float(peak_column - center_column + column_offset),
        "valid_fraction": valid_fraction,
        "reference_texture_std": reference_texture,
        "moving_texture_std": moving_texture,
        "peak": peak,
        "peak_ratio": peak_ratio,
    }


def _control_windows(
    shape: tuple[int, int], *, rows: int, columns: int, maximum_shift: int
) -> list[tuple[str, tuple[int, int, int, int]]]:
    row_edges = np.linspace(0, shape[0], rows + 1, dtype=np.int64)
    column_edges = np.linspace(0, shape[1], columns + 1, dtype=np.int64)
    windows: list[tuple[str, tuple[int, int, int, int]]] = []
    minimum_side = 2 * maximum_shift + 5
    for row in range(rows):
        for column in range(columns):
            row_start, row_stop = int(row_edges[row]), int(row_edges[row + 1])
            column_start, column_stop = int(column_edges[column]), int(column_edges[column + 1])
            if row_stop - row_start < minimum_side or column_stop - column_start < minimum_side:
                raise ValueError(
                    "problem grid is too small for the frozen independent control-window layout"
                )
            windows.append(
                (
                    f"r{row:02d}c{column:02d}",
                    (row_start, row_stop, column_start, column_stop),
                )
            )
    return windows


def _measure_controls(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    config: CoregistrationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for control_id, bounds in _control_windows(
        reference.shape,
        rows=config.control_grid_rows,
        columns=config.control_grid_columns,
        maximum_shift=config.maximum_shift_pixels,
    ):
        row_start, row_stop, column_start, column_stop = bounds
        slices = (slice(row_start, row_stop), slice(column_start, column_stop))
        try:
            measured = phase_correlation_shift(
                reference[slices],
                moving[slices],
                reference_mask[slices],
                moving_mask[slices],
                maximum_shift_pixels=config.maximum_shift_pixels,
                highpass_sigma_pixels=config.highpass_sigma_pixels,
                minimum_valid_fraction=config.minimum_valid_fraction,
                minimum_texture_standard_deviation=config.minimum_texture_standard_deviation,
            )
            record = {
                "control_id": control_id,
                "window_bounds": [row_start, row_stop, column_start, column_stop],
                "center_row": 0.5 * (row_start + row_stop - 1),
                "center_column": 0.5 * (column_start + column_stop - 1),
                **measured,
            }
            if measured["peak_ratio"] < config.minimum_peak_ratio:
                rejected.append({**record, "failure_reason": "peak_ratio_below_threshold"})
            else:
                accepted.append(record)
        except ValueError as exc:
            rejected.append(
                {
                    "control_id": control_id,
                    "window_bounds": [row_start, row_stop, column_start, column_stop],
                    "failure_reason": str(exc),
                }
            )
    return accepted, rejected


def fit_global_translation(
    controls: Sequence[Mapping[str, Any]], *, config: CoregistrationConfig
) -> tuple[tuple[float, float] | None, list[str], list[dict[str, Any]], dict[str, Any]]:
    """Fit the frozen robust two-parameter translation from accepted controls."""
    config.validate()
    raw = [dict(control) for control in controls]
    if len(raw) < MIN_COREGISTRATION_ERROR_COUNT:
        return (
            None,
            [],
            [],
            {
                "method": "coordinate_median_with_radial_mad_filter",
                "raw_control_count": len(raw),
                "fit_control_count": 0,
                "initial_median_shift_pixels": None,
                "radial_distance_median_pixels": None,
                "radial_distance_mad_pixels": None,
                "radial_inlier_threshold_pixels": None,
            },
        )
    vectors = np.asarray(
        [
            (
                float(control["shift_row_pixels"]),
                float(control["shift_column_pixels"]),
            )
            for control in raw
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(vectors)):
        raise ValueError("translation-fit control vectors must be finite")
    initial = np.median(vectors, axis=0)
    distances = np.linalg.norm(vectors - initial, axis=1)
    distance_median = float(np.median(distances))
    distance_mad = float(np.median(np.abs(distances - distance_median)))
    threshold = distance_median + config.outlier_mad_multiplier * 1.4826 * distance_mad
    tolerance = 64.0 * np.finfo(np.float64).eps
    selected = [
        index for index, distance in enumerate(distances) if distance <= threshold + tolerance
    ]
    if len(selected) < MIN_COREGISTRATION_ERROR_COUNT:
        selected = sorted(
            range(len(raw)),
            key=lambda index: (float(distances[index]), str(raw[index]["control_id"])),
        )[:MIN_COREGISTRATION_ERROR_COUNT]
    selected_set = set(selected)
    fit_control_ids = [str(raw[index]["control_id"]) for index in selected]
    fit_rejections = [
        {
            "control_id": str(raw[index]["control_id"]),
            "distance_to_initial_median_pixels": float(distances[index]),
            "failure_reason": "translation_fit_radial_mad_outlier",
        }
        for index in range(len(raw))
        if index not in selected_set
    ]
    fitted = np.median(vectors[selected], axis=0)
    if not np.all(np.isfinite(fitted)) or np.any(
        np.abs(fitted) > config.maximum_shift_pixels
    ):
        raise ValueError("fitted global translation escapes the frozen search bound")
    diagnostics = {
        "method": "coordinate_median_with_radial_mad_filter",
        "raw_control_count": len(raw),
        "fit_control_count": len(selected),
        "initial_median_shift_pixels": [float(initial[0]), float(initial[1])],
        "radial_distance_median_pixels": distance_median,
        "radial_distance_mad_pixels": distance_mad,
        "radial_inlier_threshold_pixels": float(threshold),
    }
    return (
        (float(fitted[0]), float(fitted[1])),
        fit_control_ids,
        fit_rejections,
        diagnostics,
    )


def register_to_coreg_aux(
    image: np.ndarray,
    image_mask: np.ndarray,
    coreg_aux: np.ndarray,
    coreg_aux_mask: np.ndarray,
    *,
    config: CoregistrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one auditable global translation, then measure local residuals.

    ``coreg_aux`` is never copied into the model input and cannot authorize a
    rotation, scale, shear, local warp, or intensity correction.  It supplies
    only the two translation parameters needed to remove CTX pointing offset.
    The registered image remains entirely derived from the EDR/ISIS image.
    """
    config.validate()
    image = np.asarray(image, dtype=np.float64)
    image_mask = np.asarray(image_mask, dtype=bool)
    coreg_aux = np.asarray(coreg_aux, dtype=np.float64)
    coreg_aux_mask = np.asarray(coreg_aux_mask, dtype=bool)
    if (
        image.ndim != 2
        or coreg_aux.shape != image.shape
        or image_mask.shape != image.shape
        or coreg_aux_mask.shape != image.shape
    ):
        raise ValueError("coregistration grids and masks must share one 2-D target grid")
    raw_controls, raw_rejections = _measure_controls(
        coreg_aux,
        image,
        coreg_aux_mask,
        image_mask,
        config=config,
    )
    translation: tuple[float, float] | None = None
    fit_control_ids: list[str] = []
    fit_rejections: list[dict[str, Any]] = []
    translation_fit: dict[str, Any]
    registered_image = image.copy()
    registered_mask = image_mask.copy()
    translation, fit_control_ids, fit_rejections, translation_fit = (
        fit_global_translation(raw_controls, config=config)
    )
    if translation is not None:
        if translation != (0.0, 0.0):
            registered_image = ndimage.shift(
                image,
                shift=translation,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            registered_mask = ndimage.shift(
                image_mask.astype(np.uint8),
                shift=translation,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).astype(bool)
        registered_mask &= np.isfinite(registered_image)
        registered_image[~registered_mask] = np.nan
        residual_controls, residual_rejections = _measure_controls(
            coreg_aux,
            registered_image,
            coreg_aux_mask,
            registered_mask,
            config=config,
        )
    else:
        residual_controls = raw_controls
        residual_rejections = raw_rejections
    errors = np.asarray(
        [
            math.hypot(
                float(control["shift_row_pixels"]),
                float(control["shift_column_pixels"]),
            )
            for control in residual_controls
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("measured coregistration residuals are invalid")
    sufficient_controls = len(residual_controls) >= MIN_COREGISTRATION_ERROR_COUNT
    rejection_reasons = (
        [] if sufficient_controls else ["coregistration_error_count<4"]
    )
    evidence = {
        "parameters": asdict(config),
        "mode": COREGISTRATION_MODE,
        "transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "coreg_aux_influenced_primary_pixels": translation is not None,
        "coreg_aux_pixel_values_copied": False,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "rotation_scale_shear_or_local_warp_applied": False,
        "fitted_translation_to_apply_pixels": (
            list(translation) if translation is not None else None
        ),
        "fit_control_ids": fit_control_ids,
        "fit_rejections": fit_rejections,
        "translation_fit": translation_fit,
        "raw_controls": raw_controls,
        "raw_rejections": raw_rejections,
        "control_count_sufficient": sufficient_controls,
        "rejection_reasons": rejection_reasons,
        "residual_controls": residual_controls,
        "residual_rejections": residual_rejections,
        "residual_error_count": int(errors.size),
        "residual_median_target_pixels": (
            float(np.median(errors)) if errors.size else None
        ),
        "residual_p95_target_pixels": (
            float(np.percentile(errors, 95)) if errors.size else None
        ),
    }
    return registered_image, registered_mask, errors, evidence


def _verified_isis_projected_cube(
    manifest_path: Path, *, paper2_root: Path
) -> tuple[Path, Mapping[str, Any]]:
    payload = read_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported ISIS run manifest")
    if (
        payload.get("isis_version") != FROZEN_ISIS_VERSION
        or payload.get("reported_isis_version") != FROZEN_ISIS_VERSION
    ):
        raise ValueError("ISIS projection is not bound to frozen runtime 10.0.0")
    if payload.get("reference_dem_accessed") is not False:
        raise ValueError("ISIS projection accessed a forbidden reference DEM")
    record = dict(dict(payload.get("artifacts", {})).get("ctx_cam2map.cub", {}))
    cube = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=paper2_root)
    if not cube.is_file() or record.get("sha256") != file_sha256(cube):
        raise ValueError("ISIS manifest ctx_cam2map cube path/hash does not verify")
    return cube, payload


def prepare_g0_rasters(
    *,
    paper2_root: str | Path,
    product_id: str,
    problem: str | Path,
    isis_run_manifest: str | Path,
    mola_source_lock: str | Path,
    mola_coarse_cube: str | Path,
    mola_coarse_cube_manifest: str | Path,
    coreg_aux: str | Path,
    output_directory: str | Path,
    isis_prefix: str | Path,
    mola_product_id: str = "MEGT90N000FB",
    coregistration_config: CoregistrationConfig = FROZEN_COREGISTRATION_CONFIG,
    command_runner: CommandRunner = _default_command_runner,
    raster_reader: RasterReader = _rasterio_reader,
    raster_reprojector: RasterReprojector = _rasterio_reprojector,
) -> dict[str, Any]:
    """Produce reference-free target-grid image/MOLA arrays and measured QA lineage."""
    root = Path(paper2_root).resolve()
    _safe_identifier(product_id, name="product_id")
    problem_path = guard_paper2_path(problem, paper2_root=root)
    isis_manifest_path = guard_paper2_path(isis_run_manifest, paper2_root=root)
    mola_lock_path = guard_paper2_path(mola_source_lock, paper2_root=root)
    aux_path = guard_paper2_path(coreg_aux, paper2_root=root)
    output = guard_paper2_path(output_directory, paper2_root=root)
    prefix = guard_paper2_path(isis_prefix, paper2_root=root)
    runtime_command_runner = _resolve_command_runner(
        command_runner,
        paper2_root=root,
        isis_prefix=prefix,
    )
    if not aux_path.is_file():
        raise FileNotFoundError(
            f"coreg_aux is mandatory for measured registration QA; no pass can be emitted: {aux_path}"
        )
    for path in (problem_path, isis_manifest_path, mola_lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    grid = load_problem_grid(problem_path, product_id=product_id)
    projected_cube, isis_payload = _verified_isis_projected_cube(
        isis_manifest_path, paper2_root=root
    )
    coarse_cube, coarse_cube_manifest = ensure_mola_coarse_cube(
        source_lock_path=mola_lock_path,
        product_id=mola_product_id,
        coarse_cube=mola_coarse_cube,
        manifest_path=mola_coarse_cube_manifest,
        isis_prefix=prefix,
        paper2_root=root,
        command_runner=runtime_command_runner,
    )
    output.mkdir(parents=True, exist_ok=True)
    projected_export = output / "ctx-cam2map-export.tif"
    mola_export = output / "mola-megt-export.tif"
    aux_export = output / "coreg-aux-export.tif"
    gdal_translate = prefix / "bin/gdal_translate"
    export_commands = {
        "projected_image": _export_geotiff(
            projected_cube,
            projected_export,
            gdal_translate=gdal_translate,
            command_runner=runtime_command_runner,
        ),
        "mola_coarse": _export_geotiff(
            coarse_cube,
            mola_export,
            gdal_translate=gdal_translate,
            command_runner=runtime_command_runner,
        ),
        "coreg_aux": _export_geotiff(
            aux_path,
            aux_export,
            gdal_translate=gdal_translate,
            command_runner=runtime_command_runner,
        ),
    }
    projected_source = raster_reader(projected_export)
    mola_source = raster_reader(mola_export)
    aux_source = raster_reader(aux_export)
    for source in (projected_source, mola_source, aux_source):
        source.validate()
    image_unregistered, image_unregistered_mask = raster_reprojector(
        projected_source, grid, "bilinear"
    )
    coarse_dem, dem_mask = raster_reprojector(mola_source, grid, "bilinear")
    aux_grid, aux_mask = raster_reprojector(aux_source, grid, "bilinear")
    ephemeral_exports = {
        "projected_grid_export": _ephemeral_export_record(
            projected_export, paper2_root=root
        ),
        "mola_grid_export": _ephemeral_export_record(mola_export, paper2_root=root),
        "coreg_aux_grid_export": _ephemeral_export_record(
            aux_export, paper2_root=root
        ),
    }
    for record in ephemeral_exports.values():
        _remove_ephemeral_export(record, paper2_root=root)
    image_unregistered = np.asarray(image_unregistered, dtype=np.float32)
    image_unregistered_mask = np.asarray(image_unregistered_mask, dtype=bool)
    image_unregistered[~image_unregistered_mask] = np.nan
    image, image_mask, errors, registration = register_to_coreg_aux(
        image_unregistered,
        image_unregistered_mask,
        aux_grid,
        aux_mask,
        config=coregistration_config,
    )
    if image.shape != grid.shape or coarse_dem.shape != grid.shape:
        raise ValueError("reprojected arrays differ from the frozen problem grid shape")
    image = np.asarray(image, dtype=np.float32)
    image_mask = np.asarray(image_mask, dtype=bool)
    coarse_dem = np.asarray(coarse_dem, dtype=np.float32)
    dem_mask = np.asarray(dem_mask, dtype=bool)
    image[~image_mask] = np.nan
    coarse_dem[~dem_mask] = np.nan
    if not np.array_equal(image_mask, np.isfinite(image)):
        raise ValueError("image mask does not equal finite support after registration")
    if not np.array_equal(dem_mask, np.isfinite(coarse_dem)):
        raise ValueError("DEM mask does not equal finite MOLA support")

    paths = {
        "image_unregistered": output / "image-unregistered.npy",
        "image_unregistered_mask": output / "image-unregistered-mask.npy",
        "image": output / "image.npy",
        "image_mask": output / "image-mask.npy",
        "coarse_dem": output / "coarse-dem.npy",
        "dem_mask": output / "dem-mask.npy",
        "coregistration_errors": output / "coregistration-errors.npy",
        "tiepoints": output / "coregistration-tiepoints.json",
        "coarse_lineage": output / "coarse-dem-lineage.json",
        "coregistration_lineage": output / "coregistration-lineage.json",
        "preparation_manifest": output / "raster-preparation-manifest.json",
    }
    _write_array(paths["image_unregistered"], image_unregistered, paper2_root=root)
    _write_array(
        paths["image_unregistered_mask"],
        image_unregistered_mask,
        paper2_root=root,
    )
    _write_array(paths["image"], image, paper2_root=root)
    _write_array(paths["image_mask"], image_mask, paper2_root=root)
    _write_array(paths["coarse_dem"], coarse_dem, paper2_root=root)
    _write_array(paths["dem_mask"], dem_mask, paper2_root=root)
    _write_array(paths["coregistration_errors"], errors, paper2_root=root)
    tiepoint_payload: dict[str, Any] = {
        "schema": COREGISTRATION_CONTROLS_SCHEMA,
        "product_id": product_id,
        "source": COREGISTRATION_SOURCE,
        "problem_contract": dict(grid.contract),
        **registration,
    }
    tiepoint_payload["content_sha256"] = canonical_sha256(tiepoint_payload)
    write_canonical_json(paths["tiepoints"], tiepoint_payload, paper2_root=root)

    source_lock_payload = read_json(mola_lock_path)
    mola_product = _mola_source_product(source_lock_payload, product_id=mola_product_id)
    coarse_artifacts = {
        "mola_source_lock": _relative_record(mola_lock_path, paper2_root=root),
        "mola_coarse_cube_manifest": _relative_record(
            coarse_cube_manifest, paper2_root=root
        ),
        "mola_coarse_cube": _relative_record(coarse_cube, paper2_root=root),
        "problem": _relative_record(problem_path, paper2_root=root),
        "coarse_dem": _relative_record(paths["coarse_dem"], paper2_root=root),
        "dem_mask": _relative_record(paths["dem_mask"], paper2_root=root),
    }
    coarse_lineage: dict[str, Any] = {
        "schema": COARSE_DEM_LINEAGE_SCHEMA,
        "state": "complete",
        "algorithm": COARSE_DEM_ALGORITHM,
        "product_id": product_id,
        "reference_dem_accessed": False,
        "mola_source": {
            "product_id": mola_product_id,
            "image_sha256": mola_product["image_sha256"],
            "label_sha256": mola_product["label_sha256"],
            "coarse_cube_sha256": file_sha256(coarse_cube),
            "coarse_cube_manifest_sha256": file_sha256(coarse_cube_manifest),
        },
        "problem_contract": dict(grid.contract),
        "target_grid": {
            "shape": list(grid.shape),
            "transform_hex": [float(value).hex() for value in grid.transform],
            "crs_wkt_sha256": canonical_sha256(grid.crs_wkt),
        },
        "resampling": "bilinear_values_nearest_validity",
        "artifacts": coarse_artifacts,
        "ephemeral_exports": {
            "mola_grid_export": ephemeral_exports["mola_grid_export"]
        },
        "commands": [export_commands["mola_coarse"]],
        "operations": [
            {
                "name": "rasterio_reproject",
                "source": "mola_grid_export",
                "destination": "coarse_dem",
                "resampling": "bilinear_values_nearest_validity",
                "threads": 1,
            }
        ],
    }
    coarse_lineage["content_sha256"] = canonical_sha256(coarse_lineage)
    coarse_lineage = _write_or_resume_completed_lineage(
        paths["coarse_lineage"], coarse_lineage, paper2_root=root
    )

    coregistration_artifacts = {
        "isis_run_manifest": _relative_record(isis_manifest_path, paper2_root=root),
        "projected_cube": _relative_record(projected_cube, paper2_root=root),
        "coreg_aux": _relative_record(aux_path, paper2_root=root),
        "problem": _relative_record(problem_path, paper2_root=root),
        "image_unregistered": _relative_record(
            paths["image_unregistered"], paper2_root=root
        ),
        "image_unregistered_mask": _relative_record(
            paths["image_unregistered_mask"], paper2_root=root
        ),
        "image": _relative_record(paths["image"], paper2_root=root),
        "image_mask": _relative_record(paths["image_mask"], paper2_root=root),
        "coregistration_errors": _relative_record(
            paths["coregistration_errors"], paper2_root=root
        ),
        "tiepoints": _relative_record(paths["tiepoints"], paper2_root=root),
    }
    coregistration_lineage: dict[str, Any] = {
        "schema": COREGISTRATION_LINEAGE_SCHEMA,
        "state": "complete",
        "algorithm": COREGISTRATION_ALGORITHM,
        "product_id": product_id,
        "source": COREGISTRATION_SOURCE,
        "reference_dem_accessed": False,
        "coreg_aux_is_primary": False,
        "coreg_aux_influenced_primary_pixels": registration[
            "coreg_aux_influenced_primary_pixels"
        ],
        "coreg_aux_pixel_values_copied": False,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "registration_transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "rotation_scale_shear_or_local_warp_applied": False,
        "problem_contract": dict(grid.contract),
        "projected_cube_sha256": file_sha256(projected_cube),
        "isis_run_manifest_sha256": file_sha256(isis_manifest_path),
        "target_grid": {
            "shape": list(grid.shape),
            "transform_hex": [float(value).hex() for value in grid.transform],
            "crs_wkt_sha256": canonical_sha256(grid.crs_wkt),
        },
        "artifacts": coregistration_artifacts,
        "ephemeral_exports": {
            name: ephemeral_exports[name]
            for name in ("projected_grid_export", "coreg_aux_grid_export")
        },
        "commands": [
            export_commands["projected_image"],
            export_commands["coreg_aux"],
        ],
        "operations": [
            {
                "name": "rasterio_reproject",
                "sources": ["projected_grid_export", "coreg_aux_grid_export"],
                "resampling": "bilinear_values_nearest_validity",
                "threads": 1,
            },
            {
                "name": COREGISTRATION_ALGORITHM,
                "parameters": asdict(coregistration_config),
                "mode": COREGISTRATION_MODE,
                "transform_family": COREGISTRATION_TRANSFORM_FAMILY,
                "pixel_adjustment_applied": registration[
                    "coreg_aux_influenced_primary_pixels"
                ],
                "coreg_aux_pixel_values_copied": False,
                "rotation_scale_shear_or_local_warp_applied": False,
                "fitted_translation_to_apply_pixels": registration[
                    "fitted_translation_to_apply_pixels"
                ],
                "measured_control_file_sha256": file_sha256(paths["tiepoints"]),
                "error_vector_sha256": file_sha256(paths["coregistration_errors"]),
            },
        ],
        "registration": {
            "mode": COREGISTRATION_MODE,
            "transform_family": COREGISTRATION_TRANSFORM_FAMILY,
            "coreg_aux_influenced_primary_pixels": registration[
                "coreg_aux_influenced_primary_pixels"
            ],
            "coreg_aux_pixel_values_copied": False,
            "primary_intensity_source": "ctx_edr_isis_cam2map_only",
            "rotation_scale_shear_or_local_warp_applied": False,
            "fitted_translation_to_apply_pixels": registration[
                "fitted_translation_to_apply_pixels"
            ],
            "fit_control_ids": list(registration["fit_control_ids"]),
            "raw_control_count": len(registration["raw_controls"]),
            "residual_control_count": len(registration["residual_controls"]),
            "residual_error_count": int(errors.size),
            "control_count_sufficient": registration["control_count_sufficient"],
            "rejection_reasons": list(registration["rejection_reasons"]),
            "residual_median_target_pixels": (
                float(np.median(errors)) if errors.size else None
            ),
            "residual_p95_target_pixels": (
                float(np.percentile(errors, 95)) if errors.size else None
            ),
        },
    }
    coregistration_lineage["content_sha256"] = canonical_sha256(coregistration_lineage)
    coregistration_lineage = _write_or_resume_completed_lineage(
        paths["coregistration_lineage"],
        coregistration_lineage,
        paper2_root=root,
    )

    manifest: dict[str, Any] = {
        "schema": RASTER_PREPARATION_SCHEMA,
        "state": (
            "complete"
            if errors.size >= MIN_COREGISTRATION_ERROR_COUNT
            else "qa_rejected"
        ),
        "product_id": product_id,
        "reference_dem_accessed": False,
        "coregistration_source": COREGISTRATION_SOURCE,
        "registration_transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "coreg_aux_influenced_primary_pixels": registration[
            "coreg_aux_influenced_primary_pixels"
        ],
        "coreg_aux_pixel_values_copied": False,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "rotation_scale_shear_or_local_warp_applied": False,
        "problem_contract": dict(grid.contract),
        "input_bindings": {
            "problem_sha256": file_sha256(problem_path),
            "isis_run_manifest_sha256": file_sha256(isis_manifest_path),
            "projected_cube_sha256": file_sha256(projected_cube),
            "mola_source_lock_sha256": file_sha256(mola_lock_path),
            "mola_coarse_cube_manifest_sha256": file_sha256(coarse_cube_manifest),
            "mola_coarse_cube_sha256": file_sha256(coarse_cube),
            "coreg_aux_sha256": file_sha256(aux_path),
            "isis_projection_terrain": isis_payload.get("projection_terrain"),
        },
        "outputs": {
            name: _relative_record(path, paper2_root=root)
            for name, path in paths.items()
            if name != "preparation_manifest"
        },
        "qa_measurements": {
            "image_coverage_fraction": float(np.mean(image_mask)),
            "finite_dem_fraction": float(np.mean(dem_mask)),
            "alignment_median_pixels": (
                float(np.median(errors)) if errors.size else None
            ),
            "alignment_p95_pixels": (
                float(np.percentile(errors, 95)) if errors.size else None
            ),
            "coregistration_error_count": int(errors.size),
        },
        "qa_rejection_reasons": list(registration["rejection_reasons"]),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    write_canonical_json(paths["preparation_manifest"], manifest, paper2_root=root)
    return read_json(paths["preparation_manifest"])
