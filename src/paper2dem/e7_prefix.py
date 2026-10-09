from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import io
import json
import time

import numpy as np

from .audit_footprints import read_audit_path_projections, write_audit_projection_prefix
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    write_canonical_json,
    write_immutable_bytes,
)
from .matrix import ArtifactDigest, ExpectedCell, ExpectedMatrix, InferenceManifest, MatrixKey
from .provenance import mechanism_parameters_sha256


E7_PATH_COUNTS = (2, 4, 8, 16, 32)
E7_STEP_COUNTS = (10, 20, 40)


def e7_condition(path_count: int, step_count: int) -> str:
    if path_count not in E7_PATH_COUNTS or step_count not in E7_STEP_COUNTS:
        raise ValueError("E7 requires M in {2,4,8,16,32} and N in {10,20,40}")
    return f"prefix-M{path_count}-N{step_count}"


def paired_path_seed(path_seed_set: str) -> int:
    if not path_seed_set:
        raise ValueError("paired path-seed set cannot be empty")
    return int(canonical_sha256({"path_seed_set": path_seed_set})[:16], 16) % (
        2**63 - 1
    )


def validate_e7_key(key: MatrixKey, *, source: bool = False) -> None:
    if key.experiment != "E7":
        raise ValueError("E7 prefix contract requires experiment=E7")
    expected = e7_condition(key.path_count, key.step_count)
    if key.condition != expected:
        raise ValueError(f"E7 condition must be {expected}")
    if source and key.path_count != max(E7_PATH_COUNTS):
        raise ValueError("E7 source inference must generate M=32 paths")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def ordered_path_sha256(path_stack: np.ndarray) -> tuple[str, ...]:
    paths = np.asarray(path_stack)
    if paths.ndim != 3 or paths.shape[0] <= 0:
        raise ValueError("E7 path stack must have shape (M,H,W)")
    if not np.all(np.isfinite(paths)):
        raise ValueError("E7 path stack must be finite")
    return tuple(_array_sha256(path) for path in paths)


def prefix_statistics(
    path_stack: np.ndarray,
    path_count: int,
    *,
    calibration_scale: float = 1.0,
    calibration_floor_variance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    paths = np.asarray(path_stack)
    if paths.ndim != 3 or path_count <= 0 or path_count > paths.shape[0]:
        raise ValueError("prefix count must select a nonempty strict path-stack prefix")
    if not np.isfinite(calibration_scale) or calibration_scale <= 0:
        raise ValueError("calibration scale must be finite and positive")
    if not np.isfinite(calibration_floor_variance) or calibration_floor_variance < 0:
        raise ValueError("calibration floor must be finite and non-negative")
    selected = np.asarray(paths[:path_count], dtype=np.float64)
    mean = np.mean(selected, axis=0, dtype=np.float64)
    variance = np.var(selected, axis=0, ddof=1, dtype=np.float64)
    variance = np.maximum(
        calibration_scale * variance,
        calibration_floor_variance,
    )
    return mean.astype(np.float32), variance.astype(np.float32)


def _prefix_record(
    path_stack: np.ndarray,
    path_count: int,
    *,
    path_hashes: Sequence[str],
    calibration_scale: float,
    calibration_floor_variance: float,
) -> dict[str, Any]:
    mean, variance = prefix_statistics(
        path_stack,
        path_count,
        calibration_scale=calibration_scale,
        calibration_floor_variance=calibration_floor_variance,
    )
    return {
        "path_count": path_count,
        "ordered_path_sha256": list(path_hashes[:path_count]),
        "mean_sha256": _array_sha256(mean),
        "variance_sha256": _array_sha256(variance),
    }


def build_e7_source_payload(
    *,
    key: MatrixKey,
    path_seed: int,
    path_stack: np.ndarray,
    path_stack_sha256: str,
    prediction_sha256: str,
    calibration_manifest_sha256: str,
    calibration_scale: float,
    calibration_floor_variance: float,
    mechanism_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    validate_e7_key(key, source=True)
    if path_seed != paired_path_seed(key.path_seed_set):
        raise ValueError("E7 source must use the deterministic paired path seed")
    paths = np.asarray(path_stack)
    if paths.shape[0] != key.path_count:
        raise ValueError("source path stack count differs from its matrix key")
    for digest_name, digest in (
        ("path_stack_sha256", path_stack_sha256),
        ("prediction_sha256", prediction_sha256),
        ("calibration_manifest_sha256", calibration_manifest_sha256),
    ):
        if not is_sha256(digest):
            raise ValueError(f"{digest_name} must be SHA-256")
    hashes = ordered_path_sha256(paths)
    prefixes = {
        str(count): _prefix_record(
            paths,
            count,
            path_hashes=hashes,
            calibration_scale=calibration_scale,
            calibration_floor_variance=calibration_floor_variance,
        )
        for count in E7_PATH_COUNTS
    }
    payload: dict[str, Any] = {
        "schema": "paper2.e7-prefix-source/v1",
        "key": key.to_dict(),
        "path_seed": int(path_seed),
        "path_seed_set": key.path_seed_set,
        "path_stack_sha256": path_stack_sha256,
        "prediction_sha256": prediction_sha256,
        "ordered_path_sha256": list(hashes),
        "prefixes": prefixes,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "calibration": {
            "scale": float(calibration_scale),
            "floor_variance": float(calibration_floor_variance),
        },
        "mechanism_parameters": dict(mechanism_parameters),
        "n_semantics": "independent complete reverse solve; never a truncation of N=40",
    }
    payload["source_manifest_sha256"] = canonical_sha256(payload)
    return payload


def read_e7_source_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "paper2.e7-prefix-source/v1":
        raise ValueError("unsupported E7 prefix source schema")
    observed = payload.get("source_manifest_sha256")
    expected = canonical_sha256(
        {name: value for name, value in payload.items() if name != "source_manifest_sha256"}
    )
    if observed != expected:
        raise ValueError("E7 prefix source semantic hash mismatch")
    key = MatrixKey.from_dict(payload["key"])
    validate_e7_key(key, source=True)
    return payload


def build_e7_derivation_payload(
    *,
    key: MatrixKey,
    source_inference_manifest_sha256: str,
    source_manifest_file_sha256: str,
    path_stack_sha256: str,
    prediction_sha256: str,
    prefix_record: Mapping[str, Any],
) -> dict[str, Any]:
    validate_e7_key(key)
    if key.path_count == max(E7_PATH_COUNTS):
        raise ValueError("M=32 is the E7 source cell, not a derived prefix")
    for digest in (
        source_inference_manifest_sha256,
        source_manifest_file_sha256,
        path_stack_sha256,
        prediction_sha256,
    ):
        if not is_sha256(digest):
            raise ValueError("E7 derivation bindings must be SHA-256 digests")
    if int(prefix_record.get("path_count", -1)) != key.path_count:
        raise ValueError("E7 prefix record count differs from derived matrix key")
    payload: dict[str, Any] = {
        "schema": "paper2.e7-prefix-derivation/v1",
        "key": key.to_dict(),
        "source_inference_manifest_sha256": source_inference_manifest_sha256,
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "path_stack_sha256": path_stack_sha256,
        "prediction_sha256": prediction_sha256,
        "prefix": dict(prefix_record),
    }
    payload["derivation_manifest_sha256"] = canonical_sha256(payload)
    return payload


def read_e7_derivation_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "paper2.e7-prefix-derivation/v1":
        raise ValueError("unsupported E7 prefix derivation schema")
    observed = payload.get("derivation_manifest_sha256")
    expected = canonical_sha256(
        {
            name: value
            for name, value in payload.items()
            if name != "derivation_manifest_sha256"
        }
    )
    if observed != expected:
        raise ValueError("E7 prefix derivation semantic hash mismatch")
    validate_e7_key(MatrixKey.from_dict(payload["key"]))
    return payload


def _artifact(manifest: InferenceManifest, name: str) -> ArtifactDigest:
    matches = [artifact for artifact in manifest.artifacts if artifact.name == name]
    if len(matches) != 1:
        raise ValueError(f"E7 inference manifest requires exactly one {name} artifact")
    return matches[0]


def _group_key(key: MatrixKey) -> tuple[Any, ...]:
    return (
        key.tile_id,
        key.method,
        key.budget,
        key.noise_replicate,
        key.training_seed,
        key.model_identity,
        key.path_seed_set,
        key.step_count,
        key.dc_mode,
    )


@lru_cache(maxsize=8)
def _tree_sha256(root: Path) -> str:
    return canonical_sha256(
        [
            (path.relative_to(root).as_posix(), file_sha256(path))
            for path in sorted(root.rglob("*.py"))
        ]
    )


@dataclass(frozen=True)
class _E7SourceBundle:
    root: Path
    source: InferenceManifest
    source_json_path: Path
    stack_path: Path
    audit_projection_path: Path
    source_payload: Mapping[str, Any]
    path_stack: np.ndarray
    ordered_hashes: tuple[str, ...]
    source_runtime: Mapping[str, Any]
    source_manifest_file_sha256: str
    path_stack_file_sha256: str


def _load_e7_source_bundle(
    *,
    source_inference_manifest_path: str | Path,
    paper2_root: str | Path,
) -> _E7SourceBundle:
    """Load and verify one immutable M=32 source exactly once."""

    root = Path(paper2_root).resolve()
    source_inference_path = guard_paper2_path(
        source_inference_manifest_path, paper2_root=root
    )
    source = InferenceManifest.read(source_inference_path)
    validate_e7_key(source.key, source=True)
    if source.status != "complete":
        raise ValueError("E7 prefix source inference is not complete")
    source_json_artifact = _artifact(source, "e7_prefix_source")
    stack_artifact = _artifact(source, "e7_path_stack")
    audit_projection_artifact = _artifact(source, "audit_footprint_projection")
    source_runtime_artifact = _artifact(source, "runtime")
    source_json_path = guard_paper2_path(
        root / source_json_artifact.relative_path, paper2_root=root
    )
    stack_path = guard_paper2_path(
        root / stack_artifact.relative_path, paper2_root=root
    )
    audit_projection_path = guard_paper2_path(
        root / audit_projection_artifact.relative_path, paper2_root=root
    )
    source_runtime_path = guard_paper2_path(
        root / source_runtime_artifact.relative_path, paper2_root=root
    )
    source_manifest_file_sha256 = file_sha256(source_json_path)
    path_stack_file_sha256 = file_sha256(stack_path)
    if source_manifest_file_sha256 != source_json_artifact.sha256:
        raise ValueError("E7 source-manifest artifact hash mismatch")
    if path_stack_file_sha256 != stack_artifact.sha256:
        raise ValueError("E7 path-stack artifact hash mismatch")
    if file_sha256(audit_projection_path) != audit_projection_artifact.sha256:
        raise ValueError("E7 audit projection artifact hash mismatch")
    if file_sha256(source_runtime_path) != source_runtime_artifact.sha256:
        raise ValueError("E7 source runtime artifact hash mismatch")
    source_payload = read_e7_source_payload(source_json_path)
    if MatrixKey.from_dict(source_payload["key"]) != source.key:
        raise ValueError("E7 source payload key differs from source inference")
    if source_payload["path_stack_sha256"] != path_stack_file_sha256:
        raise ValueError("E7 source payload binds a different path stack")
    if source_payload["calibration_manifest_sha256"] != dict(source.trace_hashes).get(
        "calibration_manifest_sha256"
    ):
        raise ValueError("E7 source calibration binding differs from inference trace")
    with np.load(stack_path, allow_pickle=False) as archive:
        if set(archive.files) != {"paths"}:
            raise ValueError("E7 path-stack NPZ must contain only paths")
        path_stack = np.asarray(archive["paths"], dtype=np.float32)
    hashes = ordered_path_sha256(path_stack)
    if list(hashes) != source_payload["ordered_path_sha256"]:
        raise ValueError("E7 source ordered path hashes differ from stack")
    path_stack.setflags(write=False)
    mechanism = source_payload["mechanism_parameters"]
    required_mechanism = {
        "g_min",
        "alpha",
        "ddpm_cosine_offset",
        "ddpm_eta",
        "generative",
    }
    if set(mechanism) != required_mechanism:
        raise ValueError("E7 source mechanism parameters are incomplete")
    source_runtime = json.loads(source_runtime_path.read_text(encoding="utf-8"))
    if (
        source_runtime.get("schema") != "paper2.inference-runtime/v1"
        or source_runtime.get("status") != "complete"
        or float(source_runtime.get("area_km2", 0.0)) <= 0
    ):
        raise ValueError("E7 source runtime contract is incomplete")
    return _E7SourceBundle(
        root=root,
        source=source,
        source_json_path=source_json_path,
        stack_path=stack_path,
        audit_projection_path=audit_projection_path,
        source_payload=source_payload,
        path_stack=path_stack,
        ordered_hashes=hashes,
        source_runtime=source_runtime,
        source_manifest_file_sha256=source_manifest_file_sha256,
        path_stack_file_sha256=path_stack_file_sha256,
    )


def _expected_e7_context(
    *,
    bundle: _E7SourceBundle,
    expected_matrix_path: str | Path | None,
    expected_by_id: Mapping[str, ExpectedCell] | None,
    expected_analysis_lock_sha256: str | None,
    formal: bool,
) -> tuple[Mapping[str, ExpectedCell] | None, str | None]:
    if expected_matrix_path is not None and expected_by_id is not None:
        raise ValueError("E7 derivation accepts one expected-matrix representation")
    if formal and expected_matrix_path is None and expected_by_id is None:
        raise ValueError("formal E7 derivation requires the G5 expected matrix")
    expected_analysis_sha256 = expected_analysis_lock_sha256
    if expected_by_id is not None:
        if expected_analysis_sha256 is None:
            raise ValueError("registered E7 expected cells require the analysis-lock hash")
    elif expected_matrix_path is not None:
        expected_path = guard_paper2_path(
            expected_matrix_path, paper2_root=bundle.root
        )
        expected = ExpectedMatrix.read(expected_path)
        expected_by_id = {cell.key.identifier: cell for cell in expected.cells}
        expected_analysis_sha256 = expected.analysis_lock_sha256
    if expected_by_id is not None:
        source_expected_cell = expected_by_id.get(bundle.source.key.identifier)
        if source_expected_cell is None:
            raise ValueError("E7 M=32 source key is absent from expected matrix")
        if (
            dict(bundle.source.trace_hashes).get("analysis_lock_sha256")
            != expected_analysis_sha256
        ):
            raise ValueError("E7 source inference differs from expected analysis lock")
        source_trace = dict(bundle.source.trace_hashes)
        for name, digest in source_expected_cell.locked_trace_hashes:
            if source_trace.get(name) != digest:
                raise ValueError(f"E7 source trace {name} differs from expected lock")
    return expected_by_id, expected_analysis_sha256


def _derive_e7_prefix_from_bundle(
    *,
    bundle: _E7SourceBundle,
    prefix_count: int,
    output: str | Path,
    expected_by_id: Mapping[str, ExpectedCell] | None,
    source_code_sha256: str,
    shared_setup_seconds: float,
    shared_setup_fraction: float,
) -> InferenceManifest:
    started = time.monotonic()
    source = bundle.source
    root = bundle.root
    calibration = bundle.source_payload["calibration"]
    mean, variance = prefix_statistics(
        bundle.path_stack,
        prefix_count,
        calibration_scale=float(calibration["scale"]),
        calibration_floor_variance=float(calibration["floor_variance"]),
    )
    expected_prefix = bundle.source_payload["prefixes"][str(prefix_count)]
    if (
        _array_sha256(mean) != expected_prefix["mean_sha256"]
        or _array_sha256(variance) != expected_prefix["variance_sha256"]
        or list(bundle.ordered_hashes[:prefix_count])
        != expected_prefix["ordered_path_sha256"]
    ):
        raise ValueError("derived E7 aggregate differs from registered source prefix")

    key = MatrixKey(
        experiment=source.key.experiment,
        tile_id=source.key.tile_id,
        method=source.key.method,
        budget=source.key.budget,
        noise_replicate=source.key.noise_replicate,
        training_seed=source.key.training_seed,
        condition=e7_condition(prefix_count, source.key.step_count),
        model_identity=source.key.model_identity,
        path_seed_set=source.key.path_seed_set,
        path_count=prefix_count,
        step_count=source.key.step_count,
        dc_mode=source.key.dc_mode,
    )
    expected_cell = None if expected_by_id is None else expected_by_id.get(key.identifier)
    if expected_by_id is not None and expected_cell is None:
        raise ValueError("derived E7 key is absent from expected matrix")

    output_path = guard_paper2_path(output, paper2_root=root)
    stream = io.BytesIO()
    np.savez_compressed(stream, mean=mean, variance=variance)
    write_immutable_bytes(output_path, stream.getvalue(), paper2_root=root)
    mechanism = bundle.source_payload["mechanism_parameters"]
    trace = dict(source.trace_hashes)
    trace["mechanism_parameters_sha256"] = mechanism_parameters_sha256(
        g_min=float(mechanism["g_min"]),
        alpha=float(mechanism["alpha"]),
        ddpm_cosine_offset=float(mechanism["ddpm_cosine_offset"]),
        ddpm_eta=float(mechanism["ddpm_eta"]),
        steps=key.step_count,
        paths=key.path_count,
        generative=bool(mechanism["generative"]),
    )
    trace["code_sha256"] = source_code_sha256
    if expected_cell is not None:
        for name, digest in expected_cell.locked_trace_hashes:
            if trace.get(name) != digest:
                raise ValueError(f"derived E7 trace {name} differs from expected lock")

    derivation_payload = build_e7_derivation_payload(
        key=key,
        source_inference_manifest_sha256=source.sha256,
        source_manifest_file_sha256=bundle.source_manifest_file_sha256,
        path_stack_sha256=bundle.path_stack_file_sha256,
        prediction_sha256=file_sha256(output_path),
        prefix_record=expected_prefix,
    )
    derivation_path = output_path.with_name("e7-prefix-derivation.json")
    write_canonical_json(derivation_path, derivation_payload, paper2_root=root)
    derived_audit_projection_path = output_path.with_name(
        "audit-footprint-path-projections.npz"
    )
    write_audit_projection_prefix(
        derived_audit_projection_path,
        source_projection_path=bundle.audit_projection_path,
        prefix_count=prefix_count,
        matrix_key_identifier=key.identifier,
        paper2_root=root,
    )
    runtime_path = output_path.with_name("runtime.json")
    source_runtime = bundle.source_runtime
    derived_wall = time.monotonic() - started + (
        shared_setup_seconds * shared_setup_fraction
    )
    runtime = {
        "schema": "paper2.inference-runtime/v1",
        "status": "complete",
        "wall_seconds": derived_wall,
        "area_km2": float(source_runtime["area_km2"]),
        "area_basis": source_runtime["area_basis"],
        "valid_pixel_count": int(source_runtime["valid_pixel_count"]),
        "pixel_area_m2": float(source_runtime["pixel_area_m2"]),
        "seconds_per_km2": derived_wall / float(source_runtime["area_km2"]),
        "nfe": 0,
        "incremental_model_nfe": 0,
        "shared_source_nfe": source.key.path_count * source.key.step_count,
        "shared_source_setup_seconds": shared_setup_seconds,
        "shared_source_setup_allocation_fraction": shared_setup_fraction,
        "execution_mode": "derived-strict-prefix",
        "peak_gpu_memory_bytes": 0,
        "peak_gpu_memory_allocated_bytes": 0,
        "peak_gpu_memory_reserved_bytes": 0,
        "device_index": source_runtime.get("device_index"),
        "device_name": source_runtime.get("device_name"),
        "device_uuid": source_runtime.get("device_uuid"),
        "device_identity_source": "observed_M32_source_runtime",
    }
    write_canonical_json(runtime_path, runtime, paper2_root=root)
    manifest = InferenceManifest(
        key=key,
        status="complete",
        trace_hashes=tuple(sorted(trace.items())),
        artifacts=(
            ArtifactDigest(
                "prediction",
                output_path.relative_to(root).as_posix(),
                file_sha256(output_path),
            ),
            ArtifactDigest(
                "runtime",
                runtime_path.relative_to(root).as_posix(),
                file_sha256(runtime_path),
            ),
            ArtifactDigest(
                "e7_prefix_derivation",
                derivation_path.relative_to(root).as_posix(),
                file_sha256(derivation_path),
            ),
            ArtifactDigest(
                "audit_footprint_projection",
                derived_audit_projection_path.relative_to(root).as_posix(),
                file_sha256(derived_audit_projection_path),
            ),
        ),
    )
    manifest_path = output_path.with_name("inference-manifest.json")
    manifest.write(manifest_path, paper2_root=root)
    return manifest


def derive_e7_prefixes(
    *,
    source_inference_manifest_path: str | Path,
    outputs: Mapping[int, str | Path],
    paper2_root: str | Path,
    expected_matrix_path: str | Path | None = None,
    expected_by_id: Mapping[str, ExpectedCell] | None = None,
    expected_analysis_lock_sha256: str | None = None,
    source_code_sha256: str | None = None,
    formal: bool = False,
) -> dict[int, InferenceManifest]:
    """Derive any requested strict M-prefixes after one M=32 load and audit."""

    requested = tuple(count for count in E7_PATH_COUNTS[:-1] if count in outputs)
    if not requested or set(outputs) != set(requested):
        raise ValueError("derived E7 prefixes must be a nonempty subset of 2,4,8,16")
    root = Path(paper2_root).resolve()
    resolved_outputs = tuple(
        guard_paper2_path(outputs[count], paper2_root=root) for count in requested
    )
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("derived E7 prefixes require distinct output paths")

    setup_started = time.monotonic()
    bundle = _load_e7_source_bundle(
        source_inference_manifest_path=source_inference_manifest_path,
        paper2_root=root,
    )
    registered, _ = _expected_e7_context(
        bundle=bundle,
        expected_matrix_path=expected_matrix_path,
        expected_by_id=expected_by_id,
        expected_analysis_lock_sha256=expected_analysis_lock_sha256,
        formal=formal,
    )
    code_sha256 = source_code_sha256 or _tree_sha256(root / "src")
    shared_setup_seconds = time.monotonic() - setup_started
    allocation = 1.0 / len(requested)
    return {
        count: _derive_e7_prefix_from_bundle(
            bundle=bundle,
            prefix_count=count,
            output=outputs[count],
            expected_by_id=registered,
            source_code_sha256=code_sha256,
            shared_setup_seconds=shared_setup_seconds,
            shared_setup_fraction=allocation,
        )
        for count in requested
    }


def derive_e7_prefix(
    *,
    source_inference_manifest_path: str | Path,
    prefix_count: int,
    output: str | Path,
    paper2_root: str | Path,
    expected_matrix_path: str | Path | None = None,
    expected_by_id: Mapping[str, ExpectedCell] | None = None,
    expected_analysis_lock_sha256: str | None = None,
    source_code_sha256: str | None = None,
    formal: bool = False,
) -> InferenceManifest:
    """Create one M<32 cell from the first M paths of an immutable M=32 source."""

    return derive_e7_prefixes(
        source_inference_manifest_path=source_inference_manifest_path,
        outputs={prefix_count: output},
        paper2_root=paper2_root,
        expected_matrix_path=expected_matrix_path,
        expected_by_id=expected_by_id,
        expected_analysis_lock_sha256=expected_analysis_lock_sha256,
        source_code_sha256=source_code_sha256,
        formal=formal,
    )[prefix_count]


def audit_e7_prefix_manifests(
    manifests: Sequence[InferenceManifest],
    *,
    paper2_root: str | Path,
) -> tuple[str, ...]:
    """Verify every E7 M cell is an exact prefix of one separately solved N source."""

    root = Path(paper2_root).resolve()
    groups: dict[tuple[Any, ...], dict[int, InferenceManifest]] = {}
    errors: list[str] = []
    for manifest in manifests:
        if manifest.key.experiment != "E7":
            continue
        try:
            validate_e7_key(manifest.key)
        except ValueError as exc:
            errors.append(f"{manifest.key.identifier}: {exc}")
            continue
        group = groups.setdefault(_group_key(manifest.key), {})
        if manifest.key.path_count in group:
            errors.append(
                f"{manifest.key.identifier}: duplicate E7 M={manifest.key.path_count} prefix"
            )
        group[manifest.key.path_count] = manifest

    for group_key, by_count in sorted(groups.items(), key=lambda item: repr(item[0])):
        label = canonical_sha256(group_key)
        missing = set(E7_PATH_COUNTS).difference(by_count)
        if missing:
            errors.append(f"E7 group {label}: missing M prefixes {sorted(missing)}")
            continue
        source = by_count[max(E7_PATH_COUNTS)]
        try:
            source_json_artifact = _artifact(source, "e7_prefix_source")
            stack_artifact = _artifact(source, "e7_path_stack")
            prediction_artifact = _artifact(source, "prediction")
            source_audit_artifact = _artifact(source, "audit_footprint_projection")
            source_json_path = guard_paper2_path(
                root / source_json_artifact.relative_path, paper2_root=root
            )
            stack_path = guard_paper2_path(
                root / stack_artifact.relative_path, paper2_root=root
            )
            prediction_path = guard_paper2_path(
                root / prediction_artifact.relative_path, paper2_root=root
            )
            source_audit_path = guard_paper2_path(
                root / source_audit_artifact.relative_path, paper2_root=root
            )
            if file_sha256(source_audit_path) != source_audit_artifact.sha256:
                raise ValueError("source audit projection hash mismatch")
            source_audit_metadata, source_audit_values, _, _ = (
                read_audit_path_projections(source_audit_path)
            )
            if (
                len(source_audit_values) != 32
                or source_audit_metadata.get("matrix_key_identifier")
                != source.key.identifier
            ):
                raise ValueError("source audit projection is not the M=32 source")
            source_payload = read_e7_source_payload(source_json_path)
            if MatrixKey.from_dict(source_payload["key"]) != source.key:
                raise ValueError("source payload matrix key differs from inference manifest")
            mechanism = source_payload["mechanism_parameters"]
            expected_mechanism = mechanism_parameters_sha256(
                g_min=float(mechanism["g_min"]),
                alpha=float(mechanism["alpha"]),
                ddpm_cosine_offset=float(mechanism["ddpm_cosine_offset"]),
                ddpm_eta=float(mechanism["ddpm_eta"]),
                steps=source.key.step_count,
                paths=source.key.path_count,
                generative=bool(mechanism["generative"]),
            )
            if dict(source.trace_hashes).get("mechanism_parameters_sha256") != expected_mechanism:
                raise ValueError("source mechanism parameters differ from inference trace")
            if source_payload["path_stack_sha256"] != file_sha256(stack_path):
                raise ValueError("source payload path-stack hash mismatch")
            if source_payload["prediction_sha256"] != file_sha256(prediction_path):
                raise ValueError("source payload prediction hash mismatch")
            with np.load(stack_path, allow_pickle=False) as archive:
                if set(archive.files) != {"paths"}:
                    raise ValueError("E7 path-stack NPZ must contain only paths")
                stack = np.asarray(archive["paths"], dtype=np.float32)
            if stack.shape[0] != max(E7_PATH_COUNTS):
                raise ValueError("E7 source stack must contain exactly 32 paths")
            hashes = ordered_path_sha256(stack)
            if list(hashes) != source_payload["ordered_path_sha256"]:
                raise ValueError("source ordered path hashes do not match path stack")
            calibration = source_payload["calibration"]
            expected_prefixes = {
                str(count): _prefix_record(
                    stack,
                    count,
                    path_hashes=hashes,
                    calibration_scale=float(calibration["scale"]),
                    calibration_floor_variance=float(calibration["floor_variance"]),
                )
                for count in E7_PATH_COUNTS
            }
            if source_payload["prefixes"] != expected_prefixes:
                raise ValueError("source prefix statistics/hashes do not match path stack")
            with np.load(prediction_path, allow_pickle=False) as archive:
                source_mean = np.asarray(archive["mean"], dtype=np.float32)
                source_variance = np.asarray(archive["variance"], dtype=np.float32)
            prefix32 = expected_prefixes[str(max(E7_PATH_COUNTS))]
            if (
                _array_sha256(source_mean) != prefix32["mean_sha256"]
                or _array_sha256(source_variance) != prefix32["variance_sha256"]
            ):
                raise ValueError("M=32 prediction is not the registered source aggregate")

            source_json_file_hash = file_sha256(source_json_path)
            for count in E7_PATH_COUNTS[:-1]:
                derived = by_count[count]
                derivation_artifact = _artifact(derived, "e7_prefix_derivation")
                derived_prediction_artifact = _artifact(derived, "prediction")
                derived_audit_artifact = _artifact(
                    derived, "audit_footprint_projection"
                )
                derivation_path = guard_paper2_path(
                    root / derivation_artifact.relative_path, paper2_root=root
                )
                derived_prediction_path = guard_paper2_path(
                    root / derived_prediction_artifact.relative_path, paper2_root=root
                )
                derived_audit_path = guard_paper2_path(
                    root / derived_audit_artifact.relative_path, paper2_root=root
                )
                derivation = read_e7_derivation_payload(derivation_path)
                if MatrixKey.from_dict(derivation["key"]) != derived.key:
                    raise ValueError(f"M={count} derivation key mismatch")
                if derivation["source_inference_manifest_sha256"] != source.sha256:
                    raise ValueError(f"M={count} binds a different source inference")
                if derivation["source_manifest_file_sha256"] != source_json_file_hash:
                    raise ValueError(f"M={count} binds a different source manifest")
                if derivation["path_stack_sha256"] != file_sha256(stack_path):
                    raise ValueError(f"M={count} binds a different path stack")
                if derivation["prefix"] != expected_prefixes[str(count)]:
                    raise ValueError(f"M={count} is not the strict ordered source prefix")
                derived_mechanism = mechanism_parameters_sha256(
                    g_min=float(mechanism["g_min"]),
                    alpha=float(mechanism["alpha"]),
                    ddpm_cosine_offset=float(mechanism["ddpm_cosine_offset"]),
                    ddpm_eta=float(mechanism["ddpm_eta"]),
                    steps=derived.key.step_count,
                    paths=derived.key.path_count,
                    generative=bool(mechanism["generative"]),
                )
                if (
                    dict(derived.trace_hashes).get("mechanism_parameters_sha256")
                    != derived_mechanism
                ):
                    raise ValueError(f"M={count} mechanism trace is not prefix-specific")
                source_trace = dict(source.trace_hashes)
                derived_trace = dict(derived.trace_hashes)
                for name, digest in source_trace.items():
                    if name == "mechanism_parameters_sha256":
                        continue
                    if derived_trace.get(name) != digest:
                        raise ValueError(
                            f"M={count} trace {name} differs from its M=32 source"
                        )
                if set(derived_trace) != set(source_trace):
                    raise ValueError(f"M={count} trace fields differ from its M=32 source")
                if derivation["prediction_sha256"] != file_sha256(
                    derived_prediction_path
                ):
                    raise ValueError(f"M={count} prediction hash mismatch")
                if file_sha256(derived_audit_path) != derived_audit_artifact.sha256:
                    raise ValueError(f"M={count} audit projection hash mismatch")
                audit_metadata, audit_values, _, _ = read_audit_path_projections(
                    derived_audit_path
                )
                if (
                    audit_metadata.get("matrix_key_identifier") != derived.key.identifier
                    or audit_metadata.get("source_projection_sha256")
                    != file_sha256(source_audit_path)
                    or not np.array_equal(audit_values, source_audit_values[:count])
                ):
                    raise ValueError(
                        f"M={count} audit projections are not the ordered source prefix"
                    )
                with np.load(derived_prediction_path, allow_pickle=False) as archive:
                    mean = np.asarray(archive["mean"], dtype=np.float32)
                    variance = np.asarray(archive["variance"], dtype=np.float32)
                prefix = expected_prefixes[str(count)]
                if (
                    _array_sha256(mean) != prefix["mean_sha256"]
                    or _array_sha256(variance) != prefix["variance_sha256"]
                ):
                    raise ValueError(f"M={count} prediction differs from source prefix")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"E7 group {label}: {exc}")
    return tuple(sorted(set(errors)))
