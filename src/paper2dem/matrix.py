from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .hashing import (
    PAPER2_ROOT,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)


@dataclass(frozen=True, order=True)
class MatrixKey:
    experiment: str
    tile_id: str
    method: str
    budget: int
    noise_replicate: int
    training_seed: int
    condition: str = "main"
    model_identity: str = ""
    path_seed_set: str = "default"
    path_count: int = 1
    step_count: int = 0
    dc_mode: str = "D1"

    def validate(self) -> None:
        for name, value in (
            ("experiment", self.experiment),
            ("tile_id", self.tile_id),
            ("method", self.method),
            ("condition", self.condition),
            ("path_seed_set", self.path_seed_set),
            ("dc_mode", self.dc_mode),
        ):
            if not value:
                raise ValueError(f"matrix {name} cannot be empty")
        if (
            self.budget < 0
            or self.noise_replicate < 0
            or self.training_seed < 0
            or self.path_count <= 0
            or self.step_count < 0
        ):
            raise ValueError("matrix numeric key fields must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "experiment": self.experiment,
            "tile_id": self.tile_id,
            "method": self.method,
            "budget": self.budget,
            "noise_replicate": self.noise_replicate,
            "training_seed": self.training_seed,
            "condition": self.condition,
            "model_identity": self.model_identity or self.method,
            "path_seed_set": self.path_seed_set,
            "path_count": self.path_count,
            "step_count": self.step_count,
            "dc_mode": self.dc_mode,
        }

    @property
    def identifier(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MatrixKey":
        return cls(
            experiment=str(value["experiment"]),
            tile_id=str(value["tile_id"]),
            method=str(value["method"]),
            budget=int(value["budget"]),
            noise_replicate=int(value["noise_replicate"]),
            training_seed=int(value["training_seed"]),
            condition=str(value.get("condition", "main")),
            model_identity=str(value.get("model_identity", "")),
            path_seed_set=str(value.get("path_seed_set", "default")),
            path_count=int(value.get("path_count", 1)),
            step_count=int(value.get("step_count", 0)),
            dc_mode=str(value.get("dc_mode", "D1")),
        )


@dataclass(frozen=True)
class ExpectedCell:
    key: MatrixKey
    required_trace_hashes: tuple[str, ...]
    locked_trace_hashes: tuple[tuple[str, str], ...] = ()
    sealed_reference_sha256: str | None = None
    sealed_reference_grid_sha256: str | None = None

    def validate(self) -> None:
        self.key.validate()
        if not self.required_trace_hashes:
            raise ValueError("each expected cell requires at least one trace hash")
        if len(set(self.required_trace_hashes)) != len(self.required_trace_hashes):
            raise ValueError("required trace hash fields must be unique")
        locked_names = [name for name, _ in self.locked_trace_hashes]
        if len(locked_names) != len(set(locked_names)):
            raise ValueError("locked trace hash fields must be unique")
        if not set(locked_names).issubset(self.required_trace_hashes):
            raise ValueError("locked trace hashes must also be required")
        for name in self.required_trace_hashes:
            if not name or not name.endswith("_sha256"):
                raise ValueError("trace hash field names must be nonempty and end in _sha256")
        for name, digest in self.locked_trace_hashes:
            if not is_sha256(digest):
                raise ValueError(f"locked trace {name} is not a SHA-256 digest")
        reference_values = (
            self.sealed_reference_sha256,
            self.sealed_reference_grid_sha256,
        )
        if any(value is not None for value in reference_values) and not all(
            value is not None and is_sha256(value) for value in reference_values
        ):
            raise ValueError("sealed reference file/grid hashes must be locked together")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "key": self.key.to_dict(),
            "required_trace_hashes": list(self.required_trace_hashes),
            "locked_trace_hashes": dict(self.locked_trace_hashes),
            "sealed_reference_sha256": self.sealed_reference_sha256,
            "sealed_reference_grid_sha256": self.sealed_reference_grid_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExpectedCell":
        return cls(
            key=MatrixKey.from_dict(value["key"]),
            required_trace_hashes=tuple(str(item) for item in value["required_trace_hashes"]),
            locked_trace_hashes=tuple(
                sorted((str(name), str(digest)) for name, digest in value.get("locked_trace_hashes", {}).items())
            ),
            sealed_reference_sha256=(
                None
                if value.get("sealed_reference_sha256") is None
                else str(value["sealed_reference_sha256"])
            ),
            sealed_reference_grid_sha256=(
                None
                if value.get("sealed_reference_grid_sha256") is None
                else str(value["sealed_reference_grid_sha256"])
            ),
        )


@dataclass(frozen=True)
class ExpectedMatrix:
    SCHEMA: ClassVar[str] = "paper2.expected-matrix/v1"

    analysis_lock_sha256: str
    cells: tuple[ExpectedCell, ...]

    def validate(self) -> None:
        if not is_sha256(self.analysis_lock_sha256):
            raise ValueError("analysis_lock_sha256 must be a SHA-256 digest")
        if not self.cells:
            raise ValueError("expected matrix cannot be empty")
        identifiers: set[str] = set()
        for cell in self.cells:
            cell.validate()
            if cell.key.identifier in identifiers:
                raise ValueError(f"duplicate expected matrix key: {cell.key.to_dict()}")
            identifiers.add(cell.key.identifier)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.SCHEMA,
            "analysis_lock_sha256": self.analysis_lock_sha256,
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        return write_canonical_json(path, self.to_dict(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "ExpectedMatrix":
        payload = read_json(path)
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported expected matrix schema: {payload.get('schema')}")
        result = cls(
            analysis_lock_sha256=str(payload["analysis_lock_sha256"]),
            cells=tuple(ExpectedCell.from_dict(item) for item in payload["cells"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ArtifactDigest:
    name: str
    relative_path: str
    sha256: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("artifact name cannot be empty")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("artifact paths must be relative to paper2 and cannot traverse parents")
        if not is_sha256(self.sha256):
            raise ValueError("artifact digest must be SHA-256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"name": self.name, "relative_path": self.relative_path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactDigest":
        return cls(
            name=str(value["name"]),
            relative_path=str(value["relative_path"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class InferenceManifest:
    SCHEMA: ClassVar[str] = "paper2.inference-manifest/v1"

    key: MatrixKey
    status: str
    trace_hashes: tuple[tuple[str, str], ...]
    artifacts: tuple[ArtifactDigest, ...]

    def validate(self) -> None:
        self.key.validate()
        if self.status not in {"complete", "failed", "skipped"}:
            raise ValueError(f"unsupported inference status: {self.status}")
        trace_names = [name for name, _ in self.trace_hashes]
        if len(trace_names) != len(set(trace_names)):
            raise ValueError("trace hash field names must be unique")
        for name, digest in self.trace_hashes:
            if not name.endswith("_sha256") or not is_sha256(digest):
                raise ValueError(f"invalid trace hash: {name}")
        artifact_names = [item.name for item in self.artifacts]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("artifact names must be unique")
        for artifact in self.artifacts:
            artifact.validate()
        if self.status == "complete" and "prediction" not in artifact_names:
            raise ValueError("complete inference manifests require a prediction artifact")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.SCHEMA,
            "key": self.key.to_dict(),
            "status": self.status,
            "trace_hashes": dict(self.trace_hashes),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        return write_canonical_json(path, self.to_dict(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "InferenceManifest":
        payload = read_json(path)
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported inference manifest schema: {payload.get('schema')}")
        result = cls(
            key=MatrixKey.from_dict(payload["key"]),
            status=str(payload["status"]),
            trace_hashes=tuple(
                sorted((str(name), str(digest)) for name, digest in payload["trace_hashes"].items())
            ),
            artifacts=tuple(ArtifactDigest.from_dict(item) for item in payload["artifacts"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class EvaluationManifest:
    """One sealed-reference evaluation joined one-to-one to an inference cell."""

    SCHEMA: ClassVar[str] = "paper2.evaluation-manifest/v1"

    key: MatrixKey
    status: str
    inference_manifest_sha256: str
    sealed_reference_sha256: str
    artifacts: tuple[ArtifactDigest, ...]
    sealed_reference_grid_sha256: str | None = None

    def validate(self) -> None:
        self.key.validate()
        if self.status not in {"complete", "failed", "skipped"}:
            raise ValueError(f"unsupported evaluation status: {self.status}")
        if not is_sha256(self.inference_manifest_sha256):
            raise ValueError("evaluation must bind a semantic inference-manifest SHA-256")
        if not is_sha256(self.sealed_reference_sha256):
            raise ValueError("evaluation must bind the sealed reference SHA-256")
        if self.sealed_reference_grid_sha256 is not None and not is_sha256(
            self.sealed_reference_grid_sha256
        ):
            raise ValueError("evaluation sealed reference grid hash must be SHA-256")
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("evaluation artifact names must be unique")
        for artifact in self.artifacts:
            artifact.validate()
        if self.status == "complete" and "metrics" not in names:
            raise ValueError("complete evaluations require a metrics artifact")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.SCHEMA,
            "key": self.key.to_dict(),
            "status": self.status,
            "inference_manifest_sha256": self.inference_manifest_sha256,
            "sealed_reference_sha256": self.sealed_reference_sha256,
            "sealed_reference_grid_sha256": self.sealed_reference_grid_sha256,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        return write_canonical_json(path, self.to_dict(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "EvaluationManifest":
        payload = read_json(path)
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported evaluation manifest schema: {payload.get('schema')}")
        result = cls(
            key=MatrixKey.from_dict(payload["key"]),
            status=str(payload["status"]),
            inference_manifest_sha256=str(payload["inference_manifest_sha256"]),
            sealed_reference_sha256=str(payload["sealed_reference_sha256"]),
            artifacts=tuple(ArtifactDigest.from_dict(item) for item in payload["artifacts"]),
            sealed_reference_grid_sha256=(
                None
                if payload.get("sealed_reference_grid_sha256") is None
                else str(payload["sealed_reference_grid_sha256"])
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class MatrixAudit:
    ok: bool
    expected_cells: int
    observed_manifests: int
    observed_evaluations: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    incomplete_by_method: tuple[tuple[str, int], ...]
    hash_errors: tuple[str, ...]
    artifact_errors: tuple[str, ...]
    evaluation_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "paper2.matrix-audit/v1",
            "ok": self.ok,
            "expected_cells": self.expected_cells,
            "observed_manifests": self.observed_manifests,
            "observed_evaluations": self.observed_evaluations,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "duplicate_keys": list(self.duplicate_keys),
            "incomplete_by_method": dict(self.incomplete_by_method),
            "hash_errors": list(self.hash_errors),
            "artifact_errors": list(self.artifact_errors),
            "evaluation_errors": list(self.evaluation_errors),
        }


class MatrixClosureError(RuntimeError):
    def __init__(self, audit: MatrixAudit):
        self.audit = audit
        super().__init__(
            "expected matrix is not closed: "
            f"missing={len(audit.missing_keys)}, unexpected={len(audit.unexpected_keys)}, "
            f"duplicates={len(audit.duplicate_keys)}, hash_errors={len(audit.hash_errors)}, "
            f"artifact_errors={len(audit.artifact_errors)}, "
            f"evaluation_errors={len(audit.evaluation_errors)}"
        )


def audit_expected_matrix(
    expected: ExpectedMatrix,
    manifests: tuple[InferenceManifest, ...] | list[InferenceManifest],
    evaluation_manifests: tuple[EvaluationManifest, ...] | list[EvaluationManifest] = (),
    *,
    paper2_root: str | Path = PAPER2_ROOT,
) -> MatrixAudit:
    expected.validate()
    expected_by_id = {cell.key.identifier: cell for cell in expected.cells}
    manifest_by_id: dict[str, InferenceManifest] = {}
    evaluation_by_id: dict[str, EvaluationManifest] = {}
    duplicates: list[str] = []
    for manifest in manifests:
        manifest.validate()
        identifier = manifest.key.identifier
        if identifier in manifest_by_id:
            duplicates.append(identifier)
        else:
            manifest_by_id[identifier] = manifest
    for manifest in evaluation_manifests:
        manifest.validate()
        identifier = manifest.key.identifier
        if identifier in evaluation_by_id:
            duplicates.append(f"evaluation:{identifier}")
        else:
            evaluation_by_id[identifier] = manifest
    expected_ids = set(expected_by_id)
    observed_ids = set(manifest_by_id)
    missing = sorted(expected_ids.difference(observed_ids))
    unexpected = sorted(observed_ids.difference(expected_ids))
    hash_errors: list[str] = []
    artifact_errors: list[str] = []
    evaluation_errors: list[str] = []
    incomplete: dict[str, int] = {}
    root = Path(paper2_root).resolve()
    for identifier in sorted(expected_ids.intersection(observed_ids)):
        cell = expected_by_id[identifier]
        manifest = manifest_by_id[identifier]
        if manifest.status != "complete":
            incomplete[manifest.key.method] = incomplete.get(manifest.key.method, 0) + 1
        trace = dict(manifest.trace_hashes)
        for field in cell.required_trace_hashes:
            if field not in trace:
                hash_errors.append(f"{identifier}: missing {field}")
            elif not is_sha256(trace[field]):
                hash_errors.append(f"{identifier}: invalid {field}")
        for field, locked_digest in cell.locked_trace_hashes:
            if trace.get(field) != locked_digest:
                hash_errors.append(f"{identifier}: {field} differs from expected lock")
        for artifact in manifest.artifacts:
            try:
                artifact_path = guard_paper2_path(
                    root / artifact.relative_path,
                    paper2_root=root,
                )
            except ValueError as exc:
                artifact_errors.append(f"{identifier}/{artifact.name}: {exc}")
                continue
            if not artifact_path.is_file():
                artifact_errors.append(f"{identifier}/{artifact.name}: file is missing")
            elif file_sha256(artifact_path) != artifact.sha256:
                artifact_errors.append(f"{identifier}/{artifact.name}: file hash mismatch")
        if "audit_footprint_plan_sha256" in cell.required_trace_hashes:
            projection_artifacts = [
                artifact
                for artifact in manifest.artifacts
                if artifact.name == "audit_footprint_projection"
            ]
            if len(projection_artifacts) != 1:
                artifact_errors.append(
                    f"{identifier}: formal inference requires one audit footprint projection"
                )
        evaluation = evaluation_by_id.get(identifier)
        if evaluation is None:
            evaluation_errors.append(f"{identifier}: missing evaluation manifest")
        else:
            if evaluation.status != "complete":
                evaluation_errors.append(f"{identifier}: evaluation is {evaluation.status}")
            if evaluation.inference_manifest_sha256 != manifest.sha256:
                evaluation_errors.append(
                    f"{identifier}: evaluation binds a different inference manifest"
                )
            if cell.sealed_reference_sha256 is not None and (
                evaluation.sealed_reference_sha256 != cell.sealed_reference_sha256
            ):
                evaluation_errors.append(
                    f"{identifier}: evaluation uses a different sealed reference"
                )
            if cell.sealed_reference_grid_sha256 is not None and (
                evaluation.sealed_reference_grid_sha256
                != cell.sealed_reference_grid_sha256
            ):
                evaluation_errors.append(
                    f"{identifier}: evaluation uses a different sealed reference grid"
                )
            for artifact in evaluation.artifacts:
                try:
                    artifact_path = guard_paper2_path(
                        root / artifact.relative_path,
                        paper2_root=root,
                    )
                except ValueError as exc:
                    evaluation_errors.append(f"{identifier}/{artifact.name}: {exc}")
                    continue
                if not artifact_path.is_file():
                    evaluation_errors.append(f"{identifier}/{artifact.name}: file is missing")
                elif file_sha256(artifact_path) != artifact.sha256:
                    evaluation_errors.append(f"{identifier}/{artifact.name}: file hash mismatch")
            if "audit_footprint_plan_sha256" in cell.required_trace_hashes:
                metric_artifacts = [
                    artifact for artifact in evaluation.artifacts if artifact.name == "metrics"
                ]
                if len(metric_artifacts) == 1:
                    metrics_path = guard_paper2_path(
                        root / metric_artifacts[0].relative_path, paper2_root=root
                    )
                    if metrics_path.is_file() and file_sha256(metrics_path) == metric_artifacts[
                        0
                    ].sha256:
                        payload = read_json(metrics_path)
                        audit_metrics = (
                            payload.get("metrics", {}).get("audit_footprints")
                            if isinstance(payload, dict)
                            and isinstance(payload.get("metrics"), dict)
                            else None
                        )
                        if (
                            not isinstance(audit_metrics, dict)
                            or audit_metrics.get("count") != 128
                            or audit_metrics.get("diagonal_pixel_variance_projection")
                            is not False
                            or not isinstance(payload, dict)
                            or payload.get("audit_footprint_plan_sha256")
                            != trace.get("audit_footprint_plan_sha256")
                            or payload.get("evaluation_code_sha256")
                            != trace.get("code_sha256")
                        ):
                            evaluation_errors.append(
                                f"{identifier}: held-out audit footprint metrics/trace differ"
                            )
    unexpected_evaluations = sorted(set(evaluation_by_id).difference(expected_ids))
    evaluation_errors.extend(
        f"{identifier}: unexpected evaluation key" for identifier in unexpected_evaluations
    )
    # E7 cannot close merely because all physical keys exist: every M<32 cell
    # must be derived from the ordered first M samples of its M=32 source for
    # the same independently solved N.  This audit also recomputes aggregates
    # from the immutable source stack.
    from .e7_prefix import audit_e7_prefix_manifests

    artifact_errors.extend(
        audit_e7_prefix_manifests(manifests, paper2_root=root)
    )
    artifact_errors.extend(_audit_e6_shared_uncorrected_paths(manifests))
    for identifier in missing:
        method = expected_by_id[identifier].key.method
        incomplete[method] = incomplete.get(method, 0) + 1
    ok = not (
        missing
        or unexpected
        or duplicates
        or incomplete
        or hash_errors
        or artifact_errors
        or evaluation_errors
    )
    return MatrixAudit(
        ok=ok,
        expected_cells=len(expected.cells),
        observed_manifests=len(manifests),
        observed_evaluations=len(evaluation_manifests),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        duplicate_keys=tuple(sorted(set(duplicates))),
        incomplete_by_method=tuple(sorted(incomplete.items())),
        hash_errors=tuple(sorted(set(hash_errors))),
        artifact_errors=tuple(sorted(set(artifact_errors))),
        evaluation_errors=tuple(sorted(set(evaluation_errors))),
    )


def _audit_e6_shared_uncorrected_paths(
    manifests: tuple[InferenceManifest, ...] | list[InferenceManifest],
) -> list[str]:
    groups: dict[tuple[Any, ...], dict[str, InferenceManifest]] = {}
    for manifest in manifests:
        key = manifest.key
        if key.experiment != "E6":
            continue
        group_key = (
            key.tile_id,
            key.method,
            key.budget,
            key.noise_replicate,
            key.training_seed,
            key.model_identity,
            key.path_seed_set,
            key.path_count,
            key.step_count,
        )
        mode_group = groups.setdefault(group_key, {})
        if key.dc_mode in mode_group:
            return [f"E6 {group_key}: duplicate {key.dc_mode} shared-path cell"]
        mode_group[key.dc_mode] = manifest
    errors: list[str] = []
    for group_key, modes in sorted(groups.items(), key=lambda item: str(item[0])):
        if set(modes) != {"D0", "D1", "D3"}:
            errors.append(
                f"E6 {group_key}: shared path group has modes {sorted(modes)}"
            )
            continue
        contracts = []
        for mode in ("D0", "D1", "D3"):
            manifest = modes[mode]
            trace_hash = dict(manifest.trace_hashes).get("uncorrected_path_sha256")
            artifacts = {
                artifact.name: artifact for artifact in manifest.artifacts
            }
            artifact = artifacts.get("e6_uncorrected_path_stack")
            if trace_hash is None or artifact is None or artifact.sha256 != trace_hash:
                errors.append(
                    f"E6 {group_key}/{mode}: uncorrected path trace/artifact is missing or differs"
                )
                continue
            contracts.append((artifact.relative_path, artifact.sha256))
        if len(contracts) == 3 and len(set(contracts)) != 1:
            errors.append(
                f"E6 {group_key}: D0/D1/D3 do not reuse one uncorrected path artifact"
            )
    return errors


def finalize_expected_matrix(
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    inference_manifest_paths: tuple[str | Path, ...] | list[str | Path],
    output_directory: str | Path,
    *,
    evaluation_manifest_paths: tuple[str | Path, ...] | list[str | Path],
    g7_manifest_path: str | Path,
    matrix_spec_path: str | Path,
    g5_unlock_path: str | Path,
    g6_unlock_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    g4_freeze_path: str | Path,
    paper2_root: str | Path = PAPER2_ROOT,
) -> Path:
    """Write _SUCCESS.json only after an exact, traceable matrix join succeeds."""
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    spec_path = guard_paper2_path(matrix_spec_path, paper2_root=root)
    g5_path = guard_paper2_path(g5_unlock_path, paper2_root=root)
    g6_path = guard_paper2_path(g6_unlock_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    g4_path = guard_paper2_path(g4_freeze_path, paper2_root=root)
    output = guard_paper2_path(output_directory, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("final output directory must be below paper2 root")
    from .g5 import validate_g5_unlock

    g5 = validate_g5_unlock(
        g5_path,
        matrix_spec_path=spec_path,
        expected_matrix_path=expected_path,
        analysis_lock_path=analysis_path,
        role_manifest_path=role_path,
        asset_lock_path=asset_path,
        g4_freeze_path=g4_path,
        paper2_root=root,
    )
    expected = ExpectedMatrix.read(expected_path)
    observed_analysis_hash = file_sha256(analysis_path)
    if observed_analysis_hash != expected.analysis_lock_sha256:
        raise ValueError("analysis-lock hash differs from expected-matrix")
    manifest_paths = [guard_paper2_path(path, paper2_root=root) for path in inference_manifest_paths]
    manifests = [InferenceManifest.read(path) for path in manifest_paths]
    evaluation_paths = [
        guard_paper2_path(path, paper2_root=root) for path in evaluation_manifest_paths
    ]
    evaluations = [EvaluationManifest.read(path) for path in evaluation_paths]
    from .phase_gates import validate_g6_unlock

    validate_g6_unlock(
        g6_path,
        g5_unlock_path=g5_path,
        expected_matrix_path=expected_path,
        analysis_lock_path=analysis_path,
        paper2_root=root,
        inference_manifest_paths=manifest_paths,
        evaluation_manifest_paths=evaluation_paths,
    )
    audit = audit_expected_matrix(
        expected, manifests, evaluations, paper2_root=root
    )
    if not audit.ok:
        raise MatrixClosureError(audit)
    # Import lazily to keep the matrix contracts independent from report builders.
    from .g7 import validate_g7_manifest

    g7_path = guard_paper2_path(g7_manifest_path, paper2_root=root)
    g7 = validate_g7_manifest(
        g7_path,
        expected_matrix_path=expected_path,
        analysis_lock_path=analysis_path,
        paper2_root=root,
    )
    audit_path = write_canonical_json(
        output / "matrix-audit.json",
        audit.to_dict(),
        paper2_root=root,
    )
    success = {
        "schema": "paper2.success/v1",
        "status": "complete",
        "expected_cells": len(expected.cells),
        "expected_matrix_sha256": file_sha256(expected_path),
        "analysis_lock_sha256": observed_analysis_hash,
        "matrix_spec_sha256": file_sha256(spec_path),
        "g5_unlock_sha256": file_sha256(g5_path),
        "g5_semantic_sha256": g5["g5_unlock_sha256"],
        "g6_unlock_sha256": file_sha256(g6_path),
        "g6_semantic_sha256": read_json(g6_path)["g6_unlock_sha256"],
        "role_manifest_sha256": file_sha256(role_path),
        "asset_lock_sha256": file_sha256(asset_path),
        "g4_freeze_sha256": file_sha256(g4_path),
        "matrix_audit_sha256": file_sha256(audit_path),
        "g7_manifest_sha256": file_sha256(g7_path),
        "g7_semantic_sha256": g7["g7_manifest_sha256"],
        "inference_manifests": [
            {
                "key_sha256": manifest.key.identifier,
                "manifest_sha256": file_sha256(path),
            }
            for manifest, path in sorted(
                zip(manifests, manifest_paths, strict=True),
                key=lambda pair: pair[0].key.identifier,
            )
        ],
        "evaluation_manifests": [
            {
                "key_sha256": manifest.key.identifier,
                "manifest_sha256": file_sha256(path),
            }
            for manifest, path in sorted(
                zip(evaluations, evaluation_paths, strict=True),
                key=lambda pair: pair[0].key.identifier,
            )
        ],
    }
    return write_canonical_json(output / "_SUCCESS.json", success, paper2_root=root)
