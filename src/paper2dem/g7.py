from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)
from .matrix import ExpectedMatrix


G7_JSON_ARTIFACTS = {
    "e10_mola_audit": "paper2.e10-mola-audit/v1",
    "e11_efficiency": "paper2.e11-efficiency/v1",
    "e12_tile_results": "paper2.e12-tile-results/v1",
    "e12_error_maps": "paper2.e12-error-maps/v1",
    "paper_results": "paper2.paper-results/v1",
}
G7_ARTIFACT_NAMES = frozenset({*G7_JSON_ARTIFACTS, "final_report"})
G7_REPORT_MARKER = "<!-- paper2.g7-final-report/v1 -->"


def _record(path: Path, *, root: Path) -> dict[str, str | int]:
    guarded = guard_paper2_path(path, paper2_root=root)
    if not guarded.is_file():
        raise FileNotFoundError(guarded)
    size_bytes = guarded.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"G7 artifact is empty: {guarded}")
    return {
        "relative_path": guarded.relative_to(root).as_posix(),
        "sha256": file_sha256(guarded),
        "size_bytes": size_bytes,
    }


def _validate_derived(name: str, payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != G7_JSON_ARTIFACTS[name] or payload.get("status") != "complete":
        raise ValueError(f"G7 {name} is not a completed registered artifact")
    if payload.get("content_sha256") != canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    ):
        raise ValueError(f"G7 {name} content hash does not reproduce")
    if name == "e10_mola_audit":
        if (
            payload.get("all_z0_from_locked_mola") is not True
            or payload.get("all_e3_e10_problem_and_z0_distinct") is not True
            or payload.get("truth_data_opened") is not False
        ):
            raise ValueError("G7 E10 does not prove distinct truth-blind locked-MOLA Z0")
        expected = int(payload.get("expected_tile_count", -1))
        tiles = payload.get("tiles")
        if expected not in {24, 30, 36} or not isinstance(tiles, list) or len(tiles) != expected:
            raise ValueError("G7 E10 locked CTX tile count does not close")
        for tile in tiles:
            if (
                tile.get("e3_problem_sha256") == tile.get("e10_problem_sha256")
                or tile.get("e3_z0_array_sha256") == tile.get("e10_z0_array_sha256")
                or not all(
                    is_sha256(str(tile.get(field, "")))
                    for field in (
                        "e3_problem_sha256",
                        "e10_problem_sha256",
                        "e3_z0_array_sha256",
                        "e10_z0_array_sha256",
                    )
                )
            ):
                raise ValueError("G7 E10 tile reuses the E3 problem or Z0")
    if name == "paper_results":
        if not isinstance(payload.get("claim_policy"), str):
            raise ValueError("G7 scientific results lack the locked claim policy")
        primary = payload.get("E3_primary")
        mechanism = payload.get("E4_OT_mechanism")
        if not isinstance(primary, Mapping) or not isinstance(mechanism, Mapping):
            raise ValueError("G7 scientific results lack E3/E4 analyses")
        _validate_closed_audit(payload, name=name)
    if name == "e11_efficiency":
        required = {
            "parameter_count",
            "training_updates",
            "training_stage_wall_seconds",
            "data_loading_seconds",
            "forward_seconds",
            "backward_seconds",
            "optimizer_seconds",
            "validation_seconds",
            "gpu_hours",
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
            "nfe",
            "seconds_per_km2",
        }
        if payload.get("formal") is not True or not required.issubset(
            payload.get("reported_metrics", [])
        ):
            raise ValueError("G7 E11 efficiency metrics are incomplete")
        _validate_closed_audit(payload, name=name)
    if name == "e12_tile_results":
        if (
            payload.get("formal") is not True
            or payload.get("positive_and_negative_tiles_retained") is not True
        ):
            raise ValueError("G7 E12 tile table filters negative results")
        _validate_closed_audit(payload, name=name)
    if name == "e12_error_maps":
        if (
            payload.get("formal") is not True
            or payload.get("uniform_template") is not True
            or payload.get("fixed_scale_outcome_independent") is not True
            or payload.get("complete_supplement_tile_coverage") is not True
        ):
            raise ValueError("G7 E12 error maps do not use one complete fixed template")


def _validate_closed_audit(payload: Mapping[str, Any], *, name: str) -> None:
    audit = payload.get("matrix_audit")
    if not isinstance(audit, Mapping) or audit.get("ok") is not True or payload.get(
        "matrix_audit_sha256"
    ) != canonical_sha256(audit):
        raise ValueError(f"G7 {name} does not bind a closed matrix audit")


def _derived_payloads(
    records: Mapping[str, Mapping[str, Any]], *, root: Path
) -> dict[str, Mapping[str, Any]]:
    payloads: dict[str, Mapping[str, Any]] = {}
    for name in G7_JSON_ARTIFACTS:
        artifact_path = root / str(records[name]["relative_path"])
        payload = read_json(artifact_path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"G7 {name} must be a JSON object")
        _validate_derived(name, payload)
        if name == "e10_mola_audit":
            from .e10_mola import read_e10_mola_audit

            payload = read_e10_mola_audit(
                artifact_path, paper2_root=root, verify_tiles=True
            )
        elif name == "e11_efficiency":
            from .e11_efficiency import read_e11_efficiency

            payload = read_e11_efficiency(
                artifact_path, paper2_root=root, verify_sources=True
            )
        elif name == "e12_tile_results":
            from .e12_reporting import read_e12_tile_results

            payload = read_e12_tile_results(
                artifact_path, paper2_root=root, verify_sources=True
            )
        elif name == "e12_error_maps":
            from .e12_reporting import read_e12_error_maps

            payload = read_e12_error_maps(
                artifact_path, paper2_root=root, verify_sources=True
            )
        elif name == "paper_results":
            from .paper_results import read_paper_results

            payload = read_paper_results(
                artifact_path, paper2_root=root, verify_sources=True
            )
        payloads[name] = payload
    return payloads


def _upstream_hash(payload: Mapping[str, Any], name: str) -> str:
    record = payload.get(name)
    if not isinstance(record, Mapping) or not is_sha256(str(record.get("sha256", ""))):
        raise ValueError(f"G7 derived artifact lacks {name} binding")
    return str(record["sha256"])


def _cross_validate(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    expected_matrix_sha256: str,
    analysis_lock_sha256: str,
    expected_cell_count: int,
) -> None:
    e10 = payloads["e10_mola_audit"]
    e11 = payloads["e11_efficiency"]
    table = payloads["e12_tile_results"]
    maps = payloads["e12_error_maps"]
    paper = payloads["paper_results"]
    for name, payload in (
        ("paper results", paper),
        ("E11", e11),
        ("E12 table", table),
        ("E12 maps", maps),
    ):
        if (
            _upstream_hash(payload, "expected_matrix") != expected_matrix_sha256
            or _upstream_hash(payload, "analysis_lock") != analysis_lock_sha256
        ):
            raise ValueError(f"G7 {name} binds different upstream locks")
    audit_hashes = {
        str(paper["matrix_audit_sha256"]),
        str(e11["matrix_audit_sha256"]),
        str(table["matrix_audit_sha256"]),
        str(maps["matrix_audit_sha256"]),
    }
    if len(audit_hashes) != 1:
        raise ValueError("G7 E11/E12 matrix audits differ")
    for payload in (paper, e11, table):
        audit = payload["matrix_audit"]
        if (
            int(audit.get("expected_cells", -1)) != expected_cell_count
            or int(audit.get("observed_manifests", -1)) != expected_cell_count
            or int(audit.get("observed_evaluations", -1)) != expected_cell_count
        ):
            raise ValueError("G7 derived matrix audit counts do not close")
    e10_tiles = {str(record["tile_id"]) for record in e10["tiles"]}
    table_tiles = {str(record["tile_id"]) for record in table.get("tiles", [])}
    if e10_tiles != table_tiles or len(table_tiles) != int(table.get("tile_count", -1)):
        raise ValueError("G7 E10 and E12 locked CTX tile sets differ")
    if int(maps.get("tile_count", -1)) < len(e10_tiles):
        raise ValueError("G7 E12 maps omit a locked CTX tile")
    if int(maps.get("method_count", -1)) != 5 or int(maps.get("map_count", -1)) != int(
        maps.get("tile_count", -1)
    ) * int(maps.get("method_count", -1)):
        raise ValueError("G7 E12 map tile/method cross does not close")
    tile_results = maps.get("tile_results")
    if not isinstance(tile_results, Mapping):
        raise ValueError("G7 E12 maps lack their tile table binding")


def _report_evidence(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    expected_matrix_sha256: str,
    analysis_lock_sha256: str,
) -> dict[str, Any]:
    e10 = payloads["e10_mola_audit"]
    e11 = payloads["e11_efficiency"]
    table = payloads["e12_tile_results"]
    maps = payloads["e12_error_maps"]
    paper = payloads["paper_results"]
    return {
        "schema": "paper2.g7-final-report-evidence/v1",
        "expected_matrix_sha256": expected_matrix_sha256,
        "analysis_lock_sha256": analysis_lock_sha256,
        "derived_artifact_sha256": {
            name: str(records[name]["sha256"]) for name in sorted(G7_JSON_ARTIFACTS)
        },
        "e10_locked_ctx_tiles": int(e10["audited_tile_count"]),
        "e11_gpu_hours": float(e11["summary"]["gpu_hours"]),
        "e11_peak_memory_allocated_bytes": int(
            e11["summary"]["peak_memory_allocated_bytes"]
        ),
        "e11_peak_memory_reserved_bytes": int(
            e11["summary"]["peak_memory_reserved_bytes"]
        ),
        "e11_nfe": int(e11["summary"]["nfe"]),
        "e12_positive_tiles": int(table["positive_tile_count"]),
        "e12_nonpositive_tiles": int(table["nonpositive_tile_count"]),
        "e12_error_maps": int(maps["map_count"]),
        "primary_claim_pass": bool(paper["E3_primary"]["claim_pass"]),
        "ot_geometry_attribution_pass": bool(
            paper["E4_OT_mechanism"]["ot_geometry_attribution_pass"]
        ),
        "claim_policy": str(paper["claim_policy"]),
        "matrix_audit_sha256": str(e11["matrix_audit_sha256"]),
    }


def render_g7_final_report(evidence: Mapping[str, Any]) -> bytes:
    evidence_hash = canonical_sha256(evidence)
    lines = [
        G7_REPORT_MARKER,
        "# Paper2 final experiment report",
        "",
        "This report is generated from the immutable E10-E12 deliverables. It is not a manually completed status page.",
        "",
        f"- Locked CTX tiles with independent MOLA Z0: {evidence['e10_locked_ctx_tiles']}",
        f"- Total measured GPU-hours: {float(evidence['e11_gpu_hours']):.9f}",
        f"- Peak allocated/reserved bytes: {evidence['e11_peak_memory_allocated_bytes']} / {evidence['e11_peak_memory_reserved_bytes']}",
        f"- Total measured NFE: {evidence['e11_nfe']}",
        f"- Positive/nonpositive locked tiles retained: {evidence['e12_positive_tiles']} / {evidence['e12_nonpositive_tiles']}",
        f"- Uniform fixed-scale error maps: {evidence['e12_error_maps']}",
        f"- Primary efficiency claim passed: {evidence['primary_claim_pass']}",
        f"- OT-geometry attribution passed: {evidence['ot_geometry_attribution_pass']}",
        f"- Locked claim policy: {evidence['claim_policy']}",
        "",
        f"Evidence SHA-256: `{evidence_hash}`",
        "",
        "```json",
        canonical_json_bytes(evidence).decode("utf-8"),
        "```",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def write_g7_final_report(
    path: str | Path,
    *,
    artifact_paths: Mapping[str, Path],
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    paper2_root: str | Path,
) -> Path:
    root = Path(paper2_root).resolve()
    if set(artifact_paths) != set(G7_JSON_ARTIFACTS):
        raise ValueError("G7 final report requires exactly the four E10-E12 JSON artifacts")
    records = {
        name: _record(Path(path), root=root) for name, path in sorted(artifact_paths.items())
    }
    payloads = _derived_payloads(records, root=root)
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    expected = ExpectedMatrix.read(expected_path)
    _cross_validate(
        payloads,
        expected_matrix_sha256=file_sha256(expected_path),
        analysis_lock_sha256=file_sha256(analysis_path),
        expected_cell_count=len(expected.cells),
    )
    evidence = _report_evidence(
        payloads=payloads,
        records=records,
        expected_matrix_sha256=file_sha256(expected_path),
        analysis_lock_sha256=file_sha256(analysis_path),
    )
    return write_immutable_bytes(path, render_g7_final_report(evidence), paper2_root=root)


def build_g7_manifest(
    *,
    artifact_paths: Mapping[str, Path],
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    if set(artifact_paths) != G7_ARTIFACT_NAMES:
        raise ValueError(
            f"G7 artifacts must be exactly {sorted(G7_ARTIFACT_NAMES)}"
        )
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    records = {
        name: _record(Path(path), root=root)
        for name, path in sorted(artifact_paths.items())
    }
    payloads = _derived_payloads(records, root=root)
    expected = ExpectedMatrix.read(expected_path)
    _cross_validate(
        payloads,
        expected_matrix_sha256=file_sha256(expected_path),
        analysis_lock_sha256=file_sha256(analysis_path),
        expected_cell_count=len(expected.cells),
    )
    table_binding = payloads["e12_error_maps"].get("tile_results")
    if (
        not isinstance(table_binding, Mapping)
        or table_binding.get("sha256") != records["e12_tile_results"]["sha256"]
    ):
        raise ValueError("G7 E12 error maps bind a different tile-results artifact")
    evidence = _report_evidence(
        payloads=payloads,
        records=records,
        expected_matrix_sha256=file_sha256(expected_path),
        analysis_lock_sha256=file_sha256(analysis_path),
    )
    report_path = root / str(records["final_report"]["relative_path"])
    if report_path.read_bytes() != render_g7_final_report(evidence):
        raise ValueError("G7 final report was not generated from the bound E10-E12 artifacts")
    manifest: dict[str, Any] = {
        "schema": "paper2.g7-deliverables/v1",
        "status": "complete",
        "expected_matrix_sha256": file_sha256(expected_path),
        "analysis_lock_sha256": file_sha256(analysis_path),
        "artifacts": records,
    }
    manifest["g7_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def write_g7_manifest(
    path: str | Path,
    *,
    artifact_paths: Mapping[str, Path],
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    paper2_root: str | Path,
) -> Path:
    return write_canonical_json(
        path,
        build_g7_manifest(
            artifact_paths=artifact_paths,
            expected_matrix_path=expected_matrix_path,
            analysis_lock_path=analysis_lock_path,
            paper2_root=paper2_root,
        ),
        paper2_root=paper2_root,
    )


def validate_g7_manifest(
    path: str | Path,
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    paper2_root: str | Path,
) -> Mapping[str, Any]:
    root = Path(paper2_root).resolve()
    manifest_path = guard_paper2_path(path, paper2_root=root)
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    payload = read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError("G7 manifest must be a JSON object")
    semantic = canonical_sha256(
        {key: value for key, value in payload.items() if key != "g7_manifest_sha256"}
    )
    if (
        payload.get("schema") != "paper2.g7-deliverables/v1"
        or payload.get("status") != "complete"
        or payload.get("g7_manifest_sha256") != semantic
        or payload.get("expected_matrix_sha256") != file_sha256(expected_path)
        or payload.get("analysis_lock_sha256") != file_sha256(analysis_path)
    ):
        raise ValueError("G7 manifest identity/hash/matrix binding differs")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != G7_ARTIFACT_NAMES:
        raise ValueError("G7 manifest artifact set differs")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"G7 artifact record is malformed: {name}")
        artifact_path = guard_paper2_path(
            root / str(record.get("relative_path", "")), paper2_root=root
        )
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != record.get("size_bytes")
            or file_sha256(artifact_path) != record.get("sha256")
        ):
            raise ValueError(f"G7 artifact bytes differ: {name}")
    payloads = _derived_payloads(artifacts, root=root)
    expected = ExpectedMatrix.read(expected_path)
    _cross_validate(
        payloads,
        expected_matrix_sha256=file_sha256(expected_path),
        analysis_lock_sha256=file_sha256(analysis_path),
        expected_cell_count=len(expected.cells),
    )
    table_binding = payloads["e12_error_maps"].get("tile_results")
    if (
        not isinstance(table_binding, Mapping)
        or table_binding.get("sha256") != artifacts["e12_tile_results"]["sha256"]
    ):
        raise ValueError("G7 E12 map/table binding differs")
    evidence = _report_evidence(
        payloads=payloads,
        records=artifacts,
        expected_matrix_sha256=file_sha256(expected_path),
        analysis_lock_sha256=file_sha256(analysis_path),
    )
    report_path = root / str(artifacts["final_report"]["relative_path"])
    if report_path.read_bytes() != render_g7_final_report(evidence):
        raise ValueError("G7 final report contents differ from the bound derivation")
    return payload
