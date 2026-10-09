from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

from .assets import (
    GeographicGroup,
    assign_stratified_roles,
    audit_asset_lock_geography,
    stratify_groups,
)
from .contracts import AssetFile, AssetLock, AssetQA, AssetRecord, Paper1SourceLock
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)


SCHEMAS = {
    "conflicts": "paper2.g0.conflict-groups/v1",
    "processed": "paper2.g0.processed-lock/v1",
    "edr": "paper2.g0.edr-index/v1",
    "mola": "paper2.g0.mola-source-lock/v1",
    "terrain": "paper2.g0.terrain-manifest/v1",
    "seen": "paper2.g0.paper1-seen-lock/v1",
    "draft": "paper2.g0.asset-lock-draft/v1",
}


def _require_schema(payload: Mapping[str, Any], name: str) -> None:
    expected = SCHEMAS[name]
    if payload.get("schema") != expected:
        raise ValueError(f"expected {expected}, found {payload.get('schema')}")


def _require_hash(name: str, value: Any) -> str:
    result = str(value)
    if not is_sha256(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _optional_sidecar(
    path: str | Path | None, name: str
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    sidecar_path = Path(path)
    if not sidecar_path.is_file():
        return None, None
    payload = read_json(sidecar_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} sidecar must be a JSON object")
    _require_schema(payload, name)
    return payload, file_sha256(sidecar_path)


def validate_edr_index(
    payload: Mapping[str, Any],
    *,
    catalog_sha256: str,
    conflict_groups_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    _require_schema(payload, "edr")
    if payload.get("state") != "complete":
        raise ValueError("EDR index is incomplete")
    if payload.get("catalog_sha256") != catalog_sha256:
        raise ValueError("EDR index is not bound to the frozen catalog")
    if (
        conflict_groups_sha256 is not None
        and payload.get("conflict_groups_sha256") != conflict_groups_sha256
    ):
        raise ValueError("EDR index is not bound to the frozen conflict groups")
    for key in (
        "conflict_groups_sha256",
        "conflict_groups_file_sha256",
        "assignments_file_sha256",
        "role_assignment_sha256",
        "required_observations_sha256",
        "input_bindings_sha256",
    ):
        _require_hash(key, payload.get(key))
    expected_bindings_hash = canonical_sha256(
        {
            "catalog_sha256": payload["catalog_sha256"],
            "conflict_groups_sha256": payload["conflict_groups_sha256"],
            "conflict_groups_file_sha256": payload["conflict_groups_file_sha256"],
            "assignments_file_sha256": payload["assignments_file_sha256"],
            "role_assignment_sha256": payload["role_assignment_sha256"],
            "required_observations_sha256": payload["required_observations_sha256"],
        }
    )
    if payload.get("input_bindings_sha256") != expected_bindings_hash:
        raise ValueError("EDR index input binding digest does not verify")
    entries: dict[str, dict[str, Any]] = {}
    for raw in payload.get("entries", []):
        entry = dict(raw)
        observation_id = str(entry.get("observation_id", ""))
        if not observation_id or observation_id in entries:
            raise ValueError("EDR observation IDs must be nonempty and unique")
        status = entry.get("status")
        if status not in {"resolved", "missing", "ambiguous", "rejected"}:
            raise ValueError(f"unsupported EDR status for {observation_id}: {status}")
        _require_hash("ODE response hash", entry.get("ode_response_sha256"))
        if status == "resolved":
            selected_url = str(entry.get("selected_url", ""))
            parsed = urlparse(selected_url)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or not parsed.path.lower().endswith(".img")
            ):
                raise ValueError(f"resolved EDR {observation_id} requires one HTTPS .IMG URL")
            if int(entry.get("match_count", 0)) != 1:
                raise ValueError(f"resolved EDR {observation_id} must have exactly one ODE match")
        entries[observation_id] = entry
    if int(payload.get("requested_observation_count", -1)) != len(entries):
        raise ValueError("EDR index requested observation count does not close")
    if int(payload.get("terminal_observation_count", -1)) != len(entries):
        raise ValueError("EDR index terminal observation count does not close")
    if int(payload.get("failed_observation_count", -1)) != 0:
        raise ValueError("EDR index retains failed observations")
    return entries


def validate_mola_source_lock(payload: Mapping[str, Any]) -> bool:
    _require_schema(payload, "mola")
    required = {
        "MEGT90N000FB": "terrain_score",
        "MEGR90N000FB": "isis_shape_model",
        "MEGC90N000FB": "observation_count_qa",
    }
    observed: dict[str, str] = {}
    downloads_ready = True
    for raw in payload.get("products", []):
        product = dict(raw)
        product_id = str(product.get("product_id", ""))
        if product_id in observed:
            raise ValueError(f"duplicate MOLA product: {product_id}")
        observed[product_id] = str(product.get("purpose", ""))
        for key in ("image_url", "label_url"):
            parsed = urlparse(str(product.get(key, "")))
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"MOLA {product_id} has invalid {key}")
        for key in ("image_sha256", "label_sha256"):
            value = product.get(key)
            if value is None:
                downloads_ready = False
            else:
                _require_hash(f"MOLA {product_id} {key}", value)
    if observed != required:
        raise ValueError(f"MOLA source lock products differ from required set: {observed}")
    return downloads_ready


def validate_paper1_seen_lock(
    payload: Mapping[str, Any], *, catalog_sha256: str
) -> tuple[str, ...]:
    _require_schema(payload, "seen")
    if payload.get("catalog_sha256") != catalog_sha256:
        raise ValueError("Paper1-seen lock is not bound to the frozen catalog")
    _require_hash("Paper1-seen source file hash", payload.get("source_file_sha256"))
    product_ids = tuple(str(value) for value in payload.get("paper1_seen_product_ids", []))
    if len(product_ids) != len(set(product_ids)) or any(not value for value in product_ids):
        raise ValueError("Paper1-seen product IDs must be nonempty and unique")
    if list(product_ids) != sorted(product_ids):
        raise ValueError("Paper1-seen product IDs must use canonical sorted order")
    if int(payload.get("paper1_seen_product_count", -1)) != len(product_ids):
        raise ValueError("Paper1-seen product count does not close")
    return product_ids


def validate_terrain_manifest(
    payload: Mapping[str, Any],
    *,
    conflict_groups_sha256: str,
    mola_source_lock_sha256: str,
) -> dict[str, float]:
    _require_schema(payload, "terrain")
    if payload.get("conflict_groups_sha256") != conflict_groups_sha256:
        raise ValueError("terrain manifest is not bound to conflict groups")
    if payload.get("mola_source_lock_sha256") != mola_source_lock_sha256:
        raise ValueError("terrain manifest is not bound to the MOLA source lock")
    if payload.get("algorithm") != "mola-megt-robust-relief-v1":
        raise ValueError("terrain manifest uses an unlocked scoring algorithm")
    if payload.get("state") != "complete":
        raise ValueError("terrain manifest is incomplete")
    for key in (
        "conflict_groups_file_sha256",
        "megt_image_sha256",
        "megt_label_sha256",
        "input_bindings_sha256",
    ):
        _require_hash(key, payload.get(key))
    expected_bindings_hash = canonical_sha256(
        {
            "conflict_groups_sha256": payload["conflict_groups_sha256"],
            "conflict_groups_file_sha256": payload["conflict_groups_file_sha256"],
            "mola_source_lock_sha256": payload["mola_source_lock_sha256"],
            "megt_image_sha256": payload["megt_image_sha256"],
            "megt_label_sha256": payload["megt_label_sha256"],
            "megt_image_relative_path": payload.get("megt_image_relative_path"),
            "megt_label_relative_path": payload.get("megt_label_relative_path"),
        }
    )
    if payload.get("input_bindings_sha256") != expected_bindings_hash:
        raise ValueError("terrain manifest input binding digest does not verify")
    scores: dict[str, float] = {}
    for raw in payload.get("groups", []):
        group_id = str(raw.get("group_id", ""))
        score = float(raw.get("terrain_score", np.nan))
        lower = float(raw.get("lower_quantile_m", np.nan))
        upper = float(raw.get("upper_quantile_m", np.nan))
        if (
            raw.get("status") != "complete"
            or not group_id
            or group_id in scores
            or not np.isfinite(score)
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or not np.isclose(score, upper - lower, rtol=0.0, atol=1e-9)
            or int(raw.get("finite_pixel_count", 0)) < 4
            or not is_sha256(str(raw.get("bbox_sha256", "")))
        ):
            raise ValueError("terrain group IDs must be unique and scores finite")
        scores[group_id] = score
    if int(payload.get("expected_group_count", -1)) != len(scores):
        raise ValueError("terrain manifest expected group count does not close")
    if int(payload.get("complete_group_count", -1)) != len(scores):
        raise ValueError("terrain manifest complete group count does not close")
    if int(payload.get("failed_group_count", -1)) != 0:
        raise ValueError("terrain manifest retains failed groups")
    return scores


def validate_processed_lock(
    payload: Mapping[str, Any],
    *,
    conflict_groups_sha256: str,
    edr_index_sha256: str,
    mola_source_lock_sha256: str,
    paper2_root: str | Path | None = None,
    require_source_preparation: bool = False,
) -> dict[str, dict[str, Any]]:
    _require_schema(payload, "processed")
    if payload.get("state") != "complete":
        raise ValueError("processed lock is incomplete")
    bindings = {
        "conflict_groups_sha256": conflict_groups_sha256,
        "edr_index_sha256": edr_index_sha256,
        "mola_source_lock_sha256": mola_source_lock_sha256,
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"processed lock has stale {key}")
    for key in (
        "assignments_file_sha256",
        "role_assignment_sha256",
        "processing_input_bindings_sha256",
    ):
        _require_hash(key, payload.get(key))
    _require_hash("Paper1-seen source hash", payload.get("paper1_seen_source_sha256"))
    if require_source_preparation:
        _require_hash(
            "source preparation contract hash",
            payload.get("source_preparation_contract_sha256"),
        )
    paper1_seen = tuple(str(value) for value in payload.get("paper1_seen_product_ids", []))
    if len(paper1_seen) != len(set(paper1_seen)):
        raise ValueError("Paper1-seen product IDs must be unique")
    paper1_seen_set = set(paper1_seen)
    root = None if paper2_root is None else Path(paper2_root).resolve()
    entries: dict[str, dict[str, Any]] = {}
    passed_group_ids: set[str] = set()
    for raw in payload.get("products", []):
        entry = dict(raw)
        product_id = str(entry.get("product_id", ""))
        if not product_id or product_id in entries:
            raise ValueError("processed product IDs must be nonempty and unique")
        group_id = str(entry.get("geographic_group_id", ""))
        role = str(entry.get("role", ""))
        product_order = entry.get("product_order")
        if not group_id or not role:
            raise ValueError(f"processed product lacks group/role identity: {product_id}")
        if (
            isinstance(product_order, bool)
            or not isinstance(product_order, int)
            or product_order < 0
        ):
            raise ValueError(f"processed product order is invalid: {product_id}")
        if not str(entry.get("primary_observation_id", "")):
            raise ValueError(f"processed product lacks a primary observation: {product_id}")
        qa = AssetQA.from_dict(dict(entry.get("qa", {})))
        qa.validate()
        expected_failures: list[str] = []
        if qa.image_coverage_fraction is None or qa.image_coverage_fraction < 0.95:
            expected_failures.append("coverage_fraction<0.95")
        if qa.finite_dem_fraction is None or qa.finite_dem_fraction < 0.99:
            expected_failures.append("finite_dem_fraction<0.99")
        if qa.alignment_median_pixels is None or qa.alignment_median_pixels > 1.0:
            expected_failures.append("coreg_median_target_pixels>1")
        if qa.alignment_p95_pixels is None or qa.alignment_p95_pixels > 2.0:
            expected_failures.append("coreg_p95_target_pixels>2")
        if qa.passed is not (not expected_failures) or qa.failure_reasons != tuple(
            expected_failures
        ):
            raise ValueError(f"processed QA decision does not reproduce: {product_id}")
        _require_hash("processing manifest hash", entry.get("processing_manifest_sha256"))
        _require_hash("QA evidence hash", entry.get("qa_evidence_sha256"))
        _require_hash("QA evidence content hash", entry.get("qa_evidence_content_sha256"))
        processing_path = Path(str(entry.get("processing_manifest_relative_path", "")))
        if (
            not processing_path.parts
            or processing_path.is_absolute()
            or ".." in processing_path.parts
        ):
            raise ValueError(f"processed manifest path is invalid: {product_id}")
        if root is not None:
            resolved_processing = guard_paper2_path(processing_path, paper2_root=root)
            if (
                not resolved_processing.is_file()
                or file_sha256(resolved_processing) != entry["processing_manifest_sha256"]
            ):
                raise ValueError(f"processed manifest bytes do not verify: {product_id}")
        if qa.passed:
            _require_hash("prepared manifest hash", entry.get("prepared_manifest_sha256"))
            _require_hash("problem hash", entry.get("problem_sha256"))
            if group_id in passed_group_ids:
                raise ValueError(f"multiple QA-passed products belong to group {group_id}")
            passed_group_ids.add(group_id)
        else:
            if not qa.failure_reasons:
                raise ValueError(f"failed QA product lacks measured failure reasons: {product_id}")
            if entry.get("prepared_manifest_sha256") is not None:
                raise ValueError(f"failed QA product cannot have a prepared manifest: {product_id}")
        files = tuple(AssetFile.from_dict(dict(value)) for value in entry.get("files", []))
        for asset_file in files:
            asset_file.validate()
            if root is not None and asset_file.status == "present":
                resolved_file = guard_paper2_path(str(asset_file.relative_path), paper2_root=root)
                if (
                    not resolved_file.is_file()
                    or file_sha256(resolved_file) != asset_file.sha256
                    or resolved_file.stat().st_size != asset_file.byte_count
                ):
                    raise ValueError(
                        f"processed file bytes do not verify: {product_id}/{asset_file.kind}"
                    )
        files_by_kind: dict[str, AssetFile] = {}
        for asset_file in files:
            if asset_file.kind in files_by_kind:
                raise ValueError(
                    f"processed product has duplicate file kind: {product_id}/{asset_file.kind}"
                )
            files_by_kind[asset_file.kind] = asset_file
        required_kinds = {
            "ctx_edr",
            "ctx_edr_download_receipt",
            "isis_run_manifest",
            "qa_evidence",
            "problem",
            "image",
            "image_mask",
            "coarse_dem",
            "dem_mask",
            "coregistration_errors",
            "mola_source_lock",
            "coarse_dem_lineage_manifest",
            "coregistration_lineage_manifest",
        }
        if qa.passed:
            required_kinds.add("prepared_manifest")
        if not required_kinds.issubset(files_by_kind):
            missing = sorted(required_kinds.difference(files_by_kind))
            raise ValueError(f"processed product lacks traced files {missing}: {product_id}")
        integrity = dict(entry.get("edr_integrity", {}))
        ctx_edr_file = files_by_kind["ctx_edr"]
        if (
            integrity.get("validator") != "attached-pds3-fixed-length/v1"
            or int(integrity.get("declared_bytes", -1)) <= 0
            or int(integrity.get("observed_bytes", -1)) != int(integrity.get("declared_bytes", -1))
            or ctx_edr_file.byte_count != int(integrity.get("observed_bytes", -1))
            or Path(str(ctx_edr_file.relative_path)).name.upper()
            != str(integrity.get("file_name", "")).upper()
        ):
            raise ValueError(f"processed CTX EDR integrity evidence differs: {product_id}")
        if root is not None:
            receipt_path = guard_paper2_path(
                str(files_by_kind["ctx_edr_download_receipt"].relative_path),
                paper2_root=root,
            )
            receipt_payload = read_json(receipt_path)
            if (
                not isinstance(receipt_payload, dict)
                or receipt_payload.get("integrity") != integrity
                or receipt_payload.get("sha256") != ctx_edr_file.sha256
                or int(receipt_payload.get("bytes", -1)) != ctx_edr_file.byte_count
            ):
                raise ValueError(f"processed CTX EDR receipt differs: {product_id}")
        if files_by_kind["qa_evidence"].sha256 != entry["qa_evidence_sha256"]:
            raise ValueError(f"processed QA evidence file hash differs: {product_id}")
        if qa.passed:
            if (
                files_by_kind["prepared_manifest"].sha256 != entry["prepared_manifest_sha256"]
                or files_by_kind["problem"].sha256 != entry["problem_sha256"]
            ):
                raise ValueError(f"processed prepared/problem file hash differs: {product_id}")
        elif "prepared_manifest" in files_by_kind:
            raise ValueError(f"failed QA product traces a prepared manifest: {product_id}")
        source_preparation_sha256 = entry.get("source_preparation_manifest_sha256")
        if require_source_preparation and source_preparation_sha256 is None:
            raise ValueError(
                f"formal processed product lacks source-preparation trace: {product_id}"
            )
        if source_preparation_sha256 is not None:
            _require_hash("source preparation manifest hash", source_preparation_sha256)
            if (
                "source_preparation_manifest" not in files_by_kind
                or files_by_kind["source_preparation_manifest"].sha256
                != source_preparation_sha256
            ):
                raise ValueError(
                    f"processed source-preparation trace differs: {product_id}"
                )
        if qa.passed and (
            entry.get("coreg_aux_influenced_primary_pixels") is not True
            or entry.get("coreg_aux_pixel_values_copied") is not False
            or entry.get("registration_transform_family") != "translation_2d_only"
            or entry.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
            or entry.get("rotation_scale_shear_or_local_warp_applied") is not False
        ):
            raise ValueError(
                f"processed QA pass violates the rigid CTX registration contract: {product_id}"
            )
        entry["_qa"] = qa
        entry["_files"] = files
        declared_seen = bool(entry.get("seen_by_paper1", False))
        locked_seen = product_id in paper1_seen_set
        if declared_seen != locked_seen:
            raise ValueError(f"processed Paper1-seen flag differs from lock for {product_id}")
        entries[product_id] = entry
    expected_groups = int(payload.get("expected_group_count", -1))
    passed_groups = int(payload.get("qa_passed_group_count", -1))
    if expected_groups < 1 or passed_groups != expected_groups:
        raise ValueError("processed lock does not close every expected group with QA pass")
    selections: dict[str, dict[str, Any]] = {}
    products_in_selection_records: set[str] = set()
    for raw in payload.get("group_selections", []):
        selection = dict(raw)
        group_id = str(selection.get("group_id", ""))
        role = str(selection.get("role", ""))
        if not group_id or not role or group_id in selections:
            raise ValueError("processed group selections require unique group/role identities")
        fixed_products = [str(value) for value in selection.get("fixed_product_ids", [])]
        if not fixed_products or len(fixed_products) != len(set(fixed_products)):
            raise ValueError(f"processed group has invalid fixed product order: {group_id}")
        attempts = [dict(value) for value in selection.get("attempts", [])]
        attempt_products = [str(value.get("product_id", "")) for value in attempts]
        if attempt_products != fixed_products[: len(attempt_products)]:
            raise ValueError(f"processed group did not follow fixed product order: {group_id}")
        selected_product_id = str(selection.get("selected_product_id", ""))
        if (
            not attempts
            or attempt_products[-1] != selected_product_id
            or attempts[-1].get("status") != "qa_passed"
        ):
            raise ValueError(f"processed group lacks a final QA-passed selection: {group_id}")
        if any(
            attempt.get("status")
            not in {"terminal_unavailable", "qa_failed", "source_rejected"}
            for attempt in attempts[:-1]
        ):
            raise ValueError(f"processed group used a nonterminal fallback: {group_id}")
        for order, attempt in enumerate(attempts):
            product_id = attempt_products[order]
            status = attempt.get("status")
            if status in {"qa_failed", "qa_passed"}:
                entry = entries.get(product_id)
                if entry is None:
                    raise ValueError(
                        f"processed group lacks measured record: {group_id}/{product_id}"
                    )
                if (
                    entry.get("geographic_group_id") != group_id
                    or entry.get("role") != role
                    or entry.get("product_order") != order
                    or entry["_qa"].passed is not (status == "qa_passed")
                ):
                    raise ValueError(
                        f"processed group/record binding differs: {group_id}/{product_id}"
                    )
                if product_id in products_in_selection_records:
                    raise ValueError(f"processed product appears in multiple groups: {product_id}")
                products_in_selection_records.add(product_id)
        _require_hash("group state hash", selection.get("group_state_sha256"))
        group_state_path = Path(str(selection.get("group_state_relative_path", "")))
        if (
            not group_state_path.parts
            or group_state_path.is_absolute()
            or ".." in group_state_path.parts
        ):
            raise ValueError(f"processed group state path is invalid: {group_id}")
        if root is not None:
            resolved_group_state = guard_paper2_path(group_state_path, paper2_root=root)
            if (
                not resolved_group_state.is_file()
                or file_sha256(resolved_group_state) != selection["group_state_sha256"]
            ):
                raise ValueError(f"processed group state bytes do not verify: {group_id}")
        selections[group_id] = selection
    if len(selections) != expected_groups or set(selections) != passed_group_ids:
        raise ValueError("processed lock group-selection closure differs from QA-passed groups")
    if products_in_selection_records != set(entries):
        raise ValueError("processed records differ from measured group-selection attempts")
    if len(passed_group_ids) != expected_groups:
        raise ValueError("processed lock QA-passed product count differs from expected groups")
    return entries


def _group_product_order(group: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(group["representative_product_id"]),
        *(str(value) for value in group.get("fallback_product_ids", [])),
    )


def _primary_edr(
    product_id: str,
    *,
    edr_entries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str] | None:
    observations = tuple(value for value in product_id.split("__") if value)
    for observation_id in observations:
        entry = edr_entries.get(observation_id)
        if entry is not None and entry.get("status") == "resolved":
            return observation_id, str(entry["selected_url"])
        if entry is None:
            return None
    return None


def _selected_product(
    group: Mapping[str, Any],
    *,
    edr_entries: Mapping[str, Mapping[str, Any]],
    processed_entries: Mapping[str, Mapping[str, Any]],
    source_rejected_products: frozenset[str] = frozenset(),
) -> tuple[str, str, str, Mapping[str, Any]] | None:
    """Use fixed product order, skipping only terminal EDR absence or measured QA failure."""
    for product_id in _group_product_order(group):
        if product_id in source_rejected_products:
            continue
        observations = tuple(value for value in product_id.split("__") if value)
        observation_entries = [edr_entries.get(value) for value in observations]
        if any(entry is None for entry in observation_entries):
            return None
        edr = _primary_edr(product_id, edr_entries=edr_entries)
        if edr is None:
            terminal = {"missing", "ambiguous", "rejected"}
            if all(str(entry.get("status")) in terminal for entry in observation_entries):
                continue
            return None
        processed = processed_entries.get(product_id)
        if processed is None:
            return None
        qa = processed["_qa"]
        if edr is not None and qa.passed:
            return product_id, edr[0], edr[1], processed
        # A measured QA failure is terminal and permits the next fixed fallback.
        if not qa.failure_reasons:
            return None
    return None


def build_g0_bundle(
    audit: Mapping[str, Any],
    *,
    paper1_lock: Mapping[str, str],
    paper2_root: str | Path | None = None,
    processed_lock_path: str | Path | None = None,
    edr_index_path: str | Path | None = None,
    mola_source_lock_path: str | Path | None = None,
    terrain_manifest_path: str | Path | None = None,
    paper1_seen_lock_path: str | Path | None = None,
    assignment_override: Sequence[Mapping[str, Any]] | None = None,
    assignment_parent_file_sha256: str | None = None,
) -> tuple[dict[str, Any], AssetLock | None, dict[str, Any]]:
    """Build a draft lock and return a formal lock only when every G0 gate closes."""
    conflicts = dict(audit["conflict_groups"])
    conflict_hash = canonical_sha256(conflicts)
    conflicts["content_sha256"] = conflict_hash
    blockers = list(audit.get("errors", []))
    sidecars: dict[str, str | None] = {"conflict_groups_sha256": conflict_hash}
    if assignment_override is not None:
        if assignment_parent_file_sha256 is None or not is_sha256(
            assignment_parent_file_sha256
        ):
            blockers.append("assignment override lacks its validated parent file SHA256")
        else:
            sidecars["assignment_parent_file_sha256"] = assignment_parent_file_sha256
    elif assignment_parent_file_sha256 is not None:
        blockers.append("assignment parent hash was supplied without an assignment override")

    edr_payload, edr_hash = _optional_sidecar(edr_index_path, "edr")
    mola_payload, mola_hash = _optional_sidecar(mola_source_lock_path, "mola")
    terrain_payload, terrain_hash = _optional_sidecar(terrain_manifest_path, "terrain")
    seen_payload, seen_hash = _optional_sidecar(paper1_seen_lock_path, "seen")
    processed_payload, processed_hash = _optional_sidecar(processed_lock_path, "processed")
    sidecars.update(
        {
            "processed_lock_file_sha256": processed_hash,
            "edr_index_file_sha256": edr_hash,
            "mola_source_lock_file_sha256": mola_hash,
            "terrain_manifest_file_sha256": terrain_hash,
            "paper1_seen_lock_file_sha256": seen_hash,
        }
    )

    edr_entries: dict[str, dict[str, Any]] = {}
    mola_downloads_ready = False
    terrain_scores: dict[str, float] = {}
    processed_entries: dict[str, dict[str, Any]] = {}
    paper1_seen_products: set[str] = set()
    paper1_seen_ready = False
    if seen_payload is None:
        blockers.append("Paper1-seen lock sidecar is missing")
    else:
        try:
            paper1_seen_products = set(
                validate_paper1_seen_lock(
                    seen_payload, catalog_sha256=str(audit["inputs"]["catalog"]["sha256"])
                )
            )
            paper1_seen_ready = True
        except ValueError as exc:
            blockers.append(str(exc))
    if edr_payload is None:
        blockers.append("EDR index sidecar is missing")
    else:
        try:
            edr_entries = validate_edr_index(
                edr_payload,
                catalog_sha256=str(audit["inputs"]["catalog"]["sha256"]),
                conflict_groups_sha256=conflict_hash,
            )
        except ValueError as exc:
            blockers.append(str(exc))
    if mola_payload is None or mola_hash is None:
        blockers.append("MOLA source lock sidecar is missing")
    else:
        try:
            mola_downloads_ready = validate_mola_source_lock(mola_payload)
            if not mola_downloads_ready:
                blockers.append("MOLA source files have not all been downloaded and hashed")
        except ValueError as exc:
            blockers.append(str(exc))
    if terrain_payload is None or terrain_hash is None or mola_hash is None:
        blockers.append("terrain manifest sidecar is missing or cannot be bound")
    else:
        try:
            terrain_scores = validate_terrain_manifest(
                terrain_payload,
                conflict_groups_sha256=conflict_hash,
                mola_source_lock_sha256=mola_hash,
            )
        except ValueError as exc:
            blockers.append(str(exc))
    if processed_payload is None or processed_hash is None or edr_hash is None or mola_hash is None:
        blockers.append("processed lock sidecar is missing or cannot be bound")
    else:
        try:
            processed_entries = validate_processed_lock(
                processed_payload,
                conflict_groups_sha256=conflict_hash,
                edr_index_sha256=edr_hash,
                mola_source_lock_sha256=mola_hash,
                paper2_root=paper2_root,
                require_source_preparation=True,
            )
            if seen_payload is not None:
                if (
                    processed_payload.get("paper1_seen_source_sha256")
                    != seen_payload.get("source_file_sha256")
                    or set(processed_payload.get("paper1_seen_product_ids", []))
                    != paper1_seen_products
                ):
                    blockers.append("processed lock differs from the independent Paper1-seen lock")
        except ValueError as exc:
            blockers.append(str(exc))

    eligible_groups = list(conflicts["eligible_groups"])
    all_catalog_products = {
        str(product_id)
        for group in [*conflicts["eligible_groups"], *conflicts.get("excluded_groups", [])]
        for product_id in group["product_ids"]
    }
    unknown_seen = sorted(paper1_seen_products.difference(all_catalog_products))
    if unknown_seen:
        blockers.append(
            f"Paper1-seen lock contains {len(unknown_seen)} products outside the catalog"
        )
        paper1_seen_ready = False
    in_band = [
        group for group in eligible_groups if -65.0 <= float(group["center_latitude"]) <= 65.0
    ]
    missing_terrain = sorted(
        str(group["group_id"]) for group in in_band if str(group["group_id"]) not in terrain_scores
    )
    if missing_terrain:
        blockers.append(f"terrain scores are missing for {len(missing_terrain)} in-band groups")
    extra_terrain = sorted(
        set(terrain_scores).difference(str(group["group_id"]) for group in in_band)
    )
    if extra_terrain:
        blockers.append(f"terrain scores contain {len(extra_terrain)} unexpected groups")

    assignments: dict[str, str] = {}
    stratified: tuple[GeographicGroup, ...] = ()
    if in_band and not missing_terrain and paper1_seen_ready:
        groups = tuple(
            GeographicGroup(
                group_id=str(group["group_id"]),
                product_ids=tuple(str(value) for value in group["product_ids"]),
                observation_ids=tuple(str(value) for value in group["observation_ids"]),
                bbox_lonlat=tuple(float(value) for value in group["bbox_lonlat"]),
                center_latitude=float(group["center_latitude"]),
                terrain_score=terrain_scores[str(group["group_id"])],
                seen_by_paper1=any(
                    str(product_id) in paper1_seen_products for product_id in group["product_ids"]
                ),
            )
            for group in in_band
        )
        try:
            stratified = stratify_groups(groups)
            assignments = assign_stratified_roles(
                stratified, seed_material=str(audit["seed_material"])
            )
        except ValueError as exc:
            blockers.append(str(exc))

    strata = {group.group_id: [group.latitude_band, group.terrain_quartile] for group in stratified}
    group_by_id = {str(group["group_id"]): group for group in eligible_groups}
    if assignment_override is not None and assignments:
        try:
            records = [dict(raw) for raw in assignment_override]
            record_by_group = {str(raw.get("group_id", "")): raw for raw in records}
            if len(records) != 108 or len(record_by_group) != len(records):
                raise ValueError("assignment override must contain 108 unique groups")
            if not set(record_by_group).issubset(group_by_id):
                raise ValueError("assignment override contains a group outside the G0 catalog")
            expected_role_counts = Counter(assignments.values())
            observed_role_counts = Counter(str(raw.get("role", "")) for raw in records)
            if observed_role_counts != expected_role_counts:
                raise ValueError("assignment override changes the frozen formal role counts")
            for group_id, raw in record_by_group.items():
                group = group_by_id[group_id]
                expected_products = [
                    str(group["representative_product_id"]),
                    *(str(value) for value in group.get("fallback_product_ids", [])),
                ]
                observed_products = [
                    str(raw.get("representative_product_id", "")),
                    *(str(value) for value in raw.get("fallback_product_ids", [])),
                ]
                if observed_products != expected_products:
                    raise ValueError(
                        f"assignment override changes fixed product order for {group_id}"
                    )
                if [raw.get("latitude_band"), raw.get("terrain_quartile")] != strata.get(
                    group_id
                ):
                    raise ValueError(
                        f"assignment override changes the frozen stratum for {group_id}"
                    )
                role = str(raw["role"])
                if role != "bridge_train" and any(
                    product_id in paper1_seen_products for product_id in group["product_ids"]
                ):
                    raise ValueError(
                        f"assignment override places Paper1-seen group outside bridge_train: {group_id}"
                    )
            assignments = {
                group_id: str(record_by_group[group_id]["role"])
                for group_id in sorted(record_by_group)
            }
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(str(exc))
            assignments = {}
    if assignments and edr_payload is not None:
        assignment_contract = [
            {
                "group_id": group_id,
                "role": role,
                "representative_product_id": str(
                    group_by_id[group_id]["representative_product_id"]
                ),
                "fallback_product_ids": [
                    str(value) for value in group_by_id[group_id].get("fallback_product_ids", [])
                ],
            }
            for group_id, role in sorted(assignments.items())
        ]
        expected_observations = {
            observation_id
            for assignment in assignment_contract
            for product_id in [
                assignment["representative_product_id"],
                *assignment["fallback_product_ids"],
            ]
            for observation_id in str(product_id).split("__")
            if observation_id
        }
        if edr_payload.get("role_assignment_sha256") != canonical_sha256(assignment_contract):
            blockers.append("EDR index role assignment binding differs from frozen G0 assignment")
        if set(edr_entries) != expected_observations:
            blockers.append("EDR index observations differ from fixed representative/fallback plan")
    selected_assets: list[AssetRecord] = []
    source_rejected_products = frozenset(
        str(attempt.get("product_id", ""))
        for selection in (processed_payload or {}).get("group_selections", [])
        for attempt in selection.get("attempts", [])
        if attempt.get("status") == "source_rejected"
    )
    for group_id, role in sorted(assignments.items()):
        group = group_by_id[group_id]
        selected = _selected_product(
            group,
            edr_entries=edr_entries,
            processed_entries=processed_entries,
            source_rejected_products=source_rejected_products,
        )
        if selected is None:
            blockers.append(f"group {group_id} lacks a QA-passed fixed representative/fallback")
            continue
        product_id, primary_observation, edr_url, processed = selected
        expected_product_order = _group_product_order(group).index(product_id)
        if (
            processed.get("geographic_group_id") != group_id
            or processed.get("role") != role
            or processed.get("product_order") != expected_product_order
        ):
            blockers.append(f"processed group/role/order mismatch for {product_id}")
            continue
        if processed.get("primary_observation_id") != primary_observation:
            blockers.append(f"processed primary observation mismatch for {product_id}")
            continue
        latitude_band, terrain_quartile = strata[group_id]
        selected_assets.append(
            AssetRecord(
                product_id=product_id,
                mission="CTX",
                observation_ids=tuple(str(value) for value in group["observation_ids"]),
                geographic_group_id=group_id,
                role=role,
                bbox_lonlat=tuple(float(value) for value in group["bbox_lonlat"]),
                latitude_band=int(latitude_band),
                terrain_quartile=int(terrain_quartile),
                urls=tuple(
                    dict.fromkeys([edr_url, *(str(value) for value in processed.get("urls", []))])
                ),
                files=tuple(processed["_files"]),
                imagery_lineage=tuple(str(value) for value in processed.get("imagery_lineage", [])),
                dem_lineage=tuple(str(value) for value in processed.get("dem_lineage", [])),
                prepared_manifest_sha256=str(processed["prepared_manifest_sha256"]),
                problem_sha256=str(processed["problem_sha256"]),
                seen_by_paper1=product_id in paper1_seen_products,
                qa=processed["_qa"],
            )
        )

    blockers = sorted(set(blockers))
    draft = {
        "schema": SCHEMAS["draft"],
        "state": "ready_for_formal_freeze"
        if not blockers and len(selected_assets) == 108
        else "blocked",
        "formal_test_unlocked": False,
        "bindings": sidecars,
        "locked_latitude_bands": [[-65.0, -20.0], [-20.0, 20.0], [20.0, 65.0]],
        "terrain_quartile_rule": ("balanced-rank-on-paper1-unseen-within-fixed-latitude-band-v2"),
        "eligible_group_count": len(eligible_groups),
        "in_band_group_count": len(in_band),
        "assignments": [
            {
                "group_id": group_id,
                "role": role,
                "latitude_band": strata.get(group_id, [None, None])[0],
                "terrain_quartile": strata.get(group_id, [None, None])[1],
                "representative_product_id": group_by_id[group_id]["representative_product_id"],
                "fallback_product_ids": group_by_id[group_id]["fallback_product_ids"],
            }
            for group_id, role in sorted(assignments.items())
        ],
        "selected_qa_passed_product_count": len(selected_assets),
        "blockers": blockers,
    }
    draft["draft_sha256"] = canonical_sha256(draft)

    formal: AssetLock | None = None
    if draft["state"] == "ready_for_formal_freeze":
        formal = AssetLock(
            paper1=Paper1SourceLock(**{key: str(value) for key, value in paper1_lock.items()}),
            source_catalog_sha256=str(audit["inputs"]["catalog"]["sha256"]),
            grouping_buffer_degrees=float(audit["buffer_degrees"]),
            grouping_algorithm=str(conflicts["grouping_algorithm"]),
            assignment_seed_sha256=canonical_sha256(str(audit["seed_material"])),
            assets=tuple(selected_assets),
        )
        geography = audit_asset_lock_geography(formal)
        if not geography.ok:
            draft["state"] = "blocked"
            draft["blockers"] = list(geography.errors)
            draft["draft_sha256"] = canonical_sha256(
                {key: value for key, value in draft.items() if key != "draft_sha256"}
            )
            formal = None
    return draft, formal, conflicts


def write_g0_bundle(
    *,
    draft: Mapping[str, Any],
    conflicts: Mapping[str, Any],
    formal: AssetLock | None,
    draft_path: str | Path,
    conflicts_path: str | Path,
    formal_path: str | Path | None,
    paper2_root: str | Path,
) -> dict[str, str | None]:
    conflict_output = write_canonical_json(conflicts_path, conflicts, paper2_root=paper2_root)
    draft_path = str(draft_path).replace("{sha256}", str(draft["draft_sha256"]))
    draft_output = write_canonical_json(draft_path, draft, paper2_root=paper2_root)
    formal_output = None
    if formal is not None:
        if formal_path is None:
            raise ValueError("formal asset lock is ready but no output path was supplied")
        formal_output = formal.write(formal_path, paper2_root=paper2_root)
    return {
        "conflict_groups": str(conflict_output),
        "draft_asset_lock": str(draft_output),
        "formal_asset_lock": None if formal_output is None else str(formal_output),
    }
