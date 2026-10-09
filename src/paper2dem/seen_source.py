from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import re

from .hashing import file_sha256


PAPER1_DATA_DOMAINS = (
    "benchmark",
    "information-v2",
    "pilot",
    "pilot-sensitivity",
    "image-pilot",
    "screening-biased-catalog-20260719",
    "sensitivity",
)
_OBSERVATION = r"[A-Z][0-9]{2}_[0-9]{6}_[0-9]{4}_[A-Z]{2}_[0-9]{2}[NS][0-9]{3}W"
_CTX_DIRECTORY = re.compile(
    rf"^(?P<first>{_OBSERVATION})__(?P<second>{_OBSERVATION})(?:__.+)?$"
)
_TIER_ORDER = {"dir_only": 0, "partial_file": 1, "audit_only": 2, "prepared": 3}


def _sorted_lines_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence_tier(directory: Path) -> str:
    if (directory / "prepared-manifest.json").is_file():
        return "prepared"
    if (directory / "asset-audit.json").is_file():
        return "audit_only"
    if any(path.is_file() for path in directory.rglob("*")):
        return "partial_file"
    return "dir_only"


def _catalog_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("Paper1 formal catalog must be a JSON list of objects")
    return payload


def derive_paper1_seen_source(
    *,
    paper1_data_root: str | Path,
    catalog: str | Path,
    paper1_source_lock: str | Path | None = None,
    benchmark_lock: str | Path | None = None,
    domains: tuple[str, ...] = PAPER1_DATA_DOMAINS,
) -> dict[str, Any]:
    """Project every Paper1 CTX processing footprint onto the current CTX catalog."""
    root = Path(paper1_data_root).resolve()
    catalog_path = Path(catalog).resolve()
    if (paper1_source_lock is None) != (benchmark_lock is None):
        raise ValueError("Paper1 source lock and benchmark lock must be supplied together")
    if len(domains) != len(set(domains)) or not domains:
        raise ValueError("Paper1 seen-source domains must be nonempty and unique")
    missing_domains = [domain for domain in domains if not (root / domain).is_dir()]
    if missing_domains:
        raise FileNotFoundError(f"Paper1 data domains are missing: {missing_domains}")

    directory_records: list[dict[str, str]] = []
    evidence_by_tile: dict[str, list[dict[str, str]]] = defaultdict(list)
    tiles_by_domain: dict[str, set[str]] = defaultdict(set)
    for domain in domains:
        domain_path = root / domain
        for directory in sorted(path for path in domain_path.rglob("*") if path.is_dir()):
            match = _CTX_DIRECTORY.fullmatch(directory.name)
            if match is None:
                continue
            tile_id = f"{match.group('first')}__{match.group('second')}"
            relative_path = directory.relative_to(root).as_posix()
            tier = _evidence_tier(directory)
            record = {
                "relative_directory": relative_path,
                "standardized_tile_id": tile_id,
                "evidence_tier": tier,
            }
            directory_records.append(record)
            evidence_by_tile[tile_id].append(record)
            tiles_by_domain[domain].add(tile_id)

    if not directory_records:
        raise ValueError("no Paper1 CTX processing directories were discovered")
    directory_records.sort(
        key=lambda value: (value["relative_directory"], value["standardized_tile_id"])
    )
    raw_tiles = sorted(evidence_by_tile)
    seen_observations = sorted(
        {observation for tile in raw_tiles for observation in tile.split("__")[:2]}
    )

    evidence_records = []
    for tile_id in raw_tiles:
        records = evidence_by_tile[tile_id]
        strongest = max(
            (record["evidence_tier"] for record in records), key=_TIER_ORDER.__getitem__
        )
        evidence_records.append(
            {
                "tile_id": tile_id,
                "evidence_tier": strongest,
                "relative_directories": sorted(
                    record["relative_directory"] for record in records
                ),
            }
        )

    rows = _catalog_rows(catalog_path)
    catalog_ctx: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if "ctx" not in str(row.get("collection", "")).lower():
            continue
        tile_id = str(row.get("tile_id", ""))
        observations = tuple(value for value in tile_id.split("__") if value)
        if len(observations) < 2 or tile_id in catalog_ctx:
            raise ValueError(f"invalid or duplicate CTX catalog tile: {tile_id}")
        catalog_ctx[tile_id] = observations[:2]
    seen_set = set(seen_observations)
    projected = sorted(
        tile_id
        for tile_id, observations in catalog_ctx.items()
        if seen_set.intersection(observations)
    )
    direct_catalog_tiles = sorted(set(raw_tiles).intersection(catalog_ctx))
    tier_counts = Counter(record["evidence_tier"] for record in evidence_records)
    domain_counts = Counter(
        record["relative_directory"].split("/", 1)[0] for record in directory_records
    )
    snapshot_lines = [
        f"{record['relative_directory']}\t{record['standardized_tile_id']}"
        for record in directory_records
    ]
    paper1_release: dict[str, Any] | None = None
    if paper1_source_lock is not None and benchmark_lock is not None:
        source_lock_path = Path(paper1_source_lock).resolve()
        benchmark_lock_path = Path(benchmark_lock).resolve()
        source_payload = json.loads(source_lock_path.read_text(encoding="utf-8"))
        if source_payload.get("schema") != "information-v2-source-lock":
            raise ValueError("Paper1 source lock schema differs")
        release_commit = str(source_payload.get("release_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", release_commit):
            raise ValueError("Paper1 release commit is invalid")
        paper1_release = {
            "release_commit": release_commit,
            "source_lock_sha256": file_sha256(source_lock_path),
            "benchmark_lock_sha256": file_sha256(benchmark_lock_path),
        }
    return {
        "schema": "paper2.paper1-seen-source/v2",
        "catalog_sha256": file_sha256(catalog_path),
        "paper1_release": paper1_release,
        "source_domains": list(domains),
        "derivation_rule": (
            "Normalize every matching Paper1 processing directory to its first two CTX "
            "observation IDs, union observations, then mark every current catalog CTX "
            "tile sharing either observation as seen. Directory-only attempts are included."
        ),
        "hash_rule": "SHA256 of UTF-8 canonical sorted lines with one trailing LF per item",
        "source_directory_record_count": len(directory_records),
        "source_directory_counts_by_domain": dict(sorted(domain_counts.items())),
        "source_unique_tile_counts_by_domain": {
            domain: len(tiles_by_domain[domain]) for domain in sorted(domains)
        },
        "source_directory_snapshot_sha256": _sorted_lines_sha256(snapshot_lines),
        "raw_attempted_tile_count": len(raw_tiles),
        "raw_attempted_tile_ids": raw_tiles,
        "raw_attempted_tile_ids_sha256": _sorted_lines_sha256(raw_tiles),
        "seen_observation_count": len(seen_observations),
        "seen_observation_ids": seen_observations,
        "seen_observation_ids_sha256": _sorted_lines_sha256(seen_observations),
        "direct_catalog_tile_count": len(direct_catalog_tiles),
        "direct_catalog_tile_ids": direct_catalog_tiles,
        "catalog_projected_tile_count": len(projected),
        "catalog_projected_tile_ids": projected,
        "catalog_projected_tile_ids_sha256": _sorted_lines_sha256(projected),
        "paper1_seen_product_ids": projected,
        "evidence_tier_counts": dict(sorted(tier_counts.items())),
        "evidence_records": evidence_records,
        "source_directory_records": directory_records,
    }
