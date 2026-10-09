from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import fcntl

import numpy as np

from .ale_spiceql_compat import (
    OFFICIAL_SPICEQL_URL,
    WEB_QUERY_FUNCTIONS,
    WEB_REQUEST_ATTEMPTS,
    WEB_REQUEST_TIMEOUT_SECONDS,
)
from .isis_preferences import verify_spiceql_preferences
from .isis_environment import verify_isis_environment
from .isis_runtime_assets import DEFAULT_LOCK as ISIS_BASE_KERNEL_LOCK
from .isis_runtime_assets import verify_isis_base_kernel_lock


MOLA_PREPARATION_SCHEMA = "paper2.mola-isis-preparation/v1"
ISIS_STAGE_TIMEOUT_SECONDS = 2 * 60 * 60
ISIS_METADATA_TIMEOUT_SECONDS = 120
SPICEQL_STAGE_MAX_ATTEMPTS = 4
SPICEQL_STAGE_RETRY_MAX_SECONDS = 4.0
# Kept as a compatibility alias for manifests and callers written before the
# C++ ISIS applications were observed to surface the same transient SpiceQL
# transport failure as ALE's isd_generate.
ISD_GENERATE_MAX_ATTEMPTS = SPICEQL_STAGE_MAX_ATTEMPTS
ISD_GENERATE_RETRY_MAX_SECONDS = SPICEQL_STAGE_RETRY_MAX_SECONDS
SPICEQL_RETRYABLE_STAGES = frozenset(
    {
        "mroctx2isis",
        "ctxcal",
        "ctxevenodd",
        "isd_generate",
        "csminit",
        "cam2map",
    }
)
ALE_SPICEQL_COMPAT_SCHEMA = "paper2.ale-spiceql-compatibility/v2"
ALE_SPICEQL_WRAPPER = Path("bin/isd_generate")
ALE_SPICEQL_MODULE = Path("src/paper2dem/ale_spiceql_compat.py")


def is_transient_spiceql_failure(record: Mapping[str, Any]) -> bool:
    """Recognize a transport-only SpiceQL failure from a frozen CTX stage."""

    argv = record.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or argv[0] not in SPICEQL_RETRYABLE_STAGES
        or (argv[0] == "isd_generate" and argv[1:] == ["--version"])
    ):
        return False
    # isd_generate is an explicitly online operation, so a timeout is itself a
    # transport failure. Other ISIS applications can also be compute-bound;
    # retry their timeouts only when the diagnostic explicitly names SpiceQL.
    if record.get("timed_out") is True and argv[0] == "isd_generate":
        return True
    diagnostic = f"{record.get('stdout', '')}\n{record.get('stderr', '')}".lower()
    if "spiceql" not in diagnostic:
        return False
    return any(
        marker in diagnostic
        for marker in (
            "error parsing json",
            "unexpected end of input",
            "empty response",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "remote end closed",
            "temporary failure",
            "tls",
            "ssl",
            "http error",
            "network",
        )
    )


def is_transient_isd_generate_failure(record: Mapping[str, Any]) -> bool:
    """Backward-compatible narrow predicate used by existing audit callers."""

    argv = record.get("argv")
    return (
        isinstance(argv, list)
        and bool(argv)
        and argv[0] == "isd_generate"
        and is_transient_spiceql_failure(record)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_ale_spiceql_compatibility(paper2_root: Path) -> dict[str, Any]:
    """Bind the exact launcher and validated official-web transport implementation."""
    root = paper2_root.resolve()
    wrapper = assert_inside(root, root / ALE_SPICEQL_WRAPPER)
    module = assert_inside(root, root / ALE_SPICEQL_MODULE)
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise FileNotFoundError(f"executable ALE compatibility wrapper is missing: {wrapper}")
    if not module.is_file():
        raise FileNotFoundError(f"ALE compatibility module is missing: {module}")
    return {
        "schema": ALE_SPICEQL_COMPAT_SCHEMA,
        "strategy": "validated_requests_transport_with_serialized_ale_executors",
        "wrapper_relative_path": ALE_SPICEQL_WRAPPER.as_posix(),
        "wrapper_sha256": file_sha256(wrapper),
        "module_relative_path": ALE_SPICEQL_MODULE.as_posix(),
        "module_sha256": file_sha256(module),
        "upstream_ale_version": "1.2.0",
        "official_spiceql_endpoint": OFFICIAL_SPICEQL_URL,
        "patched_web_functions": list(WEB_QUERY_FUNCTIONS),
        "web_request_attempts": WEB_REQUEST_ATTEMPTS,
        "web_request_timeout_seconds": list(WEB_REQUEST_TIMEOUT_SECONDS),
        "response_integrity_checks": [
            "http_status",
            "content_length_when_identity_encoded",
            "utf8_json",
            "api_status",
            "return_and_kernels_shape",
        ],
        "upstream_source_sha256": {
            "ale.base.base": (
                "571bdf4f94c3a322bcbeb4f8793f24f61b8585b7e8c1fadecef233e0ad2fb95c"
            ),
            "ale.base.data_naif": (
                "fcbcd18a09246a54af4613a1c6d48d31e276241f37e68834c300ad5d29aa76d3"
            ),
            "ale.transformation": (
                "06ac4ba4224bfcda46cd8378ba4a65105518a4e647a3ec59579e9ec35ca9ae8d"
            ),
        },
        "effective_web_worker_cap": 1,
    }


def assert_inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Paper2 write path escapes root: {resolved}")
    return resolved


def primary_ctx_observation(stereo_product_id: str, available: Iterable[str]) -> tuple[str, str]:
    observations = tuple(part for part in stereo_product_id.split("__") if part)
    if len(observations) < 2:
        raise ValueError("CTX stereo product ID must contain two observations separated by '__'")
    available_set = set(available)
    if observations[0] in available_set:
        return observations[0], "first_observation"
    if observations[1] in available_set:
        return observations[1], "first_missing_used_second"
    raise FileNotFoundError(f"neither CTX observation is available for {stereo_product_id}")


@dataclass(frozen=True)
class CTXPipelineConfig:
    paper2_root: Path
    isis_prefix: Path
    isis_data: Path
    spice_cache: Path
    conda_cache: Path
    mola_radius_label: Path
    mola_radius_raw_cube: Path
    mola_radius_prepared_cube: Path
    map_template: Path
    isis_preferences: Path
    isis_version: str = "10.0.0"
    isis_build: str = "h1f94ec8_1"
    ale_version: str = "1.2.0"
    ale_build: str = "py313h7037e92_1"
    csm_version: str = "3.1.0"
    csm_build: str = "hb700be7_0"
    usgscsm_version: str = "2.1.0"
    usgscsm_build: str = "h71e3a03_3"
    spice_backend: str = "ale_spiceql_csm"
    spiceql_rest_url: str = "https://astrogeology.usgs.gov/apis/spiceql/latest/"
    spiceql_https_proxy: str | None = None

    def validate(self) -> None:
        root = self.paper2_root.resolve()
        for path in (
            self.isis_prefix,
            self.isis_data,
            self.spice_cache,
            self.conda_cache,
            self.mola_radius_label,
            self.mola_radius_raw_cube,
            self.mola_radius_prepared_cube,
            self.map_template,
            self.isis_preferences,
        ):
            assert_inside(root, path)
        if self.isis_version != "10.0.0":
            raise ValueError("formal protocol freezes USGS ISIS 10.0.0")
        if self.isis_build != "h1f94ec8_1":
            raise ValueError("formal protocol freezes the ISIS 10.0.0 Linux build")
        if (self.ale_version, self.ale_build) != ("1.2.0", "py313h7037e92_1"):
            raise ValueError("formal protocol freezes ALE 1.2.0 for CPython 3.13")
        if (self.csm_version, self.csm_build) != ("3.1.0", "hb700be7_0"):
            raise ValueError("formal protocol freezes CSM 3.1.0")
        if (self.usgscsm_version, self.usgscsm_build) != ("2.1.0", "h71e3a03_3"):
            raise ValueError("formal protocol freezes USGSCSM 2.1.0")
        if self.spice_backend != "ale_spiceql_csm":
            raise ValueError("formal CTX preparation freezes the ALE/SpiceQL/CSM backend")
        if self.spiceql_rest_url != "https://astrogeology.usgs.gov/apis/spiceql/latest/":
            raise ValueError("formal protocol freezes the official SpiceQL REST endpoint")
        if self.spiceql_https_proxy is not None:
            parsed = urlparse(self.spiceql_https_proxy)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.port is None
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError("SpiceQL CONNECT proxy must be an unauthenticated loopback HTTP URL")
        runtime_preferences = (self.isis_prefix / "IsisPreferences").resolve()
        if self.isis_preferences.resolve() != runtime_preferences:
            raise ValueError(
                "ISIS preferences must be the active conda-prefix IsisPreferences file"
            )
        verify_spiceql_preferences(runtime_preferences)

    def environment(self) -> dict[str, str]:
        self.validate()
        environment = dict(os.environ)
        for name in (
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            environment.pop(name, None)
        if self.spiceql_https_proxy is not None:
            environment["HTTPS_PROXY"] = self.spiceql_https_proxy
            environment["https_proxy"] = self.spiceql_https_proxy
        existing_path = environment.get("PATH", "")
        existing_library_path = environment.get("LD_LIBRARY_PATH", "")
        environment.update(
            {
                "ISISROOT": str(self.isis_prefix),
                "CONDA_PREFIX": str(self.isis_prefix),
                "ISISDATA": str(self.isis_data),
                "CONDA_PKGS_DIRS": str(self.conda_cache),
                "XDG_CACHE_HOME": str(self.paper2_root / ".cache"),
                "TMPDIR": str(self.paper2_root / ".tmp"),
                "SPICEQL_REST_URL": self.spiceql_rest_url,
                "ALESPICEQL_LOG_LEVEL": "debug",
                "PATH": (
                    f"{self.paper2_root / 'bin'}:{self.isis_prefix / 'bin'}:{existing_path}"
                ),
                "LD_LIBRARY_PATH": (
                    f"{self.isis_prefix / 'lib'}:{existing_library_path}"
                    if existing_library_path
                    else str(self.isis_prefix / "lib")
                ),
            }
        )
        return environment


def mola_preparation_manifest_path(config: CTXPipelineConfig) -> Path:
    return config.mola_radius_prepared_cube.with_name(
        f"{config.mola_radius_prepared_cube.name}.manifest.json"
    )


def verify_mola_preparation(config: CTXPipelineConfig) -> dict[str, Any]:
    """Verify the atomic MOLA pds2isis+demprep publication and all input bytes."""
    config.validate()
    root = config.paper2_root.resolve()
    manifest_path = assert_inside(root, mola_preparation_manifest_path(config))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MOLA preparation manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MOLA_PREPARATION_SCHEMA:
        raise ValueError("MOLA preparation manifest schema differs")
    if payload.get("isis_version") != config.isis_version:
        raise ValueError("MOLA preparation ISIS version differs")
    expected_paths = {
        "source_label": config.mola_radius_label,
        "raw_cube": config.mola_radius_raw_cube,
        "prepared_cube": config.mola_radius_prepared_cube,
    }
    for name, path in expected_paths.items():
        path = assert_inside(root, path)
        record = dict(payload.get("artifacts", {}).get(name, {}))
        if record.get("relative_path") != path.relative_to(root).as_posix():
            raise ValueError(f"MOLA preparation {name} path differs")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"MOLA preparation {name} is missing or empty")
        if record.get("sha256") != file_sha256(path):
            raise ValueError(f"MOLA preparation {name} hash differs")
        if int(record.get("byte_count", -1)) != path.stat().st_size:
            raise ValueError(f"MOLA preparation {name} byte count differs")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if declared != hashlib.sha256(canonical).hexdigest():
        raise ValueError("MOLA preparation manifest content hash differs")
    return payload


def mola_preparation_ready(config: CTXPipelineConfig) -> bool:
    try:
        verify_mola_preparation(config)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def mola_isis_commands(config: CTXPipelineConfig) -> tuple[tuple[str, ...], ...]:
    """Convert the official MEGR radius product to an ISIS-ready shape DEM."""
    config.validate()
    return (
        (
            "pds2isis",
            f"from={config.mola_radius_label}",
            f"to={config.mola_radius_raw_cube}",
        ),
        (
            "demprep",
            f"from={config.mola_radius_raw_cube}",
            f"to={config.mola_radius_prepared_cube}",
        ),
    )


def ctx_isis_commands(
    *,
    edr: Path,
    output_directory: Path,
    config: CTXPipelineConfig,
    spatial_summing: int,
) -> tuple[tuple[str, ...], ...]:
    config.validate()
    output_directory = assert_inside(config.paper2_root, output_directory)
    edr = assert_inside(config.paper2_root, edr)
    raw = output_directory / "ctx_raw.cub"
    calibrated = output_directory / "ctx_cal.cub"
    evenodd = output_directory / "ctx_evenodd.cub"
    isd = output_directory / "ctx_isd.json"
    projected = output_directory / "ctx_cam2map.cub"
    if spatial_summing < 1:
        raise ValueError("SpatialSumming must be a positive integer")
    commands: list[tuple[str, ...]] = [
        ("mroctx2isis", f"from={edr}", f"to={raw}"),
        ("ctxcal", f"from={raw}", f"to={calibrated}"),
    ]
    csm_input = calibrated
    if spatial_summing != 2:
        commands.append(("ctxevenodd", f"from={calibrated}", f"to={evenodd}"))
        csm_input = evenodd
    commands.extend(
        (
            (
                "isd_generate",
                "--use_web_spice",
                "--only_naif_spice",
                "--out",
                str(isd),
                str(csm_input),
            ),
            (
                "csminit",
                f"from={csm_input}",
                f"isd={isd}",
                f"shapemodel={config.mola_radius_prepared_cube}",
            ),
        )
    )
    commands.append(
        (
            "cam2map",
            f"from={csm_input}",
            f"to={projected}",
            f"map={config.map_template}",
        )
    )
    return tuple(commands)


def read_spatial_summing(
    raw_cube: Path, *, config: CTXPipelineConfig
) -> tuple[int, dict[str, Any]]:
    """Read CTX Instrument/SpatialSumming from the converted ISIS cube."""
    command = (
        "getkey",
        f"from={raw_cube}",
        "grp=Instrument",
        "keyword=SpatialSumming",
    )
    completed = subprocess.run(
        command,
        env=config.environment(),
        check=False,
        text=True,
        capture_output=True,
        timeout=ISIS_METADATA_TIMEOUT_SECONDS,
    )
    record = {
        "argv": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode:
        raise RuntimeError("ISIS getkey could not read Instrument/SpatialSumming")
    match = next(
        (token for token in completed.stdout.replace("=", " ").split() if token.isdigit()), None
    )
    if match is None or int(match) < 1:
        raise ValueError(f"invalid SpatialSumming value from ISIS: {completed.stdout!r}")
    return int(match), record


def read_isis_version(*, config: CTXPipelineConfig) -> tuple[str, dict[str, Any]]:
    command = ("isisversion",)
    completed = subprocess.run(
        command,
        env=config.environment(),
        check=False,
        text=True,
        capture_output=True,
        timeout=ISIS_METADATA_TIMEOUT_SECONDS,
    )
    record = {
        "argv": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode:
        raise RuntimeError("ISIS version query failed")
    reported_versions = set(re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", completed.stdout))
    if not reported_versions:
        raise ValueError(f"could not parse ISIS version: {completed.stdout!r}")
    if config.isis_version not in reported_versions:
        raise ValueError(
            f"runtime ISIS versions {sorted(reported_versions)} do not include frozen "
            f"{config.isis_version}"
        )
    return config.isis_version, record


def read_ale_version(*, config: CTXPipelineConfig) -> tuple[str, dict[str, Any]]:
    command = ("isd_generate", "--version")
    completed = subprocess.run(
        command,
        env=config.environment(),
        check=False,
        text=True,
        capture_output=True,
        timeout=ISIS_METADATA_TIMEOUT_SECONDS,
    )
    record = {
        "argv": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode:
        raise RuntimeError("ALE version query failed")
    reported_versions = set(re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", completed.stdout))
    if config.ale_version not in reported_versions:
        raise ValueError(
            f"runtime ALE versions {sorted(reported_versions)} do not include frozen "
            f"{config.ale_version}"
        )
    return config.ale_version, record


def verify_csm_environment(config: CTXPipelineConfig) -> dict[str, Any]:
    """Verify exact Conda records and the loadable USGSCSM plugin bytes."""
    config.validate()
    return verify_isis_environment(
        prefix=config.isis_prefix,
        paper2_root=config.paper2_root,
        require_usgscsm=True,
    )


def run_ctx_isis(
    *,
    edr: Path,
    output_directory: Path,
    config: CTXPipelineConfig,
    dry_run: bool = False,
    spatial_summing: int | None = None,
    spiceql_stage_max_attempts: int = SPICEQL_STAGE_MAX_ATTEMPTS,
    retry_sleeper: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    config.validate()
    if not 1 <= spiceql_stage_max_attempts <= SPICEQL_STAGE_MAX_ATTEMPTS:
        raise ValueError(
            "spiceql_stage_max_attempts must be between 1 and "
            f"{SPICEQL_STAGE_MAX_ATTEMPTS}"
        )
    edr = assert_inside(config.paper2_root, edr)
    output_directory = assert_inside(config.paper2_root, output_directory)
    all_mola_commands = mola_isis_commands(config)
    mola_commands = () if mola_preparation_ready(config) else all_mola_commands
    if dry_run and spatial_summing is None:
        raise ValueError("dry-run requires an explicit SpatialSumming value")
    provisional_summing = spatial_summing or 1
    commands = ctx_isis_commands(
        edr=edr,
        output_directory=output_directory,
        config=config,
        spatial_summing=provisional_summing,
    )
    executables = ["isisversion", *(command[0] for command in (*mola_commands, *commands))]
    if spatial_summing is None:
        executables.append("getkey")
    missing = [
        executable
        for executable in executables
        if shutil.which(executable, path=config.environment().get("PATH")) is None
    ]
    if missing and not dry_run:
        raise RuntimeError(f"ISIS executables unavailable: {', '.join(sorted(set(missing)))}")
    output_directory.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    transient_command_failures: list[dict[str, Any]] = []

    def execute(command: tuple[str, ...]) -> None:
        retryable_stage = command[0] in SPICEQL_RETRYABLE_STAGES and not (
            command[0] == "isd_generate" and command[1:] == ("--version",)
        )
        maximum_attempts = spiceql_stage_max_attempts if retryable_stage else 1
        for command_attempt in range(1, maximum_attempts + 1):
            record: dict[str, Any] = {"argv": list(command)}
            if dry_run:
                logs.append(record)
                return
            try:
                completed = subprocess.run(
                    command,
                    env=config.environment(),
                    cwd=output_directory,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=ISIS_STAGE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                record.update(
                    {
                        "returncode": None,
                        "timed_out": True,
                        "timeout_seconds": ISIS_STAGE_TIMEOUT_SECONDS,
                        "stdout": str(exc.stdout or "")[-4000:],
                        "stderr": str(exc.stderr or "")[-4000:],
                    }
                )
                if retryable_stage and command_attempt < maximum_attempts:
                    failure = {
                        **record,
                        "stage": command[0],
                        "attempt": command_attempt,
                        "retry_reason": "bounded_spiceql_transport_failure",
                    }
                    if is_transient_spiceql_failure(failure):
                        transient_command_failures.append(failure)
                        retry_sleeper(
                            min(
                                float(2 ** (command_attempt - 1)),
                                SPICEQL_STAGE_RETRY_MAX_SECONDS,
                            )
                        )
                        continue
                logs.append(record)
                raise RuntimeError(f"ISIS stage timed out: {command[0]}") from exc
            record.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                }
            )
            if completed.returncode:
                if retryable_stage and command_attempt < maximum_attempts:
                    failure = {
                        **record,
                        "stage": command[0],
                        "attempt": command_attempt,
                        "retry_reason": "bounded_spiceql_transport_failure",
                    }
                    if is_transient_spiceql_failure(failure):
                        transient_command_failures.append(failure)
                        retry_sleeper(
                            min(
                                float(2 ** (command_attempt - 1)),
                                SPICEQL_STAGE_RETRY_MAX_SECONDS,
                            )
                        )
                        continue
                logs.append(record)
                raise RuntimeError(
                    f"ISIS stage failed: {command[0]}; stderr={completed.stderr[-1000:]!r}"
                )
            logs.append(record)
            return
        raise AssertionError("unreachable ISIS command retry loop")

    reported_isis_version = config.isis_version
    reported_ale_version = config.ale_version
    ale_spiceql_compatibility: dict[str, Any] | None = None
    csm_environment: dict[str, Any] = {
        "packages": {
            "isis": {"version": config.isis_version, "build": config.isis_build},
            "ale": {"version": config.ale_version, "build": config.ale_build},
            "csm": {"version": config.csm_version, "build": config.csm_build},
            "usgscsm": {
                "version": config.usgscsm_version,
                "build": config.usgscsm_build,
            },
        },
        "plugins": [],
    }
    if dry_run:
        logs.append({"argv": ["isisversion"], "reported_version": config.isis_version})
        logs.append(
            {
                "argv": ["isd_generate", "--version"],
                "reported_version": config.ale_version,
            }
        )
        for command in mola_commands:
            execute(command)
    else:
        ale_spiceql_compatibility = verify_ale_spiceql_compatibility(config.paper2_root)
        verify_isis_base_kernel_lock(paper2_root=config.paper2_root)
        reported_isis_version, version_record = read_isis_version(config=config)
        logs.append(version_record)
        reported_ale_version, ale_version_record = read_ale_version(config=config)
        logs.append(ale_version_record)
        csm_environment = verify_csm_environment(config)
        lock_path = config.mola_radius_prepared_cube.parent / ".mola-preparation.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if not mola_preparation_ready(config):
                with tempfile.TemporaryDirectory(
                    prefix=".mola-preparation-attempt-",
                    dir=config.mola_radius_prepared_cube.parent,
                ) as attempt_name:
                    attempt = Path(attempt_name)
                    temporary_raw = attempt / config.mola_radius_raw_cube.name
                    temporary_prepared = attempt / config.mola_radius_prepared_cube.name
                    temporary_commands = (
                        (
                            "pds2isis",
                            f"from={config.mola_radius_label}",
                            f"to={temporary_raw}",
                        ),
                        (
                            "demprep",
                            f"from={temporary_raw}",
                            f"to={temporary_prepared}",
                        ),
                    )
                    for command in temporary_commands:
                        execute(command)
                    if (
                        not temporary_raw.is_file()
                        or temporary_raw.stat().st_size <= 0
                        or not temporary_prepared.is_file()
                        or temporary_prepared.stat().st_size <= 0
                    ):
                        raise RuntimeError("ISIS MOLA preparation produced a missing/empty cube")
                    config.mola_radius_raw_cube.parent.mkdir(parents=True, exist_ok=True)
                    temporary_raw.replace(config.mola_radius_raw_cube)
                    temporary_prepared.replace(config.mola_radius_prepared_cube)
                    mola_payload: dict[str, Any] = {
                        "schema": MOLA_PREPARATION_SCHEMA,
                        "isis_version": reported_isis_version,
                        "artifacts": {
                            name: {
                                "relative_path": path.relative_to(config.paper2_root).as_posix(),
                                "sha256": file_sha256(path),
                                "byte_count": path.stat().st_size,
                            }
                            for name, path in {
                                "source_label": config.mola_radius_label,
                                "raw_cube": config.mola_radius_raw_cube,
                                "prepared_cube": config.mola_radius_prepared_cube,
                            }.items()
                        },
                    }
                    canonical = json.dumps(
                        mola_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    mola_payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
                    manifest_path = mola_preparation_manifest_path(config)
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".{manifest_path.name}.",
                        suffix=".part",
                        dir=manifest_path.parent,
                    )
                    try:
                        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                            json.dump(
                                mola_payload,
                                stream,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            stream.write("\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        Path(temporary_name).replace(manifest_path)
                    finally:
                        Path(temporary_name).unlink(missing_ok=True)
            verify_mola_preparation(config)
    execute(commands[0])
    if spatial_summing is None:
        spatial_summing, getkey_record = read_spatial_summing(
            output_directory / "ctx_raw.cub", config=config
        )
        logs.append(getkey_record)
        commands = ctx_isis_commands(
            edr=edr,
            output_directory=output_directory,
            config=config,
            spatial_summing=spatial_summing,
        )
    for command in commands[1:]:
        execute(command)
    artifact_names = (
        "ctx_raw.cub",
        "ctx_cal.cub",
        "ctx_evenodd.cub",
        "ctx_isd.json",
        "ctx_cam2map.cub",
    )
    artifacts = {
        name: {
            "relative_path": (output_directory / name).relative_to(config.paper2_root).as_posix(),
            "sha256": file_sha256(output_directory / name),
        }
        for name in artifact_names
        if (output_directory / name).is_file()
    }
    return {
        "schema_version": 1,
        "isis_version": config.isis_version,
        "reported_isis_version": reported_isis_version,
        "isis_build": config.isis_build,
        "ale_version": config.ale_version,
        "reported_ale_version": reported_ale_version,
        "ale_build": config.ale_build,
        "csm_version": config.csm_version,
        "csm_build": config.csm_build,
        "usgscsm_version": config.usgscsm_version,
        "usgscsm_build": config.usgscsm_build,
        "csm_environment": csm_environment,
        "edr_sha256": file_sha256(edr) if edr.exists() else None,
        "mola_radius_label_sha256": (
            file_sha256(config.mola_radius_label) if config.mola_radius_label.exists() else None
        ),
        "mola_radius_prepared_sha256": (
            file_sha256(config.mola_radius_prepared_cube)
            if config.mola_radius_prepared_cube.exists()
            else None
        ),
        "isis_preferences_sha256": file_sha256(config.isis_preferences),
        "map_template_sha256": file_sha256(config.map_template),
        "mola_preparation_manifest_sha256": (
            file_sha256(mola_preparation_manifest_path(config))
            if mola_preparation_manifest_path(config).is_file()
            else None
        ),
        "isis_base_kernel_lock_sha256": (
            file_sha256(config.paper2_root / ISIS_BASE_KERNEL_LOCK)
            if not dry_run
            else None
        ),
        "spice_backend": config.spice_backend,
        "spiceql_rest_url": config.spiceql_rest_url,
        "spiceql_transport": (
            "loopback_https_connect_proxy"
            if config.spiceql_https_proxy is not None
            else "direct_https"
        ),
        "ale_spiceql_compatibility": ale_spiceql_compatibility,
        "spatial_summing": spatial_summing,
        "ctxevenodd_applied": spatial_summing != 2,
        "csm_attached_cube": (
            "ctx_evenodd.cub" if spatial_summing != 2 else "ctx_cal.cub"
        ),
        "commands": logs,
        "transient_command_failures": transient_command_failures,
        "artifacts": artifacts,
        "reference_dem_accessed": False,
        "projection_terrain": "MOLA_MEGR_32PPD_PLANETARY_RADIUS_DEMPREP",
    }


@dataclass(frozen=True)
class ImageQA:
    coverage_fraction: float
    finite_dem_fraction: float
    coreg_median_target_pixels: float
    coreg_p95_target_pixels: float

    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.coverage_fraction < 0.95:
            failures.append("coverage_fraction<0.95")
        if self.finite_dem_fraction < 0.99:
            failures.append("finite_dem_fraction<0.99")
        if self.coreg_median_target_pixels > 1.0:
            failures.append("coreg_median_target_pixels>1")
        if self.coreg_p95_target_pixels > 2.0:
            failures.append("coreg_p95_target_pixels>2")
        return tuple(failures)

    @property
    def passed(self) -> bool:
        return not self.failures()


def image_qa(
    image_mask: np.ndarray,
    dem_mask: np.ndarray,
    coregistration_errors_target_pixels: Sequence[float],
) -> ImageQA:
    image_mask = np.asarray(image_mask, dtype=bool)
    dem_mask = np.asarray(dem_mask, dtype=bool)
    if image_mask.shape != dem_mask.shape or image_mask.size == 0:
        raise ValueError("image and DEM masks must be non-empty and share shape")
    errors = np.asarray(coregistration_errors_target_pixels, dtype=np.float64)
    if errors.ndim != 1 or not len(errors) or np.any(errors < 0) or not np.all(np.isfinite(errors)):
        raise ValueError("coregistration errors must be a finite non-negative vector")
    return ImageQA(
        coverage_fraction=float(np.mean(image_mask)),
        finite_dem_fraction=float(np.mean(dem_mask)),
        coreg_median_target_pixels=float(np.median(errors)),
        coreg_p95_target_pixels=float(np.percentile(errors, 95)),
    )


def write_prepared_manifest(
    output: Path,
    *,
    paper2_root: Path,
    tile_id: str,
    role: str,
    primary_observation: str,
    primary_reason: str,
    problem_sha256: str,
    image: Path,
    image_mask: Path,
    coarse_dem: Path,
    qa: ImageQA,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    output = assert_inside(paper2_root, output)
    assets = {"image": image, "image_mask": image_mask, "coarse_dem": coarse_dem}
    for path in assets.values():
        assert_inside(paper2_root, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "tile_id": tile_id,
        "role": role,
        "primary_observation": primary_observation,
        "primary_observation_reason": primary_reason,
        "problem_sha256": problem_sha256,
        "assets": {
            name: {"path": str(path.relative_to(paper2_root)), "sha256": file_sha256(path)}
            for name, path in assets.items()
        },
        "qa": {**asdict(qa), "passed": qa.passed, "failures": list(qa.failures())},
        "lineage": dict(lineage),
        "reference_dem_accessed": False,
        "coreg_aux_is_primary": False,
    }
    if not qa.passed:
        raise ValueError(f"prepared asset failed frozen QA: {', '.join(qa.failures())}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    serialized = json.dumps(payload, indent=2).encode("utf-8")
    if output.exists():
        if output.is_file() and output.read_bytes() == serialized:
            return payload
        raise FileExistsError(f"refusing to replace immutable prepared manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload
