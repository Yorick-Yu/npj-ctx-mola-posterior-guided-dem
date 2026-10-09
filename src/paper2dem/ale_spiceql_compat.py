"""Audited ALE 1.2.0 compatibility shim for CTX web-SPICE generation.

The frozen C++ ``pyspiceql`` web transport intermittently accepts an empty or
truncated response body from the official SpiceQL service.  The same exact URL
is reliable through ``requests`` and the Paper2 CONNECT proxy.  This module
therefore replaces only ``useWeb=True`` calls with a validated Python transport
and caps ALE's three exact, source-hashed executors at one worker for the
lifetime of ``isd_generate``.  Offline calls, endpoint identity, arguments,
ALE/ISIS package bytes, and the resulting ISD contract remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import sys
import time

import numpy as np
import requests


EXPECTED_ALE_VERSION = "1.2.0"
OFFICIAL_SPICEQL_URL = "https://astrogeology.usgs.gov/apis/spiceql/latest/"
WEB_QUERY_FUNCTIONS = (
    "extractExactCkTimes",
    "findMissionKeywords",
    "findTargetKeywords",
    "frameTrace",
    "getExactTargetOrientations",
    "getTargetFrameInfo",
    "getTargetOrientationsRanged",
    "getTargetStates",
    "getTargetStatesRanged",
    "searchForKernelsets",
    "strSclkToEt",
    "translateNameToCode",
)
WEB_REQUEST_ATTEMPTS = 5
WEB_REQUEST_TIMEOUT_SECONDS = (15.0, 120.0)
EXPECTED_ALE_SOURCE_SHA256 = {
    "ale.base.base": "571bdf4f94c3a322bcbeb4f8793f24f61b8585b7e8c1fadecef233e0ad2fb95c",
    "ale.base.data_naif": (
        "fcbcd18a09246a54af4613a1c6d48d31e276241f37e68834c300ad5d29aa76d3"
    ),
    "ale.transformation": (
        "06ac4ba4224bfcda46cd8378ba4a65105518a4e647a3ec59579e9ec35ca9ae8d"
    ),
}


class _RetriableSpiceQLError(RuntimeError):
    """A transport or response-integrity failure that may succeed on retry."""


class _PermanentSpiceQLError(RuntimeError):
    """A deterministic API/configuration failure that must not be retried."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialized_executor_factory(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Return the narrowly scoped one-worker replacement used by the launcher."""

    def build(max_workers: int | None = None, *args: Any, **kwargs: Any) -> Any:
        if max_workers is None or isinstance(max_workers, bool) or max_workers < 1:
            raise RuntimeError(
                "frozen ALE web executor requested an invalid worker count"
            )
        return original(1, *args, **kwargs)

    return build


def _normalise_official_url(value: str | None) -> str:
    observed = (value or OFFICIAL_SPICEQL_URL).rstrip("/") + "/"
    if observed != OFFICIAL_SPICEQL_URL:
        raise RuntimeError(
            "SPICEQL_REST_URL differs from the frozen official endpoint: "
            f"observed {observed!r}"
        )
    return observed


def _stringify_web_args(function_args: Mapping[str, Any]) -> dict[str, Any]:
    """Match ALE 1.2.0's web argument representation without mutating callers."""
    cleaned: dict[str, Any] = {}
    for key, value in function_args.items():
        if key == "useWeb":
            continue
        if isinstance(value, np.ndarray):
            cleaned[key] = str(value.tolist())
        elif isinstance(value, list):
            cleaned[key] = str(value)
        elif isinstance(value, bool):
            cleaned[key] = str(value)
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def _bind_web_args(
    function_name: str,
    positional: tuple[Any, ...],
    keyword: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the one positional form used by frozen ALE; reject ambiguous forms."""
    bound = dict(keyword)
    if positional:
        if function_name != "searchForKernelsets" or len(positional) != 1:
            raise RuntimeError(
                f"unsupported positional web arguments for frozen {function_name} call"
            )
        if "spiceqlNames" in bound:
            raise RuntimeError("searchForKernelsets supplied spiceqlNames twice")
        bound["spiceqlNames"] = positional[0]
    bound.pop("useWeb", None)
    return bound


class SpiceQLWebClient:
    """Strict official SpiceQL client with bounded integrity retries."""

    def __init__(
        self,
        *,
        session: requests.Session | Any | None = None,
        base_url: str = OFFICIAL_SPICEQL_URL,
        attempts: int = WEB_REQUEST_ATTEMPTS,
        timeout: tuple[float, float] = WEB_REQUEST_TIMEOUT_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError("SpiceQL attempts must be positive")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/") + "/"
        self.attempts = attempts
        self.timeout = timeout
        self.sleeper = sleeper

    @staticmethod
    def _decode_response(response: Any) -> tuple[Any, Any]:
        status_code = int(response.status_code)
        if status_code != 200:
            message = f"SpiceQL HTTP status {status_code}"
            if status_code >= 500 or status_code in {408, 425, 429}:
                raise _RetriableSpiceQLError(message)
            raise _PermanentSpiceQLError(message)

        content = bytes(response.content)
        headers = response.headers
        content_length = headers.get("Content-Length")
        content_encoding = headers.get("Content-Encoding", "").strip().lower()
        transfer_encoding = headers.get("Transfer-Encoding", "").strip().lower()
        if content_length and content_encoding in {"", "identity"} and not transfer_encoding:
            try:
                expected_length = int(content_length)
            except ValueError as exc:
                raise _RetriableSpiceQLError("invalid SpiceQL Content-Length") from exc
            if expected_length != len(content):
                raise _RetriableSpiceQLError(
                    "truncated SpiceQL response: "
                    f"received {len(content)} bytes, expected {expected_length}"
                )
        if not content:
            raise _RetriableSpiceQLError("empty SpiceQL response body")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _RetriableSpiceQLError("invalid SpiceQL JSON response") from exc
        if not isinstance(payload, Mapping):
            raise _RetriableSpiceQLError("SpiceQL response is not a JSON object")

        api_status = payload.get("statusCode")
        if api_status != 200:
            message = f"SpiceQL API status {api_status!r}"
            if isinstance(api_status, int) and api_status >= 500:
                raise _RetriableSpiceQLError(message)
            raise _PermanentSpiceQLError(message)
        body = payload.get("body")
        if not isinstance(body, Mapping) or "return" not in body:
            raise _RetriableSpiceQLError("SpiceQL response body lacks return value")
        kernels = body.get("kernels", {})
        if not isinstance(kernels, (dict, list)):
            raise _RetriableSpiceQLError("SpiceQL kernels field has an invalid type")
        return body["return"], kernels

    def call(self, function_name: str, function_args: Mapping[str, Any]) -> tuple[Any, Any]:
        if function_name not in WEB_QUERY_FUNCTIONS:
            raise _PermanentSpiceQLError(
                f"SpiceQL function is not frozen for Paper2: {function_name}"
            )
        clean_args = _stringify_web_args(function_args)
        url = self.base_url + function_name
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
            "Content-Type": "application/json",
        }
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                if function_name == "getTargetStates" and "ets" in clean_args:
                    response = self.session.post(
                        url,
                        data=json.dumps(clean_args, separators=(",", ":")),
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                else:
                    response = self.session.get(
                        url,
                        params=clean_args,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                return self._decode_response(response)
            except _PermanentSpiceQLError:
                raise
            except (requests.RequestException, _RetriableSpiceQLError) as exc:
                last_error = exc
                if attempt == self.attempts:
                    break
                delay = float(2 ** (attempt - 1))
                print(
                    "PAPER2_SPICEQL_RETRY "
                    f"function={function_name} attempt={attempt}/{self.attempts} "
                    f"error={type(exc).__name__}:{exc}",
                    file=sys.stderr,
                    flush=True,
                )
                self.sleeper(delay)
        raise RuntimeError(
            f"official SpiceQL request failed after {self.attempts} attempts: {function_name}"
        ) from last_error

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if close is not None:
            close()


def _web_function_wrapper(
    function_name: str,
    original: Callable[..., Any],
    client: SpiceQLWebClient,
) -> Callable[..., Any]:
    """Route only explicit web calls to the audited requests implementation."""

    @wraps(original)
    def call(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("useWeb") is not True:
            return original(*args, **kwargs)
        return client.call(function_name, _bind_web_args(function_name, args, kwargs))

    return call


def install_serialized_web_queries() -> tuple[dict[str, Any], Callable[[], None]]:
    """Verify exact upstream files, install in-memory shims, and return restore."""
    import ale
    import pyspiceql
    from ale import transformation
    from ale.base import base
    from ale.base import data_naif

    if ale.__version__ != EXPECTED_ALE_VERSION:
        raise RuntimeError(
            f"ALE version differs: observed {ale.__version__}, expected {EXPECTED_ALE_VERSION}"
        )
    modules = {
        "ale.base.base": base,
        "ale.base.data_naif": data_naif,
        "ale.transformation": transformation,
    }
    observed_source_sha256: dict[str, str] = {}
    executor_originals: dict[str, Callable[..., Any]] = {}
    for name, module in modules.items():
        source = Path(module.__file__).resolve()
        observed = _file_sha256(source)
        observed_source_sha256[name] = observed
        if observed != EXPECTED_ALE_SOURCE_SHA256[name]:
            raise RuntimeError(
                f"{name} bytes differ from the frozen compatibility target: observed {observed}"
            )
        executor_originals[name] = module.ThreadPoolExecutor
    for name, module in modules.items():
        module.ThreadPoolExecutor = serialized_executor_factory(executor_originals[name])

    base_url = _normalise_official_url(os.environ.get("SPICEQL_REST_URL"))
    client = SpiceQLWebClient(base_url=base_url)
    spiceql_originals: dict[str, Callable[..., Any]] = {}
    for function_name in WEB_QUERY_FUNCTIONS:
        original = getattr(pyspiceql, function_name)
        spiceql_originals[function_name] = original
        setattr(
            pyspiceql,
            function_name,
            _web_function_wrapper(function_name, original, client),
        )

    def restore() -> None:
        for name, module in modules.items():
            module.ThreadPoolExecutor = executor_originals[name]
        for function_name, original in spiceql_originals.items():
            setattr(pyspiceql, function_name, original)
        client.close()

    return (
        {
            "schema": "paper2.ale-spiceql-transport/v3",
            "ale_version": EXPECTED_ALE_VERSION,
            "source_sha256": observed_source_sha256,
            "patched_executor_modules": sorted(modules),
            "effective_web_worker_cap": 1,
            "patched_web_functions": list(WEB_QUERY_FUNCTIONS),
            "spiceql_endpoint": base_url,
            "web_transport": "python-requests-validated",
            "web_request_attempts": WEB_REQUEST_ATTEMPTS,
            "web_request_timeout_seconds": list(WEB_REQUEST_TIMEOUT_SECONDS),
            "scope": "isd_generate_process_only",
        },
        restore,
    )


def main() -> int | None:
    _audit, restore = install_serialized_web_queries()
    try:
        from ale.isd_generate import main as ale_isd_generate_main

        return ale_isd_generate_main()
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(main())
