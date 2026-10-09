#!/usr/bin/env python3
"""Run the frozen v35 mechanism with exact terminal footprint interpolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import evaluate_local_laser_projection_v35 as v35


def main() -> int:
    v35.LOCAL_NOISE_M = 0.0
    output_index = sys.argv.index("--output") + 1
    output = Path(sys.argv[output_index]).resolve()
    return_code = v35.main()
    payload = json.loads(output.read_text())
    payload["schema"] = "mars-sparse-detail-exact-laser-projection/v1"
    payload["mechanism"]["local_laser_noise_m"] = 0.0
    payload["mechanism"]["terminal_constraint"] = "exact footprint interpolation"
    payload["bindings"]["v35_implementation_sha256"] = hashlib.sha256(
        Path(v35.__file__).read_bytes()
    ).hexdigest()
    payload["bindings"]["v36_wrapper_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
