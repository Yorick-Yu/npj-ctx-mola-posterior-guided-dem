# Posterior-guided transport for CTX-derived Martian DEMs

Research code accompanying **Posterior-Guided Optimal Transport for Terrain Reconstruction from CTX-Derived Coarse DEMs** (npj Space Exploration submission 02fc5d8c-51ec-4c7a-943c-2982ea21e673).

The `paper2dem` package implements terrain preparation, Gaussian-process assimilation, transport, point-observation correction, and evaluation metrics. `scripts/` contains the v30–v36 real-MOLA training and frozen external-evaluation scripts used for the main results. `experiments/ablation`, `experiments/hirise`, and `experiments/physical` contain the corresponding revision analyses. The three primary model checkpoints and twelve final ablation checkpoints are under `checkpoints/`; `MANIFEST.sha256.json` records their checksums.

## Install and checks

Python 3.10+ is required. In a new environment:

```bash
python -m pip install -e '.[test,geo,ml]'
python -m pytest -q tests
python scripts/run_sparse_detail_v32.py --help
python scripts/run_frozen_external_v36.py --help
```

CUDA is needed for the full training workload; geospatial preparation also requires independently installed ISIS and official CTX/MOLA data. The command-line entry points accept project and input paths rather than downloading or bundling the study data. Historical protocol files may contain their original machine paths; adapt those paths to your environment without changing frozen scientific parameters.

## Data and reproducibility boundary

Raw CTX, MOLA, and reference DTM products are available from NASA PDS and USGS Astrogeology. Processed training tiles, held-out evaluation arrays, and other large derived data are **not** included in this repository. They are available from the corresponding author on reasonable request. Consequently, the full paper tables cannot be regenerated from this repository alone.

The real-MOLA scripts use a point-observation approximation and a three-step deterministic inference path followed by low-frequency and point-consistency projection. This is distinct from the controlled study's stochastic bridge experiments. The code is an archived research implementation; the included unit tests check software behavior, not the truth of every reported scientific result.

The HiRISE comparison has an additional provenance limitation: the six DEM arrays provided for the latest analysis were reconstructed from a rendered color figure, not recovered as original model-output rasters. The scripts are included for transparency; they should not be used to claim exact independent regeneration of those table values or terrain derivatives. See `experiments/hirise/audit_digitized_dem.py`.

## Layout

- `src/paper2dem/`: reusable package.
- `scripts/`: primary training, inference, frozen-candidate, and evaluation code.
- `experiments/`: ablation and additional terrain analyses.
- `checkpoints/primary/`: three primary training endpoints.
- `checkpoints/ablation/`: twelve final revision-ablation endpoints.
- `tests/`: selected unit tests for the public code path.

No third-party dataset or public reuse license is included. Copyright in the code remains with its contributors.
