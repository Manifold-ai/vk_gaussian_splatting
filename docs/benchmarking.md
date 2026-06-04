# Profiling and Benchmarking

To properly assess the performance of the pipelines, you should **deactivate vertical synchronization (V-Sync)**. This can be done from the **View > V-Sync** menu, the **Renderer Properties** panel, or the **Profiler** window. If V-Sync is not deactivated, the system does not run at optimal performance, and the reported timings in the **Window Top Bar** and **Profiler** window are generally higher (and fps lower) than the achievable performance. The V-Sync option is enabled by default to save energy.

The system also provides a means to run automatic benchmarks, which are detailed in the **Performance Results** sections of the documentation sub-pages. In Benchmarking mode, V-Sync is automatically disabled.

## Automated benchmarks (`benchmark.py`)

The repository includes `benchmark.py` to drive the application through scripted parameter sequences (`.cfg` files), collect timing logs, and—for some datasets—generate PSNR/FPS charts and comparison images against a raster reference.

### Python dependencies

Install the required packages:

``` sh
python -m pip install matplotlib numpy pillow
```

### Running a benchmark

Build the application first (see [Getting Started](getting-started.md#building-from-source)), then run from the repository root:

``` sh
python benchmark.py <benchmark.cfg> <dataset> <dataset_path> <output.csv> [options]
```

Example (billboard bounding-mode study):

``` sh
python benchmark.py benchmark_billboards.cfg 3DGS_BILLBOARDS <path_to_vkgs_dataset> benchmark_billboards.csv --headless
```

Other dataset-specific configuration files include `benchmark_3dgs.cfg`, `benchmark_3dgrt.cfg`, and `benchmark_3dgut.cfg`. Their usage and published charts are described in the **Performance Results** sections of the corresponding [deep dives](index.md).

### Useful options

| Flag | Description |
|------|-------------|
| `--headless` | Run without a window (recommended for batch benchmarks) |
| `--verbose` / `-v` | Stream application output to the console |
| `--forcegpu ID` | Forward `--forcegpu` to the app (Vulkan physical device index) |
| `--skip-exec` | Skip launching the app; re-analyze existing logs and screenshots in `_benchmark/` |

Outputs are written under `_benchmark/` (logs, screenshots, CSV, and PNG charts).

For `benchmark_billboards.cfg` / `3DGS_BILLBOARDS`, charts are grouped by sort strategy and geometry:

| Charts | Sort | Geometry |
|--------|------|----------|
| `14–17_icosa_*` | Stochastic any-hit | Icosahedron billboards |
| `14–17_aabb_*` | Stochastic any-hit | AABB billboards |
| `18–21_aabb_allpass_*` | All-pass | AABB billboards (stochastic AABB modes except NoShorten; icosa does not support all-pass) |

All-pass timing sequences use `--sequenceframes 64` / `--sequenceaverages 32` / `--sequenceresetframes 0` so the profiler writes `Timeline` lines into the log. Shorter sequences (for example 5 frames) produce empty `ParameterSequence` blocks and the all-pass charts are skipped.

`13_summary_table.csv` lists timing and PSNR for all modes.
