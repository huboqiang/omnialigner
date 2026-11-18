# Original Figure 2 spatial-omics benchmark

Open the [original Fig. 2 notebook](https://github.com/huboqiang/omnialigner/blob/e272a30c311de06727253db5305e60b425440b95/benchmarks/rev1/OmniAligner/notebooks/Fig2_spatial_omics_benchmark_reproduce.ipynb) in a fresh Python
3.10/3.11 notebook kernel. The current branch uses that filename for the [Deep benchmark](../notebooks/Fig2_spatial_omics_benchmark_reproduce.ipynb). Its opening cells download the public assets, clone
the `rev1` repository and install pinned CPU dependencies. No private path,
full expression H5AD, pretrained model or registration runtime is required.

## What executes

The default notebook recomputes ARI, NMI and CLC for all five methods on every
adjacent SmallIntestine section pair. CLC is calculated both on the archived
evaluation neighbour indices and on neighbours freshly selected from coordinates.
The indices contain no cached metric scores. They preserve historical choices
among equal-distance neighbours, which can otherwise differ across platforms.
It also transparently aggregates the
archived per-pair scores for the complete nine-dataset, five-method comparison,
checking all 135 means against the original source table. It identifies these
two execution scopes separately. `RUN_FULL = True` downloads the per-dataset
lossless scoring-input archives and recalculates every method's scores.

The independent Linux validation has completed all 45 dataset/method conditions:
1,955 adjacent pairs, 5,865 metric scores and all 135 method means. Every mean
matches the original source table within 2.23e-16. See
`validation/fig2/full_recompute_status.json` and `completed_method_means.csv` for
the exact environment and checks. This completed validation does not change the
default notebook's explicitly smaller fresh-execution scope.

This is the original manuscript comparison, not the later NCS six-metric table.
The original methods are STalign, Loki, SPACEL, Spateo and OmniAligner. No method
is removed to match the different scope of the revision benchmark.

## Inputs and metrics

ARI and NMI compare adjacent categorical image grids, including background.
CLC uses original labels, cell order, layer identities and the first two aligned
coordinate columns. The original MNN-named function falls back to directional
1-NN for nonmutual matches; this produces a directional 1-NN mapping. Its KNN
neighbourhood is 2.5% of the mean section cell count, including the central cell.
The bounded-memory implementation preserves this definition and is shown in full
inside the notebook. It does not introduce a different k or sample cells.

Per-dataset archives retain exact scoring values while omitting expression
matrices. Integer rasters are losslessly stored as int16 after range checks;
coordinate precision is retained. NPZ contents and archives have SHA-256 hashes.
Source H5AD paths and sizes are recorded; the 254 GB full H5AD collection was not
rehash-scanned during this export. The hashes authenticate the exported scoring
inputs, not an unperformed full-source-file audit.

The small example uses approximately 2 GB RAM. For the full run, allow 16 GB RAM
and sufficient time for large point-neighbourhood queries. Archives are split
by dataset and are each below GitHub Release's 2 GB asset limit. The notebook
processes datasets and methods sequentially and CLC queries in blocks. The full
run reports every difference from the historical table. It does not relax a
numerical assertion to conceal coordinate-neighbour tie differences.
Historical coordinate-based results were generated with SciPy 1.15.3 on Linux
x86_64. The portable example fixes the neighbour support explicitly; on macOS,
fresh equal-distance neighbour choices are shown separately from this exact
historical-support reconstruction.

## Historical display processing

The historical plot is reproduced separately from unmodified metric scores.
Its source changes the openST reference ARI to 0.8 and all three Spateo/E9.5
scores to 1.0, then computes `(score - raw)/(reference - raw)` and clips values
to [0, 1]. The notebook displays the original values and overrides explicitly.
These changes are not passed off as calculated metric values. The original
plotting notebook contains a tenth dataset; the manuscript panel uses nine.

Source paths on the original server:

- `isoST/script/mouse_brain/benchmark.py`: original metric implementation.
- `isoST/script/mouse_brain/pyvista3d.ipynb`: zero-based cells 32–37 aggregate,
  42 defines the historical display and 43 exports the plot.
- `isoST/script/mouse_brain/Table2_benchmark_v2.csv`: original score table.

Original source file hashes accompany the score asset. Numerical validation
reports are kept under `validation/fig2*`; pending full execution is not reported
as completed. The notebook reproduces scoring and a readable equivalent plot,
not the full multi-panel figure assembly or algorithm training.
