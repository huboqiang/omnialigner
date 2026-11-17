# Original Fig. 5A–B: CRC benchmark

Run [the notebook](notebooks/Fig5_CRC_original_benchmark.ipynb) with Python 3.10/3.11. Its first cell uses public `git clone`, `wget` and `pip` commands. Subsequent cells expose preprocessing, marker mapping, the pseudo-spot grid, averaging, and Pearson calculation directly. It needs no private directories, GPU, H5AD installation, or training checkpoint.

The independent data download is `rev1-fig5-crc-data.tar.gz` (228,140,259 bytes), SHA-256 `3e8464fad986d2717a24aaa0e1ee058c917a5966caee15fbc13ddd7be80d3368`. It contains the original frozen expression/prediction matrices and cell identities, coordinates, ordered marker mappings, source excerpts, and source-derived reference scores/aggregates.

The source evaluates 563,315 cells for H&E WD-76845-096 paired with measured CyCIF WD-76845-097. The original pseudo-spot rule produces 1,684 spots containing 561,994 cells. The replay verifies all 148 method/marker/support PCC values and the original cached OmniPred/SpatialEx pseudo-spot means, 0.7905389511977857 and 0.6845747704032076.

## What is preserved and corrected

- Original frozen predictions, numerical transforms, cell cohort, grid settings and method-specific marker coverage are preserved. This is distinct from the later revision's protein-region benchmark.
- GigaTIME has 9 mapped markers, HEX 17, SpatialEx and OmniPred 24 each. Method distributions do not use a common marker panel.
- The original code labelled GigaTIME/HEX scores with an unordered `list(set(...))`. The notebook corrects this by preserving each scored column's ordered mapping. Boxplot distributions are unchanged; the corrected per-marker plot is explicitly named `Fig5B_corrected_marker_labels`.
- Each PCC measures one marker across cells or pseudo-spots, not expression-profile correlation within a cell. Coordinates are retained in archived units without asserting physical calibration.

Outputs include both numerical tables, coverage, validation JSON, the Fig. 5A distribution PDF/PNG, and the corrected marker-labelled PDF/PNG. [The recovered source record](Fig5_original_source_status.md) gives exact notebook cells, hashes and input paths. This notebook scores frozen predictions and does not claim to reproduce training or validate independent patients/pathology.

Cross-platform validation retains the original float32 calculations. PCC tolerance is 1e-7; predicted aggregate relative tolerance is 3e-7. Exact cell memberships, IDs and marker mappings are checked separately. The Linux source replay agreed within 3.3e-15 PCC, and the macOS ARM numerical check within 2.9e-8 PCC.
