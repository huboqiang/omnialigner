# Original Fig. 4A: BRCA benchmark

Run [the notebook](notebooks/Fig4_BRCA_original_benchmark.ipynb) with a fresh Python 3.10/3.11 kernel. Its first cell clones the public repository, downloads the data, and installs the pinned numerical/plotting dependencies. The computation requires no private server paths, credentials, GPU, H&E images, or AnnData installation.

The approximately 650 MiB download contains measured expression, frozen OmniPred and SpatialEx predictions, gene names, cell IDs, and the original prediction-row mapping. It supports the original S2→S1 and S1→S2 comparisons, with 164,000 and 111,555 evaluation cells, respectively, and 313 genes. It is distinct from the later revision benchmark.

Data asset: `rev1-fig4-brca-data.tar.gz` (681,559,920 bytes).

SHA-256: `76e2c5a31e9c85a2e2ed6365ada32365396efbfe2e3770e05e165c9de2e0d89a`.

## What is calculated

The notebook exposes the Pearson calculation directly: centering across cells, computing norms, and summing normalized products. It preserves the original float32 arithmetic and compares all 1,252 correlations with the archived source calculation. One plotted value represents one gene's correlation across cells. It does not represent the correlation of a single cell's expression profile.

Measured expression was reconstructed from the original Xenium count matrices with the original filtering, total-count normalization and `log1p` preprocessing. It exactly matched the saved ground-truth matrices in both evaluation cohorts. For S2, the original 118,370 prediction rows are mapped by restored cell identity to the 111,555-cell evaluation subset; the full row mapping is included.

## Provenance and outputs

The archive includes source-cell excerpts, source/input hashes, the complete row mapping, and the frozen gene-score reference. The source notebook is `Tutorial 1: OmniPred.ipynb`, zero-based cells 2, 28, 31 and 33. Its original output is `BRCA_corr_compare.pdf`.

The notebook writes the full gene-score table, summary statistics, validation JSON, and an editable Matplotlib PDF/PNG for Fig. 4A. The refreshed labels clarify the correlation axis; this does not reconstruct the final manuscript's multi-panel layout or rerun predictor training. Both sections come from the same tissue block.
