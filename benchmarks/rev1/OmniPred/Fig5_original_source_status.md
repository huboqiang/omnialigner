# Original Fig. 5 benchmark: recovered source

The original plotting notebook has been recovered:

`/Users/dr.claw/Documents/projects/pdac3d/omni_data/3d_ST/analysis/imputation_3d.ipynb`

SHA-256: `74d320906d9f6d9fd2b63baf37788f390ac97e9129867400980907a04cc5ee69`. Modification time: 22 February 2026; 47 cells. A second copy at `omni_data/3d_ST/omnialigner/notebooks/imputation_3d.ipynb` has identical scoring-cell source, although the full notebook hash differs.

## Exact plotting chain

Cell indices are zero-based.

| Cells | Computation/output |
|---|---|
| 1 | Defines the 24-channel panel and `Generate_pseudo_spot1`. |
| 17–18 | Loads measured CyCIF WD-76845-097 and OmniPred/SpatialEx predictions for H&E WD-76845-096; applies the historical transforms. |
| 19–20 | Calculates marker-wise correlations across cells; saves `CRC3D_predict_096_corr_compare.pdf`. |
| 21 | Saves `CRC3D_predict_096_corr_compare_bygenes.pdf`. |
| 22–24 | Aggregates pseudo-spots, calculates correlations across them; saves `CRC3D_predict_096_corr_compare_spot.pdf`. |
| 25 | Saves `CRC3D_predict_096_corr_compare_spot_bygenes.pdf`. |

The source evaluates 563,315 cells. Pseudo-spots use `spatial_3d` XY coordinates, X spacing 15, Y spacing `15*sqrt(3)`, the original second-grid X offset 50, nearest-center assignment within 27.5 coordinate units, at least 25 cells per retained spot, and the arithmetic mean of processed expression. Cached outputs report 1,684 retained pseudo-spots containing 561,994 cells. These coordinate units are not asserted to be calibrated micrometres.

## Inputs and model boundary

Measured inputs/coordinates are under the historical server directory `/cluster/home/bqhu_jh/projects/scGaussian3dGen/output/CRC/v1/dhy_20250226/20260105/`: `adata_alignHE_WD-76845-097.h5ad`, `annotated3d_WD-76845-097.h5ad`, `adata_gigaTME_WD-76845-097.h5ad`, and `adata_hex_WD-76845-097.h5ad`.

Exact frozen predictions under `/cluster/home/bqhu_jh/projects/bqhu_jh/SpatialEx/tutorials/`:

- OmniPred: `spatranslator_v5_celltypes_train5/modified_model/WD-76845-096_A2R_predict_conch.h5ad`.
- SpatialEx: `spatialEx_crc_train1/WD-76845-096_A2R_predict_uni.h5ad`.

The code applies `log1p` to measured values after float64 conversion, `log1p(10*X)` to the sparse OmniPred prediction, and `log1p(X)` to SpatialEx. GigaTIME/HEX outputs are evaluated as stored on their available mapped marker subsets. Frozen-prediction replay does not establish a stronger checkpoint-to-prediction provenance claim than these recovered files provide.

## Historical marker-label issue

The original plotting code pairs GigaTIME/HEX score vectors with names built using `list(set(...))`, losing matrix-column order. This does not affect method boxplot distributions, but the historical per-marker labels are not reliable.

The reproduction preserves the ordered `gene_CRC3d` mapping and discloses this correction. It does not certify the original unordered labels or mix these results with the later revision's protein-region comparison. Method-specific marker coverage remains explicit.

Numerical replay verifies all 148 marker/support PCC values and all retained pseudo-spot aggregates against the recovered source function. The historical cached OmniPred/SpatialEx means also agree within the declared cross-platform tolerances; exact cell identities and spot memberships are checked separately. See [the executable notebook](notebooks/Fig5_CRC_original_benchmark.ipynb) and its validation records for the download-and-run reproduction.
