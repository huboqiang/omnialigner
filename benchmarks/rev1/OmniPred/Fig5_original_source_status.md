# Original Fig. 5 benchmark: source recovery status

The original CRC four-method plot has **not yet been recovered as a complete executable source chain**. No notebook in this package claims to reproduce that plot. An embedded original PDF would not constitute a computational reproduction.

The historical panel compares OmniPred, SpatialEx, HEX and the method labelled GigaTME in the manuscript, on WD-76845-096. These original cell/pseudo-spot results must not be replaced with the later revision's CRC protein-region comparison.

## Located artifacts

All paths below are provenance on the historical `zyyh` server, not runtime dependencies for a public notebook. Base directory:

`/cluster/home/bqhu_jh/projects/bqhu_jh/SpatialEx/tutorials/`

| Artifact | What has been verified |
|---|---|
| `inference_on_CRC3d.py` | GigaTIME and HEX inference functions; contains the HE-096 / CyCIF-097 pairing. |
| `se_on_CRC3d.py` | SpatialEx training/prediction function and prediction H5AD export. |
| `celltype_network/train_CRC3d_v5.py` | Historical OmniPred training candidate; the checkpoint used for the original figure is not yet bound to this entry point. |
| `celltype_network/CRC3D_predict_096_corr_compare.pdf` | Original cell-level comparison output exists. |
| `celltype_network/CRC3D_predict_096_corr_compare_spot.pdf` | Original pseudo-spot comparison output exists. |
| `celltype_network/CRC3D_predict_096_corr_compare_spot_bygenes.pdf` | Original marker comparison output exists. |

`eval_HE2ST.ipynb` contains a training-section-count analysis. Its plots do not establish provenance for the original four-method panel.

## Required before a reproduction notebook is released

1. Recover the notebook or script that saved the three `CRC3D_predict_096_corr_compare*` outputs (the PDFs date to 10 February 2026), including an archived version if the current file was overwritten.
2. Bind each of the four methods to its exact prediction file and model checkpoint; record source hashes.
3. Recover the common marker mapping, prediction/measurement cell mapping, cell-inclusion criteria, pseudo-spot boundaries, aggregation rule and correlation axis.
4. Export those arrays and identities, calculate the metrics explicitly, and reconcile with the original figure values before packaging a public download.

The shortest remaining provenance search is the original figure-producing notebook/version from around 10 February 2026. Current tutorial notebooks, nearby evaluation notebooks, and selected editor-history versions have not supplied that final plotting cell. The original PDFs and inference scripts alone cannot resolve its scoring support.
