# Revision 1 reproducibility

The notebooks start with public `wget` and `git clone` commands and run without private server directories. Use a Python 3.10/3.11 Jupyter kernel; the computation was validated on Python 3.10, CPU.

| Notebook | What it reproduces |
|---|---|
| [NCS submitted table](OmniAligner/notebooks/NCS_submission_table_reproduce.ipynb) | All included algorithm rows, six metrics, and rank/count matrix |
| [NCS selected-coordinate scoring](OmniAligner/notebooks/OmniAligner_NCS_reproduce.ipynb) | 44 adjacent pairs, 36 OmniAligner dataset means from released expression inputs and selected coordinates |
| [Weights → coordinates → scores](OmniAligner/notebooks/OmniAligner_weights_to_scores_DLPFC.ipynb) | DLPFC checkpoint inference from original coordinates, followed by all six metrics |
| [OmniPred benchmark tables](OmniPred/notebooks/OmniPred_submission_benchmarks.ipynb) | Cell RNA, regional RNA, native tiles, and CRC protein-region tables, each on its own support |

The [rev1 release](https://github.com/huboqiang/omnialigner/releases/tag/rev1) contains the 145 MB data archive. Six selected coordinate NPZs (13.23 MB) and nine original geometry checkpoints plus the portable DLPFC checkpoint are in Git. Full H5ADs are not in Git; [original download links](OmniAligner/docs/DATA_DOWNLOADS.md) are provided separately.

The NCS snapshot includes completed PASTE2 MLN (18/18) and STalign E14–16 (15/15). CAST MLN remains incomplete and is excluded from ranking. The selected OmniAligner configuration is the sole OmniAligner representative. Loki and CPD are not included. Undefined metrics remain NA. Ranking is within eligible finite results on the disclosed supports; it is not a significance test.

Only the selected OmniAligner scoring and the DLPFC weight example rerun coordinate-level computations. Other NCS methods are reconstructed from archived means or complete pair-score records; OmniPred tables are reaggregated from frozen gene/seed/marker scores, not retrained. See the package READMEs for scope and provenance.
