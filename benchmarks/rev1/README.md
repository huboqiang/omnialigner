# Revision 1 reproducibility

The notebooks start with public `wget` and `git clone` commands and run without private server directories. Use a Python 3.10/3.11 Jupyter kernel; the computation was validated on Python 3.10, CPU.

| Notebook | What it reproduces |
|---|---|
| [OmniAligner alignment + agent-refinement walkthrough](OmniAligner/notebooks/OmniAligner_alignment_workflow_DLPFC.ipynb) | DLPFC input, matching/stage evidence, recorded agent decisions and optimizer curves, then fresh coordinate inference and scoring |
| [OmniAligner benchmark table](OmniAligner/notebooks/NCS_submission_table_reproduce.ipynb) | Visible metric functions; freshly computed OmniAligner scores; all algorithm rows and rank/count matrix |
| [OmniAligner selected-coordinate scoring](OmniAligner/notebooks/OmniAligner_NCS_reproduce.ipynb) | 44 adjacent pairs, 36 OmniAligner dataset means from released expression inputs and selected coordinates |
| [Weights → coordinates → scores](OmniAligner/notebooks/OmniAligner_weights_to_scores_DLPFC.ipynb) | DLPFC checkpoint inference from original coordinates, followed by all six metrics |
| [OmniPred benchmark tables](OmniPred/notebooks/OmniPred_submission_benchmarks.ipynb) | Cell RNA, regional RNA, native tiles, and CRC protein-region tables, each on its own support |

The [rev1 release](https://github.com/huboqiang/omnialigner/releases/tag/rev1) contains the 145 MB data archive. Six selected coordinate NPZs (13.23 MB) and nine original geometry checkpoints plus the portable DLPFC checkpoint are in Git. Full H5ADs are not in Git; [original download links](OmniAligner/docs/DATA_DOWNLOADS.md) are provided separately.

The NCS snapshot includes completed PASTE2 MLN (18/18) and STalign E14–16 (15/15). CAST MLN remains incomplete and is excluded from ranking. The selected OmniAligner configuration is the sole OmniAligner representative. Loki and CPD are not included. Undefined metrics remain NA. Ranking is within eligible finite results on the disclosed supports; it is not a significance test.

The full comparison notebook and selected-coordinate notebook recompute all OmniAligner metrics from expression inputs and coordinates; the DLPFC examples also replay weight inference. Other NCS methods are reconstructed from archived means or complete pair-score records; OmniPred tables are reaggregated from frozen gene/seed/marker scores, not retrained. See the package READMEs for scope and provenance.

The DLPFC workflow archive (26 MB, in the same release) contains the actual recorded stage evidence and agent reviews. The walkthrough does not make a fresh LLM call or rerun training; it recomputes inference/scoring and retains the original final geometric-admission status.
