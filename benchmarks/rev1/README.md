# Revision 1 reproducibility

The notebooks start with public `wget` and `git clone` commands and run without private server directories. Use a Python 3.10/3.11 Jupyter kernel; the computation was validated on Python 3.10, CPU.

| Notebook | What it reproduces |
|---|---|
| [OmniAligner alignment + agent-refinement walkthrough](OmniAligner/notebooks/OmniAligner_alignment_workflow_DLPFC.ipynb) | DLPFC input, matching/stage evidence, recorded agent decisions and optimizer curves, then fresh coordinate inference and scoring |
| [OmniAligner NCS benchmark](OmniAligner/notebooks/NCS_submission_table_reproduce.ipynb) | Six NCS cohorts; nine families; nine metrics; explicit NPZ scoring, source checks and PDF figures |
| [Weights → coordinates → scores](OmniAligner/notebooks/OmniAligner_weights_to_scores_DLPFC.ipynb) | DLPFC checkpoint inference from original coordinates, followed by all six metrics |
| [Controlled rotations and crops](OmniAligner/notebooks/OmniAligner_rotation_crop_benchmark_reproduce.ipynb) | All 748 configuration–condition results freshly scored from NPZ; six visible metric functions, family ranks and radar/crop/variance figures |
| [Deep spatial-omics benchmark](OmniAligner/notebooks/Fig2_spatial_omics_benchmark_reproduce.ipynb) | Current Deep benchmark: five cohorts >20 sections, seven completed methods, nine metrics and PDF figures |
| [Original Fig. 3: FAIR landmarks](OmniAligner/notebooks/Fig3_FAIR_landmark_benchmark.ipynb) | Ten portable landmark arrays, TRE and both aTRE definitions; fresh calculations separated from historical display substitutions |
| [Original Fig. 4: BRCA prediction](OmniPred/notebooks/Fig4_BRCA_original_benchmark.ipynb) | True expression and identity-mapped two-method predictions; 1,252 freshly calculated gene-wise PCCs in both directions |
| [OmniPred benchmark tables](OmniPred/notebooks/OmniPred_submission_benchmarks.ipynb) | Cell RNA, regional RNA, native tiles, and CRC protein-region tables, each on its own support |

The [rev1 release](https://github.com/huboqiang/omnialigner/releases/tag/rev1) contains the 145 MB data archive. Six selected coordinate NPZs (13.23 MB) and nine original geometry checkpoints plus the portable DLPFC checkpoint are in Git. Full H5ADs are not in Git; [original download links](OmniAligner/docs/DATA_DOWNLOADS.md) are provided separately.

The current alignment entry points are the three [OmniAligner benchmark notebooks](OmniAligner/README.md): Robustness, NCS and Deep. Loki, CAST and CPD are excluded. NCS has nine families; Deep has seven completed methods, with unfinished SANTO/MOSAICField retained as diagnostics. NCS/Deep include native six metrics plus ARI/NMI/CLC. Raw missing values remain NA; the derived quality score applies documented zero credit to failed/ineligible pairs. Source corrections and different native supports are disclosed in the audit.

The unified NCS notebook recomputes all OmniAligner metrics from expression inputs and coordinates; the DLPFC examples also replay weight inference. Other NCS methods are reconstructed from archived means or complete pair-score records; OmniPred tables are reaggregated from frozen gene/seed/marker scores, not retrained. See the package READMEs for scope and provenance.

The DLPFC workflow archive (26 MB, in the same release) contains the actual recorded stage evidence and agent reviews. The walkthrough does not make a fresh LLM call or rerun training; it recomputes inference/scoring and retains the original final geometric-admission status.

The [rotation/crop reproduction](OmniAligner/docs/PERTURBATION_REPRODUCTION.md) uses a separate 130 MB NPZ Release asset and six numerical/plotting dependencies. It recomputes all reported configurations, including the four predefined isotropic crop retries, without SSH or private data paths. Figures use `om.pl.plt` with system Arial and Type 42 PDF/PS fonts. Registration fitting is not rerun. Earlier assets and checksums are unchanged.

## Original main-figure reproduction

The new figure-specific data are separate Release assets; the existing benchmark archive and its checksum are unchanged. Inputs, hashes and exact calculation scopes are described in [Fig. 2](OmniAligner/docs/FIG2_REPRODUCTION.md), [Fig. 3](OmniAligner/docs/Fig3_FAIR_reproduction.md), and [Fig. 4](OmniPred/Fig4_original_benchmark.md). Each notebook displays its metric functions and exports CSV/PDF results. Original-figure cohorts are distinct from the revision's fixed-cohort benchmarks.

[Fig. 5 source recovery remains incomplete](OmniPred/Fig5_original_source_status.md): the final four-method scoring code, prediction/checkpoint binding and pseudo-spot rules have not all been recovered. The protein-region revision table is not substituted for that historical experiment.
