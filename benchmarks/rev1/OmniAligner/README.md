# OmniAligner revision benchmarks

Open any notebook in `notebooks/` with a fresh Python 3.10/3.11 Jupyter kernel. Its first code cell downloads the public release archive, clones the `rev1` branch, checks SHA-256, and extracts the inputs; the second installs the pinned dependencies. No private server path or existing H5AD is required.

Start with [the DLPFC alignment and agent-refinement walkthrough](notebooks/OmniAligner_alignment_workflow_DLPFC.ipynb) to see the complete recorded stage sequence, actual agent decisions and both optimization trajectories. It verifies the source records and freshly recomputes coarse/final coordinates and scores. Archived detector/agent/training steps are explicitly distinguished from executed inference.

1. **NCS_submission_table_reproduce.ipynb** shows the six metric definitions directly, freshly computes all 44 OmniAligner adjacent pairs / 36 means, then reconstructs the full submitted comparison, including all method rows and the rank/count figure. It verifies values, missingness, eligibility and all 36 ranks against the frozen submission reference.
2. **OmniAligner_NCS_reproduce.ipynb** recomputes all 44 adjacent pairs and 36 selected OmniAligner means from original expression values/annotations and the six selected coordinate NPZs. Allow 48–64 GB RAM for MLN.
3. **OmniAligner_weights_to_scores_DLPFC.ipynb** loads the portable selected geometry checkpoint, transforms original raw coordinates through the frozen coarse frame and learned nonlinear field, and computes the six scores on all three DLPFC pairs. Saved final coordinates are used only for validation. This replays inference, not the coarse matcher, agent or optimizer.

The release expression inputs retain exact source `.X` values for the frozen evaluation-gene panel, original cell IDs, labels and raw coordinates. They are exported losslessly as sparse NPZs, with per-file and source-H5AD hashes. Full source H5ADs are [linked separately](docs/DATA_DOWNLOADS.md). They are not needed to reproduce the submitted table.

## Included records

- `data/selected_coordinates`: six coordinate NPZs (13.23 MB), with cell IDs and fixed cohort masks.
- `data/weights`: nine original checkpoints and a tensor-only portable DLPFC checkpoint. The latter preserves the original learned parameters and frozen pre-nonlinear coordinate frame; it contains no expected final coordinate array.
- `src`: frozen native SABench scorer, pair scoring and portable geometry inference.
- `data/pair_scores` and `data/sources`: complete pair scores, archived baseline means, submitted references and eligibility/status records.
- `validation`: executed numerical checks; `tables`: the matching submitted XLSX/PDF and rank figure.

## Scope

There is one selected OmniAligner result per dataset; no historical OmniAligner alternative enters the ranks. Loki and CPD are excluded. PASTE2 MLN is complete (18/18); CAST MLN remains incomplete and unranked. Failed, undefined and diagnostic entries are retained. Green means rank 1; gray means another rank or unavailable comparison.

The full-table notebook reconstructs every method row; it does not refit all comparators. Historical baseline means come from archived scored records, and added comparisons from their complete pair-score files. The full-table notebook itself recalculates OmniAligner from expression inputs and coordinates. MLN PASTE-family uses 5,000 cells per section; OmniAligner retains its recorded full-cell support. Fitting information and tuning budgets differ. These are descriptive development results, not evidence of independent-patient generalization.

The native scoring implementation is from [SABench](https://github.com/Yunzhi-Yan/SABench). See `THIRD_PARTY_NOTICES.md` and the provenance manifests. Benchmark-frame coordinates are not calibrated micrometre measurements.

## Original main figures

- [Fig. 2 spatial-omics benchmark](notebooks/Fig2_spatial_omics_benchmark_reproduce.ipynb): explicit ARI/NMI/CLC with original raster/coordinate/label inputs, transparent historical scaling and overrides. [Scope](docs/FIG2_REPRODUCTION.md).
- [Fig. 3 FAIR benchmark](notebooks/Fig3_FAIR_landmark_benchmark.ipynb): direct landmark calculations and separately reproduced historical display. [Scope](docs/Fig3_FAIR_reproduction.md).

These original-figure experiments use different datasets and metrics from the NCS revision comparison above. They do not change its scores, ranks or eligibility rules.
