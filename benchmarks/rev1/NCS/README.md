# OmniAligner NCS revision benchmark

Open any notebook in `notebooks/` with a fresh Python 3.10/3.11 Jupyter kernel. Its first code cell downloads the public release archive, clones the `rev1` branch, checks SHA-256, and extracts the inputs; the second installs the pinned dependencies. No private server path or existing H5AD is required.

1. **NCS_submission_table_reproduce.ipynb** reconstructs the full submitted comparison, including all method rows and the rank/count figure. It verifies values, missingness, eligibility and all 36 ranks against the frozen submission reference.
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

The full-table notebook reconstructs every method row; it does not refit all comparators. Historical baseline means come from archived scored records, and added comparisons from their complete pair-score files. The coordinate replay independently checks OmniAligner. MLN PASTE-family uses 5,000 cells per section; OmniAligner retains its recorded full-cell support. Fitting information and tuning budgets differ. These are descriptive development results, not evidence of independent-patient generalization.

The native scoring implementation is from [SABench](https://github.com/Yunzhi-Yan/SABench). See `THIRD_PARTY_NOTICES.md` and the provenance manifests. Benchmark-frame coordinates are not calibrated micrometre measurements.
