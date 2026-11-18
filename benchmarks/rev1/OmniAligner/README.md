# OmniAligner: revision benchmark reproduction

Three notebooks are the current benchmark entry points:

| Benchmark | Runnable notebook | Main scope | Publication PDFs |
|---|---|---|---|
| Robustness | [Rotation/crop](notebooks/OmniAligner_rotation_crop_benchmark_reproduce.ipynb) | DLPFC/MHPR; 68 perturbations; 748 configurations; nine algorithm families | [Rotation/crop](figures/current/OmniAligner_rotation_crop.pdf) |
| NCS | [NCS nine metrics](notebooks/NCS_submission_table_reproduce.ipynb) | Six datasets; nine families; six native metrics + ARI/NMI/CLC | [Ranks](figures/current/NCS_dataset_nine_metric_ranks.pdf), [summary](figures/current/NCS_grouped_performance_summary.pdf) |
| Deep | [Deep nine metrics](notebooks/Fig2_spatial_omics_benchmark_reproduce.ipynb) | Five datasets >20 sections; seven completed methods; nine metrics | [Ranks](figures/current/Deep_dataset_nine_metric_ranks.pdf), [summary](figures/current/Deep_grouped_performance_summary.pdf) |

All three have passed execution from public downloads in a clean directory and
isolated Python 3.10 environment. Use Python 3.10/3.11. Each begins with `!wget`
and `!git clone`, then installs its dependencies. No SSH or private directory is
needed. The three notebooks include saved figure outputs for preview on open. Figures use
`import omnialigner as om; om.pl.plt`, Arial and Type 42 fonts.

Loki, CAST and CPD are excluded from all current comparisons. Deep SANTO and
MOSAICField native runs are unfinished and retained only as diagnostics; the
main Deep comparison is explicitly seven methods, not nine. There is one selected
OmniAligner representative per cohort, without configuration tags in the table.

[NCS details](docs/NCS_BENCHMARK_REPRODUCTION.md), [Deep details](docs/DEEP_BENCHMARK_REPRODUCTION.md),
[full categorical recalculation](docs/FULL_CATEGORICAL_RECALC.md), and the
[source audit](docs/SCORING_AUDIT_20261118.md) explain inputs, support differences,
field/mask corrections, missingness and the precise ranking policy. Complete
experiments preserve native-mean order. Raw NA is not replaced with zero MSE.

The fast default displays actual metric functions and recalculates real inputs;
large full ARI/NMI/CLC calculations are optional. Complete coordinate/label/raster
NPZs are checksummed per-dataset Release assets (NCS 332 MB; Deep 1.48 GB), not Git
payloads. [Original H5AD downloads](docs/DATA_DOWNLOADS.md) remain separate.

The current NCS and Deep notebooks replace the overlapping NCS notebook and
the historical six-metric rank snapshot. Earlier tracked versions remain in Git history.

Additional focused examples remain available: [DLPFC weight inference](notebooks/OmniAligner_weights_to_scores_DLPFC.ipynb),
[recorded agent-refinement walkthrough](notebooks/OmniAligner_alignment_workflow_DLPFC.ipynb),
and [original FAIR landmark reproduction](notebooks/Fig3_FAIR_landmark_benchmark.ipynb).
