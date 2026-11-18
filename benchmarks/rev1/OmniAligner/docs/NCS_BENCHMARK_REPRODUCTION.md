# NCS benchmark: nine algorithms and nine metrics

Run [NCS_submission_table_reproduce.ipynb](../notebooks/NCS_submission_table_reproduce.ipynb)
in a clean Python 3.10/3.11 kernel. The first two lines download a checksummed
public archive and clone the public repository. Sparse checkout downloads the
benchmark package and plotting code without the unrelated vendor history.

The notebook visibly calculates all 44 adjacent pairs for the six selected
OmniAligner outputs, reproducing all 36 original six-metric means within declared
numeric tolerances. Cosine uses float32 dot products and permits eight float32
epsilons across CPUs; all other endpoints use absolute 1e-10. Raw references are
not overwritten. The ARI/NMI/CLC functions and their pair loops are also visible.
One real categorical pair is recalculated by default; complete verified pair
records reconstruct the comparison. An optional full-input mode recalculates
all three categorical metrics directly from public NPZs.

| Dataset | Sections | Adjacent pairs | Assay class |
|---|---:|---:|---|
| DLPFC | 4 | 3 | sequencing-based ST |
| MHPR | 5 | 4 | imaging-based ST |
| MMPC | 3 | 2 | imaging-based ST |
| MPC | 3 | 2 | imaging-based ST |
| E14–16 | 16 | 15 | sequencing-based ST |
| MLN | 19 | 18 | sequencing-based ST |

The nine families are OmniAligner, PASTE, PASTE2, Spateo, SPACEL, STalign, SANTO,
SLAT and MOSAICField. Loki, CAST and CPD are excluded. There is one selected
OmniAligner representative per cohort. Algorithm configurations are fixed before
pair aggregation; configurations are not selected independently for each pair.

All 54 conditions have resolved source/scoring status: 53 have complete native
outputs; MMPC–PASTE2 failed during GLM-PCA before transport and has no aligned
pair. Its native values remain NA. Other native undefined values and finite
fallbacks retain their original values and eligibility flags. See the source
pair CSVs for exact per-endpoint validity.

MLN PASTE and PASTE2 use their actual original 19 × 5,000-cell native evaluation
support (95,000 cells), whereas full-cell methods keep their recorded larger
supports. No full-cell projection is invented. The comparison therefore retains
different native fitting/evaluation supports and is not a significance test.

[The audit](SCORING_AUDIT_20261118.md) records two necessary source corrections:
actual PASTE MLN coordinates and the frozen PASTE2 E14 cell mask. The original
selected OmniAligner six-metric results are unchanged within numeric tolerances.

The notebook exports complete native means, observed-only means, pair coverage,
coverage-adjusted quality, fixed-nine ranks, grouped rank credits, and two
publication figures. PDFs use `om.pl.plt`, Arial, and Type 42 fonts. These readouts
measure distinct kinds of consistency, not independent anatomical validation.
