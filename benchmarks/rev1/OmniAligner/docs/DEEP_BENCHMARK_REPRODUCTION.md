# Deep spatial-omics benchmark

Run [Fig2_spatial_omics_benchmark_reproduce.ipynb](../notebooks/Fig2_spatial_omics_benchmark_reproduce.ipynb).
It is the current five-dataset Deep entry point, replacing the former historical
Figure 2 notebook. Historical reproduction notebooks remain explicitly archived.

The main comparison contains seven completed methods: OmniAligner, PASTE,
PASTE2, Spateo, SPACEL, STalign and SLAT. Loki is excluded. Unfinished SANTO and
MOSAICField runs remain diagnostic records rather than being declared failures;
they are not used in the seven-method main ranking.

| Dataset | Sections | Adjacent pairs | Assay class |
|---|---:|---:|---|
| ST brain | 75 | 74 | sequencing-based ST |
| E9.5 | 70 | 69 | sequencing-based ST |
| ABCA1 | 129 | 128 | imaging-based ST |
| ABCA2 | 54 | 53 | imaging-based ST |
| ABCA3 | 22 | 21 | imaging-based ST |

All five datasets have more than 20 sections. The sequencing group has two
datasets/145 sections and the imaging group three datasets/205 sections.
Imaging-based here describes spatial molecular assays, not a histology-image
registration task.

All 35 method–dataset conditions are source-bound and categorical scoring is
complete. The native six-metric table has 14,490 actual pair–metric slots;
original undefined values and support flags are retained. ARI/NMI/CLC add 7,245
pair–metric slots. One current ST-brain pair is freshly scored for all nine
metrics by default, with exact cell IDs and coordinates shared by its native
and categorical inputs. The complete pair records rebuild all tables/figures.
The optional full mode uses 1.48 GB of expression-free NPZs, selected per dataset,
to recalculate every categorical pair with the displayed functions.

ABCA1/3 retain their documented formal selected outputs; ABCA2 retains its
recorded selected development condition; ST brain/E9.5 retain current selected
400-step results. Historical results are not relabelled as new checkpoint fits.
No registration fitting or fresh agent call is performed by this scoring book.

The notebook exports per-pair sources/coverage, native means, quality/ranks,
and grouped/dataset-specific PDF/SVG/PNG figures through `om.pl.plt` with Arial
and Type 42 PDF/PS fonts. No statistical significance or biological validation
claim is inferred from these ranks.
