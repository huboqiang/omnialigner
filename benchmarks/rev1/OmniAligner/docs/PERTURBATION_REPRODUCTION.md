# Rotation and crop result reproduction

Open [the executable notebook](../notebooks/OmniAligner_rotation_crop_benchmark_reproduce.ipynb) in a fresh Python 3.10/3.11 Jupyter kernel and run all cells. Its first commands download the public NPZ archive, clone `rev1`, and install the six pinned numerical/plotting dependencies. No server account, private directory, original H5AD download, GPU, or model fitting is needed.

The notebook **recomputes scores from saved alignment coordinates**. It displays the spatial-bin construction, six metric functions, explicit per-gene calculation loop, family ranking, and plotting code. It also reconstructs every input rotation/crop and checks the saved input coordinates and row identities. It does not rerun agent matching or optimization.

## Released inputs

The separate [rotation/crop NPZ archive](https://github.com/huboqiang/omnialigner/releases/download/rev1/rev1-OmniAligner-rotation-crop-20261118.tar.gz) is 129,997,636 bytes (about 130 MB). Existing Release assets and their checksums are unchanged.

SHA-256:

```text
4e01c2dd154820de82547cd67ade34cd1622c1ecb0a5c4bc1fd249fc6da9dd79
```

The archive contains original measured expression in sparse CSR NPZ form, original cell identities and region labels, original and perturbed coordinates, and all 748 aligned-coordinate NPZs. It has two fixed section pairs, 68 perturbation settings, and 11 configurations representing nine algorithm families. The manifest binds every file to its checksum and original score/fit receipts. H5ADs and private source directories are not included.

## Exact calculation scope

- DLPFC: sections 151673→151674; 123 ordered gene arguments, including repeated arguments. MHPR: sections 0.04→0.09; 155 gene arguments.
- Rotation: 0°–360° in 15° steps. Both 0° and 360° enter statistics; the radar displays 15°–360°.
- Cropping: nominal retained x-ranges of 20%–100%. The original strict `x < threshold` rule also applies to the fixed 100% reference. These are x-ranges, not measured overlap areas or cell fractions.
- OmniAligner: 68 fixed-400-step results. Four previously failed crop cases (DLPFC 20%; MHPR 20%, 30%, 40%) use the declared isotropic coarse-transform retry; the original 64 results are unchanged. Whole case rows are retained, without choosing a checkpoint separately for each metric.
- The five molecular metrics use the original density-filtered 10×10 overlap grid. The original expression dtype, pandas grid means, lexicographic grid order, per-gene MI seed 0 and equal gene weights are retained. A single undefined gene invalidates that endpoint; no `nanmean` or gene dropping is used.
- Region is moving-to-reference nearest-neighbor label agreement inside the closed overlap box. Completed native zero-overlap failures remain explicitly flagged zeros. Unsupported molecular fallbacks remain ineligible.
- Ranks compare nine families. PASTE/PASTE-p0 and Spateo rigid/nonrigid contribute one best eligible family value per condition and metric. Missing endpoints do not enter the denominator. MSE is minimized; the other metrics are maximized.

Geometry-review markers are retained from recorded audits and do not filter numerical scores. The annotations used in the fitting/evaluation conditions are not independent anatomical ground truth. These perturbations on one pair per dataset are technical settings, not biological replicates; no significance test or confidence interval is claimed.

## Outputs and checks

The notebook exports all configuration scores, all per-gene scores, OmniAligner rank/count tables, and PDF/SVG/PNG figures to `perturbation_reproduced/`. Expected records are opened only after the fresh calculations. The checks compare all six metrics and eligibility/NA positions, all per-gene values, and the independently archived 408 OmniAligner rank/count positions.

Plots use `import omnialigner as om` and `om.pl.plt`, selecting system Arial when available and embedding Type 42 PDF/PS fonts. `OMNIALIGNER_FONT_DIR` can point to a locally installed font directory; an unavailable Arial falls back to the standard portable font without preventing score reproduction. No server-specific font path is required.

NA positions, eligibility flags and ranks must match exactly. Numeric checks use absolute 1e-10 except for float32 cosine dot products, which allow eight float32 epsilons across CPU implementations. The full ARM/x86 comparison found a maximum cosine mean difference of 2.50e-8 (all four-decimal values identical); all other endpoint means matched exactly. This does not replace or round the released canonical scores.

Metric conventions and the plotting design follow the pinned [SABench code](https://github.com/Yunzhi-Yan/SABench/blob/54d4e17c6dff5eab7ad93fee2a9bd70385ac750b/Tutorial/SABench.py) and [robustness notebook](https://github.com/Yunzhi-Yan/SABench/blob/54d4e17c6dff5eab7ad93fee2a9bd70385ac750b/Plot/robustness.ipynb). All calculation functions are visible in the new notebook and also supplied in [the plain Python metric source](../src/perturbation_metrics.py).
