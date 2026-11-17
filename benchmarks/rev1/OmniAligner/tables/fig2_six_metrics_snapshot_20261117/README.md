**Historical-coordinate diagnostic; not the newer agent-assisted selected OmniAligner model.**

# Additional Fig. 2 six-metric evaluation: interim snapshot

Three datasets have complete scoring for five historical Fig. 2 methods: SmallIntestine (7 adjacent pairs), STARmap (9), ST_brain (74). This is distinct from the six-dataset NCS benchmark and uses the historical Fig. 2 coordinate outputs.

OmniAligner: SmallIntestine ranks 3/5 on PCC, CosSim, SSIM, MI and MSE, and 2/5 on Region; STARmap ranks 2/5 on all six. SPACEL is first on all 12 eligible comparisons. OmniAligner is first on 0/12 rankable entries.

ST_brain OmniAligner is not ranked: molecular metrics include insufficient-grid fallback at pairs 64 and 70, and undefined PCC/CosSim/SSIM for one gene at pair 68. Region has zero support at pairs 64 and 70. Native means/defaults are retained in the diagnostic CSV; they are not valid whole-sequence performance estimates. The remaining six datasets are not included in this snapshot.

One method is counted once. Rank uses unrounded values, lower MSE and higher other endpoints. Failed or undefined endpoints are not treated as zeros for ranking. Per-gene evaluation panels are fixed independently of alignment results. No statistical significance is inferred. Figures remain standalone and have not been inserted into the manuscript.
