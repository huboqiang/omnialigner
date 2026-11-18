# Fast run and full ARI/NMI/CLC recalculation

The NCS and Deep notebooks display the complete ARI, NMI and CLC implementation.
The fast default calculates a real NPZ example and reconstructs every full-table
entry from hash-bound actual pair scores. NCS additionally recalculates all 44
selected OmniAligner native six-metric pairs from expression/coordinate NPZs.
Deep calculates all nine metrics on its portable current example.

To recalculate every categorical pair, set `FULL_CATEGORICAL_RECALC = True`.
Choose `CATEGORICAL_DATASETS = ["DLPFC"]` or `["ST_brain"]` for a smaller trial,
or leave all cohorts selected. The visible loops download each checksummed
NPZ asset, calculate ARI/NMI from the saved fixed evaluation rasters and CLC from
native coordinates/labels, and compare each new score to an independent record.
Confirmed native failures retain NA and have no synthetic coordinates.

NCS full inputs total 332 MB; Deep full inputs total 1.48 GB. They contain
coordinates, cell identities, labels and categorical rasters, with no expression
matrix, and are independent Release assets rather than Git payloads. Historical
and fresh raster layouts may be `(layers,1,1280,1280)` or `(layers,1280,1280)`;
the score loop safely removes the singleton channel. Independent native pair
outputs remain independent pair NPZs rather than a fabricated global volume.

ARI/NMI are usually quick. Exact CLC neighbourhood comparisons dominate large
cohorts; a full run can require tens of minutes per large dataset or hours for
the whole Deep comparison. Full inputs are never downloaded by the default run.
Source output coordinates are never resized to improve the evaluation results.
