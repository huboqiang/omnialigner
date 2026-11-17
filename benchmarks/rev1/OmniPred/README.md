# OmniPred revision benchmark reproduction

Run `notebooks/OmniPred_submission_benchmarks.ipynb` to rebuild the current submitted cell-RNA, regional-RNA, native-tile and CRC protein-region comparison tables from included frozen scoring records. SHA-256 checks bind every input. No original H5AD or private server directory is needed for table reproduction.

The notebook recomputes gene and seed summaries and compares them with archived table values. Within-cell profile PCC and cosine use saved per-seed scores; regional support counts use archived fixed-support records. This does not retrain models or replay prediction arrays. Historical selected OmniPred and the repaired-triplet development model are not interchangeable; this notebook retains the former in the primary table and native regional comparison.

No pooled ranking across assays or support units is produced. GHIST fallback and constant genes remain visible. S1/S2 are within one tissue block, not independent-patient validation.
