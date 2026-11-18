# Scoring audit, 18 November 2026

This audit corrects source/adapter errors; it does not add or retune alignment fits.

1. **MLN PASTE coordinate field.** The former scoring selected the inherited
   `spatial_3d_aligned` field. All 19 layers were checked against the input and
   the actual PASTE implementation, which updates `spatial`. The current native
   95,000-cell output is scored from `spatial`, with exactly the original IDs,
   float32 expression, fixed 100-gene panel and metric functions. PCC changes from
   the historical wrong-field value 0.7876 to 0.3891230048408033. The corrected
   ARI/NMI/CLC are 0.4345076445, 0.1427089257 and 0.2052574222. Old values are retained
   separately as legacy evidence, not combined with current results.
2. **E14–16 PASTE2 support.** The old six-score file included 145 cells excluded
   by the current frozen mask in section 14. Only pairs 13→14 and 14→15 required
   fresh scoring on 15,150 cells; the other 13 pairs were hash-verified and reused.
   Current PCC/Cosine/SSIM/MI/MSE/Region means are 0.3953081491, 0.9435746869,
   0.3983732666, 0.1732663232, 0.1364643826 and 0.5983348848. No fitting changed.
3. **Serial-frame adapter.** Later references in E14–16 SANTO/STalign and MLN
   SANTO are the preceding aligned moving section. The first adapter incorrectly
   required every reference to be raw. The corrected adapter verifies exact IDs
   and coordinate continuity, then uses only the first input gauge for the whole
   chain. All 15/15/18 real pairs are present and scored; these were not native
   partial failures. Incorrect intermediate partial states were archived.
4. **Input precision and chart units.** SPACEL float32/median preprocessing was
   replayed bit-for-bit rather than loosening a geometric gate. MLN full-method
   input charts were established from saved prealignment inputs, not output
   bounding-box fitting. Source coordinate values were not modified.
5. **Incomplete previews.** An early local test omitted native main scores and
   incorrectly plotted unloaded cells as 9/9. It was withdrawn. Current exports
   require every expected input or a proved native failure before ranking.

Native raw missing measurements remain NA. A failed/ineligible pair receives
zero only in the separate quality score, and every expected adjacent pair stays
in the denominator. A common affine metric scale preserves native-mean ordering
among complete experiments; pair ranks are not averaged. Grouped figures average
normalized endpoint-rank credit with equal dataset and endpoint weights.

Current comparisons exclude Loki, CAST and CPD. NCS uses nine algorithm families;
Deep uses the seven completed families and retains unfinished SANTO/MOSAICField
runs as diagnostics. Source/receipt hashes, original arrays and raw-score records
are retained. Ranks do not imply independent biological or statistical validation.
