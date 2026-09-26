# Completed L4 component study — 24 September 2026

Follow-up, 26 September: [all results collected under research/findings](research/findings/README.md), with [a measured data-scale and memory-retention analysis](research/findings/memory_and_data_scale.md). The larger-data hypothesis is separate from the completed scores below.

All **60/60 runs completed**: 12 variants, five seeds each, 200 epochs per run.
Completion time: 17:05:07 UTC (22:35:07 IST). Modal app:
[ap-DVkT46ml1rq8l5dQhGj14M](https://modal.com/apps/prathampansare25/main/ap-DVkT46ml1rq8l5dQhGj14M).

The best observed TGNN mean test F1 is **TGNN + GRU + additional attention:
86.39 ± 3.27%**. It improves on plain TGNN by **1.87 percentage points**, while
GNN + rehearsal remains **0.23 points higher**, at **86.61 ± 4.84%**.
No tested TGNN variant exceeds the GNN reference's mean test F1 in this study.
The small gap and seed variation do not establish a general superiority claim.

## All results

Values are percentages, mean ± sample standard deviation over seeds 0–4.
All variants retain rehearsal. TGNN means GNN + time + rehearsal.
GRU processes the four past-neighbor embeddings in chronological order, resetting
its state per event. Memory is persistent endpoint EMA state. Additional attention
is a separate attention readout; the inherited GNN already has graph attention.

| Variant | Test F1 | Test PR-AP |
|---|---:|---:|
| GNN + rehearsal | **86.61 ± 4.84** | 88.18 ± 5.30 |
| TGNN | 84.52 ± 2.63 | 94.22 ± 0.49 |
| TGNN + GRU | 85.45 ± 1.38 | 93.64 ± 0.66 |
| TGNN + memory | 83.86 ± 4.01 | **94.77 ± 0.48** |
| TGNN + GRU + memory | 85.52 ± 2.57 | 94.67 ± 0.95 |
| TGNN + additional attention | 83.03 ± 3.08 | 92.88 ± 1.59 |
| TGNN + GRU + additional attention | **86.39 ± 3.27** | 94.33 ± 0.40 |
| TGNN + memory + additional attention | 76.72 ± 6.29 | 94.29 ± 0.98 |
| TGNN + GRU + memory + additional attention | 84.15 ± 4.03 | 94.27 ± 0.96 |
| TGNN + normalization | 82.26 ± 4.89 | 90.87 ± 4.03 |
| TGNN + weighted loss | 85.26 ± 2.94 | 94.64 ± 0.74 |
| GNN + rehearsal + additional attention | 80.90 ± 8.23 | 84.70 ± 4.39 |

## Interpretation

- GRU + attention has the highest observed mean TGNN test F1, but improves on
  plain TGNN in only three of five paired seeds. The paired F1 change is
  +1.87 ± 5.02 percentage points. Against GNN it wins two seeds, loses two and
  ties one; paired change −0.23 ± 7.34 points. These SDs describe differences
  across seeds, not uncertainty across independent datasets.
- Plain GRU has the lowest observed TGNN F1 SD (1.38 points), with mean F1 85.45%.
- Memory alone gives the best mean PR-AP (94.77%), but F1 falls to 83.86%.
  Thus improved score ranking does not guarantee improved F1 at the frozen
  validation-selected threshold.
- The full GRU + memory + attention model has 100% precision and zero test false
  positives in all five seeds, but only 72.80% mean recall. Its F1 is 84.15%.
- The best-F1 TGNN has 95.18% mean precision and 79.20% mean recall, compared with
  plain TGNN's 97.95% and 74.40%. Its improvement comes with more recovered
  positive flows and a modest precision reduction.
- More components did not consistently improve detection. Attention alone,
  normalization, and memory + attention reduced mean F1 here.

## Validation ties and claim boundary

**Nine variants tie at 100% mean validation F1:** plain TGNN, TGNN+A, TGNN+M,
TGNN+M+A, TGNN+GRU, TGNN+GRU+A, TGNN+GRU+M, TGNN+GRU+M+A, and TGNN+normalization.
The automated summary's `validation_preferred: TGNN` is the first entry among
equal maxima; it is **not a uniquely better validation result**. This clarification
supersedes that interpretation of the generated report's “preferred” label.

The best observed test configuration is an exploratory result from the already
inspected CTU-13 scenario-6 partition, which contains only 25 positive flows.
Thresholds were selected using validation only; no new test-based threshold or
training changes were made. The next scientific step is a preregistered comparison
of GNN+rehearsal, plain TGNN and TGNN+GRU+attention on an untouched capture with
adequate positive support. That new experiment has not been launched here.

## Resources and verification

| Model | Parameters | Mean training seconds | Peak allocated GPU MiB, mean |
|---|---:|---:|---:|
| GNN + rehearsal | 35,458 | 148.14 | 74.46 |
| Plain TGNN | 35,714 | 156.72 | 75.74 |
| TGNN + GRU + attention | 183,938 | 275.49 | 87.66 |

All runs used the same L4, dataset hash, split, features, graph, optimizer,
rehearsal policy, epoch count, seed set and threshold rule. Added modules increase
capacity; the best-F1 TGNN has about 5.15 times plain TGNN's parameter count.

Local audit checked complete grid membership; all 60 checkpoint architectures and
finite weights; all saved prediction probabilities, labels and row IDs; validation
thresholds; per-seed metrics; aggregate means/SDs and paired differences; source/data
hashes; and training metadata. **Every check passed.**

The ten rerun reference checkpoints and prediction arrays are **bit-identical** to
the original GNN and plain-TGNN runs. The frozen 86.61/84.52 reference is reproduced
exactly despite the execution optimizations.

- [Per-seed report](artifacts/components_200e_v1/TGNN_COMPONENT_REPORT.md)
- [Full machine-readable summary](artifacts/components_200e_v1/summary.json)
- [Independent local verification](artifacts/components_200e_v1/local_verification.json)
- [Registered protocol](TGNN_COMPONENT_PROTOCOL.md)
- [Frozen reference](REFERENCE_RESULTS.json)

Summary SHA256: `1be3420aae31cd555189f82d92456563e1bf8a04bed7e63202462381badc70db`.
Manifest SHA256: `015e9327ff04305df122c092e46319e86225b75d96cdc8cf8038398bfd7c48e5`.
All original source snapshots, checkpoints and predictions remain preserved.
