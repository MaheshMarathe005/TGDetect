# DARPA TC Engagement 3 bounded pilot

Datasets: CADETS E3 and THEIA E3, first published raw archive of each,
downloaded directly into Modal Volume tg-detect-darpa-e3-pilot-v1.
Sources: DARPA's Transparent-Computing README-E3.md; Orthrus acquisition scripts;
ProvenanceAnalytics/ground-truth (now ubc-provenance/ground-truth) node CSVs.
Raw archive and label CSV SHA256 values are saved on the Volume.

This is an adapted pilot of GNN_H versus TGNN_Q_A_C, not an exact reproduction
of the base paper on its own data. The task is association with independently
published malicious-node labels: an event is positive when any participating
subject/predicate UUID is listed. Unlisted events form the benchmark negatives;
they are not independently verified benign. This is distinct from Windows-APT
Wazuh-annotation prediction and from verified event-level attack classification.

Sample: smallest 100,000 SHA256 values of event UUID plus seed 20260927,
independent of labels. All selected events are ordered by integer nanosecond
timestamp, ties kept together. Chronological train/validation/test 60/20/20.
Training requires both classes in every partition. If the bounded source lacks
coverage, record failure and extend raw coverage without consulting model scores.
Do not randomly distribute neighboring attack events across partitions to pass.

Feature whitelist: hashed raw event type, raw system-call name, raw executable
name, size and its missingness. Exclude UUID, ground-truth attributes, attack
names, label-derived fields and absolute timestamps from feature vectors.
UUIDs construct incident-event histories; relative seconds encode temporal ages.
Fit scaling on training events only. The four-neighbor history is built from
the sampled stream, so this pilot loses intermediate context from full telemetry.
Report that limitation and shared-node exposure across chronological partitions.

If coverage passes: both existing architectures, fresh weights, Adam lr .001,
weight decay 1e-5, batch size 256, 10% reservoir and 26 replay events, 20 epochs,
seeds 0/1/2 for a bounded initial comparison on a Modal L4. Same split,
feature tensors, replay schedule and epochs for both. Threshold learned from
validation F1 with largest-threshold tie break; test used only for evaluation.
Save checkpoints, probabilities, confusion matrices, F1, precision, recall,
PR-AP, ROC-AUC, FPR, GPU/runtime metadata and source hashes. Training begins
only after local correctness tests and the data coverage gate pass.
