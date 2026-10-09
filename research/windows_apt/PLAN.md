# Windows-APT matched comparison, v1

Registered 26 September 2026 before fitting models. Execution uses ~/main_env.

## Task and scope

Compare the existing paper-adapted GNN+rehearsal (`GNN_H`) against
GNN+time+local GRU+additional attention+causal CNN+rehearsal (`TGNN_Q_A_C`).
The exact paper implementation remains underspecified. Both arms receive the
same Windows event representation; this is an explicit new task adaptation.
Neither arm has persistent endpoint memory. The GRU resets for each target.

The publisher describes 36 Caldera scenarios:
https://data.mendeley.com/datasets/b8fmtzvpy8/3
The supplied CSV has no attack/benign or scenario ground truth. Proposed target:
presence of a nonempty JSON ATT&CK ID list in `_source.rule.mitre.id`.
This is Wazuh annotation prediction, not verified attack detection. Absent
annotations mean unknown/unannotated, not benign. User explicitly authorized this weak-label comparison on 26 September 2026.
Training may proceed after all acceptance gates pass.

## Sequential acceptance gates

1. **Source audit:** verify every supplied checksum, parse CSV records correctly,
   compare per-period event IDs with combined IDs, check column widths/IDs/time,
   audit feature availability. Save machine-readable evidence.
2. **Preprocessing:** use all combined records once, strip whitespace missingness,
   sort by timestamp then original row index; keep tied timestamps in one split.
   Chronological 60/20/20, train-only standardization. No resampling or label-based
   filtering. Preserve row identity. All graph history must be strictly earlier.
3. **Feature/graph verification:** use a fixed allowlist of raw event fields.
   No Wazuh rules, MITRE fields, full messages, ruleName, hashes, identities,
   absolute timestamps, scenario IDs or free text enter the feature vector.
   64 deterministic signed hash bins encode field-qualified eventID, provider,
   channel, executable basename, parent executable basename, protocol,
   integrityLevel and eventType. Eight numeric features: log1p lengths of image,
   parentImage, commandLine and targetFilename; source/destination ports /65535;
   two missing-port flags. Labels never influence graph/features/scaler.
   These field choices may still correlate strongly with Wazuh annotations;
   feature-only controls will measure this limitation.
   Each event is a graph node. Endpoints are agent and agent-scoped process GUID;
   if GUID is absent use a unique per-event endpoint, never a shared missing ID.
   Self plus four most recent strictly earlier events sharing either endpoint.
   This host/process incident graph is adapted, not a full provenance graph.
   No graph endpoint IDs are passed as features. Age is log1p elapsed seconds,
   consumed only by the temporal arm. Save feature and graph audit statistics.
4. **Local correctness:** test forbidden-field and label independence, future
   prefix invariance, equal-time exclusion, train-only scaling, padding and batch
   invariance, connected gradients and optimizer updates for both 72-input models,
   matching common initial weights, and original CNN/replay correctness tests.
5. **Package/upload:** create ZIP with original combined CSV and metadata,
   processed tensors, protocol, source snapshot, local gates, manifest containing
   SHA256 for every member. Upload ZIP to a dedicated Modal Volume. Extract on
   Modal CPU, reject unsafe paths, verify ZIP/member hashes, independently
   regenerate preprocessing from raw CSV and compare arrays exactly. Persist gate.
6. **Actual L4 gates:** rerun tests on CUDA, benchmark training-only batches on
   both models at concurrency 1/2/4/8/10. Select highest useful aggregate throughput
   within 5% (ties favor fewer workers). Never increase scientific batch/epochs to
   occupy VRAM. Record real timings and peak memory and one-second telemetry.
7. **Training:** only after gates 1–6 PASS and label scope is resolved. Ten runs,
   2 models x seeds 0–4, 200 epochs each, batch256, Adam lr .001/wd .00001,
   unweighted mean CE(live)+mean CE(replay), dropout .2, float32 deterministic.
   Reservoir floor(0.1*N_train), sample up to26 before admitting live batch,
   Algorithm R RNG seed+937, persists across epochs. Same saved replay schedule
   for both models per seed. No validation/test in reservoir. Final epoch only,
   threshold selected on validation F1 (largest on ties), never optimize test.
8. **Results:** save all losses, checkpoint, per-event probabilities, thresholds,
   confusion counts, F1/precision/recall/FPR/ROC-AUC/PR-AP, timing, memory,
   source/data hashes. Recompute saved metrics and report paired seed differences,
   mean/sample SD. CPU feature-only logistic/tree controls contextualize results.
   Download and independently verify artifacts. Save failures and resume only
   against identical source/data/protocol fingerprints.

Longer capture duration alone does not enlarge the four-event memory horizon.
No accuracy prediction or comparison to CTU F1 is registered. No test-based feature,
threshold, epoch, graph, or model selection is allowed. Capacity/compute differ.
