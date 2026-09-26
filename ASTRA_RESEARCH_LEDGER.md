# Astra research ledger

Date: 2026-09-26
Current phase: Completed and independently verified causal temporal-CNN comparison (C014)
Question: Does temporal state improve detection when data, task, features, graph, training budget, replay and evaluation are held fixed?
Hypothesis: A temporal mechanism may help, but existing comparisons change several of these variables and cannot identify its contribution.
Evidence: Fresh source inspection and independently extracted local PDF only. Previous notes, results and source comments containing performance claims are not evidence.
Experiment/config: `/home/mahesh/main_env/bin/python`; ADAPTED_PROTOCOL.md records the pilot and all amendments before real-data training.
Result: All 60 component runs complete and independently verified. GNN+rehearsal F1 86.61±4.84%; plain TGNN 84.52±2.63%; highest observed TGNN mean F1 is GRU+additional attention at 86.39±3.27%. Ten rerun anchor checkpoints and prediction arrays are bit-identical to the frozen reference. Full details: TGNN_COMPONENT_FINDINGS.md.
Interpretation: More continuous data could reveal benefits from historical context, but the 0.23-point F1 gap has no established cause. Best TGNN uses local four-neighbor GRU, not persistent memory; tested persistent EMA rapidly discounts old endpoint updates. The sample spans the entire 2.15-hour capture while retaining only 2.93% of flows.
Confounds: Underspecified paper implementation; local source NPZs verified to be synthetic; duplicated baselines; legacy evaluation and causality failures.
Decision: Preserve verified results and run the user's requested CNN comparison under the frozen matched protocol. The separate larger-data memory advantage remains an untested hypothesis.
Next action: Use an untouched capture and equal-capacity controls to test whether CNN gains persist; separately test full-resolution event density and learned persistent memory. Later data-scale/memory experiments remain proposed in research/findings/memory_and_data_scale.md.
Status: PRIOR 60/60 RUNS VERIFIED; CNN STUDY VERIFIED COMPLETE — 40/40 RUNS

## Reset record

- Full workspace backup: `/tmp/tgdetect-before-reset-20260923T071151Z.tar.gz`. Every archived file was checked against its original SHA-256 before removal. This is temporary storage; copy it elsewhere if long-term retention is needed.
- `RESET_INVENTORY.json` records all 115 original files, hashes, removal decisions and preserved files.
- Consolidated the two reset wrapper directories into the project root. Their executable source and data were byte-identical to root copies.
- Preserved the original paper and `BASE_PAPER_RESET_COPY.pdf`: different bytes, identical fresh `pdftotext -layout` output.
- Removed prior notes, presentations, logs, result JSONs, caches and disposable `.labtest` experiment/debug scripts from the active workspace. Archived the old ledger and initial audit; neither is research evidence.
- Preserved eight root Python source files, both legacy helper modules and all NPZ data/cache files. Data subdirectories are intentional dependencies, not redundant project copies.
- `.labtest/harness.py` is executable infrastructure, so retained and fixed: future fixtures use temporary storage; real scikit-learn replaces the stub injection. Preserved input arrays remain unchanged.

## Change log

### C014 — causal 1D temporal CNN and measured L4 utilization

Date: 2026-09-26
Current phase: Registered implementation and acceptance gates
Question: Can a temporal 1D CNN, alone or combined with TGNN/GRU/attention, improve the matched detection results, and can independent runs better occupy the L4?
Hypothesis: Causal convolution may learn useful ordered local patterns. Independent concurrency and removing per-event Python replay work may improve useful GPU throughput; allocating unused VRAM is not an efficiency metric.
Evidence: TEMPORAL_CNN_PROTOCOL.md; temporal_cnn.py; run_temporal_cnn.py; modal_temporal_cnn.py; test_temporal_cnn.py. Primary TCN study and official Modal/PyTorch documentation reviewed.
Experiment/config: Eight arms × five seeds, 200 epochs each: GNN+H, TGNN, TGNN+Q, TGNN+Q+A, CNN+H, TGNN+C, TGNN+Q+C, TGNN+Q+A+C. Same data, four past neighbors, Adam, batch256, CE, rehearsal and validation threshold. Two left-padded temporal Conv1d layers, kernel3, dilations1/2, hidden128. Standalone CNN replaces graph encoder; hybrids add convolution on GNN node representations. No persistent memory or longer-data claim. Exact replay indices precomputed without changing RNG/admission order. One L4; training-only 1/2/4/8/16-worker concurrency calibration; utilization/memory/power recorded every second; durable per-run results.
Result: First native-Conv1d calibration measured 2.26/4.47/9.03/12.16/12.06 aggregate steps/s for 1/2/4/8/16 workers. Stopped that slow attempt ap-PA6IRh5XpaOUrDdCrcVqnV; exclude all partial work. Replaced short-window convolution execution with mathematically identical causal tap-gather/GEMM, verifying outputs and input/kernel gradients against native Conv1d. All 62 local tests pass; actual L4 repeats all 17 CNN gates in 9.79 s. Final v2 job completed 40/40 runs using 16 workers. Full telemetry: 661 one-second samples, mean GPU utilization 95.99%, maximum 100%, mean memory 4903.9 MiB, maximum 6319 MiB, mean power 57.22 W. Independent local audit passed all 40 checkpoints/metrics/hashes; ten prior controls are bit-identical. Architecture/grid/optimization unchanged.
Interpretation: CNN+H is 97.11±1.16% F1, higher than GNN+H 86.61±4.84%; TGNN+C 85.51±4.19%; TGNN+Q+C 87.07±4.81%; TGNN+Q+A+C 87.16±6.41%. CNN alone is a strong local-window result, not evidence of persistent long-term-memory superiority. Hybrid gains over their prior models are +0.99/+1.62/+0.77 points with wide paired SDs. Local five-position context remains a short-history experiment. GPU memory occupancy alone does not establish compute efficiency. Optimized calibration achieved 25.08/50.46/96.38/202.18/299.91 aggregate steps/s at 1/2/4/8/16 workers, selecting 16 workers on the same L4.
Confounds: Previously inspected small capture and 25 test positives; additional CNN parameters; concurrent timings include contention; concurrency benchmark represents the largest hybrid and may not perfectly predict every mixed workload.
Decision: Run the complete registered grid after final local and L4 gates. Select worker count by measured aggregate training throughput, keeping each run's scientific settings fixed.
Next action: Use an untouched capture and equal-capacity controls to test whether CNN gains persist; separately test full-resolution event density and learned persistent memory.
Status: VERIFIED COMPLETE — 40/40 RUNS; LOCAL ARTIFACT AUDIT PASS; https://modal.com/apps/prathampansare25/main/ap-7y95Pbrwfia3sP44n9PnMW

### C013 — larger-data and long-term-memory hypothesis review

Date: 2026-09-26
Current phase: Research synthesis and measured implementation/data audit
Question: Did limited data suppress memory benefit, and will much larger real-time streams improve TGNN?
Hypothesis: Dense continuous histories containing predictive delayed dependencies may favor a suitable persistent-memory model; amount of data alone is insufficient.
Evidence: research/findings/component_results.md, component_runs.csv, component_summary.json, memory_and_data_scale.md, memory_horizon_audit.json; audit_memory_horizon.py; original TGN, Temporal Graph Benchmark, ORTHRUS, METANOIA v2 and CTU owner sources, linked in the research note.
Experiment/config: Read-only audit of frozen NPZ, metadata, component definitions, training/evaluation state lifetime and existing 60-run summary. No retraining, threshold changes or GPU execution.
Result: Exported all 12 aggregate results and 60 detailed seed rows. Retained 16384/558919 flows (2.9314%, 34.11-fold raw/sample ratio), spanning the complete 7749.87-second range (2.15274 h). Test span 23.52 min, 25 positives. Prior endpoint activity exists for 3265/3277 test events; 3240 have four valid graph neighbors. The 86.39% model has T+H+Q+A and no M. Q resets per target; persistent M uses 0.5 EMA, retaining only 0.09765625% of a given old state after ten further endpoint updates. Memory persists between batches and across evaluation partitions; training gradients detach between 256-event blocks. Matched M F1 effects across plain/attention/GRU/GRU+attention backbones: −0.66/−6.31/+0.07/−2.24 points.
Interpretation: Thinning may delete useful intermediate evidence and limited positives may impair learning, but full scenario 6 adds density rather than days of history. Current EMA is a weak long-retention test and higher event rates can shorten its effective clock-time horizon. Existing tests neither prove long-term memory useless nor guarantee gains from more data. Published evidence includes both memory benefits and successful stateless temporal designs.
Confounds: Single inspected botnet-flow capture, non-independent flows, limited labels, small selected test, larger-model capacity, short-context baseline already sees past graph events, fixed EMA and truncated training gradients. Diagnosis is not a causal experiment on data size.
Decision: Record the user's hypothesis as plausible and unproven; propose separate controls for amount of labeled training data, density of available context, elapsed duration, learned retention, capacity and compute. Keep rehearsal/continual forgetting distinct from endpoint memory.
Next action: Preregister a small matched study and delayed-cue learning gate, followed by dense/longer real streams and untouched-capture confirmation. No new training initiated in this phase.
Status: VERIFIED DIAGNOSTICS; RESEARCH NOTE COMPLETE; SCALING HYPOTHESIS UNTESTED

### C012 — completed component study and independent result audit

Date: 2026-09-24
Current phase: Results verified
Question: Did any registered addition improve TGNN and exceed the GNN reference?
Hypothesis: Component effects can differ between F1 and ranking metrics.
Evidence: artifacts/components_200e_v1/{summary.json,verification.json,local_verification.json}; audit_component_results.py; TGNN_COMPONENT_FINDINGS.md; all 60 checkpoints/prediction files and frozen source snapshots.
Experiment/config: Registered 12-arm, five-seed, 200-epoch single-L4 study, completed at 17:05:07 UTC. No grid/threshold/epoch changes after training began.
Result: All 60 runs complete. GNN+H 86.61±4.84% F1; TGNN 84.52±2.63%; TGNN+GRU 85.45±1.38%; +memory 83.86±4.01%; +GRU+memory 85.52±2.57%; +attention 83.03±3.08%; +GRU+attention 86.39±3.27%; +memory+attention 76.72±6.29%; +GRU+memory+attention 84.15±4.03%; +normalization 82.26±4.89%; +weighted loss 85.26±2.94%; GNN+H+attention 80.90±8.23%. Memory alone has the highest mean PR-AP, 94.77±0.48%. Full-model precision is 100% with recall 72.80%. All complete-grid, checkpoint-loading/finite-weight, prediction/threshold/metric, aggregate/paired-summary, source/data hash and metadata checks pass. Ten anchor checkpoints and predictions reproduce originals bit for bit.
Interpretation: Best observed TGNN F1 improves 1.87 points versus plain TGNN, remains 0.23 below GNN. GRU+attention beats plain TGNN in 3/5 seeds; no consistent universal improvement. Nine variants tie at 100% validation F1; generated summary chooses first-in-order TGNN, not a unique validation winner.
Confounds: Previously inspected single capture, 25 test positives, model-selection multiplicity, greater capacity (183938 vs 35714 parameters for best-F1 TGNN vs plain), longer compute. Mean differences are descriptive, not proof of general superiority.
Decision: Preserve all results, report metric-specific findings and validation ties; keep the original reference unchanged.
Next action: Register independent-capture confirmation of GNN+H, plain TGNN and TGNN+GRU+attention; no new experiment launched during this audit.
Status: VERIFIED COMPLETE

### C011 — independent GRU, memory, attention and training additions

Date: 2026-09-24
Current phase: Registered component comparison
Question: Which additions improve the matched TGNN and how do they compare with GNN+rehearsal?
Hypothesis: History GRU, persistent memory, additional attention, normalization or class weighting may improve F1; interactions may help or harm.
Evidence: User request; TGNN_COMPONENT_PROTOCOL.md; tgnn_components.py; run_tgnn_components.py; modal_tgnn_components.py; test_tgnn_components.py. Primary TGN/TGAT sources linked in protocol. REFERENCE_RESULTS.json independently verifies ten frozen reference runs.
Experiment/config: 12 arms × five seeds, all 200 epochs on one Modal L4. All eight TGNN combinations of Q (past-neighbor GRU), M (persistent EMA endpoint memory), A (additional attention); GNN+H, GNN+H+A; TGNN+N and TGNN+W. Q is explicitly distinct from pilot R (GRU replacement of persistent update). Same data/features/graph/replay/optimizer/threshold settings. Durable per-run volume commits. No test-driven grid modification.
Result: 45 causal/component tests passed locally; foundation suite adds 5 passes and 4 known legacy expected failures. The actual L4 repeated all 29 component tests successfully (10.93 s). Training started in Modal app ap-DVkT46ml1rq8l5dQhGj14M on profile prathampansare25. GNN_H seed 0 completed 200 epochs in 149.64 s: test F1 80.8511%, PR-AP 82.0700%, TP19/FP3/FN6/TN3249, matching its frozen reference metrics. Its predictions, validation threshold, checkpoint and source snapshots were downloaded and independently verified locally. Subsequent seeds are running; saved progress is readable from Volume tg-detect-components-v1, directory components_200e_v1. Full grid results pending.
Interpretation: Code and actual GPU gates pass; no component improvement is yet claimed.
Confounds: One inspected capture, 25 test positives, increased parameters/compute, stale rehearsal snapshots, exploratory multiple comparisons. Normalization/loss changes are separately identified.
Decision: Launch only after the actual L4 repeats the component gates; save every seed result and prefer configurations using validation mean F1.
Next action: Execute Modal study on prathampansare25 and verify all saved predictions and checkpoints.
Status: COMPLETE; see C012 for final verification — https://modal.com/apps/prathampansare25/main/ap-DVkT46ml1rq8l5dQhGj14M

### C010 — missing matched rehearsal/time arm and Modal L4 handoff

Date: 2026-09-23
Current phase: Matched architecture test
Question: Does paper-specified GNN+10% rehearsal improve when time encoding is the only added model factor?
Hypothesis: `B_T_H` may outperform `B_H`, but the previous pilot did not run this exact pair and cannot answer it.
Evidence: User correction; `MATCHED_TIME_REHEARSAL_PROTOCOL.md`; `run_matched_time_rehearsal.py`; `modal_matched_time_rehearsal.py`.
Experiment/config: Same scenario-6 sample, graph/features/labels/split, 3×128 GAT, ReLU/.2, CE, Adam .001/.00001, batch 256, 10% Algorithm-R replay, validation threshold, 200 epochs and seeds 0–4. `T` is the sole paired model difference. Exact unavailable paper choices remain explicitly adapted.
Result: Local execution was stopped after five `B_H` and one `B_T_H` artifacts; these are excluded. After changing to Modal profile `prathampansare25`, run `ap-2ukQHuXPf3dCLIi5g6MtYx` completed all ten runs in one L4 function. B+H F1 86.61±4.84%, PR-AP 88.18±5.30%; B+T+H F1 84.52±2.63%, PR-AP 94.22±0.49%. Paired T effect: F1 −2.10±5.35 points; PR-AP +6.04±5.46 points. Precision rises 4.41 points, recall falls 6.40, and FPR falls 0.031 points. All CUDA/source/data/checkpoint/prediction/metric gates passed. Archive SHA256 `2d75e1168e8e0c45a2ba883abe22fd5bce62c3dc6b3b7f4c293fe92661d812c7`.
Interpretation: The matched test now answers the missing question. Time encoding improves PR-AP in all five paired seeds and improves mean precision/FPR, but does not improve mean F1 because recall falls at the validation-selected thresholds. No single all-metric superiority claim is justified.
Confounds: Test data were already inspected; this completes a missing arm rather than independent confirmation. CUDA and CPU results must not be mixed. Paper data/operator/loss/replay ambiguities remain.
Decision: Report metric-specific effects: B+T+H is better for PR-AP, precision and FPR; B+H is better for F1 and recall. Preserve partial CPU files only as excluded audit artifacts. Do not compare these adapted CTU results numerically to the paper's Wazuh result.
Next action: Repeat the fixed comparison on an untouched capture/family before making a general temporal-architecture claim.
Status: VERIFIED

### C005 — registered adaptation and direct-source data

Date: 2026-09-23
Current phase: Data and baseline
Question: Can direct-source data support a transparent adaptation?
Hypothesis: A documented protocol and provenance manifest enable a bounded pilot.
Evidence: `ADAPTED_PROTOCOL.md`, `fetch_pilot_data.py`, `causal_data.py`, CTU owner documentation, `data/ctu13_verified/` manifests.
Experiment/config: Completion proxy StartTime+Dur; label-independent evenly spaced records; chronological 60/20/20; 10 STATIC_FAIR features standardized on training only. Raw source retrieval uses HTTPS and records checksum/time/size.
Result: Scenario 11: 107251 raw records, SHA256 `cee542d4b5efe4fa1cd59b87ece5aaea9a13d8f0abe56cd07cee31590736428c`; 4096 sample positives 1/3/314, excluded before training. Scenario 6: 558919 raw records, SHA256 `801800eeda9a5a44868b1e3e492f93dc6d52b5d14566c1d8431c9e85d0d2adaa`; 8192 sample positives 41/14/9, below registered 20/10/10 support minimum. Recorded amendment doubles coverage once before training: 16384 rows, train/validation/test 9830/3277/3277 with 91/25/25 positives. Processed SHA256 `cf4c0d3c9eba674db8297923f97b664d3de5c5dea412b7ebca444410949a80c6`.
Interpretation: Real captured botnet data with retrieval lineage; no independent publisher checksum or original Wazuh task is implied.
Confounds: Dataset selection used class counts; thinning changes history; Background is uncertain negative; completion proxy lacks actual exporter arrival; only 25 test positives.
Decision: Use scenario 6 under the recorded protocol; retain failed candidate manifests and original synthetic fixtures.
Next action: Complete fixed pilot with no score-driven sample/model changes.
Status: VERIFIED

### C006 — baseline and causal temporal path

Date: 2026-09-23
Current phase: Correctness
Question: Can predictions exclude future information and train recurrent state?
Hypothesis: Strictly-past ego graphs plus score-before-update state and within-block gradients resolve the earlier counterexamples.
Evidence: `pomsathit_baseline.py`, `causal_tgn.py`, `test_causal_pilot.py`; exact class remains blocked.
Experiment/config: Adapted 3×128 single-head GAT, ReLU/.2, softmax/CE; T cosine log-age; M endpoint averaging; R GRU conditional on M; H training-only Algorithm R with frozen memory snapshots. Equal-time events share strictly prior state; detach only between training blocks.
Result: **21 passed, 4 xfailed**: 16 new-path checks plus 5 legacy passes; 4 known legacy defects preserved. Hand attention, future/prefix invariance, eval batch invariance, equal-time/new-node state, GRU gradients and updates, scaling, reservoir inclusion/boundary, replay isolation, threshold and factor guards pass. Baseline 35458 parameters. ESP records exposed a 32-bit non-port field: only TCP/UDP receive port features, documented before training.
Interpretation: New-path correctness only; no certification of the old TGNN or exact paper reconstruction.
Confounds: Recency is already in graph selection; R depends on M; memory truncates at 256 events; replay snapshots become stale; parameter counts differ.
Decision: Accept the new path for the bounded adapted pilot.
Next action: Report all registered model scores and fixed-checkpoint controls.
Status: VERIFIED

### C007 — synthetic memory-learning test

Date: 2026-09-23
Current phase: Learning acceptance
Question: Can memory learn when only the preceding event predicts the query label?
Hypothesis: Memory succeeds and its reset removes the advantage.
Evidence: `temporal_toy.py`, `artifacts/temporal_toy.json`.
Experiment/config: Five seeds; 64 train and 128 held-out cue/query pairs per seed, 60 epochs, Adam .001/.00001; fixed threshold .5. Both graphs have self loops only; independent endpoint pair per cue/query. No CTU data or rehearsal.
Result: Mean accuracy: B 49.84375%, B+M+R 100%, B+M+R with reset 48.59375%. All five temporal seeds reach 100%.
Interpretation: The memory learning path works and its removal destroys the artificial signal.
Confounds: Easy synthetic task with disabled graph history; no security performance claim.
Decision: Learning gate passes; keep synthetic scores separate from CTU results.
Next action: Real-data pilot.
Status: VERIFIED

### C008 — further research and controlled pilot runner

Date: 2026-09-23
Current phase: Controlled CPU pilot
Question: What do timing, memory, recurrent updates and replay contribute under common inputs?
Hypothesis: Effects may be small or attributable to other factors; simple controls are essential.
Evidence: `RESEARCH_UPDATE.md`, primary TGN code/TGAT/TGB, USENIX 2025 simpler-model study and PIDSMaker, ORTHRUS, METANOIA; `run_causal_pilot.py`, `ABLATION_MATRIX.json`.
Experiment/config: main_env CPU one thread, deterministic algorithms, 3 epochs, 5 seeds, 8 neural arms and logistic/tree controls. Validation-only thresholds; test AP/F1/ROC-AUC/FPR. Full pilot checkpoints receive reset/frozen/no-op memory, constant and shuffled-age interventions. Shuffled age is a counterfactual stress input, never a detector.
Result: All 40 registered neural runs, 10 feature-control fits and 50 interventions completed. Matrix includes all 28 initial-factor pairs; 40 seed rows complete and 200 explicitly unrun/dependent. Checkpoints, predictions, parameters, timings, manifests and logs are in `artifacts/causal_pilot_v1/`.
Interpretation: This bounded local pilot precedes GPU-heavy full-data training on Modal; it is not the paper's 200-epoch experiment.
Confounds: One thinned capture, 25 positive test flows, short training, larger temporal models, stale replay context, no cross-scenario forgetting or complete factor study.
Decision: Finish every registered arm and preserve all results.
Next action: Follow the result interpretation in C009 and the next-stage registration at the top of this ledger.
Status: VERIFIED

### C009 — artifact verification and pilot interpretation

Date: 2026-09-23
Current phase: Pilot completed
Question: What survives five seeds, feature-only controls and memory/time interventions?
Hypothesis: A full temporal stack may provide less value than an isolated addition.
Evidence: `report_causal_pilot.py`, `TGNN_PILOT_REPORT.md`, `artifacts/causal_pilot_v1/summary.json`, `verification.json`, `model_profiles.json`; README and architecture audit updated for the new path.
Experiment/config: Recompute validation thresholds, test metrics and all intervention metrics from saved predictions; verify raw/processed/source and every checkpoint hash; profile loaded modules. No retraining, threshold retuning or extra score-based arm selection.
Result: All artifact checks pass. Mean F1: B 83.80%, B+T 89.84%, B+M 77.78%, B+M+R 75.52%, B+T+M 86.91%, pilot TGNN 84.33%, B+H 81.05%, pilot TGNN+H 84.37%; logistic 83.64%, tree 97.96%. Full five-seed SD/AP/ROC/FPR/resources and paired contrasts are in the report. TGNN-vs-B F1 difference +0.53 percentage points, paired SD 3.20, with mixed signs. Time-only-vs-B +6.04 points, paired SD 3.60, positive in every seed.
Interpretation: No convincing full-TGNN benefit on this pilot; raw features support a strong tree classifier. Memory reset at fixed threshold drops TGNN F1 to 25.22% while increasing AP from 90.34% to 93.56%; this exposes threshold/score shift, not improved ranking from memory. Constant/shuffled age also reduce AP, indicating numeric-time sensitivity in this fitted model.
Confounds: Interventions cause distribution shift; they do not replace retraining. Only 25 test positives, one sampled capture, three epochs, extra parameters, no cross-family or forgetting evaluation. Tree seed SD zero reflects deterministic repeated fits, not generalization certainty.
Decision: Preserve all results. Do not claim temporal superiority or expand/tune models using this now-observed test set. Investigate the time-only effect and simple controls on untouched captures under a new protocol.
Next action: See the preregistration task at the top of this ledger. Full factor study remains unfinished.
Status: VERIFIED

### C001 — cleanup and fresh audit tooling

Date: 2026-09-23
Current phase: Correctness and provenance
Question: Can local validation overwrite inputs or mistake generated fixtures for real captures?
Hypothesis: The old harness can overwrite `.labtest/data/real/ctu13/s01.npz` through `s03.npz`.
Evidence: `harness.make_fixtures` directly writes those paths; no provenance manifest is present alongside the retained files.
Experiment/config: `audit_environment.py`; source/data/PDF SHA-256s in `ENVIRONMENT_AUDIT.json`; fixture seed 0, 30000 rows each, no train/validation/test split or performance metrics. Legacy harness default output moved to temporary storage and stub injection removed. Final audit checks all retained Python syntax without importing apps and verifies untouched inputs/source against the reset inventory, failing on unexpected changes.
Result: Every array in s01/s02/s03 exactly matches the legacy synthetic fixture generator. main_env CPU PyTorch smoke and deterministic RNG checks pass. Five foundation checks pass; four intentional XFAIL cases expose legacy defects (see C003).
Interpretation: A folder name is not proof that data is CTU-13.
Confounds: Derived context-cache lineage is still unknown; no conclusion about any remote Modal-volume data is implied.
Decision: Preserve every input, use temporary fixtures, and block research training on unverified data.
Next action: Obtain raw-source manifests/checksums before approving a real-data experiment.
Status: VERIFIED

### C002 — independent specification and canonical skeleton

Date: 2026-09-23
Current phase: Baseline reconstruction
Question: Can the paper uniquely specify an executable network and experiment?
Hypothesis: Matching depth and width is insufficient to reconstruct it.
Evidence: Complete fresh PDF extraction plus visual inspection of tables/figures; public full-text identity cross-check. See `BASELINE_PAPER_SPEC.md`.
Experiment/config: `pomsathit_baseline.py` encodes only explicit constants and exports `BASELINE_ARCHITECTURE.json`; both reserved model APIs raise `UnresolvedPaperSpecification` rather than guess missing details. No dataset, seed, split, loss, threshold or metrics apply to this skeleton.
Result: Three layers, 128 hidden, ReLU, .2 dropout, Adam .001/.00001 decay, batch 256, 200 epochs per scenario, five seeds and 10% reservoir verified as paper statements. VAE belongs to a separate comparator. Attention equation, encoders, label mapping, loss, replay lambda and data/split remain unresolved.
Interpretation: Exact paper reproduction is blocked; an explicitly labeled adaptation remains possible.
Confounds: A machine-readable specification is not a certified executable architecture; parameter counts/modules intentionally remain null.
Decision: Skeleton accepted for the prompt's first-session scope; no temporal architecture change.
Next action: Define the adapted protocol transparently, or obtain attributable author code/data.
Status: BLOCKED

### C003 — architecture profiling and falsification diagnostics

Date: 2026-09-23
Current phase: Correctness
Question: Do legacy implementations satisfy their static/causal/trainable-memory descriptions?
Hypothesis: Source differences and state scheduling invalidate those descriptions.
Evidence: `audit_models.py`, `test_research_foundation.py`, `EXISTING_MODEL_AUDIT.json`, `ARCHITECTURE_AUDIT.md`.
Experiment/config: main_env Python 3.10.20, torch 2.11.0+cpu, one CPU thread, seed 0; tiny three-event graphs and an eight-event context sequence. Source hashes recorded. No real-data split, rehearsal or threshold selection; BCE is used only to probe gradient connectivity, not as an invented paper loss.
Result: Three baseline paths identified; 22 shared variants indexed; parameter counts measured. Perturbing a later event changes an earlier logit. All four shared TGN GRU tensors lack detection-loss gradients; two-step detached-state test confirms the issue. Context fails prefix invariance on short streams. Pytest: **5 passed, 4 xfailed**, no unexpected failures; one upstream index_reduce beta warning.
Interpretation: XFAILs are known unresolved failures, not passing causal certification. They represent two model instances of the batch-future defect, one GRU-gradient defect and one context-prefix defect.
Confounds: These diagnostics establish counterexamples only; no F1/PR-AUC or benefit estimate is produced. Full per-function and CUDA audits remain TODO.
Decision: Preserve original model code for review; repair a canonical path only after the adapted protocol is fixed.
Next action: Resolve graph event-time semantics and differentiable memory scheduling with hand-computed toy tests.
Status: VERIFIED

### C004 — operating documentation

Date: 2026-09-23
Current phase: Reproducibility
Question: Can the work resume without inheriting old benchmark conclusions?
Hypothesis: A single ledger with source-linked artifacts and explicit gates is sufficient for the first session.
Evidence: Root README, `.gitignore`, reset inventory, spec, architecture audit and this ledger.
Experiment/config: No experiment; documentation and generated manifests only.
Result: Commands consistently use main_env; no cloud jobs, dependency installations or TGNN modifications occurred.
Interpretation: Legacy scripts are preserved audit targets; they are not the new canonical runner.
Confounds: No Git metadata exists; use recorded SHA-256 bundle identity. /tmp backup is temporary storage.
Decision: This ledger is the only authoritative research memory; companion files provide specifications and evidence.
Next action: Continue the preregistered adapted baseline work after addressing the stop gates.
Status: VERIFIED

## Table A — paper specification

| Topic | Verified paper statement | Missing detail / status |
|---|---|---|
| Architecture | 3×128, attention, ReLU/.2, node softmax | Exact operator, features and placement unresolved |
| Training | Adam .001, decay .00001, batch 256, 200 epochs, 5 runs | Loss, seeds, split, stopping/threshold absent |
| Rehearsal | 10%, reservoir; VAE is a separate comparator | Unit, schedule, lambda and replay count absent |
| Data | Wazuh enterprise + sandbox APT | No original corpus or attributable release found |

Full field-by-field table: `BASELINE_PAPER_SPEC.md`.

## Table B — existing code

| Path | Parameters measured | Main deviations |
|---|---:|---|
| Shared Pomsathit_GCN | 168453 | Event targets, edge encoder/residuals, weighted loss, batch replay |
| Standalone PomsathitGCN | 168453 | Same task adaptation; also lacks dropout/weight decay |
| V13 alias | 743457 | 2×64, embeddings, time/state, focal loss, test-tuned threshold |
| Shared TGDetect_TGN | 374948 | Adds time, memory, GRU, LayerNorm, changed activation/head; GRU gradient disconnected |

Full side-by-side and dataflow: `ARCHITECTURE_AUDIT.md`. Module/tensor profile and all code-symbol locations: `EXISTING_MODEL_AUDIT.json`.

## Table C — controlled ablations

| Planned group | Configurations | Run status / reason |
|---|---|---|
| Baseline | B | Adapted B COMPLETE, five seeds; exact original task remains unavailable |
| Single factor | F,G,T,M,R,L,N,H | T,M,H COMPLETE; F,G,L,N deferred; R requires M |
| Pairs | All 28 pairs of the initial eight factors | T+M and M+R COMPLETE; other 26 pairs explicitly unrun/dependent in matrix |
| Full / reverse | Full plus eight single-removal arms | Full eight-factor study NOT RUN. Pilot T+M+R and T+M+R+H COMPLETE; pilot remove-T/remove-R comparisons recorded |
| Strong falsification | Age shuffle, constant time, frozen/reset memory, no-op update, synthetic time-only signal | 50 fixed-checkpoint controls COMPLETE plus 5-seed synthetic test. Full order shuffle and mismatched replay deferred |
| Higher order | Remaining subsets up to all 255 nonempty sets of eight | NOT RUN: defer until lower-order effects and compute budget justify them |

`ABLATION_MATRIX.json` mechanically represents all initial-factor singles, 28 pairs,
full and reverse arms plus pilot additions. There are 240 seed rows: 40 completed,
200 unrun. The original snapshot is in `artifacts/causal_pilot_v1/ablation_matrix.json`.
Unsupported factors and R-without-M fail explicitly. Only measured pilot contrasts
are calculated; the full eight-factor study and extra legacy factors remain TODO.

## Table D — initial literature and candidate mechanisms

This is an initial primary-source sweep, not a full implementation review. Source summaries describe each method; benefit, cost, confound and test columns are our hypotheses to test. No candidate has been selected as better.

| Existing component | Candidate / primary source | Mathematical change | Potential benefit | Confound / added cost | Required test |
|---|---|---|---|---|---|
| Custom snapshot operator | [GCN](https://arxiv.org/abs/1609.02907), [GAT](https://arxiv.org/abs/1710.10903), [GraphSAGE](https://arxiv.org/abs/1706.02216), [GIN](https://arxiv.org/abs/1810.00826) | Normalized adjacency, learned attention, sampled aggregation, or sum/MLP | Test structural representation | Operator/expressivity/neighbor budget all differ | Fixed features/graph/depth; parameter and throughput control |
| Time encoding / aggregation | [TGAT](https://arxiv.org/abs/2002.07962) | Time-conditioned attention with functional time encoding | Represent temporal neighborhoods | Time features plus attention; sampled-history storage | Time-only then message-only arms; timestamp destruction |
| Node memory | [TGN](https://arxiv.org/abs/2006.10637) | Memory update plus graph embedding operator | Retain interaction history | Scheduling, truncation and O(nodes×memory width) state | Two-step gradient, reset/freeze, new-node and event-order tests |
| Recurrent/interaction mechanism | [DyRep](https://openreview.net/references/pdf?id=HkvwlRcYm) | Separate topology/activity dynamics via temporal point-process modeling | Model differing interaction timescales | Changes objective and event sampling; extra compute | Keep as separate exploratory model; isolate intensity loss |
| Sinusoidal time channels | [Linear versus sinusoidal encoder](https://openreview.net/forum?id=W6GQvdOGHg) | Change the time map while holding attention fixed | Simpler time representation | Dimension/parameter budget | Match budget, constant/shuffled-time controls |
| Label-biased batch admission | [Reservoir sampling](https://www.cs.umd.edu/~samir/498/vitter.pdf) | Uniform sample retention over a defined training stream | Remove admission bias | Replay unit/capacity must be defined; O(capacity) storage | Empirical inclusion frequencies; no evaluation IDs in buffer |
| Replay selection | [iCaRL](https://arxiv.org/abs/1611.07725) exemplar strategy | Class exemplar selection; inspect full algorithm before porting | Retain representative examples | Class balance and embedding selection introduce supervision/compute | Compare uniform, stratified and representative replay at identical bytes/count |
| Rehearsal | [Deep generative replay](https://arxiv.org/abs/1705.08690) | Generator produces historical pseudo-samples | Different storage tradeoff | Generator training and distribution error | Independent replay family, matched total compute; never inserted into exact baseline |
| APT temporal graph design | [CONTINUUM v2](https://arxiv.org/abs/2501.02981v2) | Spatiotemporal graph autoencoder; federated setting | Explore sequence/structure information | Reconstruction target, federation and cryptography add multiple factors | First isolate centralized temporal encoder and match prediction target |
| Graph construction / compression | [GraphDART](https://arxiv.org/abs/2501.02796) | Distill provenance graph | Potential efficiency improvement | Information loss and preprocessing compute | Train-only construction; account for preprocessing cost |
| Online drift handling | [METANOIA v2](https://arxiv.org/abs/2501.00438v2) | Incremental provenance detection with pseudo edges/state/path processing | Study drift adaptation | Multiple mechanisms and anomaly target differ | Freeze target/labels and evaluate drift/replay policy independently |

CONTINUUM and GraphDART appeared in January 2025; METANOIA v2 is dated March 17, 2026. The base paper itself cites CONTINUUM, so it must not be called a later implementation of Pomsathit. The DyRep search exposed a primary review PDF; the OpenReview forum was blocked by a browser challenge. Exact-paper code/supplement/data searches returned no attributable implementation; later citing-paper discovery was recorded in the spec, and its full page returned HTTP 429. Full-text method extraction, online drift alternatives and exhaustive post-2024 search remain TODO.

## Table E — leakage and correctness gates

| Channel | Classification | Test / audit result |
|---|---|---|
| Local datasets | Synthetic, not verified CTU-13 | Exact regenerated-array equality VERIFIED |
| Batch graph | Future-dependent for per-event predictions | Both shared models fail later-event perturbation invariance |
| GRU state | Detached between updates | No gradient to GRU from detection objective |
| Context early event rate | Future-length-dependent | Fails append-only prefix invariance |
| Flow duration/totals at StartTime | Future-dependent under start-time claims | Source finding; availability-time protocol unresolved |
| History features | Unknown temporal validity on unsorted raw files | Source computes state before final sort |
| Scaling | Split-dependent | Train-only fitting inspected; dynamic perturbation test TODO |
| V13 threshold | Test-dependent | Test labels used to optimize threshold VERIFIED from code |
| Batch replay | Training labels control admission | Uniform-stream reservoir claim rejected by source |
| Other state/identity paths | Mixed / unknown | Full checklist and per-function tests TODO |

Full channel list and exact code paths: `ARCHITECTURE_AUDIT.md`.

## Table F — first-session diagnostic evidence (historical)

| Model | Exact configuration | Seeds | Test protocol | F1 | PR-AUC | ROC-AUC | FPR | Latency | Memory | Parameters | CI/SD | Notes |
|---|---|---|---|---|---|---|---|---|---:|---|---|
| POMSATHIT_PAPER_SPEC | Blocked skeleton | 0 | None | — | — | — | — | — | — | — | — | No executable architecture yet |
| Shared legacy GNN | Existing source hash; 3-event CPU probe | 1 diagnostic seed | Shape/gradient/causality only | — | — | — | — | — | — | 168453 | — | Not performance evidence |
| Shared legacy TGN | Existing source hash; 3-event CPU probe | 1 diagnostic seed | Shape/gradient/causality only | — | — | — | — | — | — | 374948 | — | Known failures; not performance evidence |

## Environment and reproducibility

User-selected interpreter: `/home/mahesh/main_env/bin/python` (Python 3.10.20). Installed versions: torch 2.11.0+cpu, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, Modal 1.4.3, pytest 9.1.1. No PyG/DGL. Torch has no CUDA runtime/cuDNN in this environment. CPU tensor and repeat-seed smoke tests passed. OS, compiler/GPU-tool outputs, PDF/data hashes and final source bundle hash are in `ENVIRONMENT_AUDIT.json`.

Local smoke is deliberately distinct from old Modal containers (Python 3.11/torch 2.4.0). No claim of environment equivalence or CUDA determinism is made. Before heavy runs, verify current Modal GPU syntax/options, pin or record the selected environment, run GPU smoke, persist artifacts and fix one efficiency-comparison GPU class. No cloud experiment ran in this first session because the scientific gates failed first.

## First-session status

The following is the archived first-session state; it is superseded for the new
adapted path by C005–C009 and the second-session result below. Legacy failures and
the exact-paper block remain valid.

```text
CLEANUP:         PASS
PAPER SPEC:      PASS (extracted); exact reproduction BLOCKED
BASELINE CODE:   BLOCKED (canonical skeleton created and tested)
ARCHITECTURE:    PASS (initial map and duplicate audit); deeper audit TODO
LEAKAGE AUDIT:   FAIL (verified counterexamples; full coverage TODO)
WEB RESEARCH:    PASS (initial sweep); full review TODO
TGNN TEST:       NOT STARTED (comparison); CPU correctness probes completed
```

DONE: cleaned and flattened workspace; backed up and hash-verified originals; independently reread the paper; corrected VAE attribution; mapped code; identified all three literature aliases; created blocked canonical skeleton and manifest; measured legacy shapes/parameters; isolated fixtures; started literature sweep; ran main_env tests.

TODO: preregister adapted protocol; authenticate inputs; implement canonical baseline/operator and architecture acceptance tests; repair causal graph/features and memory learning schedule; test replay; generate full factor manifest; build fresh controlled harness; verify Modal runtime; run raw-feature controls then baseline before temporal comparisons.

BLOCKED: exact data/task reconstruction and exact neural/replay choices are unavailable; local source NPZs are synthetic; legacy per-event causal and learning-state gates fail. These activate §18 of `ASTRA_FROM_SCRATCH_PROMPT.md` (stop rather than push forward on unavailable data, future information, or an ambiguous baseline). This is a scientific stop, not an outstanding permission request.

Files changed: `.labtest/harness.py` only among original executable files; fresh ledger replaces the archived initial ledger. Added `.gitignore`, README, specification, canonical skeleton/manifest, architecture audit, environment/model audit scripts and manifests, foundation tests. Both PDFs, all 15 NPZs, `sk_stub.py` and all eight original root Python files are preserved. Detailed removed-path list and original hashes: `RESET_INVENTORY.json`.

Historical next action (now completed for the pilot): register POMSATHIT_ADAPTED,
verify sources, implement the canonical baseline and causal harness. The current
next action is the untouched-capture protocol at the top of this ledger.

## Table F — second-session real-data pilot evidence

All neural rows: five seeds, scenario 6 sample of 16384 records, chronological
60/20/20, three epochs, validation-only threshold. Values are percent mean ± sample
SD. Exact arm configurations and source hashes: `artifacts/causal_pilot_v1/run_manifest.json`.
Latency, state/replay memory, parameter counts and full metrics: `TGNN_PILOT_REPORT.md`.

| Model | F1 | PR-AP | Notes |
|---|---:|---:|---|
| Adapted B | 83.80 ± 3.43 | 88.33 ± 3.07 | Common causal ego graph |
| B+T | 89.84 ± 3.99 | 93.96 ± 0.54 | Numeric age only |
| B+M | 77.78 ± 1.62 | 90.25 ± 1.95 | Fixed persistent update |
| B+M+R | 75.52 ± 3.13 | 86.11 ± 6.54 | GRU replaces fixed update |
| B+T+M | 86.91 ± 2.46 | 91.28 ± 2.72 | Time plus fixed memory |
| Pilot TGNN (B+T+M+R) | 84.33 ± 4.07 | 90.34 ± 2.88 | Not full legacy TG-Detect |
| B+H | 81.05 ± 5.81 | 88.28 ± 3.46 | Reservoir rehearsal |
| Pilot TGNN+H | 84.37 ± 3.22 | 92.37 ± 1.57 | Stale-context replay disclosed |
| Logistic regression | 83.64 ± 0.00 | 80.93 ± 0.00 | Identical deterministic fits |
| Gradient-boosted trees | 97.96 ± 0.00 | 96.06 ± 0.00 | Identical deterministic fits |

```text
DATA:           PASS for direct-source sampled CTU-13 pilot; original Wazuh unavailable
ADAPTED CODE:   PASS; exact paper implementation remains BLOCKED
CAUSAL CHECKS:  PASS for new path; 4 known legacy failures preserved
TGNN PILOT:    COMPLETE (40 neural runs, 10 feature fits, 50 interventions)
RESULT AUDIT:   PASS (hashes, thresholds, metric recomputation, loaded profiles)
FULL STUDY:    NOT COMPLETE (unseen captures, capacity controls, full factors, Modal)
```
