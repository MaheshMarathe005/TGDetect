# Detailed source inventory

Every source file below was read in full. Python entries include full module
docstrings, class/function names, per-definition docstrings, imports and path
references in `CODEBASE_INVENTORY.json`. The pre-move scan is preserved in
`CODEBASE_INVENTORY.before.json`. No cloud jobs were launched.

Temporary browser checks and the mock backend are documented in the removal
manifest rather than listed as retained files.

Columns give current path, lines and structural purpose. Sections distinguish
the active code from historical packages and frozen experiment source.

## Active Source

| Source | Lines | Purpose |
| --- | ---: | --- |
| `research/support/harness.py` | 100 | Offline validation of modal_lab.py. |
| `web_app/backend/__init__.py` | 1 | TG-Detect research and inference dashboard. |
| `web_app/backend/backend.py` | 537 | Artifact-backed dashboard, exact checkpoint loading and causal streaming inference. |
| `web_app/backend/cadets.py` | 435 | CADETS V3 investigation API. Separate from the legacy 72D Sysmon runtime. |
| `web_app/backend/graph_datasets.py` | 370 | Persistent uploaded-log graphs, scored by the verified V3 checkpoint. |
| `web_app/backend/teammate_api.py` | 459 | Teammate dashboard contracts backed by our DEDALE artifacts and checkpoints. |
| `web_app/frontend/eslint.config.mjs` | 50 | Frontend shared hook/service/configuration |
| `web_app/frontend/next-env.d.ts` | 7 | Frontend shared hook/service/configuration |
| `web_app/frontend/next.config.ts` | 14 | Frontend shared hook/service/configuration |
| `web_app/frontend/postcss.config.mjs` | 5 | Frontend shared hook/service/configuration |
| `web_app/frontend/tailwind.config.ts` | 64 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/app/globals.css` | 333 | Frontend page/layout |
| `web_app/frontend/src/app/layout.tsx` | 41 | Frontend page/layout |
| `web_app/frontend/src/app/page.tsx` | 5 | Frontend page/layout |
| `web_app/frontend/src/app/legacy/layout.tsx` | 14 | Frontend page/layout |
| `web_app/frontend/src/app/legacy/page.tsx` | 445 | Frontend page/layout |
| `web_app/frontend/src/components/cadets/AttackInvestigation.tsx` | 925 | Frontend UI component |
| `web_app/frontend/src/components/cadets/CadetsConsole.tsx` | 1315 | Frontend UI component |
| `web_app/frontend/src/components/cadets/DetectionGraph.tsx` | 913 | Frontend UI component |
| `web_app/frontend/src/components/cadets/console.module.css` | 1438 | Frontend UI component |
| `web_app/frontend/src/components/cadets/detection.module.css` | 498 | Frontend UI component |
| `web_app/frontend/src/components/cadets/investigation.module.css` | 702 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/analytics/AnalyticsPage.tsx` | 672 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/artifacts/ArtifactsPage.tsx` | 300 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/chains/ChainsPage.tsx` | 510 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/datasets/DatasetsPage.tsx` | 974 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/events/EventsPage.tsx` | 582 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/graph/AttackGraphCanvas.tsx` | 506 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/graph/GraphPage.tsx` | 299 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/graph/NodeExplanationPanel.tsx` | 247 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/graph/TriageRail.tsx` | 125 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/model/ModalPanels.tsx` | 67 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/model/ModelPage.tsx` | 658 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/overview/OverviewPage.tsx` | 848 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/shared/BackgroundMotion.tsx` | 177 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/shared/TGDetectLoader.tsx` | 142 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/shared/legends.tsx` | 154 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/shared/pills.tsx` | 352 | Frontend UI component |
| `web_app/frontend/src/components/tgdetect/shared/temporal-graph-viz.tsx` | 591 | Frontend UI component |
| `web_app/frontend/src/components/ui/accordion.tsx` | 66 | Frontend UI component |
| `web_app/frontend/src/components/ui/alert-dialog.tsx` | 157 | Frontend UI component |
| `web_app/frontend/src/components/ui/alert.tsx` | 66 | Frontend UI component |
| `web_app/frontend/src/components/ui/aspect-ratio.tsx` | 11 | Frontend UI component |
| `web_app/frontend/src/components/ui/avatar.tsx` | 53 | Frontend UI component |
| `web_app/frontend/src/components/ui/badge.tsx` | 46 | Frontend UI component |
| `web_app/frontend/src/components/ui/breadcrumb.tsx` | 109 | Frontend UI component |
| `web_app/frontend/src/components/ui/button.tsx` | 59 | Frontend UI component |
| `web_app/frontend/src/components/ui/calendar.tsx` | 213 | Frontend UI component |
| `web_app/frontend/src/components/ui/card.tsx` | 92 | Frontend UI component |
| `web_app/frontend/src/components/ui/carousel.tsx` | 241 | Frontend UI component |
| `web_app/frontend/src/components/ui/chart.tsx` | 353 | Frontend UI component |
| `web_app/frontend/src/components/ui/checkbox.tsx` | 32 | Frontend UI component |
| `web_app/frontend/src/components/ui/collapsible.tsx` | 33 | Frontend UI component |
| `web_app/frontend/src/components/ui/command.tsx` | 184 | Frontend UI component |
| `web_app/frontend/src/components/ui/context-menu.tsx` | 252 | Frontend UI component |
| `web_app/frontend/src/components/ui/dialog.tsx` | 143 | Frontend UI component |
| `web_app/frontend/src/components/ui/drawer.tsx` | 135 | Frontend UI component |
| `web_app/frontend/src/components/ui/dropdown-menu.tsx` | 257 | Frontend UI component |
| `web_app/frontend/src/components/ui/form.tsx` | 167 | Frontend UI component |
| `web_app/frontend/src/components/ui/hover-card.tsx` | 44 | Frontend UI component |
| `web_app/frontend/src/components/ui/input-otp.tsx` | 77 | Frontend UI component |
| `web_app/frontend/src/components/ui/input.tsx` | 21 | Frontend UI component |
| `web_app/frontend/src/components/ui/label.tsx` | 24 | Frontend UI component |
| `web_app/frontend/src/components/ui/menubar.tsx` | 276 | Frontend UI component |
| `web_app/frontend/src/components/ui/navigation-menu.tsx` | 168 | Frontend UI component |
| `web_app/frontend/src/components/ui/pagination.tsx` | 127 | Frontend UI component |
| `web_app/frontend/src/components/ui/popover.tsx` | 48 | Frontend UI component |
| `web_app/frontend/src/components/ui/progress.tsx` | 31 | Frontend UI component |
| `web_app/frontend/src/components/ui/radio-group.tsx` | 45 | Frontend UI component |
| `web_app/frontend/src/components/ui/resizable.tsx` | 56 | Frontend UI component |
| `web_app/frontend/src/components/ui/scroll-area.tsx` | 58 | Frontend UI component |
| `web_app/frontend/src/components/ui/select.tsx` | 185 | Frontend UI component |
| `web_app/frontend/src/components/ui/separator.tsx` | 28 | Frontend UI component |
| `web_app/frontend/src/components/ui/sheet.tsx` | 139 | Frontend UI component |
| `web_app/frontend/src/components/ui/sidebar.tsx` | 726 | Frontend UI component |
| `web_app/frontend/src/components/ui/skeleton.tsx` | 13 | Frontend UI component |
| `web_app/frontend/src/components/ui/slider.tsx` | 63 | Frontend UI component |
| `web_app/frontend/src/components/ui/sonner.tsx` | 25 | Frontend UI component |
| `web_app/frontend/src/components/ui/switch.tsx` | 31 | Frontend UI component |
| `web_app/frontend/src/components/ui/table.tsx` | 116 | Frontend UI component |
| `web_app/frontend/src/components/ui/tabs.tsx` | 66 | Frontend UI component |
| `web_app/frontend/src/components/ui/textarea.tsx` | 18 | Frontend UI component |
| `web_app/frontend/src/components/ui/toast.tsx` | 129 | Frontend UI component |
| `web_app/frontend/src/components/ui/toaster.tsx` | 35 | Frontend UI component |
| `web_app/frontend/src/components/ui/toggle-group.tsx` | 73 | Frontend UI component |
| `web_app/frontend/src/components/ui/toggle.tsx` | 47 | Frontend UI component |
| `web_app/frontend/src/components/ui/tooltip.tsx` | 61 | Frontend UI component |
| `web_app/frontend/src/hooks/use-mobile.ts` | 15 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/hooks/use-toast.ts` | 194 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/dataset-context.tsx` | 171 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/date-utils.ts` | 22 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/db.ts` | 13 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/model-context.tsx` | 125 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/theme-context.tsx` | 51 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/utils.ts` | 6 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/chart-constants.ts` | 52 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/constants.ts` | 447 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/formatters.ts` | 367 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/types.ts` | 742 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/api/client.ts` | 133 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/graph/causality.ts` | 491 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/graph/hooks.ts` | 108 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/graph/layout.ts` | 117 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/graph/types.ts` | 129 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/services/hooks.ts` | 244 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/services/index.ts` | 143 | Frontend shared hook/service/configuration |
| `web_app/frontend/src/lib/tgdetect/services/api/index.ts` | 538 | Frontend shared hook/service/configuration |
| `web_app/model/__init__.py` | 1 | Model definitions required for dashboard checkpoint inference. |
| `web_app/model/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `web_app/model/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `web_app/model/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `web_app/model/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `web_app/model/preprocessing.py` | 89 | Inference preprocessing copied from research/dedale_data.py. |
| `web_app/model/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `web_app/model/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `web_app/model/windows_apt_models.py` | 16 | Windows input dimension adapter; preserves frozen 3x128 GAT and temporal branches. |
| `web_app/backend/scripts/download_cadets.py` | 41 | Download the exact trained V3 bundle; no GPU container is started. |
| `web_app/backend/scripts/prepare_cadets.py` | 95 | Verify downloaded V3 artifacts and atomically build the investigation index. |
| `datasets/raw/Windows-APT 2025 A Dataset for APT-Inspired Attack/scripts/load_and_merge_logs.py` | 122 | Windows-APT 2025 helper: load CSV logs and merge per-period files. |
| `datasets/raw/Windows-APT 2025 A Dataset for APT-Inspired Attack/scripts/parse_attack_mapping_and_filter.py` | 154 | Windows-APT 2025 helper: filter scenarios from scenario_manifest.csv and (optionally) filter logs. |
| `research/audit_component_results.py` | 86 | Post-run completeness, metric, checkpoint and frozen-anchor verification. |
| `research/audit_environment.py` | 131 | Read-only source/data audit; generated comparison fixtures live in a temp dir. |
| `research/audit_matched_memory_results.py` | 86 | Verify matched budgets, saved model hashes and independently computed metrics. |
| `research/audit_memory_horizon.py` | 92 | Read-only data/model diagnostics for the long-history hypothesis; no training. |
| `research/audit_models.py` | 126 | Tiny CPU diagnostics of preserved models; never runs their training entrypoints. |
| `research/audit_node_memory_v3_results.py` | 92 | Independently verify downloaded V3/previous predictions and write a report. |
| `research/audit_sliding_weaknesses.py` | 140 | Reproduce structural weaknesses; synthetic probes are not accuracy results. |
| `research/audit_temporal_cnn_results.py` | 75 | Independent audit of the completed temporal-CNN study. |
| `research/audit_windows_apt_results.py` | 30 | Independent local saved-prediction, split, checkpoint and lineage audit. |
| `research/causal_data.py` | 139 | Verified flow-completion pilot data; no legacy feature/cache imports. |
| `research/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/check_windows_apt_transfer.py` | 99 | Frozen DEDALE-checkpoint compatibility test on Windows-APT CSV telemetry. |
| `research/conftest.py` | 10 | Allow research tests to import scripts and the application from either directory. |
| `research/darpa_pilot_data.py` | 77 | Bounded, label-blind DARPA E3 event sample and independent label audit. |
| `research/data_gen_v6.py` | 175 | TG-Detect v10 Formal OOD Generator (Modal) ────────────────────────────────────────────────────────────────────── Generates 1,000,000+ events spanning a 14-day continuous timeline. Includes formalized structural topological mutations: - Standard APT (0 mutations) - Moderate OOD (+1 topological mutation/insertion) - Extreme OOD (>=3 topological mutations/shuffling) |
| `research/dedale_data.py` | 423 | Build a labeled, time-ordered DEDALE Windows-host dataset (winlogbeat/Sysmon) + causal ego-graphs on the Modal Volume, for the GNN-vs-TGNN component study. |
| `research/download_dedale_real.py` | 97 | Resume the complete DEDALE Windows host dataset and verify publisher checksums. |
| `research/download_orthrus_evaluation_data.py` | 50 | Download only checksum-verified evaluation inputs from the study volume. |
| `research/estimate_l4_budget.py` | 120 | L4 GPU budget + statistical-power estimator for the DEDALE (or any) APT study. |
| `research/evaluate_orthrus_checkpoints.py` | 103 | Evaluate saved CADETS weights on CPU without changing frozen training code. |
| `research/fetch_pilot_data.py` | 42 | Fetch the two registered textual CTU captures. |
| `research/matched_memory_training.py` | 246 | Equal additional-training comparison from shared per-seed QAC checkpoints. |
| `research/modal_bench_real.py` | 526 | TG-Detect vs Pomsathit (2025) — honest head-to-head on REAL data (CTU-13). |
| `research/modal_common.py` | 993 | Shared definitions for every TG-Detect Modal app. |
| `research/modal_darpa_pilot.py` | 123 | Download a bounded DARPA E3 sample directly into a dedicated Modal Volume. |
| `research/modal_data_real.py` | 283 | Real-data acquisition + causal normalisation on Modal. |
| `research/modal_dedale.py` | 416 | Fetch the DEDALE APT dataset (doi:10.57745/Y5JLDG) directly onto a Modal Volume. |
| `research/modal_dedale_final.py` | 370 | Final TGNN (TGNN_Q_A_C) trained on the DEDALE host 15:1 reservoir-subsampled dataset in /prepared/dedale_host. This is NOT the full-unsubsampled build and should not be described as a ~365:1 imbalance experiment. |
| `research/modal_dedale_mem.py` | 538 | Three-time-scale node-memory experiment on the DEDALE host 15:1 reservoir build. |
| `research/modal_dedale_mem_v3.py` | 562 | V3 memory fine-tuning on the existing DEDALE host dataset. |
| `research/modal_dedale_train.py` | 183 | Two-GPU DEDALE host training: GNN+rehearsal vs the full TGNN hybrid, with a threshold-honest evaluation that separates RANKING quality from fixed-threshold F1. |
| `research/modal_export_preprocessing.py` | 107 | Recover the exact missing training scaler and metadata from source telemetry. |
| `research/modal_kairos_comparison.py` | 152 | Prepare shared provenance data, then run each matched seed on one Modal L4. |
| `research/modal_lab.py` | 2794 | TG-Detect LAB — a *deployed*, warm Modal app you trigger repeatedly. |
| `research/modal_matched_memory.py` | 48 | Deploy then spawn three independent matched-seed GPU jobs. |
| `research/modal_matched_time_rehearsal.py` | 369 | One Modal L4 invocation for all matched 200-epoch B+H versus B+T+H runs. |
| `research/modal_orthrus_checkpoint_evaluation.py` | 43 | CPU-only evaluation of interrupted full-CADETS training checkpoints. |
| `research/modal_orthrus_study.py` | 130 | Durable acquisition of the official full E3 database, isolated from prior pilots. |
| `research/modal_publication_analysis.py` | 47 | Read completed cloud outputs and score diagnostics without a GPU. |
| `research/modal_publication_pdf.py` | 32 | Compile the editable IEEE manuscript on CPU only; no GPU reservation. |
| `research/modal_publication_study.py` | 120 | Explicit timeout reservations for the authorized eight-GPU-hour study. |
| `research/modal_sliding_node_mem.py` | 279 | Train the new sliding model and unchanged ORTHRUS on fukatkaamsathi. |
| `research/modal_sliding_upgrade.py` | 35 | Smoke-test and time the isolated sliding upgrade on the authorized Modal profile. |
| `research/modal_sliding_v2_benchmark.py` | 90 | Bounded read-only L4 latency check using the existing prepared CADETS prefix. |
| `research/modal_sliding_v3_benchmark.py` | 64 | Bounded L4 validation of tensor-ring TGNN on existing real prepared inputs. |
| `research/modal_sliding_v3_profile.py` | 34 | Run the paired latency experiment on the existing prepared real data. |
| `research/modal_sliding_v4.py` | 125 | V4-only training on the exact saved AttnShort corpus; bounded, unique jobs. |
| `research/modal_temporal_cnn.py` | 37 | One L4 allocation; benchmarked concurrency among independent training runs. |
| `research/modal_tgnn_components.py` | 42 | One resumable L4 job for the registered 60-run study. |
| `research/modal_three_arm_five.py` | 87 | Three isolated five-epoch CADETS arms, concurrently on the authorized profile. |
| `research/modal_train_v16.py` | 470 | TG-Detect v16: The Apex Architecture Showdown ────────────────────────────────────────────────────────────────────── Compares V16_Apex (Universal Encoder + GRL + CausalHTAConv + MemBank + Rehearsal) against Literature (Pomsathit 2025). Run: modal run modal_train_v16.py |
| `research/modal_v3_accuracy.py` | 51 | Isolated five-epoch V3 versus ORTHRUS on the existing official CADETS split. |
| `research/modal_v3_vs_orthrus.py` | 82 | Fresh paired L4 inference comparison on identical CADETS event prefixes. |
| `research/modal_v4_cnn_singlehead.py` | 137 | Deployed bounded L4 training: causal CNN + single-head V4/V3 hybrid. |
| `research/modal_v4_fast_restart.py` | 142 | One-GPU V4 execution-only recovery within the existing timeout allowance. |
| `research/modal_v4_v3_hybrid.py` | 127 | One detached, bounded L4 run: V4 temporal/episodic model + V3 encoder/state. |
| `research/modal_windows_apt.py` | 67 | ZIP-on-Volume staging, CPU verification, then one L4 allocation. |
| `research/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/node_memory_v4.py` | 128 | Experimental V4: independent endpoint admission and learned wall-time decay. |
| `research/package_windows_apt.py` | 34 | Freeze exact code/protocol/prepared tensors and raw canonical CSV into uploaded ZIP. |
| `research/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/profile_v4_last_read_cpu.py` | 52 | Local CPU-only microbenchmark; does not launch or modify Modal jobs. |
| `research/report_causal_pilot.py` | 199 | Verify completed pilot artifacts and generate a report without rerunning training. |
| `research/report_tgnn_components.py` | 87 | Verify and preserve the reference; render every registered component result. |
| `research/run_causal_pilot.py` | 304 | Reproducible bounded CPU pilot. No legacy/Modal app imports or test tuning. |
| `research/run_darpa_pilot.py` | 112 | Matched DARPA E3 association-label pilot; new weights and saved artifacts. |
| `research/run_matched_time_rehearsal.py` | 191 | Matched B+H versus B+T+H at the paper-stated 200 epochs. |
| `research/run_orthrus_checkpoint_local.py` | 84 | Evaluate and trace all saved checkpoints sequentially on local CPU. |
| `research/run_temporal_cnn.py` | 371 | Independent CUDA workers plus a measured, single-L4 experiment scheduler. |
| `research/run_tgnn_components.py` | 274 | Frozen 200-epoch component experiment; invoked by the Modal L4 wrapper. |
| `research/run_windows_apt.py` | 215 | Windows-APT weak-label matched study. Independent workers share exactly one L4. |
| `research/sync_darpa_pilot.py` | 34 | Fetch durable pilot results and verify saved checkpoint hashes. |
| `research/sync_kairos_comparison.py` | 61 | Read durable KAIROS comparison status; optionally download and audit results. |
| `research/sync_orthrus_checkpoint_results.py` | 58 | Collect and report the interrupted 20/18/13 epoch checkpoint comparison. |
| `research/sync_orthrus_study.py` | 95 | Collect committed full CADETS study outputs and rebuild the result table. |
| `research/sync_temporal_cnn.py` | 95 | Inspect/download the CNN study and independently verify all completed runs. |
| `research/sync_tgnn_components.py` | 61 | Read progress or download/verify the running study without starting GPU jobs. |
| `research/sync_windows_apt.py` | 22 | Download only new/changed durable Windows-APT artifacts from Modal Volume. |
| `research/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/temporal_toy.py` | 49 | Learning check: current query is identical; only previous cue predicts label. |
| `research/test_cadets_dashboard.py` | 161 | Regression tests: test_summary_reproduces_artifact_and_full_dataset, test_events_are_aligned_and_paginated_without_invented_train_scores, test_explanations_reproduce_node_scores_and_chain_uses_no_labels, test_invalid_requests_and_missing_entities, test_event_details_expose_actual_tokenized_endpoint_attributes, test_real_v3_replay_matches_saved_gpu_scores_and_resets, test_constructed_chains_are_directed_chronological_and_label_free, test_detection_graph_predictions_ips_and_observed_connections |
| `research/test_causal_pilot.py` | 178 | Regression tests: test_non_transport_fields_cannot_become_ports, test_hand_computed_attention_and_padding, test_baseline_architecture_and_no_temporal_modules, test_future_invariance_and_eval_batch_partition, test_graph_prefix_equal_times_and_new_nodes, test_equal_time_memory_commit_and_self_endpoint, test_gru_all_tensors_receive_detection_gradient_and_update, test_scaling_excludes_eval_and_memory_interface_excludes_labels, test_replay_does_not_mutate_live_state_and_rejects_eval, test_reservoir_inclusion_frequency, test_disabled_factors_cannot_activate_and_memory_negative_controls, test_threshold_is_validation_only_and_tie_rule |
| `research/test_darpa_pilot.py` | 15 | Regression tests: test_label_uuid_time_do_not_enter_features, test_strictly_past_history |
| `research/test_dashboard.py` | 99 | Regression tests: test_real_window_graph_and_feature_context, test_actual_checkpoint_inference_matches_saved_predictions_and_sessions_isolate, test_invalid_windows_models_and_thresholds, test_matched_training_fields_and_exact_source_artifacts, test_raw_ingestion_requires_verified_training_scaler_and_rejects_time_reversal |
| `research/test_gat_slice_audit.py` | 41 | Regression tests: test_chunk_and_segment_invariance |
| `research/test_gat_study.py` | 169 | Regression tests: test_masks_match_existing_provenance_graph, test_causal_padding_and_finite_gradients, test_role_head_content_isolated_before_output_projection, test_degree_audit_separates_query_and_history, test_gat2_full_model_is_existing_v3_and_common_weights_match, test_full_model_batch_replay_gradients_and_degree_hook, test_training_copy_only_changes_documented_imports_and_source_list, test_full_training_calibration_minority_and_timing_runner, test_colab_pilot_full_run_with_frozen_encoder |
| `research/test_graph_uploads.py` | 113 | Regression tests: test_upload_real_inference_and_dynamic_connected_neighborhoods, test_malicious_roots_keep_connected_benign_context, test_parsing_formats_errors_and_ignores_supplied_verdicts, test_upload_losses_match_original_gpu_prefix, test_timestamp_offsets_and_event_attributes_do_not_leak_from_future |
| `research/test_kairos_comparison.py` | 197 | Regression tests: test_decay_scan_matches_sequential_values_and_gradients, test_v4_causality_chunk_invariance_and_time_decay, test_v4_task_gradient_reaches_decay_and_endpoint_gates, test_v4_self_edge_admission_budget_counts_actual_writes, test_ecs_guid_no_label_features_and_no_pid_merging, test_kairos_no_current_or_future_message_leak_and_training, test_queue_online_never_changes_after_future_window, test_full_pipeline_smoke |
| `research/test_node_memory_v3.py` | 130 | Regression tests: test_scan_matches_sequential_values_and_gradients, test_context_batch_invariance_and_future_independence, test_task_gradient_reaches_writes_and_controller, test_zero_gate_baseline_and_replay_state_isolation, test_full_encoder_batch_invariance, test_penalty_counts_candidate_once |
| `research/test_orthrus_universal.py` | 136 | Regression tests: test_missing_and_mixed_types, test_permutation_and_batch_padding_invariance, test_no_current_relation_or_label_in_query, test_v4_kv_state_gradients_and_reload, test_official_model_loss_and_no_current_relation_leak, test_matched_runner_synthetic, test_pinned_tracing_and_window_evaluation |
| `research/test_research_foundation.py` | 121 | Regression tests: test_paper_contract_and_persisted_manifest, test_missing_spec_never_silently_becomes_an_exact_model, test_fixture_default_is_outside_preserved_dataset, test_existing_baseline_shape_and_binary_head_equivalence, test_legacy_model_does_not_see_later_event_features, test_legacy_gru_receives_gradient_across_two_detached_steps, test_context_is_invariant_to_appending_future_events |
| `research/test_sliding_node_mem.py` | 227 | Regression tests: test_default_detector_budget_matches_unmodified_orthrus_exactly, test_cached_tokenization_preserves_values_and_encoder_gradients, test_packed_snapshot_is_cpu_only_and_independent, test_memory_commit_updates_only_touched_rows_without_full_copy, test_optimized_forward_gradients_and_updates_match_original, test_no_gru_missing_fields_and_gradient_paths, test_stream_predictions_independent_of_batch_boundaries, test_current_relation_and_future_events_cannot_change_prediction, test_provenance_roles_and_cnn_are_causal, test_top_half_novel_retention_bounds_and_snapshot_resume, test_invalid_inputs_do_not_advance_stream_and_self_edges_work, test_rare_event_survives_even_when_attention_does_not_select_it, test_training_detach_and_reset_across_optimizer_steps, test_runner_mid_epoch_resume_matches_uninterrupted_weights |
| `research/test_sliding_upgrade.py` | 197 | Regression tests: test_parameter_budget_unchanged, test_missing_fields_and_empty_windows_finite, test_path_order_and_field_order, test_current_and_future_relation_causality, test_multiple_windows_equal_single_event_inference, test_gradients_compression_and_memory_and_resume, test_local_rarity_not_global, test_scoring_monotone_and_hub_penalty, test_query_cache_canonical_and_target_masked, test_prefetch_preserves_order_and_empty_stream, test_end_to_end_runner_and_calibration, test_sparse_memory_large_identity_space, test_vector_window_planner_matches_sequential_with_self_edges, test_exact_orthrus_frontend_and_graph_feed |
| `research/test_sliding_v2.py` | 287 | Regression tests: test_dense_cnn_outputs_and_gradients, test_compatible_predictions_gradients_and_optimizer_updates, test_causality_and_batch_invariance, test_balanced_history_keeps_quiet_endpoint, test_balanced_vectorized_windows_match_sequential, test_order_sensitive_encoder_and_field_permutation, test_inference_cache_eviction_invalidation_and_gradients, test_compaction_preserves_predictions_and_snapshot, test_partial_compaction_preserves_shared_rows_and_finishes_later, test_cross_batch_credit_survives_unroll, test_planned_memory_scan_matches_reference_and_gradients, test_elapsed_memory_decay_and_self_edges, test_training_unroll_two_updates, test_regular_traffic_does_not_erase_alert, test_retention_bounded_without_sticky_selection_and_floor, test_columnar_scores_and_no_topk_alert_cap, test_reject_mismatched_stream_config, test_runner_checkpoint_selection_and_interrupted_resume |
| `research/test_sliding_v3.py` | 330 | Regression tests: test_exact_negative_pair_index_preserves_samples, test_frozen_encoder_invalidation_and_training, test_causal_attention_padding_and_future_independence, test_temporal_graph_cpu_and_training_fallback, test_static_short_read_sequential_and_gradients, test_offline_encoder_matches_online_and_gradients, test_compatible_sequential_equivalence_against_current_v2, test_chunk_invariance_causality_and_snapshot, test_count_min_matches_sequential_with_collisions, test_training_gradients_and_cross_batch_unroll, test_two_hash_does_not_add_embedding_parameters, test_elapsed_affine_scan_matches_current_v2, test_compatible_gradient_and_update_parity, test_compatible_sticky_selection_reaches_quiet_endpoint, test_corrected_balanced_quota_and_no_sticky_retention, test_sparse_ring_allocation_growth_and_cross_batch_credit, test_preparation_pretraining_runner_and_resume |
| `research/test_sliding_v4.py` | 381 | Regression tests: test_temporal_exact_backends_masks_time_and_gradients, test_dependency_schedule, test_cnn_single_head_masking_causality_and_gradients, test_cnn_single_head_hybrid_batch_parity_and_training, test_hybrid_v3_components_causality_sequential_and_resume, test_hybrid_gradients_through_parallel_memory_and_episodic_unroll, test_cpu_prefetch_preserves_batch_columns_order_and_start, test_cpu_prefetch_propagates_failure_and_closes_producer, test_temporal_last_read_outputs_and_all_gradients, test_execution_migration_rejects_training_and_model_changes, test_sequential_batch_parity_self_edges_and_resume, test_target_and_future_causality, test_cpu_memory_partition_preserves_scores_and_gradients, test_array_recurrence_matches_reference, test_compiled_array_training_matches_interpreted_unroll, test_last_read_matches_full_model_training_unroll, test_gradients_encoder_memory_and_repeated_optimizer_steps, test_archive_exact_bytes_idempotence_and_causal_retrieval, test_training_checkpoint_resume_matches_uninterrupted |
| `research/test_teammate_dashboard.py` | 72 | Regression tests: test_copied_ui_has_real_dataset_graph_and_score_mapping, test_architecture_history_and_evaluation_match_artifacts, test_uploaded_stream_uses_checkpoint_and_never_invents_ground_truth, test_ui_upload_rejects_arbitrary_paths_and_invalid_chronology |
| `research/test_temporal_cnn.py` | 187 | Regression tests: test_convolution_strictly_causal_and_padding, test_cnn_event_axis_and_current_last_order_hand_computed, test_short_sequence_gemm_matches_conv1d_outputs_and_all_gradients, test_every_arm_causal_batch_independent_padding_gradient_and_update, test_precomputed_replay_exact_rng_and_training_only, test_shared_initialization_and_zero_cnn_equivalence, test_optimized_training_matches_original_replay_path |
| `research/test_tgnn_components.py` | 154 | Regression tests: test_reference_equivalence_outputs_gradients_and_update, test_all_arms_causality_replay_padding_gradients, test_exact_reservoir_rng_and_snapshots, test_memory_controls_and_backwards_guard, test_training_across_optimizer_steps_and_rehearsal |
| `research/test_tgnn_final.py` | 101 | Regression tests: test_encoder_outputs_gradients_and_state_names_match_original, test_frozen_lookup_and_training_invalidation, test_v3_prefixed_weights_and_model_roundtrip, test_source_manifest_and_standalone_import |
| `research/test_windows_apt.py` | 60 | Regression tests: test_features_exclude_annotations_and_identity, test_missing_fields_ports_and_endpoint_isolation, test_prefix_and_tie_safety, test_scaler_train_only, test_models_gradients_padding_and_batch, test_shared_initialization |
| `research/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/threshold_calibration.py` | 194 | Frozen post-hoc threshold calibration for the Windows-APT weak-label study. |
| `research/trigger.py` | 348 | Thin local client for the deployed TG-Detect lab. |
| `research/trigger_v4_cnn_singlehead.py` | 36 | Trigger the deployed function once; retain its call ID and budget reservation. |
| `research/universal_events.py` | 52 | Schema-independent normalization before the shared V4-KV event encoder. |
| `research/windows_apt_controls.py` | 16 | Feature-only weak-label controls, no model/feature selection based on test. |
| `research/windows_apt_data.py` | 145 | Audited Windows-APT weak-label preparation; never silently treats missing as benign. |
| `research/windows_apt_models.py` | 16 | Windows input dimension adapter; preserves frozen 3x128 GAT and temporal branches. |
| `research/encoder_colab/build_notebook.py` | 226 | Build a standalone notebook with a pinned snapshot of the real V3 encoder. |
| `research/encoder_colab/build_rarity_notebook.py` | 147 | Create the focused follow-up without overwriting the first study notebook. |
| `research/encoder_colab/build_stability_notebook.py` | 122 | Research model, experiment or shared utility |
| `research/encoder_colab/experiment.py` | 691 | Self-contained encoder pilot; bundled with unmodified V3 sources in Colab. |
| `research/encoder_colab/optimizations.py` | 225 | Optional measured execution variants and schema coverage probes. |
| `research/encoder_colab/rarity_followup.py` | 292 | Controlled rarity follow-up. No masked-pretraining recipe in this matrix. |
| `research/encoder_colab/rarity_stability.py` | 255 | Longer-training and count robustness study; two encoder architectures. |
| `research/encoder_colab/test_experiment.py` | 278 | Regression tests: test_original_parity_and_shared_gradients, test_mask_hides_value_kind_number_and_rarity, test_dynamic_rarity_never_deduplicates_final_embedding, test_static_field_cache_and_compact_parity, test_counts_read_before_insert_and_eval_frozen, test_threshold_is_strict_and_bootstrap_is_paired, test_schema_probes_cover_tail_and_order, test_all_arms_end_to_end_and_optional_cpu_paths, test_notebook_cells_compile_and_baseline_snapshot_is_verbatim, test_frozen_reference_counts_are_disjoint_and_shuffling_preserves_marginals, test_tail_scaling_does_not_cap_extreme_evidence, test_followup_all_arms_two_seeds_and_split_exports, test_followup_notebook_compiles, test_vector_counts_match_python_and_collision_overcount, test_stability_two_budgets_and_count_invariance_for_original, test_stability_notebook_compiles |
| `research/gat_study/__init__.py` | 4 | Experimental graph operators; none is promoted into TGNN Final yet. |
| `research/gat_study/build_colab.py` | 198 | Embed a real prepared CADETS prefix and standalone code into one notebook. |
| `research/gat_study/colab_pilot.py` | 217 | Small real-CADETS old-vs-role4 pilot, with frozen common KV encoder. |
| `research/gat_study/diagnostics.py` | 41 | Separate untimed degree audit; avoid GPU->CPU synchronization in benchmarks. |
| `research/gat_study/model.py` | 28 | Substitute only the GAT region of corrected V3 for full-model experiments. |
| `research/gat_study/operators.py` | 86 | Dense nine-slot causal provenance operators with role-aware message content. |
| `research/gat_study/run.py` | 143 | Real prepared provenance experiment with degree audit and full-model timing. |
| `research/gat_study/slice_audit.py` | 81 | Score chronological slices using exact corrected-V3 windows, without labels. |
| `research/gat_study/training.py` | 185 | Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence. |
| `research/kairos_comparison/__init__.py` | 1 | Paper-traced KAIROS / TGNN comparison on shared DEDALE provenance events. |
| `research/kairos_comparison/audit.py` | 81 | Recompute saved held-out metrics and require matching data/features/event IDs. |
| `research/kairos_comparison/data.py` | 310 | DEDALE ECS/Sysmon -> shared provenance messages, with explicit coverage accounting. |
| `research/kairos_comparison/detection.py` | 74 | KAIROS paper §4.3 queues and §4.4 weighted Louvain investigation. |
| `research/kairos_comparison/model.py` | 78 | KAIROS encoder/decoder traced to the authors' CADETS_E3 implementation. |
| `research/kairos_comparison/run.py` | 385 | Train/evaluate KAIROS, original V3 control, and improved V4 on one archive. |
| `research/orthrus_comparison/__init__.py` | 1 | ORTHRUS paper reproduction and structured-event TGNN research. |
| `research/orthrus_comparison/acquisition.py` | 93 | Download only CADETS from the authors' public, uncompressed tar bundle. |
| `research/orthrus_comparison/models.py` | 59 | TGNN V3 relation-reconstruction adapter and proposed V4 with shared KV encoder. |
| `research/orthrus_comparison/official.py` | 56 | Load unmodified model class definitions from the pinned authors' release. |
| `research/orthrus_comparison/prepare_official.py` | 113 | Run pinned upstream preprocessing against a restored full CADETS database. |
| `research/orthrus_comparison/protocol.py` | 41 | Paper split registration, identity policy, and conservative feature contract. |
| `research/orthrus_comparison/report.py` | 62 | Collect real study outputs without turning pending arms into zero scores. |
| `research/orthrus_comparison/run.py` | 182 | Matched reconstruction study on officially featurized CADETS E3 graphs. |
| `research/orthrus_comparison/tracing.py` | 103 | Apply the pinned ORTHRUS dependency tracing algorithm to every study arm. |
| `research/orthrus_comparison/universal.py` | 145 | One shared, bounded-vocabulary encoder for unordered typed key/value sets. |
| `research/orthrus_official/run.sh` | 18 | Experiment/application command entry point |
| `research/orthrus_official/src/__init__.py` | 0 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/config.py` | 647 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database.py` | 23 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/data_utils.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/decoders.py` | 41 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/encoders.py` | 88 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/factory.py` | 149 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/labelling.py` | 110 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/model.py` | 78 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/orthrus.py` | 114 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/provnet_utils.py` | 488 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/temporal.py` | 95 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/__init__.py` | 6 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/cadets_e3.py` | 261 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/cadets_e5.py` | 266 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/clearscope_e3.py` | 246 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/clearscope_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/theia_e3.py` | 244 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/create_database/theia_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/detection/__init__.py` | 3 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/detection/evaluation.py` | 65 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/detection/evaluation_utils.py` | 613 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/detection/node_evaluation.py` | 123 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/detection/orthrus_gnn_testing.py` | 138 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/detection/orthrus_gnn_training.py` | 101 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/featurization/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/featurization/build_feature_word2vec.py` | 146 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/featurization/embed_edges_feature_word2vec.py` | 167 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/preprocessing/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/preprocessing/build_orthrus_graphs.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/triage/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/triage/tracing.py` | 136 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/triage/tracing_methods/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/triage/tracing_methods/depimpact.py` | 201 | Pinned upstream ORTHRUS reference implementation |
| `research/orthrus_official/src/triage/tracing_methods/depimpact_utils.py` | 486 | Pinned upstream ORTHRUS reference implementation |
| `research/publication/__init__.py` | 1 | Reproducible bounded publication study; no work runs on import. |
| `research/publication/analyze.py` | 39 | CPU-only operating-rule diagnostics from preserved edge scores. |
| `research/publication/data.py` | 69 | Immutable CADETS metadata and disclosed KAIROS hierarchical-hash adapter. |
| `research/publication/latency.py` | 83 | Trained-checkpoint inference on identical chronological 1024-event batches. |
| `research/publication/models.py` | 45 | Explicit study baselines and contribution ablations, not invented reproductions. |
| `research/publication/report.py` | 169 | Build manuscript tables and vector figures exclusively from saved results. |
| `research/publication/run.py` | 114 | Run registered detector-level comparisons; preserve measured provenance. |
| `research/publication/sources/kairos_upstream/config.py` | 145 | Research model, experiment or shared utility |
| `research/publication/sources/kairos_upstream/embedding.py` | 139 | Research model, experiment or shared utility |
| `research/publication/sources/kairos_upstream/model.py` | 64 | Research model, experiment or shared utility |
| `research/publication/sources/kairos_upstream/train.py` | 133 | Research model, experiment or shared utility |
| `research/sliding_node_mem/__init__.py` | 4 | Sliding provenance windows, temporal CNN, selective short and long memory. |
| `research/sliding_node_mem/budget.py` | 27 | Require equal detector parameters before either matched training arm starts. |
| `research/sliding_node_mem/cloud_copy.py` | 54 | Copy prepared data directly between the two user-authorized workspaces. |
| `research/sliding_node_mem/encoding.py` | 43 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/sliding_node_mem/evaluate_saved.py` | 86 | Score a saved TGNN sliding checkpoint without running training. |
| `research/sliding_node_mem/model.py` | 413 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/sliding_node_mem/performance.py` | 89 | Disposable L4 comparison against the archived implementation, same data/weights. |
| `research/sliding_node_mem/run.py` | 174 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/sliding_node_mem/sync.py` | 97 | Collect the new experiment into its own directory; never edit old ledgers. |
| `research/sliding_node_mem/transfer.py` | 90 | Copy checksum-verified existing CADETS inputs between authorized Modal profiles. |
| `research/sliding_node_mem_upgrade/__init__.py` | 1 | Isolated experimental upgrade; saved original experiments are immutable. |
| `research/sliding_node_mem_upgrade/collect_three.py` | 51 | Collect original five-epoch arm outputs and render comparable operating points. |
| `research/sliding_node_mem_upgrade/encoding.py` | 108 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/sliding_node_mem_upgrade/model.py` | 454 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/sliding_node_mem_upgrade/orthrus_frontend.py` | 69 | Exact authors' ORTHRUS encoder/graph feed with TGDetect temporal processing. |
| `research/sliding_node_mem_upgrade/pipeline.py` | 61 | Bounded CPU query preparation overlapped with ordered GPU batches. |
| `research/sliding_node_mem_upgrade/run.py` | 196 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/sliding_node_mem_upgrade/scoring.py` | 74 | Benign-only relation calibration and repeated-evidence node scoring. |
| `research/sliding_node_mem_upgrade/smoke.py` | 79 | Real benign-prefix disposable training timing; never reports research accuracy. |
| `research/sliding_node_mem_upgrade/windows.py` | 33 | Vectorized chronological neighborhood planning for many overlapping windows. |
| `research/sliding_node_mem_v2/__init__.py` | 4 | Isolated latency optimizations and corrected sliding-memory experiments. |
| `research/sliding_node_mem_v2/benchmark.py` | 122 | Compare warm streaming latency; synthetic timings are not detection results. |
| `research/sliding_node_mem_v2/encoding.py` | 143 | Sequence-aware value encoding and bounded, weight-versioned inference cache. |
| `research/sliding_node_mem_v2/layers.py` | 27 | Same dense causal convolution weights, evaluated as a small-window GEMM. |
| `research/sliding_node_mem_v2/memory.py` | 64 | Exact affine memory scan with CPU-planned indices and bounded scan depth. |
| `research/sliding_node_mem_v2/model.py` | 393 | V2: exact execution optimizations plus separately selectable detection fixes. |
| `research/sliding_node_mem_v2/run.py` | 216 | Chronological V2 training, resumable unrolls and uncapped calibrated alerts. |
| `research/sliding_node_mem_v2/runtime.py` | 23 | Controlled cyclic GC at stream boundaries, with restoration on exceptions. |
| `research/sliding_node_mem_v2/scoring.py` | 119 | Benign-calibrated anomaly evidence without a top-30 alert cap. |
| `research/sliding_node_mem_v2/tensors.py` | 33 | Bulk CPU metadata conversion and shared-block compressed-token gathers. |
| `research/sliding_node_mem_v2/training.py` | 29 | Bounded chronological unrolls retain cross-batch memory-write gradients. |
| `research/sliding_node_mem_v2/windows.py` | 46 | Balanced endpoint history with chronological, batch-independent sampling. |
| `research/sliding_node_mem_v3/__init__.py` | 3 | Research model, experiment or shared utility |
| `research/sliding_node_mem_v3/ablate.py` | 37 | Run independent three-seed temporal ablations on the official prepared split. |
| `research/sliding_node_mem_v3/benchmark.py` | 95 | Prepared-input benchmark; preprocessing is timed separately from inference. |
| `research/sliding_node_mem_v3/compare_orthrus.py` | 79 | Authors' ORTHRUS architecture with V3's held-out selection/calibration protocol. |
| `research/sliding_node_mem_v3/graphs.py` | 34 | Bounded inference-only CUDA graph cache for a fixed temporal region. |
| `research/sliding_node_mem_v3/memory.py` | 65 | GPU segmented affine scan with optional elapsed-time fast/slow decay. |
| `research/sliding_node_mem_v3/model.py` | 241 | Tensor-ring TGNN. No Python loop over events in the forward path. |
| `research/sliding_node_mem_v3/prepare.py` | 139 | Compile official prepared graphs once into token-ID/event memmaps. |
| `research/sliding_node_mem_v3/pretrain.py` | 84 | Train Word2Vec's skip-gram/negative-sampling objective on benign path text. |
| `research/sliding_node_mem_v3/profile_latency.py` | 97 | Paired real-prefix inference ablations; separate instrumented CUDA profile. |
| `research/sliding_node_mem_v3/rings.py` | 220 | Device-side chronological ring planning, including repeated batch endpoints. |
| `research/sliding_node_mem_v3/run.py` | 185 | Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence. |
| `research/sliding_node_mem_v3/temporal.py` | 24 | Optional full causal attention for short fixed windows, requiring retraining. |
| `research/sliding_node_mem_v3/tokens.py` | 194 | Offline typed token rows and a vectorized memmap/resident token bank. |
| `research/sliding_node_mem_v4/__init__.py` | 2 | TGDetect V4: small encoder, exact temporal attention and causal entity memory. |
| `research/sliding_node_mem_v4/archive.py` | 89 | Persistent byte-preserving event archive, independent of neural admission. |
| `research/sliding_node_mem_v4/array_memory.py` | 155 | Fused CPU recurrence with explicit first-order reverse-mode derivatives. |
| `research/sliding_node_mem_v4/compiled_memory.py` | 117 | Compiled execution of the existing CPU recurrence; no fast-math or stale state. |
| `research/sliding_node_mem_v4/memory.py` | 122 | Prediction-before-write entity memory; independent entities execute together. |
| `research/sliding_node_mem_v4/model.py` | 267 | TGDetect V4: causal RoPE attention, full episodic tokens, surprise entity state. |
| `research/sliding_node_mem_v4/prefetch.py` | 26 | Bounded CPU batch preparation; the consumer alone advances model state. |
| `research/sliding_node_mem_v4/report.py` | 49 | Compare only completed V4 runs with saved, seed-matched AttnShort results. |
| `research/sliding_node_mem_v4/run.py` | 216 | Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence. |
| `research/sliding_node_mem_v4/temporal.py` | 96 | Exact causal attention with rotary positions and per-head elapsed-time bias. |
| `research/tgnn_final/__init__.py` | 5 | TGNN Final: build and validate one component at a time. |
| `research/tgnn_final/_encoding_upgrade.py` | 108 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/tgnn_final/_encoding_v2.py` | 143 | Sequence-aware value encoding and bounded, weight-versioned inference cache. |
| `research/tgnn_final/_universal.py` | 145 | One shared, bounded-vocabulary encoder for unordered typed key/value sets. |
| `research/tgnn_final/encoder.py` | 194 | Offline typed token rows and a vectorized memmap/resident token bank. |
| `research/tgnn_final/model.py` | 31 | Encoder-only foundation for TGNN Final. |
| `web_app/__init__.py` | 1 | TG-Detect application packages. |

## Historical Package

| Source | Lines | Purpose |
| --- | ---: | --- |
| `research/archives/new/dashboard/__init__.py` | 1 | TG-Detect research and inference dashboard. |
| `research/archives/new/dashboard/backend.py` | 537 | Artifact-backed dashboard, exact checkpoint loading and causal streaming inference. |
| `research/archives/new/dashboard/cadets.py` | 435 | CADETS V3 investigation API. Separate from the legacy 72D Sysmon runtime. |
| `research/archives/new/dashboard/graph_datasets.py` | 370 | Persistent uploaded-log graphs, scored by the verified V3 checkpoint. |
| `research/archives/new/dashboard/teammate_api.py` | 459 | Teammate dashboard contracts backed by our DEDALE artifacts and checkpoints. |
| `research/archives/new/dashboard/frontend/eslint.config.mjs` | 50 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/next-env.d.ts` | 7 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/next.config.ts` | 14 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/postcss.config.mjs` | 5 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/tailwind.config.ts` | 64 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/app/globals.css` | 405 | Frontend page/layout |
| `research/archives/new/dashboard/frontend/src/app/layout.tsx` | 41 | Frontend page/layout |
| `research/archives/new/dashboard/frontend/src/app/page.tsx` | 5 | Frontend page/layout |
| `research/archives/new/dashboard/frontend/src/app/legacy/layout.tsx` | 14 | Frontend page/layout |
| `research/archives/new/dashboard/frontend/src/app/legacy/page.tsx` | 445 | Frontend page/layout |
| `research/archives/new/dashboard/frontend/src/components/cadets/AttackInvestigation.tsx` | 926 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/cadets/CadetsConsole.tsx` | 1312 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/cadets/DetectionGraph.tsx` | 874 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/cadets/console.module.css` | 2193 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/cadets/detection.module.css` | 1059 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/cadets/investigation.module.css` | 1035 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/analytics/AnalyticsPage.tsx` | 672 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/artifacts/ArtifactsPage.tsx` | 300 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/chains/ChainsPage.tsx` | 510 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/datasets/DatasetsPage.tsx` | 974 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/events/EventsPage.tsx` | 582 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/graph/AttackGraphCanvas.tsx` | 506 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/graph/GraphPage.tsx` | 299 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/graph/NodeExplanationPanel.tsx` | 247 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/graph/TriageRail.tsx` | 125 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/model/ModalPanels.tsx` | 67 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/model/ModelPage.tsx` | 658 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/overview/OverviewPage.tsx` | 848 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/shared/BackgroundMotion.tsx` | 177 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/shared/TGDetectLoader.tsx` | 142 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/shared/legends.tsx` | 154 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/shared/pills.tsx` | 352 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/tgdetect/shared/temporal-graph-viz.tsx` | 591 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/accordion.tsx` | 66 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/alert-dialog.tsx` | 157 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/alert.tsx` | 66 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/aspect-ratio.tsx` | 11 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/avatar.tsx` | 53 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/badge.tsx` | 46 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/breadcrumb.tsx` | 109 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/button.tsx` | 59 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/calendar.tsx` | 213 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/card.tsx` | 92 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/carousel.tsx` | 241 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/chart.tsx` | 353 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/checkbox.tsx` | 32 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/collapsible.tsx` | 33 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/command.tsx` | 184 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/context-menu.tsx` | 252 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/dialog.tsx` | 143 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/drawer.tsx` | 135 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/dropdown-menu.tsx` | 257 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/form.tsx` | 167 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/hover-card.tsx` | 44 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/input-otp.tsx` | 77 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/input.tsx` | 21 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/label.tsx` | 24 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/menubar.tsx` | 276 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/navigation-menu.tsx` | 168 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/pagination.tsx` | 127 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/popover.tsx` | 48 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/progress.tsx` | 31 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/radio-group.tsx` | 45 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/resizable.tsx` | 56 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/scroll-area.tsx` | 58 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/select.tsx` | 185 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/separator.tsx` | 28 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/sheet.tsx` | 139 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/sidebar.tsx` | 726 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/skeleton.tsx` | 13 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/slider.tsx` | 63 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/sonner.tsx` | 25 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/switch.tsx` | 31 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/table.tsx` | 116 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/tabs.tsx` | 66 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/textarea.tsx` | 18 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/toast.tsx` | 129 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/toaster.tsx` | 35 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/toggle-group.tsx` | 73 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/toggle.tsx` | 47 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/components/ui/tooltip.tsx` | 61 | Frontend UI component |
| `research/archives/new/dashboard/frontend/src/hooks/use-mobile.ts` | 15 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/hooks/use-toast.ts` | 194 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/dataset-context.tsx` | 171 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/date-utils.ts` | 22 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/db.ts` | 13 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/model-context.tsx` | 125 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/theme-context.tsx` | 51 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/utils.ts` | 6 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/chart-constants.ts` | 52 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/constants.ts` | 447 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/formatters.ts` | 367 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/types.ts` | 742 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/api/client.ts` | 133 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/graph/causality.ts` | 491 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/graph/hooks.ts` | 108 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/graph/layout.ts` | 117 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/graph/types.ts` | 129 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/services/hooks.ts` | 244 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/services/index.ts` | 143 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/lib/tgdetect/services/api/index.ts` | 538 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/frontend/src/types/cytoscape-fcose.d.ts` | 4 | Frontend shared hook/service/configuration |
| `research/archives/new/dashboard/runtime/__init__.py` | 1 | Model definitions required for dashboard checkpoint inference. |
| `research/archives/new/dashboard/runtime/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/archives/new/dashboard/runtime/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/archives/new/dashboard/runtime/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/archives/new/dashboard/runtime/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/archives/new/dashboard/runtime/preprocessing.py` | 89 | Inference preprocessing copied from research/dedale_data.py. |
| `research/archives/new/dashboard/runtime/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/archives/new/dashboard/runtime/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/archives/new/dashboard/runtime/windows_apt_models.py` | 16 | Windows input dimension adapter; preserves frozen 3x128 GAT and temporal branches. |
| `research/archives/new/dashboard/scripts/download_cadets.py` | 41 | Download the exact trained V3 bundle; no GPU container is started. |
| `research/archives/new/dashboard/scripts/prepare_cadets.py` | 95 | Verify downloaded V3 artifacts and atomically build the investigation index. |
| `research/archives/new/research/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/archives/new/research/check_windows_apt_v3_transfer.py` | 181 | Frozen CADETS V3 transfer diagnostic against Windows Wazuh weak annotations. |
| `research/archives/new/research/conftest.py` | 10 | Allow research tests to import scripts and the application from either directory. |
| `research/archives/new/research/modal_sliding_v2_benchmark.py` | 90 | Bounded read-only L4 latency check using the existing prepared CADETS prefix. |
| `research/archives/new/research/modal_sliding_v3_benchmark.py` | 64 | Bounded L4 validation of tensor-ring TGNN on existing real prepared inputs. |
| `research/archives/new/research/modal_sliding_v3_profile.py` | 34 | Run the paired latency experiment on the existing prepared real data. |
| `research/archives/new/research/modal_v3_accuracy.py` | 51 | Isolated five-epoch V3 versus ORTHRUS on the existing official CADETS split. |
| `research/archives/new/research/modal_v3_vs_orthrus.py` | 82 | Fresh paired L4 inference comparison on identical CADETS event prefixes. |
| `research/archives/new/research/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/archives/new/research/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/archives/new/research/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/archives/new/research/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/archives/new/research/test_cadets_dashboard.py` | 161 | Regression tests: test_summary_reproduces_artifact_and_full_dataset, test_events_are_aligned_and_paginated_without_invented_train_scores, test_explanations_reproduce_node_scores_and_chain_uses_no_labels, test_invalid_requests_and_missing_entities, test_event_details_expose_actual_tokenized_endpoint_attributes, test_real_v3_replay_matches_saved_gpu_scores_and_resets, test_constructed_chains_are_directed_chronological_and_label_free, test_detection_graph_predictions_ips_and_observed_connections |
| `research/archives/new/research/test_graph_uploads.py` | 113 | Regression tests: test_upload_real_inference_and_dynamic_connected_neighborhoods, test_malicious_roots_keep_connected_benign_context, test_parsing_formats_errors_and_ignores_supplied_verdicts, test_upload_losses_match_original_gpu_prefix, test_timestamp_offsets_and_event_attributes_do_not_leak_from_future |
| `research/archives/new/research/test_orthrus_universal.py` | 136 | Regression tests: test_missing_and_mixed_types, test_permutation_and_batch_padding_invariance, test_no_current_relation_or_label_in_query, test_v4_kv_state_gradients_and_reload, test_official_model_loss_and_no_current_relation_leak, test_matched_runner_synthetic, test_pinned_tracing_and_window_evaluation |
| `research/archives/new/research/test_sliding_v2.py` | 287 | Regression tests: test_dense_cnn_outputs_and_gradients, test_compatible_predictions_gradients_and_optimizer_updates, test_causality_and_batch_invariance, test_balanced_history_keeps_quiet_endpoint, test_balanced_vectorized_windows_match_sequential, test_order_sensitive_encoder_and_field_permutation, test_inference_cache_eviction_invalidation_and_gradients, test_compaction_preserves_predictions_and_snapshot, test_partial_compaction_preserves_shared_rows_and_finishes_later, test_cross_batch_credit_survives_unroll, test_planned_memory_scan_matches_reference_and_gradients, test_elapsed_memory_decay_and_self_edges, test_training_unroll_two_updates, test_regular_traffic_does_not_erase_alert, test_retention_bounded_without_sticky_selection_and_floor, test_columnar_scores_and_no_topk_alert_cap, test_reject_mismatched_stream_config, test_runner_checkpoint_selection_and_interrupted_resume |
| `research/archives/new/research/test_sliding_v3.py` | 330 | Regression tests: test_exact_negative_pair_index_preserves_samples, test_frozen_encoder_invalidation_and_training, test_causal_attention_padding_and_future_independence, test_temporal_graph_cpu_and_training_fallback, test_static_short_read_sequential_and_gradients, test_offline_encoder_matches_online_and_gradients, test_compatible_sequential_equivalence_against_current_v2, test_chunk_invariance_causality_and_snapshot, test_count_min_matches_sequential_with_collisions, test_training_gradients_and_cross_batch_unroll, test_two_hash_does_not_add_embedding_parameters, test_elapsed_affine_scan_matches_current_v2, test_compatible_gradient_and_update_parity, test_compatible_sticky_selection_reaches_quiet_endpoint, test_corrected_balanced_quota_and_no_sticky_retention, test_sparse_ring_allocation_growth_and_cross_batch_credit, test_preparation_pretraining_runner_and_resume |
| `research/archives/new/research/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/archives/new/research/orthrus_comparison/__init__.py` | 1 | ORTHRUS paper reproduction and structured-event TGNN research. |
| `research/archives/new/research/orthrus_comparison/acquisition.py` | 93 | Download only CADETS from the authors' public, uncompressed tar bundle. |
| `research/archives/new/research/orthrus_comparison/models.py` | 59 | TGNN V3 relation-reconstruction adapter and proposed V4 with shared KV encoder. |
| `research/archives/new/research/orthrus_comparison/official.py` | 56 | Load unmodified model class definitions from the pinned authors' release. |
| `research/archives/new/research/orthrus_comparison/prepare_official.py` | 113 | Run pinned upstream preprocessing against a restored full CADETS database. |
| `research/archives/new/research/orthrus_comparison/protocol.py` | 41 | Paper split registration, identity policy, and conservative feature contract. |
| `research/archives/new/research/orthrus_comparison/report.py` | 62 | Collect real study outputs without turning pending arms into zero scores. |
| `research/archives/new/research/orthrus_comparison/run.py` | 182 | Matched reconstruction study on officially featurized CADETS E3 graphs. |
| `research/archives/new/research/orthrus_comparison/tracing.py` | 103 | Apply the pinned ORTHRUS dependency tracing algorithm to every study arm. |
| `research/archives/new/research/orthrus_comparison/universal.py` | 145 | One shared, bounded-vocabulary encoder for unordered typed key/value sets. |
| `research/archives/new/research/orthrus_official/run.sh` | 18 | Experiment/application command entry point |
| `research/archives/new/research/orthrus_official/src/__init__.py` | 0 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/config.py` | 647 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database.py` | 23 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/data_utils.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/decoders.py` | 41 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/encoders.py` | 88 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/factory.py` | 149 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/labelling.py` | 110 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/model.py` | 78 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/orthrus.py` | 114 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/provnet_utils.py` | 488 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/temporal.py` | 95 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/__init__.py` | 6 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/cadets_e3.py` | 261 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/cadets_e5.py` | 266 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/clearscope_e3.py` | 246 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/clearscope_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/theia_e3.py` | 244 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/create_database/theia_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/detection/__init__.py` | 3 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/detection/evaluation.py` | 65 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/detection/evaluation_utils.py` | 613 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/detection/node_evaluation.py` | 123 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/detection/orthrus_gnn_testing.py` | 138 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/detection/orthrus_gnn_training.py` | 101 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/featurization/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/featurization/build_feature_word2vec.py` | 146 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/featurization/embed_edges_feature_word2vec.py` | 167 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/preprocessing/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/preprocessing/build_orthrus_graphs.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/triage/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/triage/tracing.py` | 136 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/triage/tracing_methods/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/triage/tracing_methods/depimpact.py` | 201 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/orthrus_official/src/triage/tracing_methods/depimpact_utils.py` | 486 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/new/research/sliding_node_mem/__init__.py` | 4 | Sliding provenance windows, temporal CNN, selective short and long memory. |
| `research/archives/new/research/sliding_node_mem/budget.py` | 27 | Require equal detector parameters before either matched training arm starts. |
| `research/archives/new/research/sliding_node_mem/cloud_copy.py` | 54 | Copy prepared data directly between the two user-authorized workspaces. |
| `research/archives/new/research/sliding_node_mem/encoding.py` | 43 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/archives/new/research/sliding_node_mem/evaluate_saved.py` | 86 | Score a saved TGNN sliding checkpoint without running training. |
| `research/archives/new/research/sliding_node_mem/model.py` | 413 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/archives/new/research/sliding_node_mem/performance.py` | 89 | Disposable L4 comparison against the archived implementation, same data/weights. |
| `research/archives/new/research/sliding_node_mem/run.py` | 174 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/archives/new/research/sliding_node_mem/sync.py` | 97 | Collect the new experiment into its own directory; never edit old ledgers. |
| `research/archives/new/research/sliding_node_mem/transfer.py` | 90 | Copy checksum-verified existing CADETS inputs between authorized Modal profiles. |
| `research/archives/new/research/sliding_node_mem_upgrade/__init__.py` | 1 | Isolated experimental upgrade; saved original experiments are immutable. |
| `research/archives/new/research/sliding_node_mem_upgrade/collect_three.py` | 51 | Collect original five-epoch arm outputs and render comparable operating points. |
| `research/archives/new/research/sliding_node_mem_upgrade/encoding.py` | 108 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/archives/new/research/sliding_node_mem_upgrade/model.py` | 454 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/archives/new/research/sliding_node_mem_upgrade/orthrus_frontend.py` | 69 | Exact authors' ORTHRUS encoder/graph feed with TGDetect temporal processing. |
| `research/archives/new/research/sliding_node_mem_upgrade/pipeline.py` | 61 | Bounded CPU query preparation overlapped with ordered GPU batches. |
| `research/archives/new/research/sliding_node_mem_upgrade/run.py` | 196 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/archives/new/research/sliding_node_mem_upgrade/scoring.py` | 74 | Benign-only relation calibration and repeated-evidence node scoring. |
| `research/archives/new/research/sliding_node_mem_upgrade/smoke.py` | 79 | Real benign-prefix disposable training timing; never reports research accuracy. |
| `research/archives/new/research/sliding_node_mem_upgrade/windows.py` | 33 | Vectorized chronological neighborhood planning for many overlapping windows. |
| `research/archives/new/research/sliding_node_mem_v2/__init__.py` | 4 | Isolated latency optimizations and corrected sliding-memory experiments. |
| `research/archives/new/research/sliding_node_mem_v2/benchmark.py` | 122 | Compare warm streaming latency; synthetic timings are not detection results. |
| `research/archives/new/research/sliding_node_mem_v2/encoding.py` | 143 | Sequence-aware value encoding and bounded, weight-versioned inference cache. |
| `research/archives/new/research/sliding_node_mem_v2/layers.py` | 27 | Same dense causal convolution weights, evaluated as a small-window GEMM. |
| `research/archives/new/research/sliding_node_mem_v2/memory.py` | 64 | Exact affine memory scan with CPU-planned indices and bounded scan depth. |
| `research/archives/new/research/sliding_node_mem_v2/model.py` | 393 | V2: exact execution optimizations plus separately selectable detection fixes. |
| `research/archives/new/research/sliding_node_mem_v2/run.py` | 216 | Chronological V2 training, resumable unrolls and uncapped calibrated alerts. |
| `research/archives/new/research/sliding_node_mem_v2/runtime.py` | 23 | Controlled cyclic GC at stream boundaries, with restoration on exceptions. |
| `research/archives/new/research/sliding_node_mem_v2/scoring.py` | 119 | Benign-calibrated anomaly evidence without a top-30 alert cap. |
| `research/archives/new/research/sliding_node_mem_v2/tensors.py` | 33 | Bulk CPU metadata conversion and shared-block compressed-token gathers. |
| `research/archives/new/research/sliding_node_mem_v2/training.py` | 29 | Bounded chronological unrolls retain cross-batch memory-write gradients. |
| `research/archives/new/research/sliding_node_mem_v2/windows.py` | 46 | Balanced endpoint history with chronological, batch-independent sampling. |
| `research/archives/new/research/sliding_node_mem_v3/__init__.py` | 3 | Research model, experiment or shared utility |
| `research/archives/new/research/sliding_node_mem_v3/ablate.py` | 37 | Run independent three-seed temporal ablations on the official prepared split. |
| `research/archives/new/research/sliding_node_mem_v3/benchmark.py` | 95 | Prepared-input benchmark; preprocessing is timed separately from inference. |
| `research/archives/new/research/sliding_node_mem_v3/compare_orthrus.py` | 79 | Authors' ORTHRUS architecture with V3's held-out selection/calibration protocol. |
| `research/archives/new/research/sliding_node_mem_v3/graphs.py` | 34 | Bounded inference-only CUDA graph cache for a fixed temporal region. |
| `research/archives/new/research/sliding_node_mem_v3/memory.py` | 65 | GPU segmented affine scan with optional elapsed-time fast/slow decay. |
| `research/archives/new/research/sliding_node_mem_v3/model.py` | 241 | Tensor-ring TGNN. No Python loop over events in the forward path. |
| `research/archives/new/research/sliding_node_mem_v3/prepare.py` | 139 | Compile official prepared graphs once into token-ID/event memmaps. |
| `research/archives/new/research/sliding_node_mem_v3/pretrain.py` | 84 | Train Word2Vec's skip-gram/negative-sampling objective on benign path text. |
| `research/archives/new/research/sliding_node_mem_v3/profile_latency.py` | 97 | Paired real-prefix inference ablations; separate instrumented CUDA profile. |
| `research/archives/new/research/sliding_node_mem_v3/rings.py` | 220 | Device-side chronological ring planning, including repeated batch endpoints. |
| `research/archives/new/research/sliding_node_mem_v3/run.py` | 185 | Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence. |
| `research/archives/new/research/sliding_node_mem_v3/temporal.py` | 24 | Optional full causal attention for short fixed windows, requiring retraining. |
| `research/archives/new/research/sliding_node_mem_v3/tokens.py` | 194 | Offline typed token rows and a vectorized memmap/resident token bank. |
| `research/archives/new/scripts/download_local_data.py` | 64 | Download or install the checksum-verified CADETS/V3 release data bundle. |
| `research/archives/new/scripts/load_windows_apt.py` | 55 | Score local Windows-APT telemetry with frozen V3 and register its dashboard graph. |
| `research/archives/new/scripts/verify_package.py` | 21 | Verify the standalone package against its recorded SHA-256 inventory. |
| `research/archives/old/dashboard/__init__.py` | 1 | TG-Detect research and inference dashboard. |
| `research/archives/old/dashboard/backend.py` | 537 | Artifact-backed dashboard, exact checkpoint loading and causal streaming inference. |
| `research/archives/old/dashboard/cadets.py` | 435 | CADETS V3 investigation API. Separate from the legacy 72D Sysmon runtime. |
| `research/archives/old/dashboard/graph_datasets.py` | 370 | Persistent uploaded-log graphs, scored by the verified V3 checkpoint. |
| `research/archives/old/dashboard/teammate_api.py` | 459 | Teammate dashboard contracts backed by our DEDALE artifacts and checkpoints. |
| `research/archives/old/dashboard/frontend/eslint.config.mjs` | 50 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/next-env.d.ts` | 7 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/next.config.ts` | 14 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/postcss.config.mjs` | 5 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/tailwind.config.ts` | 64 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/app/globals.css` | 333 | Frontend page/layout |
| `research/archives/old/dashboard/frontend/src/app/layout.tsx` | 41 | Frontend page/layout |
| `research/archives/old/dashboard/frontend/src/app/page.tsx` | 5 | Frontend page/layout |
| `research/archives/old/dashboard/frontend/src/app/legacy/layout.tsx` | 14 | Frontend page/layout |
| `research/archives/old/dashboard/frontend/src/app/legacy/page.tsx` | 445 | Frontend page/layout |
| `research/archives/old/dashboard/frontend/src/components/cadets/AttackInvestigation.tsx` | 925 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/cadets/CadetsConsole.tsx` | 1315 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/cadets/DetectionGraph.tsx` | 913 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/cadets/console.module.css` | 1438 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/cadets/detection.module.css` | 498 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/cadets/investigation.module.css` | 702 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/analytics/AnalyticsPage.tsx` | 672 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/artifacts/ArtifactsPage.tsx` | 300 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/chains/ChainsPage.tsx` | 510 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/datasets/DatasetsPage.tsx` | 974 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/events/EventsPage.tsx` | 582 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/graph/AttackGraphCanvas.tsx` | 506 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/graph/GraphPage.tsx` | 299 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/graph/NodeExplanationPanel.tsx` | 247 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/graph/TriageRail.tsx` | 125 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/model/ModalPanels.tsx` | 67 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/model/ModelPage.tsx` | 658 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/overview/OverviewPage.tsx` | 848 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/shared/BackgroundMotion.tsx` | 177 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/shared/TGDetectLoader.tsx` | 142 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/shared/legends.tsx` | 154 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/shared/pills.tsx` | 352 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/tgdetect/shared/temporal-graph-viz.tsx` | 591 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/accordion.tsx` | 66 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/alert-dialog.tsx` | 157 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/alert.tsx` | 66 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/aspect-ratio.tsx` | 11 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/avatar.tsx` | 53 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/badge.tsx` | 46 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/breadcrumb.tsx` | 109 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/button.tsx` | 59 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/calendar.tsx` | 213 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/card.tsx` | 92 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/carousel.tsx` | 241 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/chart.tsx` | 353 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/checkbox.tsx` | 32 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/collapsible.tsx` | 33 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/command.tsx` | 184 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/context-menu.tsx` | 252 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/dialog.tsx` | 143 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/drawer.tsx` | 135 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/dropdown-menu.tsx` | 257 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/form.tsx` | 167 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/hover-card.tsx` | 44 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/input-otp.tsx` | 77 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/input.tsx` | 21 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/label.tsx` | 24 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/menubar.tsx` | 276 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/navigation-menu.tsx` | 168 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/pagination.tsx` | 127 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/popover.tsx` | 48 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/progress.tsx` | 31 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/radio-group.tsx` | 45 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/resizable.tsx` | 56 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/scroll-area.tsx` | 58 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/select.tsx` | 185 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/separator.tsx` | 28 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/sheet.tsx` | 139 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/sidebar.tsx` | 726 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/skeleton.tsx` | 13 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/slider.tsx` | 63 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/sonner.tsx` | 25 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/switch.tsx` | 31 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/table.tsx` | 116 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/tabs.tsx` | 66 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/textarea.tsx` | 18 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/toast.tsx` | 129 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/toaster.tsx` | 35 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/toggle-group.tsx` | 73 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/toggle.tsx` | 47 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/components/ui/tooltip.tsx` | 61 | Frontend UI component |
| `research/archives/old/dashboard/frontend/src/hooks/use-mobile.ts` | 15 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/hooks/use-toast.ts` | 194 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/dataset-context.tsx` | 171 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/date-utils.ts` | 22 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/db.ts` | 13 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/model-context.tsx` | 125 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/theme-context.tsx` | 51 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/utils.ts` | 6 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/chart-constants.ts` | 52 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/constants.ts` | 447 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/formatters.ts` | 367 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/types.ts` | 742 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/api/client.ts` | 133 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/graph/causality.ts` | 491 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/graph/hooks.ts` | 108 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/graph/layout.ts` | 117 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/graph/types.ts` | 129 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/services/hooks.ts` | 244 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/services/index.ts` | 143 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/frontend/src/lib/tgdetect/services/api/index.ts` | 538 | Frontend shared hook/service/configuration |
| `research/archives/old/dashboard/runtime/__init__.py` | 1 | Model definitions required for dashboard checkpoint inference. |
| `research/archives/old/dashboard/runtime/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/archives/old/dashboard/runtime/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/archives/old/dashboard/runtime/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/archives/old/dashboard/runtime/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/archives/old/dashboard/runtime/preprocessing.py` | 89 | Inference preprocessing copied from research/dedale_data.py. |
| `research/archives/old/dashboard/runtime/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/archives/old/dashboard/runtime/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/archives/old/dashboard/runtime/windows_apt_models.py` | 16 | Windows input dimension adapter; preserves frozen 3x128 GAT and temporal branches. |
| `research/archives/old/dashboard/scripts/download_cadets.py` | 41 | Download the exact trained V3 bundle; no GPU container is started. |
| `research/archives/old/dashboard/scripts/prepare_cadets.py` | 95 | Verify downloaded V3 artifacts and atomically build the investigation index. |
| `research/archives/old/research/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/archives/old/research/check_windows_apt_v3_transfer.py` | 181 | Frozen CADETS V3 transfer diagnostic against Windows Wazuh weak annotations. |
| `research/archives/old/research/conftest.py` | 10 | Allow research tests to import scripts and the application from either directory. |
| `research/archives/old/research/modal_sliding_v2_benchmark.py` | 90 | Bounded read-only L4 latency check using the existing prepared CADETS prefix. |
| `research/archives/old/research/modal_sliding_v3_benchmark.py` | 64 | Bounded L4 validation of tensor-ring TGNN on existing real prepared inputs. |
| `research/archives/old/research/modal_sliding_v3_profile.py` | 34 | Run the paired latency experiment on the existing prepared real data. |
| `research/archives/old/research/modal_v3_accuracy.py` | 51 | Isolated five-epoch V3 versus ORTHRUS on the existing official CADETS split. |
| `research/archives/old/research/modal_v3_vs_orthrus.py` | 82 | Fresh paired L4 inference comparison on identical CADETS event prefixes. |
| `research/archives/old/research/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/archives/old/research/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/archives/old/research/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/archives/old/research/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/archives/old/research/test_cadets_dashboard.py` | 161 | Regression tests: test_summary_reproduces_artifact_and_full_dataset, test_events_are_aligned_and_paginated_without_invented_train_scores, test_explanations_reproduce_node_scores_and_chain_uses_no_labels, test_invalid_requests_and_missing_entities, test_event_details_expose_actual_tokenized_endpoint_attributes, test_real_v3_replay_matches_saved_gpu_scores_and_resets, test_constructed_chains_are_directed_chronological_and_label_free, test_detection_graph_predictions_ips_and_observed_connections |
| `research/archives/old/research/test_graph_uploads.py` | 113 | Regression tests: test_upload_real_inference_and_dynamic_connected_neighborhoods, test_malicious_roots_keep_connected_benign_context, test_parsing_formats_errors_and_ignores_supplied_verdicts, test_upload_losses_match_original_gpu_prefix, test_timestamp_offsets_and_event_attributes_do_not_leak_from_future |
| `research/archives/old/research/test_orthrus_universal.py` | 136 | Regression tests: test_missing_and_mixed_types, test_permutation_and_batch_padding_invariance, test_no_current_relation_or_label_in_query, test_v4_kv_state_gradients_and_reload, test_official_model_loss_and_no_current_relation_leak, test_matched_runner_synthetic, test_pinned_tracing_and_window_evaluation |
| `research/archives/old/research/test_sliding_v2.py` | 287 | Regression tests: test_dense_cnn_outputs_and_gradients, test_compatible_predictions_gradients_and_optimizer_updates, test_causality_and_batch_invariance, test_balanced_history_keeps_quiet_endpoint, test_balanced_vectorized_windows_match_sequential, test_order_sensitive_encoder_and_field_permutation, test_inference_cache_eviction_invalidation_and_gradients, test_compaction_preserves_predictions_and_snapshot, test_partial_compaction_preserves_shared_rows_and_finishes_later, test_cross_batch_credit_survives_unroll, test_planned_memory_scan_matches_reference_and_gradients, test_elapsed_memory_decay_and_self_edges, test_training_unroll_two_updates, test_regular_traffic_does_not_erase_alert, test_retention_bounded_without_sticky_selection_and_floor, test_columnar_scores_and_no_topk_alert_cap, test_reject_mismatched_stream_config, test_runner_checkpoint_selection_and_interrupted_resume |
| `research/archives/old/research/test_sliding_v3.py` | 330 | Regression tests: test_exact_negative_pair_index_preserves_samples, test_frozen_encoder_invalidation_and_training, test_causal_attention_padding_and_future_independence, test_temporal_graph_cpu_and_training_fallback, test_static_short_read_sequential_and_gradients, test_offline_encoder_matches_online_and_gradients, test_compatible_sequential_equivalence_against_current_v2, test_chunk_invariance_causality_and_snapshot, test_count_min_matches_sequential_with_collisions, test_training_gradients_and_cross_batch_unroll, test_two_hash_does_not_add_embedding_parameters, test_elapsed_affine_scan_matches_current_v2, test_compatible_gradient_and_update_parity, test_compatible_sticky_selection_reaches_quiet_endpoint, test_corrected_balanced_quota_and_no_sticky_retention, test_sparse_ring_allocation_growth_and_cross_batch_credit, test_preparation_pretraining_runner_and_resume |
| `research/archives/old/research/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/archives/old/research/orthrus_comparison/__init__.py` | 1 | ORTHRUS paper reproduction and structured-event TGNN research. |
| `research/archives/old/research/orthrus_comparison/acquisition.py` | 93 | Download only CADETS from the authors' public, uncompressed tar bundle. |
| `research/archives/old/research/orthrus_comparison/models.py` | 59 | TGNN V3 relation-reconstruction adapter and proposed V4 with shared KV encoder. |
| `research/archives/old/research/orthrus_comparison/official.py` | 56 | Load unmodified model class definitions from the pinned authors' release. |
| `research/archives/old/research/orthrus_comparison/prepare_official.py` | 113 | Run pinned upstream preprocessing against a restored full CADETS database. |
| `research/archives/old/research/orthrus_comparison/protocol.py` | 41 | Paper split registration, identity policy, and conservative feature contract. |
| `research/archives/old/research/orthrus_comparison/report.py` | 62 | Collect real study outputs without turning pending arms into zero scores. |
| `research/archives/old/research/orthrus_comparison/run.py` | 182 | Matched reconstruction study on officially featurized CADETS E3 graphs. |
| `research/archives/old/research/orthrus_comparison/tracing.py` | 103 | Apply the pinned ORTHRUS dependency tracing algorithm to every study arm. |
| `research/archives/old/research/orthrus_comparison/universal.py` | 145 | One shared, bounded-vocabulary encoder for unordered typed key/value sets. |
| `research/archives/old/research/orthrus_official/run.sh` | 18 | Experiment/application command entry point |
| `research/archives/old/research/orthrus_official/src/__init__.py` | 0 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/config.py` | 647 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database.py` | 23 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/data_utils.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/decoders.py` | 41 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/encoders.py` | 88 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/factory.py` | 149 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/labelling.py` | 110 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/model.py` | 78 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/orthrus.py` | 114 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/provnet_utils.py` | 488 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/temporal.py` | 95 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/__init__.py` | 6 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/cadets_e3.py` | 261 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/cadets_e5.py` | 266 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/clearscope_e3.py` | 246 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/clearscope_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/theia_e3.py` | 244 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/create_database/theia_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/detection/__init__.py` | 3 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/detection/evaluation.py` | 65 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/detection/evaluation_utils.py` | 613 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/detection/node_evaluation.py` | 123 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/detection/orthrus_gnn_testing.py` | 138 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/detection/orthrus_gnn_training.py` | 101 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/featurization/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/featurization/build_feature_word2vec.py` | 146 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/featurization/embed_edges_feature_word2vec.py` | 167 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/preprocessing/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/preprocessing/build_orthrus_graphs.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/triage/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/triage/tracing.py` | 136 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/triage/tracing_methods/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/triage/tracing_methods/depimpact.py` | 201 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/orthrus_official/src/triage/tracing_methods/depimpact_utils.py` | 486 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/old/research/publication/build_figures.py` | 180 | Publication figures from retained experiment artifacts; no synthetic results. |
| `research/archives/old/research/publication/verify_paper.py` | 44 | Check retained data, source manifests and the compiled paper. |
| `research/archives/old/research/sliding_node_mem/__init__.py` | 4 | Sliding provenance windows, temporal CNN, selective short and long memory. |
| `research/archives/old/research/sliding_node_mem/budget.py` | 27 | Require equal detector parameters before either matched training arm starts. |
| `research/archives/old/research/sliding_node_mem/cloud_copy.py` | 54 | Copy prepared data directly between the two user-authorized workspaces. |
| `research/archives/old/research/sliding_node_mem/encoding.py` | 43 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/archives/old/research/sliding_node_mem/evaluate_saved.py` | 86 | Score a saved TGNN sliding checkpoint without running training. |
| `research/archives/old/research/sliding_node_mem/model.py` | 413 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/archives/old/research/sliding_node_mem/performance.py` | 89 | Disposable L4 comparison against the archived implementation, same data/weights. |
| `research/archives/old/research/sliding_node_mem/run.py` | 174 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/archives/old/research/sliding_node_mem/sync.py` | 97 | Collect the new experiment into its own directory; never edit old ledgers. |
| `research/archives/old/research/sliding_node_mem/transfer.py` | 90 | Copy checksum-verified existing CADETS inputs between authorized Modal profiles. |
| `research/archives/old/research/sliding_node_mem_upgrade/__init__.py` | 1 | Isolated experimental upgrade; saved original experiments are immutable. |
| `research/archives/old/research/sliding_node_mem_upgrade/collect_three.py` | 51 | Collect original five-epoch arm outputs and render comparable operating points. |
| `research/archives/old/research/sliding_node_mem_upgrade/encoding.py` | 108 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/archives/old/research/sliding_node_mem_upgrade/model.py` | 454 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/archives/old/research/sliding_node_mem_upgrade/orthrus_frontend.py` | 69 | Exact authors' ORTHRUS encoder/graph feed with TGDetect temporal processing. |
| `research/archives/old/research/sliding_node_mem_upgrade/pipeline.py` | 61 | Bounded CPU query preparation overlapped with ordered GPU batches. |
| `research/archives/old/research/sliding_node_mem_upgrade/run.py` | 196 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/archives/old/research/sliding_node_mem_upgrade/scoring.py` | 74 | Benign-only relation calibration and repeated-evidence node scoring. |
| `research/archives/old/research/sliding_node_mem_upgrade/smoke.py` | 79 | Real benign-prefix disposable training timing; never reports research accuracy. |
| `research/archives/old/research/sliding_node_mem_upgrade/windows.py` | 33 | Vectorized chronological neighborhood planning for many overlapping windows. |
| `research/archives/old/research/sliding_node_mem_v2/__init__.py` | 4 | Isolated latency optimizations and corrected sliding-memory experiments. |
| `research/archives/old/research/sliding_node_mem_v2/benchmark.py` | 122 | Compare warm streaming latency; synthetic timings are not detection results. |
| `research/archives/old/research/sliding_node_mem_v2/encoding.py` | 143 | Sequence-aware value encoding and bounded, weight-versioned inference cache. |
| `research/archives/old/research/sliding_node_mem_v2/layers.py` | 27 | Same dense causal convolution weights, evaluated as a small-window GEMM. |
| `research/archives/old/research/sliding_node_mem_v2/memory.py` | 64 | Exact affine memory scan with CPU-planned indices and bounded scan depth. |
| `research/archives/old/research/sliding_node_mem_v2/model.py` | 393 | V2: exact execution optimizations plus separately selectable detection fixes. |
| `research/archives/old/research/sliding_node_mem_v2/run.py` | 216 | Chronological V2 training, resumable unrolls and uncapped calibrated alerts. |
| `research/archives/old/research/sliding_node_mem_v2/runtime.py` | 23 | Controlled cyclic GC at stream boundaries, with restoration on exceptions. |
| `research/archives/old/research/sliding_node_mem_v2/scoring.py` | 119 | Benign-calibrated anomaly evidence without a top-30 alert cap. |
| `research/archives/old/research/sliding_node_mem_v2/tensors.py` | 33 | Bulk CPU metadata conversion and shared-block compressed-token gathers. |
| `research/archives/old/research/sliding_node_mem_v2/training.py` | 29 | Bounded chronological unrolls retain cross-batch memory-write gradients. |
| `research/archives/old/research/sliding_node_mem_v2/windows.py` | 46 | Balanced endpoint history with chronological, batch-independent sampling. |
| `research/archives/old/research/sliding_node_mem_v3/__init__.py` | 3 | Research model, experiment or shared utility |
| `research/archives/old/research/sliding_node_mem_v3/ablate.py` | 37 | Run independent three-seed temporal ablations on the official prepared split. |
| `research/archives/old/research/sliding_node_mem_v3/benchmark.py` | 95 | Prepared-input benchmark; preprocessing is timed separately from inference. |
| `research/archives/old/research/sliding_node_mem_v3/compare_orthrus.py` | 79 | Authors' ORTHRUS architecture with V3's held-out selection/calibration protocol. |
| `research/archives/old/research/sliding_node_mem_v3/graphs.py` | 34 | Bounded inference-only CUDA graph cache for a fixed temporal region. |
| `research/archives/old/research/sliding_node_mem_v3/memory.py` | 65 | GPU segmented affine scan with optional elapsed-time fast/slow decay. |
| `research/archives/old/research/sliding_node_mem_v3/model.py` | 241 | Tensor-ring TGNN. No Python loop over events in the forward path. |
| `research/archives/old/research/sliding_node_mem_v3/prepare.py` | 139 | Compile official prepared graphs once into token-ID/event memmaps. |
| `research/archives/old/research/sliding_node_mem_v3/pretrain.py` | 84 | Train Word2Vec's skip-gram/negative-sampling objective on benign path text. |
| `research/archives/old/research/sliding_node_mem_v3/profile_latency.py` | 97 | Paired real-prefix inference ablations; separate instrumented CUDA profile. |
| `research/archives/old/research/sliding_node_mem_v3/rings.py` | 220 | Device-side chronological ring planning, including repeated batch endpoints. |
| `research/archives/old/research/sliding_node_mem_v3/run.py` | 185 | Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence. |
| `research/archives/old/research/sliding_node_mem_v3/temporal.py` | 24 | Optional full causal attention for short fixed windows, requiring retraining. |
| `research/archives/old/research/sliding_node_mem_v3/tokens.py` | 194 | Offline typed token rows and a vectorized memmap/resident token bank. |
| `research/archives/old/scripts/download_local_data.py` | 64 | Download or install the checksum-verified CADETS/V3 release data bundle. |
| `research/archives/old/scripts/verify_package.py` | 21 | Verify the standalone package against its recorded SHA-256 inventory. |
| `research/archives/tgdetectv3/backend/dashboard/__init__.py` | 1 | TG-Detect research and inference dashboard. |
| `research/archives/tgdetectv3/backend/dashboard/backend.py` | 537 | Artifact-backed dashboard, exact checkpoint loading and causal streaming inference. |
| `research/archives/tgdetectv3/backend/dashboard/cadets.py` | 444 | CADETS V3 investigation API. Separate from the legacy 72D Sysmon runtime. |
| `research/archives/tgdetectv3/backend/dashboard/decoder.py` | 72 | Resolve graph IDs through retained metadata, never invert lossy token hashes. |
| `research/archives/tgdetectv3/backend/dashboard/graph_datasets.py` | 378 | Persistent uploaded-log graphs, scored by the verified V3 checkpoint. |
| `research/archives/tgdetectv3/backend/dashboard/teammate_api.py` | 459 | Teammate dashboard contracts backed by our DEDALE artifacts and checkpoints. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/__init__.py` | 1 | Model definitions required for dashboard checkpoint inference. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/preprocessing.py` | 89 | Inference preprocessing copied from research/dedale_data.py. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/archives/tgdetectv3/backend/dashboard/runtime/windows_apt_models.py` | 16 | Windows input dimension adapter; preserves frozen 3x128 GAT and temporal branches. |
| `research/archives/tgdetectv3/backend/dashboard/scripts/download_cadets.py` | 41 | Download the exact trained V3 bundle; no GPU container is started. |
| `research/archives/tgdetectv3/backend/dashboard/scripts/prepare_cadets.py` | 95 | Verify downloaded V3 artifacts and atomically build the investigation index. |
| `research/archives/tgdetectv3/backend/research/causal_tgn.py` | 101 | Small controlled temporal additions to the canonical adapted graph encoder. |
| `research/archives/tgdetectv3/backend/research/conftest.py` | 10 | Allow research tests to import scripts and the application from either directory. |
| `research/archives/tgdetectv3/backend/research/modal_sliding_v2_benchmark.py` | 90 | Bounded read-only L4 latency check using the existing prepared CADETS prefix. |
| `research/archives/tgdetectv3/backend/research/modal_sliding_v3_benchmark.py` | 64 | Bounded L4 validation of tensor-ring TGNN on existing real prepared inputs. |
| `research/archives/tgdetectv3/backend/research/modal_sliding_v3_profile.py` | 34 | Run the paired latency experiment on the existing prepared real data. |
| `research/archives/tgdetectv3/backend/research/modal_v3_accuracy.py` | 51 | Isolated five-epoch V3 versus ORTHRUS on the existing official CADETS split. |
| `research/archives/tgdetectv3/backend/research/modal_v3_vs_orthrus.py` | 82 | Fresh paired L4 inference comparison on identical CADETS event prefixes. |
| `research/archives/tgdetectv3/backend/research/node_memory_models.py` | 369 | TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection. |
| `research/archives/tgdetectv3/backend/research/node_memory_v3.py` | 175 | Causal, batch-invariant selective memory with separate endpoint reads. |
| `research/archives/tgdetectv3/backend/research/pomsathit_baseline.py` | 155 | Canonical first-session skeleton, deliberately not an invented reproduction. |
| `research/archives/tgdetectv3/backend/research/temporal_cnn.py` | 111 | Causal convolutions along event order, never along feature or batch axes. |
| `research/archives/tgdetectv3/backend/research/test_cadets_dashboard.py` | 161 | Regression tests: test_summary_reproduces_artifact_and_full_dataset, test_events_are_aligned_and_paginated_without_invented_train_scores, test_explanations_reproduce_node_scores_and_chain_uses_no_labels, test_invalid_requests_and_missing_entities, test_event_details_expose_actual_tokenized_endpoint_attributes, test_real_v3_replay_matches_saved_gpu_scores_and_resets, test_constructed_chains_are_directed_chronological_and_label_free, test_detection_graph_predictions_ips_and_observed_connections |
| `research/archives/tgdetectv3/backend/research/test_graph_uploads.py` | 113 | Regression tests: test_upload_real_inference_and_dynamic_connected_neighborhoods, test_malicious_roots_keep_connected_benign_context, test_parsing_formats_errors_and_ignores_supplied_verdicts, test_upload_losses_match_original_gpu_prefix, test_timestamp_offsets_and_event_attributes_do_not_leak_from_future |
| `research/archives/tgdetectv3/backend/research/test_metadata_decoder.py` | 49 | Regression tests: test_chain_decoder_matches_retained_event_identities, test_missing_values_are_not_invented_and_commands_preserve_whitespace, test_uploaded_decoder_uses_each_events_local_original_values |
| `research/archives/tgdetectv3/backend/research/test_orthrus_universal.py` | 136 | Regression tests: test_missing_and_mixed_types, test_permutation_and_batch_padding_invariance, test_no_current_relation_or_label_in_query, test_v4_kv_state_gradients_and_reload, test_official_model_loss_and_no_current_relation_leak, test_matched_runner_synthetic, test_pinned_tracing_and_window_evaluation |
| `research/archives/tgdetectv3/backend/research/test_sliding_v2.py` | 287 | Regression tests: test_dense_cnn_outputs_and_gradients, test_compatible_predictions_gradients_and_optimizer_updates, test_causality_and_batch_invariance, test_balanced_history_keeps_quiet_endpoint, test_balanced_vectorized_windows_match_sequential, test_order_sensitive_encoder_and_field_permutation, test_inference_cache_eviction_invalidation_and_gradients, test_compaction_preserves_predictions_and_snapshot, test_partial_compaction_preserves_shared_rows_and_finishes_later, test_cross_batch_credit_survives_unroll, test_planned_memory_scan_matches_reference_and_gradients, test_elapsed_memory_decay_and_self_edges, test_training_unroll_two_updates, test_regular_traffic_does_not_erase_alert, test_retention_bounded_without_sticky_selection_and_floor, test_columnar_scores_and_no_topk_alert_cap, test_reject_mismatched_stream_config, test_runner_checkpoint_selection_and_interrupted_resume |
| `research/archives/tgdetectv3/backend/research/test_sliding_v3.py` | 330 | Regression tests: test_exact_negative_pair_index_preserves_samples, test_frozen_encoder_invalidation_and_training, test_causal_attention_padding_and_future_independence, test_temporal_graph_cpu_and_training_fallback, test_static_short_read_sequential_and_gradients, test_offline_encoder_matches_online_and_gradients, test_compatible_sequential_equivalence_against_current_v2, test_chunk_invariance_causality_and_snapshot, test_count_min_matches_sequential_with_collisions, test_training_gradients_and_cross_batch_unroll, test_two_hash_does_not_add_embedding_parameters, test_elapsed_affine_scan_matches_current_v2, test_compatible_gradient_and_update_parity, test_compatible_sticky_selection_reaches_quiet_endpoint, test_corrected_balanced_quota_and_no_sticky_retention, test_sparse_ring_allocation_growth_and_cross_batch_credit, test_preparation_pretraining_runner_and_resume |
| `research/archives/tgdetectv3/backend/research/tgnn_components.py` | 156 | Independent additions to the frozen TGNN; no changes to the reference files. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/__init__.py` | 1 | ORTHRUS paper reproduction and structured-event TGNN research. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/acquisition.py` | 93 | Download only CADETS from the authors' public, uncompressed tar bundle. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/models.py` | 59 | TGNN V3 relation-reconstruction adapter and proposed V4 with shared KV encoder. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/official.py` | 56 | Load unmodified model class definitions from the pinned authors' release. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/prepare_official.py` | 113 | Run pinned upstream preprocessing against a restored full CADETS database. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/protocol.py` | 41 | Paper split registration, identity policy, and conservative feature contract. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/report.py` | 62 | Collect real study outputs without turning pending arms into zero scores. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/run.py` | 182 | Matched reconstruction study on officially featurized CADETS E3 graphs. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/tracing.py` | 103 | Apply the pinned ORTHRUS dependency tracing algorithm to every study arm. |
| `research/archives/tgdetectv3/backend/research/orthrus_comparison/universal.py` | 145 | One shared, bounded-vocabulary encoder for unordered typed key/value sets. |
| `research/archives/tgdetectv3/backend/research/orthrus_official/run.sh` | 18 | Experiment/application command entry point |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/__init__.py` | 0 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/config.py` | 647 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database.py` | 23 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/data_utils.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/decoders.py` | 41 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/encoders.py` | 88 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/factory.py` | 149 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/labelling.py` | 110 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/model.py` | 78 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/orthrus.py` | 114 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/provnet_utils.py` | 488 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/temporal.py` | 95 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/__init__.py` | 6 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/cadets_e3.py` | 261 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/cadets_e5.py` | 266 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/clearscope_e3.py` | 246 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/clearscope_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/theia_e3.py` | 244 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/create_database/theia_e5.py` | 245 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/detection/__init__.py` | 3 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/detection/evaluation.py` | 65 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/detection/evaluation_utils.py` | 613 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/detection/node_evaluation.py` | 123 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/detection/orthrus_gnn_testing.py` | 138 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/detection/orthrus_gnn_training.py` | 101 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/featurization/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/featurization/build_feature_word2vec.py` | 146 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/featurization/embed_edges_feature_word2vec.py` | 167 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/preprocessing/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/preprocessing/build_orthrus_graphs.py` | 267 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/triage/__init__.py` | 1 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/triage/tracing.py` | 136 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/triage/tracing_methods/__init__.py` | 2 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/triage/tracing_methods/depimpact.py` | 201 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/orthrus_official/src/triage/tracing_methods/depimpact_utils.py` | 486 | Pinned upstream ORTHRUS reference implementation |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/__init__.py` | 4 | Sliding provenance windows, temporal CNN, selective short and long memory. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/budget.py` | 27 | Require equal detector parameters before either matched training arm starts. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/cloud_copy.py` | 54 | Copy prepared data directly between the two user-authorized workspaces. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/encoding.py` | 43 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/evaluate_saved.py` | 86 | Score a saved TGNN sliding checkpoint without running training. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/model.py` | 413 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/performance.py` | 89 | Disposable L4 comparison against the archived implementation, same data/weights. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/run.py` | 174 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/sync.py` | 97 | Collect the new experiment into its own directory; never edit old ledgers. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem/transfer.py` | 90 | Copy checksum-verified existing CADETS inputs between authorized Modal profiles. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/__init__.py` | 1 | Isolated experimental upgrade; saved original experiments are immutable. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/collect_three.py` | 51 | Collect original five-epoch arm outputs and render comparable operating points. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/encoding.py` | 108 | Cache only deterministic token IDs, never learned event embeddings. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/model.py` | 454 | TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/orthrus_frontend.py` | 69 | Exact authors' ORTHRUS encoder/graph feed with TGDetect temporal processing. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/pipeline.py` | 61 | Bounded CPU query preparation overlapped with ordered GPU batches. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/run.py` | 196 | Isolated CADETS comparison; prior experiment source/results are untouched. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/scoring.py` | 74 | Benign-only relation calibration and repeated-evidence node scoring. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/smoke.py` | 79 | Real benign-prefix disposable training timing; never reports research accuracy. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_upgrade/windows.py` | 33 | Vectorized chronological neighborhood planning for many overlapping windows. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/__init__.py` | 4 | Isolated latency optimizations and corrected sliding-memory experiments. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/benchmark.py` | 122 | Compare warm streaming latency; synthetic timings are not detection results. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/encoding.py` | 143 | Sequence-aware value encoding and bounded, weight-versioned inference cache. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/layers.py` | 27 | Same dense causal convolution weights, evaluated as a small-window GEMM. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/memory.py` | 64 | Exact affine memory scan with CPU-planned indices and bounded scan depth. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/model.py` | 393 | V2: exact execution optimizations plus separately selectable detection fixes. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/run.py` | 216 | Chronological V2 training, resumable unrolls and uncapped calibrated alerts. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/runtime.py` | 23 | Controlled cyclic GC at stream boundaries, with restoration on exceptions. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/scoring.py` | 119 | Benign-calibrated anomaly evidence without a top-30 alert cap. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/tensors.py` | 33 | Bulk CPU metadata conversion and shared-block compressed-token gathers. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/training.py` | 29 | Bounded chronological unrolls retain cross-batch memory-write gradients. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v2/windows.py` | 46 | Balanced endpoint history with chronological, batch-independent sampling. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/__init__.py` | 3 | Research model, experiment or shared utility |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/ablate.py` | 37 | Run independent three-seed temporal ablations on the official prepared split. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/benchmark.py` | 95 | Prepared-input benchmark; preprocessing is timed separately from inference. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/compare_orthrus.py` | 79 | Authors' ORTHRUS architecture with V3's held-out selection/calibration protocol. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/graphs.py` | 34 | Bounded inference-only CUDA graph cache for a fixed temporal region. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/memory.py` | 65 | GPU segmented affine scan with optional elapsed-time fast/slow decay. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/model.py` | 241 | Tensor-ring TGNN. No Python loop over events in the forward path. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/prepare.py` | 139 | Compile official prepared graphs once into token-ID/event memmaps. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/pretrain.py` | 84 | Train Word2Vec's skip-gram/negative-sampling objective on benign path text. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/profile_latency.py` | 97 | Paired real-prefix inference ablations; separate instrumented CUDA profile. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/rings.py` | 220 | Device-side chronological ring planning, including repeated batch endpoints. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/run.py` | 185 | Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/temporal.py` | 24 | Optional full causal attention for short fixed windows, requiring retraining. |
| `research/archives/tgdetectv3/backend/research/sliding_node_mem_v3/tokens.py` | 194 | Offline typed token rows and a vectorized memmap/resident token bank. |
| `research/archives/tgdetectv3/frontend/eslint.config.mjs` | 50 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/next-env.d.ts` | 7 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/next.config.ts` | 15 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/postcss.config.mjs` | 5 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/tailwind.config.ts` | 64 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/app/globals.css` | 225 | Frontend page/layout |
| `research/archives/tgdetectv3/frontend/src/app/layout.tsx` | 55 | Frontend page/layout |
| `research/archives/tgdetectv3/frontend/src/app/page.tsx` | 5 | Frontend page/layout |
| `research/archives/tgdetectv3/frontend/src/app/legacy/layout.tsx` | 14 | Frontend page/layout |
| `research/archives/tgdetectv3/frontend/src/app/legacy/page.tsx` | 432 | Frontend page/layout |
| `research/archives/tgdetectv3/frontend/src/components/cadets/AmbientBackdrop.tsx` | 40 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/AttackInvestigation.tsx` | 923 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/BrandMark.tsx` | 9 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/CadetsConsole.tsx` | 1273 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/DecodedEntity.tsx` | 33 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/DetectionGraph.tsx` | 735 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/ThemeControl.tsx` | 14 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/ambient.module.css` | 12 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/console.module.css` | 206 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/decoded.module.css` | 7 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/detection.module.css` | 83 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/graph-theme.ts` | 63 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/cadets/investigation.module.css` | 99 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/analytics/AnalyticsPage.tsx` | 672 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/artifacts/ArtifactsPage.tsx` | 300 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/chains/ChainsPage.tsx` | 510 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/datasets/DatasetsPage.tsx` | 974 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/events/EventsPage.tsx` | 582 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/graph/AttackGraphCanvas.tsx` | 506 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/graph/GraphPage.tsx` | 299 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/graph/NodeExplanationPanel.tsx` | 247 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/graph/TriageRail.tsx` | 125 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/model/ModalPanels.tsx` | 68 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/model/ModelPage.tsx` | 658 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/overview/OverviewPage.tsx` | 848 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/shared/BackgroundMotion.tsx` | 177 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/shared/TGDetectLoader.tsx` | 142 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/shared/legends.tsx` | 154 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/shared/pills.tsx` | 352 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/tgdetect/shared/temporal-graph-viz.tsx` | 591 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/accordion.tsx` | 66 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/alert-dialog.tsx` | 157 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/alert.tsx` | 66 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/aspect-ratio.tsx` | 11 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/avatar.tsx` | 53 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/badge.tsx` | 46 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/breadcrumb.tsx` | 109 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/button.tsx` | 59 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/calendar.tsx` | 213 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/card.tsx` | 92 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/carousel.tsx` | 241 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/chart.tsx` | 353 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/checkbox.tsx` | 32 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/collapsible.tsx` | 33 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/command.tsx` | 184 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/context-menu.tsx` | 252 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/dialog.tsx` | 143 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/drawer.tsx` | 135 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/dropdown-menu.tsx` | 257 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/form.tsx` | 167 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/hover-card.tsx` | 44 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/input-otp.tsx` | 77 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/input.tsx` | 21 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/label.tsx` | 24 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/menubar.tsx` | 276 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/navigation-menu.tsx` | 168 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/pagination.tsx` | 127 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/popover.tsx` | 48 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/progress.tsx` | 31 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/radio-group.tsx` | 45 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/resizable.tsx` | 56 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/scroll-area.tsx` | 58 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/select.tsx` | 185 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/separator.tsx` | 28 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/sheet.tsx` | 139 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/sidebar.tsx` | 726 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/skeleton.tsx` | 13 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/slider.tsx` | 63 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/sonner.tsx` | 25 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/switch.tsx` | 31 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/table.tsx` | 116 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/tabs.tsx` | 66 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/textarea.tsx` | 18 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/toast.tsx` | 129 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/toaster.tsx` | 35 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/toggle-group.tsx` | 73 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/toggle.tsx` | 47 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/components/ui/tooltip.tsx` | 61 | Frontend UI component |
| `research/archives/tgdetectv3/frontend/src/hooks/use-mobile.ts` | 15 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/hooks/use-toast.ts` | 194 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/dataset-context.tsx` | 172 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/date-utils.ts` | 22 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/model-context.tsx` | 112 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/theme-context.tsx` | 42 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/theme.ts` | 8 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/utils.ts` | 6 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/chart-constants.ts` | 52 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/constants.ts` | 447 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/formatters.ts` | 367 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/types.ts` | 742 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/api/client.ts` | 133 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/graph/causality.ts` | 491 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/graph/hooks.ts` | 108 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/graph/layout.ts` | 117 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/graph/types.ts` | 129 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/services/hooks.ts` | 244 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/services/index.ts` | 143 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/lib/tgdetect/services/api/index.ts` | 538 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/frontend/src/types/cytoscape-fcose.d.ts` | 4 | Frontend shared hook/service/configuration |
| `research/archives/tgdetectv3/scripts/download_local_data.py` | 91 | Install the checksum-verified CADETS/V3 release bundle into backend/. |
| `research/archives/tgdetectv3/scripts/run_backend.py` | 18 | Start the unchanged dashboard API from the reorganized backend directory. |
| `research/archives/tgdetectv3/scripts/update_manifest.py` | 21 | Refresh the source inventory after an intentional change; excludes local data. |
| `research/archives/tgdetectv3/scripts/verify_package.py` | 34 | Verify source and release-data checksums; --code-only works without data. |

## Recorded Source Snapshot

| Source | Lines | Purpose |
| --- | ---: | --- |
