# TG-Detect results ledger

This ledger records completed experiments found under `artifacts/`. Metrics are not interchangeable across sections: datasets, labels, splits, thresholds, training budgets, and objectives differ. Percentages are shown as percentages where the source report uses percentages; decimal metrics are shown as decimals elsewhere.

## 1. Shared KAIROS/DEDALE comparison

Source: [supervised shared-feature report](../artifacts/kairos_dedale_shared41/results/seed0/RESULTS.md), [benign-only report](../artifacts/kairos_dedale_shared41_benign/results/seed0/RESULTS.md).

The shared preparation contains 1,886,734 supported events, 41 features, nine relations, and chronological 60/20/20 partitions. The KAIROS adaptation uses wholly benign windows; supervised TGNN uses attack labels; benign TGNN uses wholly benign windows with a one-class binary objective. Each run used one seed.

### Supervised TGNN run, five epochs

| Arm | Online window F1 | Window PR-AP | Event F1 | Event PR-AP | Event ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KAIROS | 0.7692 | 0.6219 | 0.0000 | 0.0076 | 0.7503 |
| TGNN Node Memory V3 | 0.8333 | 1.0000 | 0.7553 | 0.8479 | 0.9962 |
| TGNN Node Memory V4 | 0.8333 | 1.0000 | **0.7899** | **0.8574** | 0.9958 |

KAIROS’s event row is an extra diagnostic; its primary target is anomalous windows/queues.

### Benign-only TGNN run, two epochs

| Arm | Online window F1 | Window PR-AP | Window recall | Window FPR | Event F1 | Event PR-AP |
|---|---:|---:|---:|---:|---:|---:|
| KAIROS | **0.6667** | **0.6125** | 0.5714 | 0.0116 | 0.0000 | 0.0052 |
| TGNN Node Memory V3 benign | 0.2609 | 0.3255 | **0.8571** | 0.3837 | 0.0314 | 0.0028 |
| TGNN Node Memory V4 benign | 0.2609 | 0.3249 | **0.8571** | 0.3837 | 0.0289 | 0.0029 |

The benign TGNN classifier head is not the same reconstruction objective as KAIROS. It shows that removing attack labels without adding a relation-reconstruction decoder is insufficient.

## 2. Verified node-memory studies on the earlier DEDALE protocol

Source: [V3 report](../artifacts/dedale_host_mem_v3/RESULTS.md), [matched memory report](../artifacts/dedale_host_mem_matched/RESULTS.md).

These are three-seed studies on the earlier DEDALE protocol with QAC warm starts and additional fine-tuning; they are separate from the shared 41-feature KAIROS experiment.

| Arm | F1 | PR-AP | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| GNN_H | 0.8537 | 0.8496 | 0.9932 | 0.7832 | 0.9385 |
| TGNN_Q_A_C | 0.8543 | 0.8932 | 0.9963 | 0.7783 | 0.9480 |
| TGNN_Node_Mem_Sel | 0.8463 | 0.8667 | 0.9959 | 0.7709 | 0.9385 |
| TGNN_Node_Mem_V3 | **0.9298** | **0.9870** | **0.9997** | **0.9051** | **0.9608** |

Matched additional-training comparison:

| Arm | F1 | PR-AP | ROC-AUC |
|---|---:|---:|---:|
| TGNN_Q_A_C | 0.8543 | 0.8932 | 0.9963 |
| TGNN_Node_Mem_V3_Fixed | 0.8929 | 0.9786 | 0.9987 |
| TGNN_Node_Mem_V3 | **0.9326** | **0.9877** | **0.9997** |

## 3. Component ablation study

Source: [component report](../artifacts/components_200e_v1/TGNN_COMPONENT_REPORT.md). Five seeds, 200 epochs per arm.

| Variant | Test F1 (%) | PR-AP (%) | Precision (%) | Recall (%) | Parameters |
|---|---:|---:|---:|---:|---:|
| GNN_H | **86.61 ± 4.84** | 88.18 ± 5.30 | 93.53 ± 4.02 | 80.80 ± 6.57 | 35,458 |
| GNN_H_A | 80.90 ± 8.23 | 84.70 ± 4.39 | 88.93 ± 4.45 | 75.20 ± 13.08 | 84,610 |
| TGNN | 84.52 ± 2.63 | 94.22 ± 0.49 | 97.95 ± 2.81 | 74.40 ± 3.58 | 35,714 |
| TGNN_A | 83.03 ± 3.08 | 92.88 ± 1.59 | 96.94 ± 2.81 | 72.80 ± 5.22 | 84,866 |
| TGNN_M | 83.86 ± 4.01 | 94.77 ± 0.48 | 98.09 ± 2.62 | 73.60 ± 7.27 | 35,714 |
| TGNN_M_A | 76.72 ± 6.29 | 94.29 ± 0.98 | 99.05 ± 2.13 | 63.20 ± 9.96 | 84,866 |
| TGNN_Q | 85.45 ± 1.38 | 93.64 ± 0.66 | 95.10 ± 0.13 | 77.60 ± 2.19 | 134,786 |
| TGNN_Q_A | 86.39 ± 3.27 | 94.33 ± 0.40 | 95.18 ± 0.31 | 79.20 ± 5.22 | 183,938 |
| TGNN_Q_M | 85.52 ± 2.57 | 94.67 ± 0.95 | 98.10 ± 2.61 | 76.00 ± 4.90 | 134,786 |
| TGNN_Q_M_A | 84.15 ± 4.03 | 94.27 ± 0.96 | 100.00 ± 0.00 | 72.80 ± 5.93 | 183,938 |
| TGNN_N | 82.26 ± 4.89 | 90.87 ± 4.03 | 97.87 ± 2.94 | 71.20 ± 7.16 | 35,970 |
| TGNN_W | 85.26 ± 2.94 | 94.64 ± 0.74 | 96.09 ± 2.20 | 76.80 ± 5.22 | 35,714 |

## 4. Temporal CNN study

Source: [temporal CNN report](../artifacts/temporal_cnn_200e_v2/TEMPORAL_CNN_REPORT.md). Five seeds, 200 epochs.

| Model | F1 (%) | PR-AP (%) | Parameters |
|---|---:|---:|---:|
| GNN_H | 86.61 ± 4.84 | 88.18 ± 5.30 | 35,458 |
| TGNN | 84.52 ± 2.63 | 94.22 ± 0.49 | 35,714 |
| TGNN_Q | 85.45 ± 1.38 | 93.64 ± 0.66 | 134,786 |
| TGNN_Q_A | 86.39 ± 3.27 | 94.33 ± 0.40 | 183,938 |
| CNN_H | **97.11 ± 1.16** | **96.49 ± 0.22** | 100,226 |
| TGNN_C | 85.51 ± 4.19 | 92.52 ± 3.00 | 134,274 |
| TGNN_Q_C | 87.07 ± 4.81 | 93.97 ± 0.77 | 233,346 |
| TGNN_Q_A_C | 87.16 ± 6.41 | 93.15 ± 2.77 | 282,498 |

## 5. Causal and matched-time studies

Sources: [causal pilot](../artifacts/causal_pilot_v1/summary.json), [matched-time summary](../artifacts/modal_matched_time_rehearsal_200e/summary.json). Five seeds; decimal means and sample SD.

| Arm | F1 | PR-AP | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| B | 0.8380 ± 0.0343 | 0.8833 ± 0.0307 | 0.9868 ± 0.0035 | 0.8715 | 0.8080 |
| B_T | 0.8984 ± 0.0399 | 0.9396 ± 0.0054 | 0.9857 ± 0.0013 | 0.9465 | 0.8560 |
| B_M | 0.7778 ± 0.0162 | 0.9025 ± 0.0195 | 0.9977 ± 0.0012 | 0.8993 | 0.6880 |
| B_MR | 0.7552 ± 0.0313 | 0.8611 ± 0.0654 | 0.9980 ± 0.0014 | 0.8923 | 0.6560 |
| B_TM | 0.8691 ± 0.0246 | 0.9128 ± 0.0272 | 0.9930 ± 0.0011 | 0.9523 | 0.8000 |
| B_TMR | 0.8433 ± 0.0407 | 0.9034 ± 0.0288 | 0.9946 ± 0.0012 | 0.9497 | 0.7600 |
| B_H | 0.8105 ± 0.0581 | 0.8828 ± 0.0346 | 0.9860 ± 0.0031 | 0.8225 | 0.8000 |
| B_TMRH | 0.8437 ± 0.0322 | 0.9237 ± 0.0157 | 0.9943 ± 0.0015 | 0.9498 | 0.7600 |
| Logistic Regression | 0.8364 | 0.8093 | 0.9600 | 0.7667 | 0.9200 |
| HistGradientBoosting | **0.9796** | **0.9606** | 0.9875 | 1.0000 | 0.9600 |

Matched-time B_H vs B_T_H:

| Arm | F1 | PR-AP | ROC-AUC |
|---|---:|---:|---:|
| B_H | 0.8661 ± 0.0484 | 0.8818 ± 0.0530 | 0.9897 ± 0.0016 |
| B_T_H | 0.8452 ± 0.0263 | **0.9422 ± 0.0049** | 0.9894 ± 0.0058 |

## 6. DARPA E3 pilot

Source: [CADETS E3 summary](../artifacts/darpa_e3_pilot_v1/remote/results/cadets_e3/summary.json). Values are three-seed means (seeds 0, 1, 2). This sampled pilot uses incident-to-node proxy labels and incomplete April 2–6 coverage; its scores are not comparable to the full ORTHRUS protocol.

| Arm | F1 | PR-AP | ROC-AUC | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| GNN_H | **0.9555 ± 0.0076** | **0.9834 ± 0.0025** | 0.9907 | 0.9576 | 0.9536 | 0.0028 |
| TGNN_Q_A_C | 0.9461 ± 0.0039 | 0.9757 ± 0.0011 | 0.9898 | 0.9387 | 0.9536 | 0.0041 |

## 7. Windows-APT transfer diagnostics

Source: [transfer report](../artifacts/windows_apt_transfer_check/RESULTS.md). These are frozen DEDALE checkpoints transferred to weak Wazuh MITRE-presence labels; they are not Windows-trained held-out results.

| Frozen model | Precision | Recall | F1 | PR-AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TGNN_Q_A_C | 94.21% | 7.78% | 14.37% | 85.25% | 79.11% |
| TGNN_Node_Mem_V3_Fixed | 97.79% | 6.25% | 11.75% | 96.20% | 95.15% |
| TGNN_Node_Mem_V3 | 99.75% | 2.15% | 4.22% | 95.77% | 94.69% |

Frozen DEDALE learned-memory thresholding missed 54,014 of 55,203 annotated Windows events. Separately trained Windows-APT checkpoints had original held-out F1 means of 98.28% (GNN) and 98.97% (QAC); those are separate experiments.

## Interpretation and limits

- The strongest direct DEDALE node-memory result is TGNN Node Memory V3 in the earlier three-seed protocol (F1 0.9298 mean; matched treatment 0.9326).
- V4 improved the shared-feature supervised event F1 from 0.7553 to 0.7899 in the one-seed five-epoch run.
- In the benign-only shared-feature run, KAIROS outperformed the current TGNN classifier head. This does not test a TGNN relation-reconstruction decoder.
- CNN_H and HistGradientBoosting are strong baselines in their respective experiments, but their datasets/protocols are not the shared KAIROS protocol.
- Transfer performance demonstrates threshold and feature-domain shift; ranking AP can remain high while a frozen operating threshold has very low recall.
- Several studies use small positive counts, warm starts, exploratory ablations, or separate inference settings. None alone establishes cross-dataset generalization.

## ORTHRUS E3 follow-up (2026-10-08)

See [protocol and implementation status](orthrus_comparison/README.md) and [collected measured results](../artifacts/orthrus_e3_study/RESULTS.md). Saved-checkpoint comparison is complete at 20/18/13 epochs; a matched 20-epoch comparison was not completed because credits ended. The complete authors' CADETS E3 dump has been acquired in the dedicated Modal volume (1,465,153,615 bytes; SHA256 `ab3fae64a90d9c8a92a84382195cfd04223a9d8335c1e1690e3cb67c3a441c7a`). All three arms use benign relation reconstruction and the same released preprocessing/splits, followed by identical dependency tracing. The result collector records detection, tracing, data/source hashes, runtime versions, checkpoints, and timing. Earlier benign-only TGNN scores used a constant benign binary target and attack-labeled validation; they are not a matched reconstruction experiment. Measured checkpoint scores are recorded in the local CPU comparison section below.

### Prepared full-data audit

All released preprocessing stages completed. [Data audit](../artifacts/orthrus_e3_study/DATA_AUDIT.md): training has 3,526,541 fused edges over 645 windows; validation has 105,481 over 22; test has 1,905,942 over 333. The test universe exactly matches the paper: 268,153 nodes and 68 positives. There are 72 distinct ground-truth database nodes, four outside the test graph universe. The three planned 20-epoch L4 jobs ended at epochs 20, 18, and 13 after credits ended. Saved-checkpoint evaluation and tracing completed on local CPU; the results are recorded below.

## Local CPU evaluation of saved 20/18/13 epoch checkpoints

# ORTHRUS / TGNN V3 / TGNN V4-KV: E3 CADETS

All locally measured rows below use the shared manifest, benign relation reconstruction, and the saved checkpoints (ORTHRUS: 20 epochs; V3: 18; V4: 13). A missing row is pending. Synthetic checks and older sampled pilots are excluded.

| Model | Stage | TP | FP | FN | Precision | Recall | F1 | MCC | PR-AP | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ORTHRUS_Reference | threshold_only | 6 | 2 | 62 | 0.7500 | 0.0882 | 0.1579 | 0.2572 | 0.1453 | 0.7746 |
| ORTHRUS_Reference | released_top30_kmeans | 6 | 0 | 62 | 1.0000 | 0.0882 | 0.1622 | 0.2970 | 0.1453 | 0.7746 |
| ORTHRUS_Reference | tracing_released_last_window | 15 | 13 | 53 | 0.5357 | 0.2206 | 0.3125 | 0.3437 | 0.1453 | 0.7746 |
| ORTHRUS_Reference | tracing_global_union | 19 | 22 | 49 | 0.4634 | 0.2794 | 0.3486 | 0.3597 | 0.1453 | 0.7746 |
| TGNN_Node_Mem_V3_Reconstruction | threshold_only | 3 | 33 | 65 | 0.0833 | 0.0441 | 0.0577 | 0.0605 | 0.0131 | 0.9020 |
| TGNN_Node_Mem_V3_Reconstruction | released_top30_kmeans | 0 | 13 | 68 | 0.0000 | 0.0000 | 0.0000 | -0.0001 | 0.0131 | 0.9020 |
| TGNN_Node_Mem_V3_Reconstruction | tracing_released_last_window | 7 | 2078 | 61 | 0.0034 | 0.1029 | 0.0065 | 0.0173 | 0.0131 | 0.9020 |
| TGNN_Node_Mem_V3_Reconstruction | tracing_global_union | 10 | 2135 | 58 | 0.0047 | 0.1471 | 0.0090 | 0.0249 | 0.0131 | 0.9020 |
| TGNN_Node_Mem_V4_KV | threshold_only | 0 | 33 | 68 | 0.0000 | 0.0000 | 0.0000 | -0.0002 | 0.0132 | 0.8797 |
| TGNN_Node_Mem_V4_KV | released_top30_kmeans | 0 | 20 | 68 | 0.0000 | 0.0000 | 0.0000 | -0.0001 | 0.0132 | 0.8797 |
| TGNN_Node_Mem_V4_KV | tracing_released_last_window | 0 | 3947 | 68 | 0.0000 | 0.0000 | 0.0000 | -0.0019 | 0.0132 | 0.8797 |
| TGNN_Node_Mem_V4_KV | tracing_global_union | 0 | 3991 | 68 | 0.0000 | 0.0000 | 0.0000 | -0.0020 | 0.0132 | 0.8797 |

## Published paper reference (not measured in this run)

Source: supplied `sec25cycle1-prepub-103-jiang-baoxiang.pdf`, Table 10. These figures establish a reference; exact reproduction is not assumed.

| Paper system | TP | FP | TN | FN | Precision | MCC |
|---|---:|---:|---:|---:|---:|---:|
| ORTHRUS anomaly detection | 10 | 0 | 268085 | 58 | 1.00 | 0.38 |
| ORTHRUS with tracing | 25 | 23 | 268062 | 43 | 0.52 | 0.44 |

The released tracing evaluation assigns a repeated node its last-window prediction. Global-union tracing is also reported to expose that choice. The original upstream orchestration selected the epoch with best test MCC; this comparison selects the latest saved checkpoints without test labels.

## Run records

- ORTHRUS_Reference: complete; tracing: complete.
  Parameters: 764,202; training seconds: 1558.5; evaluation seconds: 53.1; missing ground truth IDs: [336477, 337065, 370874, 370897].
- TGNN_Node_Mem_V3_Reconstruction: complete; tracing: complete.
  Parameters: 350,516; training seconds: 11121.6; evaluation seconds: 217.4; missing ground truth IDs: [336477, 337065, 370874, 370897].
- TGNN_Node_Mem_V4_KV: complete; tracing: complete.
  Parameters: 1,406,516; training seconds: 10441.2; evaluation seconds: 267.6; missing ground truth IDs: [336477, 337065, 370874, 370897].

## Evaluation conditions

GPU credits ended during training. These are local CPU evaluations of the unchanged saved weights, with no further training. All arms share the full test and validation data, cold state, scoring rules, and tracing algorithm. Epoch counts and training times differ, so this is not an equal-training-budget comparison. PR-AP and ROC-AUC on tracing rows retain the detector ranking; tracing changes binary node predictions.

| Model | Epoch | Last training loss | Training minutes | CPU evaluation minutes |
|---|---:|---:|---:|---:|
| ORTHRUS_Reference | 20 | 0.451810 | 26.0 | 0.9 |
| TGNN_Node_Mem_V3_Reconstruction | 18 | 0.203025 | 185.4 | 3.6 |
| TGNN_Node_Mem_V4_KV | 13 | 0.161853 | 174.0 | 4.5 |

[Interpretation and operating-point diagnostics](../artifacts/orthrus_e3_study/local_checkpoint_comparison/ANALYSIS.md).
