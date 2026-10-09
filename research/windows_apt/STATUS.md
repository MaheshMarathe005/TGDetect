# Windows-APT execution status

Training started on 26 September 2026. This is the user-authorized **weak-label
Wazuh MITRE annotation-presence comparison**, not verified attack detection.

[Modal app](https://modal.com/apps/prathampansare25/main/ap-SPFyF3clXWG0WZjadBGKH8)

| Gate | Verified status |
|---|---|
| Source integrity | All 19 supplied checksums PASS; combined/period ID multisets equal |
| Preprocessing | All 102,011 rows; 72 shared features; train-only scaling; causal graph |
| Split | 61,206 / 20,404 / 20,401; equal-time events stay in one partition |
| Local tests | 69 passed |
| ZIP upload / extraction | 44.1 MiB ZIP uploaded to dedicated Volume and extracted remotely |
| Remote verification | ZIP and member hashes, exact raw-to-tensor reconstruction, replay training-only PASS |
| Actual L4 tests | 24 passed in 10.03 seconds |
| Calibration | 1/2/4/8/10 workers: 35.76/47.94/99.77/201.71/249.20 aggregate steps/s |
| Training | All 10 runs started on one L4; 5 seeds per model; 200 epochs each |
| Final results and audit | 10/10 runs complete; remote verification PASS; local audit PASS |

All ten runs completed 200/200 epochs. Mean training time was 783.50 seconds
(13.06 minutes) for GNN_H and 1,947.54 seconds (32.46 minutes) for TGNN_Q_A_C;
these are concurrent ten-worker timings. Mean peak allocated memory was 256.83
MiB and 278.10 MiB, respectively. Training telemetry contained 1,841 samples:
mean GPU utilization 97.63%, maximum 99%, mean memory 3,876.84 MiB, maximum
5,631 MiB, and mean power 54.61 W. No artificial VRAM allocations were used.

The ten-run production wall time was about 33 minutes after preflight. This
replaces earlier estimates based on the smaller CTU feature representation and
two-worker scheduling.

The models have 43,394 and 290,434 parameters with the 72-feature adapter. Both
retain the previously tested architectures; only their input dimension changes.
The hybrid uses local GRU history, time encoding, extra attention and causal CNN;
it has no persistent endpoint-memory branch.

Volume: `tg-detect-windows-apt-v1`.
ZIP: `/upload/windows_apt_v1.zip`.
Prepared bundle: `/windows_apt_v1/bundle/`.
Progress/checkpoints/predictions/telemetry: `/windows_apt_v1/results/`.

Local evidence: `artifacts/windows_apt_v1/{local_gates.json,remote_verification.json,
execution_status.json,prepared/data_manifest.json,remote/results/gates.json}`.

Use `~/main_env/bin/python sync_windows_apt.py` to download current artifacts.
The independent local audit verified all ten saved metrics, labels, row IDs,
thresholds, checkpoint hashes and sampled checkpoint predictions. Evidence is in
`artifacts/windows_apt_v1/remote/results/` and
`artifacts/windows_apt_v1/local_result_verification.json`.

## Final weak-label results

| Model | Parameters | Test F1 | PR-AP | ROC-AUC | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| GNN_H | 43,394 | 98.28 ± 0.41% | 99.48 ± 0.11% | 99.47 ± 0.19% | 98.20 ± 0.32% | 98.36 ± 0.60% | 2.40 ± 0.43% |
| TGNN_Q_A_C | 290,434 | **98.97 ± 0.07%** | **99.81 ± 0.02%** | 99.79 ± 0.02% | 98.56 ± 0.24% | **99.38 ± 0.26%** | **1.93 ± 0.33%** |

The paired hybrid-minus-GNN differences are +0.69 ± 0.39 F1 points, +0.33 ±
0.13 PR-AP points, +0.32 ± 0.18 ROC-AUC points, +1.02 ± 0.35 recall points,
and −0.47 ± 0.66 FPR points. The hybrid wins F1 in all five paired seeds.

These scores predict Wazuh MITRE annotation presence. They are not attack-detection
scores: unannotated records may be benign, missed, or simply outside the rule
mapping. Feature-only controls are already extremely strong: logistic regression
gets 99.04% test F1 and histogram gradient boosting gets 99.08%, slightly above
the hybrid. The architecture result is therefore a weak-label benchmark finding,
not evidence that the temporal model is intrinsically superior.

## Frozen threshold calibration (2026-09-26)

The post-hoc experiment is saved under
`artifacts/windows_apt_v1/threshold_calibration/`. All ten checkpoints were
SHA-256 verified and replayed with the exact prepared tensors; replayed
probabilities differed from the official files by at most 1.9e-5. Models were
not retrained or modified. Thresholds were selected only from deterministic
validation subsets and then frozen before test evaluation.

For the primary 50-positive/50-negative calibration, GNN F1 changed from
98.278% to 98.231% (−0.047 points); TGNN hybrid changed from 98.967% to
97.992% (−0.974 points). The GNN threshold was relatively stable (0.724–0.744
across seeds), while TGNN thresholds ranged 0.636–0.809 and had broad bootstrap
intervals. Larger negative pools changed thresholds and operating points sharply.
Precision and FPR improved because recall was sacrificed; PR-AUC and ROC-AUC
were unchanged, as required for fixed continuous scores. This is evidence of
small-sample threshold instability, not evidence that the previous TGNN result
was threshold-limited. The entire result remains a weak-label annotation
experiment, not verified attack/benign detection.
