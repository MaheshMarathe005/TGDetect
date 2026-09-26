# Matched GNN+rehearsal versus temporal GNN+rehearsal

Completed 23 September 2026 in one Modal invocation on an NVIDIA L4.

## Direct answer

This is the missing matched comparison. Both models used the same verified CTU-13
sample, labels, causal graph, static features, split, 3×128 attention architecture,
ReLU, dropout 0.2, softmax/CE, Adam 0.001 with 0.00001 weight decay, batch 256,
10% Algorithm-R rehearsal, 200 epochs, and seeds 0–4. `B_T_H` adds only learned
neighbor-age encoding (256 parameters) to `B_H`.

| Test metric | GNN + rehearsal (`B_H`) | TGNN-time + rehearsal (`B_T_H`) | Paired change from time |
|---|---:|---:|---:|
| F1 | **86.61 ± 4.84%** | 84.52 ± 2.63% | −2.10 ± 5.35 points |
| PR-AP | 88.18 ± 5.30% | **94.22 ± 0.49%** | **+6.04 ± 5.46 points** |
| Precision | 93.53 ± 4.02% | **97.95 ± 2.81%** | **+4.41 ± 3.24 points** |
| Recall | **80.80 ± 6.57%** | 74.40 ± 3.58% | −6.40 ± 7.80 points |
| ROC-AUC | **98.97 ± 0.16%** | 98.94 ± 0.58% | −0.03 ± 0.64 points |
| False-positive rate | 0.043 ± 0.028% | **0.012 ± 0.017%** | **−0.031 ± 0.022 points** |

Values are mean ± sample SD over five paired initialization seeds. The F1 threshold
was selected independently for each trained model using validation labels, then
frozen before test. The test partition contains 25 positive flows, so one additional
miss changes recall by four percentage points.

The temporal model improves **ranking quality (PR-AP), precision, and false-positive
rate**, with a positive PR-AP difference in all five paired seeds. It does not improve
the selected-threshold F1: its higher precision comes with lower recall. Therefore
the evidence supports “time encoding improves PR-AP and alert precision under this
protocol,” not the broader statement that it improves every detection metric.

## Architecture and resource comparison

| Measurement | `B_H` | `B_T_H` |
|---|---:|---:|
| Parameters | 35,458 | 35,714 |
| Added parameters | — | 256 (+0.72%) |
| Mean training time/run | 182.46 s | 186.98 s |
| Mean peak allocated GPU memory | 74.87 MiB | 75.66 MiB |

The extra time channel costs about 0.79 MiB peak allocated GPU memory and 4.52
seconds per 200-epoch run in this implementation. The workload is small and includes
Python-controlled replay, so these timings do not measure maximum L4 throughput.

## Environment and verification

- Modal run: `ap-2ukQHuXPf3dCLIi5g6MtYx`, active profile `prathampansare25`.
- GPU: NVIDIA L4, 23,034 MiB reported by `nvidia-smi`; driver 580.95.05.
- Python 3.10.17; PyTorch 2.7.1+cu126; CUDA 12.6; deterministic algorithms enabled.
- CUDA repeatability smoke, identical paired shared initialization, and tiny
  time-encoder backward check passed before training.
- All ten checkpoints, source/data hashes, validation thresholds, saved predictions,
  and recomputed test metrics passed verification inside the same Modal function.
- Download archive SHA-256:
  `2d75e1168e8e0c45a2ba883abe22fd5bce62c3dc6b3b7f4c293fe92661d812c7`.

Artifacts are in `artifacts/modal_matched_time_rehearsal_200e/`; the original archive
is `artifacts/modal_matched_time_rehearsal_200e.tar.gz`. Partial local CPU artifacts
in `artifacts/matched_time_rehearsal_200e/` are excluded from this result.

## Claim boundary

This tests the paper-specified 3×128 architecture and stated training settings under
one common adapted protocol. It is not a literal reconstruction of the paper's
unreleased Wazuh data or its unspecified attention equation, feature encoder, loss,
split, replay schedule/weight, and threshold. The test partition had already been
inspected during the earlier pilot; this completes a missing controlled arm and is
not independent confirmation on an untouched capture.

