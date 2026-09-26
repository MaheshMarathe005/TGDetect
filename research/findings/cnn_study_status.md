# Temporal CNN experiment — registered and launched

Date: 26 September 2026. [Protocol](../../TEMPORAL_CNN_PROTOCOL.md).
[Completed Modal app](https://modal.com/apps/prathampansare25/main/ap-7y95Pbrwfia3sP44n9PnMW)
on profile `prathampansare25`, one L4.

Eight arms × five seeds × 200 epochs = 40 runs. Four controls: GNN+rehearsal,
TGNN, TGNN+GRU, TGNN+GRU+attention. Four new models: CNN+rehearsal,
TGNN+CNN, TGNN+GRU+CNN, TGNN+GRU+attention+CNN.

Local acceptance: **62 passed**. Actual-L4 CNN acceptance: **17 passed in 9.79 s**.
The checks include causal temporal convolution and zero future gradients, a
hand-computed time-axis check, padding isolation, all-arm training/gradient updates,
shared initialization, zero-CNN reference equivalence, exact replay RNG and exact
optimized-versus-original training updates.

The first native-Conv1d attempt (`ap-PA6IRh5XpaOUrDdCrcVqnV`) was explicitly
stopped after slow calibration: 2.26 steps/s at one worker, 4.47 at two, 9.03 at
four, 12.16 at eight, 12.06 at sixteen. Any partial work is excluded. Its source
and diagnostic artifacts remain under `temporal_cnn_200e_v1` for audit.
The active v2 evaluates the identical convolution via causal tap gathering and
matrix multiplication. Outputs and input/kernel gradients match native Conv1d
in float64 acceptance tests on CPU and the L4. Architecture and training settings
are unchanged; all reported experiment results will come from v2.

GPU throughput is measured with independent processes at 1/2/4/8/16 workers;
selection is based on useful training steps per second. Each run keeps batch256,
FP32, the same data and optimizer settings. VRAM allocation is not padded with
unused tensors. Actual GPU utilization, allocated device memory and power are
recorded every second. Concurrent timings include contention.

Live artifacts: Modal Volume `tg-detect-temporal-cnn-v1`, directory
`temporal_cnn_200e_v2`. Local download location:
`../../artifacts/temporal_cnn_200e_v2/`.

```bash
~/main_env/bin/python sync_temporal_cnn.py
~/main_env/bin/python sync_temporal_cnn.py --download
```

The study is now complete: **40/40 runs finished and independently audited**.
See [final findings](../../TEMPORAL_CNN_FINDINGS.md).
The summary, per-seed JSON, checkpoints, predictions, epoch losses, concurrency
benchmark, GPU telemetry and verification are saved under
`../../artifacts/temporal_cnn_200e_v2/`.

The final telemetry contains 661 one-second samples: mean GPU utilization
**95.99%** (maximum 100%), mean device memory **4903.9 MiB** (maximum 6319 MiB),
and mean power **57.22 W**. The local audit reports PASS for all 40 checkpoints,
predictions, metrics, hashes and aggregate statistics. The ten GNN_H/TGNN control
predictions are bit-identical to the earlier verified reference artifacts.

Optimized calibration completed: 25.08 / 50.46 / 96.38 / 202.18 / 299.91 aggregate
training steps per second at 1 / 2 / 4 / 8 / 16 workers respectively. The scheduler
selected **16 concurrent independent workers on the same L4**. The eight-worker
benchmark improves about 16.6× over the corresponding native-Conv1d attempt.
These are training-only microbenchmark rates, not F1 or end-to-end runtime claims.

Production snapshot ending 2026-09-26 06:50:26 UTC: 60 one-second samples,
mean GPU utilization **99.9%**, maximum 100%; device memory **6253 MiB**
(about 6.11 GiB), mean power 57.89 W. All 16 workers were active. These values
describe that measured window, not the full-run average. Spare VRAM does not mean
idle compute; the complete telemetry will show startup and tail effects as well.
