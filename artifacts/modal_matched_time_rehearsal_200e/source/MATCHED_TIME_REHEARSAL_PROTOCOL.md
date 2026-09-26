# Matched paper-architecture versus time-encoding experiment

Registered: 2026-09-23, before these 200-epoch runs.

## Question

On the same adapted CTU-13 task, does adding only learned neighbor-age encoding to
the paper-specified GNN plus 10% rehearsal improve performance?

This directly compares:

- `B_H`: paper-specified architecture plus matched rehearsal.
- `B_T_H`: identical system plus learned time encoding.

The sole model difference is `T`. Both use the same data, labels, causal ego graph,
static features, split, three attention layers, width 128, ReLU, dropout 0.2,
softmax head, cross-entropy, Adam, learning rate 0.001, weight decay 0.00001,
batch size 256, 200 epochs, seeds 0–4, threshold selection, and 10% Algorithm-R
training reservoir. Replay sampling uses the same seed and count in paired runs.

The architecture and training values explicitly stated in the paper are retained.
The attention equation, input encoding, loss, replay schedule/weight, dataset and
split remain the documented `POMSATHIT_ADAPTED` choices because the paper does not
fully specify or release them. Therefore `B_H` is the paper-specified architecture
under the adapted common protocol, not an exact reconstruction of the unavailable
original experiment.

## Fixed data and evaluation

Use `scenario06_pilot16384.npz`, processed SHA-256
`cf4c0d3c9eba674db8297923f97b664d3de5c5dea412b7ebca444410949a80c6`.
The chronological train/validation/test partitions contain 9830/3277/3277 events
and 91/25/25 positive labels. The data and test outcomes were already inspected in
the earlier pilot, so this is a completion of a missing factorial arm rather than
independent confirmation on untouched data.

For each run, choose the largest threshold tied for maximum validation F1, then
freeze it for test. Report F1, precision, recall, false-positive rate, average
precision and ROC-AUC. Report paired seed differences, mean and sample SD, parameter
count, training time, replay visits, checkpoint/data/source hashes, and predictions.
Do not choose the conclusion from a single seed or compare these CTU scores directly
with the paper's reported Wazuh score.

## Execution amendment — requested before completion

The project owner stopped the sequential CPU execution and requested all remaining
tests in one Modal L4 run. The partial local artifacts are retained but excluded
from the final matched result. The Modal invocation reruns all ten combinations
(two arms × five seeds) from scratch in one L4 function, so no local and cloud
results are mixed. It must record the exact GPU, PyTorch/CUDA environment, CUDA
smoke/backward checks, peak allocated GPU memory, source/data/checkpoint hashes,
saved predictions, and recomputed metrics. CUDA float32 results may differ from
the interrupted CPU run; only internally paired L4 results support the conclusion.

Execution status: **BLOCKED before GPU allocation**. Modal run
`ap-xDrw3BTxaW1k2LIJQrv9aU` successfully built the pinned PyTorch 2.7.1 CUDA image,
then Modal rejected the L4 Function because the active profile has no payment
method. The remote function never started; no CUDA result or charge was produced.
After billing is enabled, resume with exactly one invocation:

```bash
~/main_env/bin/python -m modal run modal_matched_time_rehearsal.py
```
