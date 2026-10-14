# V4/V3 hybrid: causal CNN and single-head temporal attention

This requested experiment retains the V3 64D typed-KV encoder, parallel affine
node memory, V4 full 256D episodic rings, two causal event-line GAT layers and
relation/pair objectives. It adds one causal CNN block before temporal attention
and changes that temporal attention from four heads to one. Earlier results and
their source snapshots are preserved.
The V3 encoder's field-pooling attention remains unchanged; the single-head
change applies specifically to the temporal window attention.

## Architecture and research basis

The new local block is a depthwise temporal Conv1d (128 channels, kernel 3,
left padding 2), GELU, 128-to-128 pointwise projection, residual addition and
LayerNorm. Padded input/output positions are masked. At position t the CNN can
read only t-2, t-1 and t. RoPE/elapsed-time attention remains exactly causal.
V3 temporal-importance admission uses all temporal outputs as in the preceding
hybrid; the prediction context remains V4's final-position context.

[MobileNets](https://arxiv.org/abs/1704.04861) provides the depthwise/pointwise
factorization rationale. Applied here at width 128 and kernel 3, its convolution
MAC count per position is 3*128+128*128=16,768, versus 3*128*128=49,152 for a
dense kernel-3 convolution: approximately 2.93x fewer convolution MACs. This is
an arithmetic comparison, not measured L4 speedup; biases, normalization, GELU
and residual costs are additional.

The attention width remains 128: one head has width 128 instead of four heads
of width 32. QKV/output projection and FFN costs therefore remain substantially
unchanged, and the aggregate attention QK/AV arithmetic is still proportional
to L²*128. The per-head elapsed-time bias has one output instead of four.
[PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
explains automatic kernel selection and backend input constraints. A head-count
change can change kernel behavior; fewer heads do not guarantee lower latency.
Sparse/linear attention is not introduced for a nine-position window.

The temporal module grows from 132,580 to 149,809 parameters: the CNN adds
17,280 while reducing the bias head removes 51. Total model size is consequently
819,232 parameters, versus preceding hybrid 802,003 and V3 785,327. Parameter
efficiency is not improved by this change. Full 256D episodic rings are retained,
so no claim of better memory efficiency than V3 is made.

## Tests, deployment and bounded training

Twenty local tests pass, including new CNN future independence, all-padding
safety, final-read parity, sequential-versus-batch model outputs and gradients
through the CNN, attention and affine writers over two batches. CUDA tests and
runtime acceptance precede training in the deployed function.

Deployment: `research/modal_v4_cnn_singlehead.py`, app
`tg-detect-v4-cnn-singlehead`. Trigger:
`research/trigger_v4_cnn_singlehead.py`, using
`modal.Function.from_name(...).spawn()`. The deployed function continues without
the trigger client; its call ID and reservation are saved before waiting for the
result. One L4, four CPU cores, requested 32 GiB RAM, one-hour function timeout,
zero automatic retries, zero minimum containers and two-second idle shutdown.

Training: five epochs, seed 0, batch 1,024, two-batch unroll, original prepared
CADETS split and V3 benign-only pretrained 64D table. Fresh weights/optimizer/
state; no resume from an incompatible architecture. CPU batch prefetch and
frozen FP32 evaluation lookup are retained. No baseline retraining.

Final evaluation reports actual precision/recall/F1/AP and false positives under
the existing calibration. Then three paired inference timing repetitions use
saved V3 and new trained checkpoints on the same 40 chronological batches:
8 warmup, 32 measured, batch 1,024, alternating execution order. Measure p50,
p95, p99, throughput and prefix memory; precomputation and batch-formation wait
are excluded. Full-run peak CUDA memory is recorded separately before resetting
benchmark statistics. This comparison does not establish mature-state latency.

The one-hour reservation includes GPU tests, training, final evaluation and
the paired timing. A prefix estimate must fit the remaining allowance with a
margin. Accounted upper-bound study time including this reservation is 24,756
seconds out of the existing 28,800-second cap. Failed historical reservations
remain accounted; no account-balance or invoice claim is made.

Durable output:
`/study/v4_cnn_singlehead_20261011_v1/V4_CNN_SingleHead_s0_e5`.
Final local output:
`artifacts/sliding_v4/V4_CNN_SingleHead_s0_e5.json`.

Deployment and trigger succeeded. Call:
`fc-01M4N5PYHPX571V03P6A37VNC7`. Deployment URL:
<https://modal.com/apps/fukatkaamsathi/main/deployed/tg-detect-v4-cnn-singlehead>.
Accuracy/speed superiority over V3 remains unproven until those results exist.

CUDA acceptance passed all 20 tests. Warm prefix two-batch optimizer steps
were 0.1272, 0.1178 and 0.1173 seconds, with an estimated five-epoch training
time of approximately 1,015 seconds. This does not show a clear training-speed
gain over the preceding hybrid's separate prefix benchmark; it is not a paired
comparison. Training then passed 100,352 events in epoch 1 at approximately 6 s.
Raw acceptance: `artifacts/sliding_v4/cnn_singlehead_acceptance.json`.
Active app: <https://modal.com/apps/fukatkaamsathi/main/ap-yy2JLwUghqL1p2hvPkTevI>.
