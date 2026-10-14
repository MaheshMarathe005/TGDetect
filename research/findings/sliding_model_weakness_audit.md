# Sliding-window model weakness audit — 10 October 2026

This audit covers `sliding_node_mem`, `sliding_node_mem_upgrade`, its optional
ORTHRUS frontend, shared encoding/memory, and training/scoring runners. The real
measurements below describe the **upgraded** five-epoch runs. Their recorded
source hashes match the local sources. They do not establish the accuracy of
the original sliding implementation or isolate individual component effects.

Evidence: `artifacts/three_arm_five_epochs/`,
`artifacts/sliding_model_audit/probes.json`, and reproducible diagnostics in
`research/audit_sliding_weaknesses.py`. No detector weights or training code were
changed. No remote training was launched.

## Measured problem

Same CADETS E3 split, seed 0, batch 1024, five epochs, NVIDIA L4:

| Detector | Raw top-30 F1 | Raw AP | Calibrated F1 | Training seconds | Recorded evaluation seconds |
|---|---:|---:|---:|---:|---:|
| ORTHRUS | 0.09091 | 0.02518 | Not reported | 362.45 | 46.95 |
| Sliding KV upgrade | 0 | 0.00443 | 0.04000 | 3155.09 | 305.99 |
| Sliding + ORTHRUS encoder | 0 | 0.01026 | 0 | 4014.64 | 315.68 |

The KV model takes 8.70 times the recorded training time; the hybrid takes
11.08 times. These are whole-run measurements, not isolated kernel benchmarks.
Training timing scopes differ around periodic checkpointing. Evaluation scopes
also differ: the upgrade includes additional calibration/metrics work.

There are 268,085 benign and 68 malicious scored nodes. Always predicting benign
achieves 99.97464% accuracy. Accuracy is therefore a poor primary objective.
Four additional ground-truth nodes are absent from the scored population and
reported separately by all arms. Report their coverage explicitly; do not mix
detection scores with dependency-tracing attribution scores.

The upgrade's ROC-AUC 0.92576 coexists with AP 0.00443 and zero raw-rule true
positives. The high-score tail is poor even though broad ranking is better than
chance. This is not solely a threshold problem. The exact ORTHRUS encoder does
not by itself solve it. One seed and one dataset cannot establish generality.

## Findings affecting detection

1. **Alert selection and training target do not match the detection objective.**
   `sliding_node_mem_upgrade/run.py:125,156,168` trains only benign relation
   cross-entropy, evaluates after the final epoch, uses a maximum validation
   loss, and separately clusters only the top 30 nodes. With 68 malicious scored
   nodes, top-30 selection caps recall at 30/68 = 44.12% even with perfect ranking;
   KMeans can select fewer. The maximum threshold is sensitive to one benign
   outlier. Lower relation loss does not prove improved anomaly discrimination.
   The KV training objective falls from 0.45253 to 0.03885, but validation curves
   are absent, so calling this overfitting would be premature.
   **Solution:** remove the fixed top-30 rule as the primary operational detector;
   calibrate an explicit false-alert budget on separate benign calibration data.
   Record AP, precision/recall at fixed alert budgets, alert counts per host/day,
   and per-relation validation curves. Select hyperparameters on a separate
   development capture if labeled F1 optimization is needed. Keep this test set
   untouched. Test context dropout, masked endpoint-attribute prediction, or
   temporal/structural auxiliary objectives separately from the baseline CE.

2. **The calibrated score can erase strong evidence.**
   `scoring.py:39` computes `max(0, sum(top3)/3 - log(total_incident_events))`.
   One score-12 event gives node score 4; adding 63 score-zero events reduces it
   to zero in the probe. A single attack is diluted by division by three, and
   benign traffic continues increasing the degree penalty. Actual calibration
   raises KV F1 to 0.04, but reduces hybrid F1 from threshold-only 0.02020 to zero
   while producing 160 false positives. This combined comparison does not isolate
   which calibration term is responsible.
   **Solution:** ablate relation calibration, evidence aggregation, degree
   correction, and threshold separately. Combine a single-extreme-event channel
   with repeated evidence in a fixed time window. Fit empirical benign score
   distributions conditional on activity, relation and entity type when sample
   support allows. Preserve emitted alerts; do not retroactively cancel them as
   normal traffic accumulates. Calibrate the combined detector after fusion.

3. **One busy endpoint crowds out the other endpoint's context.**
   `windows.py:5` merges the two histories and keeps only the newest eight events.
   The probe excludes the quiet endpoint's only prior event when the other has
   eight newer events. A fixed event-count window also spans very different
   durations for busy and quiet nodes.
   **Solution:** reserve slots per endpoint, deduplicate, and fill spare slots
   globally. Test recent plus time-stratified/rare-history slots at the same
   total budget before enlarging the window. Log actual window duration and
   endpoint coverage. Longer windows increase compute and require an experiment.

4. **Attention selection is neither an anomaly criterion nor a lifetime budget.**
   `model.py:359,394` takes the top half in each query, then permanently ORs
   selections while the event remains active. With tied attention and rarity/
   novelty disabled, the probe retains all eight most recent evictions. Exception
   union can increase retention further. Attention receives relation-prediction
   gradients; hard selection does not directly learn which evidence improves
   later attack detection. Local rarity also treats new entities as exceptional,
   and LRU expiration can make previously seen behavior appear rare again.
   **Solution:** measure actual eviction-retention rate and exception rate;
   choose retention at eviction using an explicit per-entity budget, temporal
   diversity and a reserved rare-event quota. Ablate novelty/rarity thresholds
   using benign retention and alert budgets. Do not assume “top half” saves half
   the computation: selection currently occurs after GAT and CNN execution.

5. **Long memory retains history longer than its learning credit.**
   `model.py:187` detaches state after every optimizer step (`run.py:133`).
   In the two single-event-batch probe, a later loss trains the memory reader
   but gives zero gradient to the earlier source/destination writes and admission
   gate. At batch 1024 this affects dependencies crossing batch boundaries;
   it does not mean all memory writes are untrained. Stored compressed tokens
   also use older encoder weights while active raw events are re-encoded.
   **Solution:** measure entity recurrence distance relative to batch boundaries;
   test bounded multi-batch truncated backpropagation with the optimizer step
   delayed until the unroll ends, or causal replay chunks with a burn-in prefix.
   Add a future-context auxiliary target within that horizon. A slowly updated
   target encoder can reduce representation drift, but changes training and
   needs a separate ablation. No unrestricted backpropagation through the entire
   stream is necessary.

6. **Every event erodes node memory; elapsed time does not control the update.**
   `model.py:370,439` imposes an admission floor of 0.05 and multiplies by fixed
   fast/slow strengths 0.5/0.005. Even unselected repetitive events halve a stored
   contribution after about 27.4 fast-memory or 2772.2 slow-memory incident
   updates; larger admissions erase it faster. A long idle gap causes no direct
   long-memory decay. This mechanism is confirmed; its F1 impact is unmeasured.
   **Solution:** test removal/reduction of the forced-write floor alongside a
   budgeted admission loss and protected rare-event memory. Add bounded elapsed-
   time decay or last-seen features to memory reads. Compare event-count and
   wall-clock retention separately; changing decay can also discard useful APT
   evidence.

7. **Compressed-memory supervision has a range and objective mismatch.**
   `model.py:134,453` constrains compressed values with Tanh, then matches them
   directly to an encoder output whose LayerNorm is not restricted to [-1,1].
   In the initialized probe 32.81% of target coordinates exceed that range;
   minimum coordinatewise MSE at that fixed target is 0.14694. The target can
   change during training, so this is not a universal training-loss lower bound.
   The target is also the observed event embedding rather than its contextual
   history, potentially favoring a shortcut that discards context.
   **Solution:** use a decoder from the compressed token into the target space,
   or normalize both representations and use a cosine objective. Test predicting
   a detached contextual representation or future attributes. Compare against
   removing this auxiliary loss; its weight 0.01 is presently a heuristic.

8. **The upgraded encoder remains lossy for command/path order.**
   `encoding.py:91` uses a weighted sum with linear positional weights. The probe
   produces equal embeddings (within floating-point tolerance) for `a b b a`
   and `b a a b`: each token has the same summed positional weight. Thus position
   weighting improves some cases but is not a full sequence encoder. Hashing
   into 5303 buckets and truncating values to 32 subtokens introduce additional
   possible information loss; their actual corpus collision/truncation rates
   have not been measured.
   **Solution:** use a small shared subtoken CNN or positional attention encoder
   for values, retaining unordered pooling over fields. Audit collisions and
   truncation first; use a measured vocabulary or multiple independent hashes
   if necessary. Reallocate parameters from dense CNN layers if matching the
   detector parameter budget remains necessary.

9. **Graph and memory propagation are restricted.**
   The base KV branch sees a one-hop incident event neighborhood. In
   `model.py:371,435`, long-memory write content is formed from local context
   and the observed event before short-memory fusion and before endpoint-memory
   reads. Older endpoint summaries therefore do not directly transfer into the
   other endpoint's write. This can limit multi-stage history propagation.
   **Solution:** first ablate relation-aware history sampling or bounded two-hop
   context. For explicit memory-to-memory propagation, incorporate pre-event
   endpoint reads into messages. Such state-dependent writes invalidate the
   current event-independent affine scan; use a new causal recurrence or a
   separately justified approximation and measure its throughput cost. Merely
   stacking the ORTHRUS frontend already failed to fix this run's alert quality.

## Findings affecting efficiency

10. **Overlapping windows repeat expensive dense convolution.**
    `model.py:354,413` processes nine positions through three dense 128-channel,
    kernel-three convolutions, then repeats the same CNN for nonempty compressed
    windows. A full nine-position path costs about 1.327 million convolution
    multiply-accumulates per event, excluding normalization and other layers;
    two paths cost about 2.654 million. Shared weights save parameters, not
    execution. The upgrade already skips empty compressed windows.
    **Solution:** profile GAT/CNN/encoding separately on the real GPU stream;
    compare depthwise-separable or narrower temporal convolutions, and attention
    pooling for compressed history. Persistent convolution caching is not a
    drop-in equivalent because each endpoint pair changes the neighborhood and
    GAT features. Frozen-inference event-embedding caching can be exact if keys
    include the full input and checkpoint identity; training must recompute
    learned embeddings after weight changes.

11. **Python metadata work interrupts the GPU path.**
    `model.py:373` copies GPU selections/novelty decisions to CPU and then loops
    over events, dictionaries, queues and lists before constructing more GPU
    tensors. `encoding.py:97` sends several separately allocated metadata arrays
    to the device. `run.py:134` also reads the loss scalar every batch. These are
    confirmed synchronization/allocation sites; their fraction of GPU runtime
    has not been profiled locally.
    **Solution:** use indexed ring buffers, batched gathers and a tensor-based
    retention path; pack metadata transfers and use reusable pinned buffers with
    properly ordered asynchronous transfers. Compile tensor-only submodules
    after profiling. Test BF16/TF32 separately, keeping the affine scan and
    score/calibration computations in FP32 until numerical parity is established.
    PyTorch's official tuning guide supports reducing CPU/GPU synchronization
    and testing fusion/mixed precision:
    https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

12. **One retained token pins its entire batch allocation.**
    `TokenRef` and `detach_state` keep the full backing tensor plus a row index.
    After other rows expire, the probe retains one 256-byte token but keeps its
    16,384-byte, 64-row allocation alive. At batch 1024 the same mechanism can
    pin a 256 KiB block for one 256-byte row. This is storage amplification,
    not a measured 1024-fold increase in whole-process memory.
    **Solution:** compact surviving detached rows into owned storage periodically
    or when block occupancy falls below a threshold; preserve one shared copy
    for duplicate references. Test prediction parity, snapshot continuation,
    storage use and compaction overhead. A pooled slab/ring allocator is a
    scalable alternative to per-token cloning.

13. **Per-entity bounds do not bound the whole stream's state.**
    History/short queues and the sparse long-memory table grow with observed
    entities; inactive entities are never retired. The hybrid additionally
    retains global ORTHRUS feature caches and neighbor tables, and stores graph
    outputs from their observation-time weights. Recorded peak allocations are
    about 1.61 GiB KV, 3.74 GiB ORTHRUS, and 6.51 GiB hybrid. The KV arm is slower
    despite using less GPU memory.
    **Solution:** use explicit entity lifecycle/TTL policies with cold-state
    offloading for histories that must survive long gaps, rather than silent
    deletion. Compact inactive raw-event records. Evaluate the frontend as a
    replacement for redundant graph work before stacking both graph encoders.

14. **Evaluation materializes large Python event collections.**
    `run.py:136` saves every validation/test tuple and writes JSON per test event;
    `scoring.py:61` then copies complete collections to NumPy arrays. This adds
    memory and serialization work to model evaluation.
    **Solution:** write buffered columnar chunks, retain streaming top-k node
    accumulators, and calibrate using memory-mapped sorted arrays or explicitly
    validated quantile sketches. Report model-only scoring time separately from
    calibration, disk I/O, metric computation and tracing.

## Experiment order and acceptance criteria

1. Re-score saved edge outputs, if available, to isolate each calibration term;
   report raw and calibrated AP and operating points. No model retraining is
   needed for this stage. Local exports currently omit per-node score arrays.
2. Implement storage compaction and profile the existing GPU runner before
   architectural changes. Require causal prediction/gradient parity for exact
   optimizations; measure events/s and peak CPU/GPU memory on identical streams.
3. Train one-change ablations: balanced/time-stratified histories; eviction-time
   retention; normalized/decoded compression target; then bounded multi-batch
   credit. Test memory decay/floor and value-sequence encoding separately.
4. Evaluate lighter CNN/short-memory processing with an explicit accuracy/runtime
   tradeoff. Keep the hybrid optional pending evidence of useful detection gains.
5. Use several seeds and a separate development capture to select changes. Keep
   a held-out attack test for final reporting. Retain chronological processing,
   current-target masking, split-state discipline and ground-truth coverage.

Seven structural diagnostics passed on local PyTorch 2.11 CPU. Existing tests:
28 passed; one failed because the historical parity fixture
`artifacts/sliding_node_mem_e3/performance_before/model.py` is absent. This is a
missing validation artifact, not a demonstrated detector failure. The saved
cloud runs use a different pinned environment. No local GPU is available, so
GPU bottleneck percentages and post-fix accuracy/speed gains remain unverified.
