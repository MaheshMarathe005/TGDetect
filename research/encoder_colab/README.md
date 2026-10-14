# V3 encoder-only Colab pilot

Latest study: `TGDetect_Rarity_Stability.ipynb` compares original and frozen
log-count rarity at 8/12/16 epochs and measures vectorized count lookup plus
stale/noisy/missing-count robustness. See [stability protocol](STABILITY_PROTOCOL.md).

After reviewing the first ZIP, use `TGDetect_Rarity_Followup.ipynb` for the focused
next experiment. It tests six arms, five seeds, and two fixed data splits with
equal eight-epoch budgets. See [research plan](RARITY_RESEARCH_PLAN.md) for each
change, its hypothesized effect, and the control needed to interpret it.
The original notebook remains the first-study artifact.

Upload `TGDetect_Encoder_L4.ipynb` to Google Colab, select a GPU runtime, and run
all cells. The notebook includes the experiment and a source snapshot of the
original V3 encoder; no repository clone, Drive mount, or credentials are needed.
The setup checks the actual GPU and available memory. L4 is the target; other
CUDA GPUs are allowed and recorded. CPU execution requires an explicit override.
The default PyTorch allocator budget is 15 GiB; the CUDA context and other
non-PyTorch allocations are outside that allocator limit. Default batch size is 128.

The small public KDD Cup 1999 10% download is deduplicated **by encoder input**
before splitting. Only benign records enter learning, category/quantile fitting,
Word2Vec initialization, rarity counters, checkpoint selection, score fusion, and
threshold calibration. Test attack labels are used only for final metrics. This
is an encoder pilot on a deliberately sampled population, not a temporal TGDetect
evaluation or a modern intrusion benchmark. Exact deduplication does not remove
all near-duplicate connections. Dataset order has no verified timestamp meaning.

The baseline is the original `PackedEncoder`, with its original tokenizer,
two-hash sum, typed field MLP, and four-head query pooling. All arms share its
SGNS token initialization and a small service-reconstruction decoder. Service is
removed from encoder inputs. There is no GAT, CNN, or endpoint memory. An
additional one-layer field interaction block is the interpretation of the
unspecified "idea 2" in the request.

Default matrix: old; old with matched extra updates; rarity only; interaction;
interaction with extra updates; interaction plus masked pretraining; rarity plus
interaction; rarity plus interaction with extra updates; full; full plus
contrastive field dropout. Three paired seeds, fixed data splits, identical
shared initial weights, identical task batches, three task epochs. Masked recipes
have an extra pretraining phase, explicitly matched by the extra-update controls
on the same encoder architecture for optimizer-update count (not FLOPs). Every arm also uses the original benign-path
SGNS initialization adapted to the training categorical-value corpus.

Field rarity uses a NEW count-min sketch over exact `(key, value)` strings.
TGDetect's endpoint/signature counters cannot supply this statistic. Training
counts are read before each record is inserted, once in a fixed ingestion order;
they are reused across epochs. Validation/calibration/test read a frozen training
snapshot, without updates. This is a causal-prefix feature construction, not a
claim of a chronological dataset split. No record counts itself. Counts are
converted with `log1p`; key-conditioned FiLM modifies the field vector. The
notebook audits sketch overestimation against exact counts. Counts are constructed
on CPU before training; encoder latency includes FiLM but excludes online counting.
An untrained rarity-only anomaly-score control is also reported.

Masked-field learning hides value, kind, numeric feature, AND rarity; only the
key remains. Numeric targets use benign-training quantile bins, categorical
targets a bounded benign-training dictionary with an explicit unknown class.
Masked scoring performs separate masked views and is not free. Its cost and the
fraction of unseen targets are reported. The label being reconstructed by the
shared service head is never a masking target.

Metrics: AP, ROC-AUC, F1, precision, recall, benign false-positive rate, confusion
counts, recall at validation-calibrated 1%/5% FPR targets, precision@k, and test
false alerts/1,000 benign records. AUC is secondary. Paired stratified-bootstrap
AP-difference intervals hold the test population fixed; three-seed spread is
reported separately. These intervals do not capture capture-to-capture variation.
Benign calibration is split: one half sets per-channel empirical-CDF fusion,
the other sets conservative thresholds. Test labels never select fusion weights,
thresholds, checkpoints, or winners.

Latency reports distinguish encoder from service-plus-masked scoring, cache
construction from warm lookup, and preparation from execution. Benchmarks use actual test identity
requests plus a disclosed synthetic repeat-heavy request stream. Field cache is
before rarity/interaction. Whole-event frozen tables require a frozen count
snapshot and are disabled for changing rarity. Cache memory/build time and cache
coverage are recorded. No unlimited-cache timing is called LRU performance.

Optional cells benchmark BF16, compact token bags, field-vector tables, frozen
event tables, `torch.compile`, fixed-shape CUDA graphs, int8 token storage, and
teacher-to-64D distillation. They do not silently replace primary FP32 results.
IDF truncation and list-order invariance are controlled synthetic probes because
KDD has neither >64 fields nor list/path values. They cannot establish security
accuracy for those changes.

Outputs are under `/content/tgdetect_encoder_run/results`: per-run checkpoints,
histories, scores, CSV/JSON summaries, raw timings, data/split/source hashes,
environment details, and a downloadable zip. Failed arms are recorded and stop
the default run rather than silently disappearing.

Sources: [KDD archive](https://kdd.ics.uci.edu/databases/kddcup99/),
[dataset description](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html),
[scikit-learn's pinned download mirror/checksum](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_kddcup99.py),
[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html),
[CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

Local validation does not require a GPU:

```bash
PYTHONPATH=research/encoder_colab:research /home/mahesh/main_env/bin/python -m pytest -q research/encoder_colab/test_experiment.py
python3 research/encoder_colab/build_notebook.py
```
