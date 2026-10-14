# Single-file old versus new GAT pilot

Upload `TGDetect_Old_vs_New_GAT.ipynb` to Colab, select GPU, and Run all.
The notebook embeds a ~3.4 MB ZIP containing real prepared CADETS prefixes and
frozen original KV encoder weights exported as NumPy arrays. No repository,
separate data upload, Drive credentials or download API is required.

Old: two original additive GAT layers. New: one role-factored four-head layer.
Default 16 epochs × three seeds × two operators; batch 256, two-batch unroll.
The new arm changes both depth and operator, so attribution needs the broader
eight-arm study later. The KV encoder is frozen from the same benign-trained
V3 checkpoint, not retrained independently. CNN, memories, decoder, planner and
losses remain the original V3 components, trained jointly outside the encoder.

Included prefixes: earliest 32,768 train / 16,384 validation / 262,144 test
events, with compact identity/node indexing and unchanged within-split ordering.
The source dataset has 72 ground-truth IDs; inspect the bundled manifest for
the exact source count and prefix coverage (13 truth nodes in test). This is a
limited pilot with node labels; attack-family/minority annotations are absent.
No full-CADETS or minority-family result should be claimed. Existing historical
test-informed development and full-benign-corpus encoder training are disclosed.
Compact node remapping can alter count-min collision patterns; it is shared
between arms and is not numerical parity with the original full-data stream.

Benign validation thirds select checkpoints, fit relation calibration and set
a 1% node-window alert budget. This budget does not guarantee node-level test
FPR. When the calibration sample cannot support the budget, threshold=null and
calibration.supported=false explicitly flag unsupported operating-point metrics.
AP can still be reported. Test truth does not select checkpoints or thresholds.

Degree auditing runs separately. Pair timings alternate arm order and replay
identical prefixes with fresh memory, three repetitions per seed, eight warmup
and 64 measured batches. Timing includes full-model forward, prepared host input
transfer, state update/detach, but not raw parsing, cache construction, scoring
or arrival wait. Original frozen FP32 KV lookup is used for both timing arms.

The automatic download is `tgdetect_old_vs_new_gat_results.zip`: raw node/edge
scores, histories, checkpoints/hashes, degree audit, source snapshot, manifest,
conditional paired AP bootstrap intervals, and per-batch timings.

Local validation: 24 tests pass, including a frozen-encoder full-model pipeline
test. Both arms also completed a one-epoch standalone-bundle CPU smoke run on
96 real events per split. Calibration was correctly flagged unsupported for that
tiny smoke sample. This proves execution only; GPU metrics remain unmeasured.
Bundled imports require NumPy, PyTorch and scikit-learn, not torch_geometric.

The prepared-data inputs and original encoder checkpoint are local trusted
artifacts. The build checks source event/token checksums and checkpoint token
configuration before repacking, then writes payload/source SHA256 hashes.
