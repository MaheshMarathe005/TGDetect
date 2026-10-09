# Research continuation — 23 September 2026

26 September continuation: [data volume, duration and persistent-memory research](research/findings/memory_and_data_scale.md), with [complete 60-run findings](research/findings/component_results.md) and [measured audit JSON](research/findings/memory_horizon_audit.json). The earlier pilot discussion below is historical and does not replace the completed 200-epoch comparison.

This note records sources examined and implementation implications. Results and
decisions remain in ASTRA_RESEARCH_LEDGER.md; the pilot protocol was registered in
ADAPTED_PROTOCOL.md before real-data training.

| Source examined | Verified point | Implication / test in this project |
|---|---|---|
| [Original TGN code](https://github.com/twitter-research/tgn/blob/master/model/tgn.py), especially compute_temporal_embeddings/get_updated_memory | The memory-at-start path computes updated memory from stored messages before embedding computation. | Gradient scheduling matters. Our adaptation uses within-block backpropagation and tests all GRU parameter gradients; it does not claim to duplicate TGN's implementation. |
| [TGAT paper](https://arxiv.org/abs/2002.07962) | Temporal attention and functional time representations are architectural choices. | Isolate a numeric-age encoder from memory. Our cosine-on-log-age map is a simplified adaptation, not exact TGAT. |
| [Temporal Graph Benchmark](https://arxiv.org/abs/2307.01026) | Evaluation across datasets finds performance varies and simple methods can be strong. | Include feature-only controls; one capture cannot establish general superiority. |
| [Sometimes Simpler is Better, USENIX 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/bilot) and [released paper](https://www.usenix.org/system/files/usenixsecurity25-bilot.pdf) | Authors compare eight provenance IDSs and demonstrate a competitive simpler network. The landing abstract and PDF differ in dataset-count wording. | Do not quote a single dataset-count claim without a version; prioritize matched simple controls and exact artifact identity. Our flow pilot cannot reproduce their endpoint benchmark. |
| [PIDSMaker artifact](https://zenodo.org/records/15603122) and [author repository](https://github.com/ubc-provenance/PIDSMaker) | A public common framework and archived artifact are available for provenance IDS comparisons. | Candidate next-stage endpoint benchmark; inspect dataset licensing, labels, preprocessing and version before adoption. No code imported this session. |
| [ORTHRUS, USENIX 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/jiang-baoxiang) | Combines temporal graph learning and dependency analysis for node-level attribution. | Attribution/reconstruction is a separate factor and target. Do not compare its reported node metrics directly to our flow F1. |
| [METANOIA v2](https://arxiv.org/abs/2501.00438v2) | Studies lifelong intrusion detection under concept drift. | Single-capture rehearsal does not measure forgetting. Need staged scenarios, retention curves and drift-specific controls. |
| [CTU-13 owner documentation](https://www.stratosphereips.org/datasets-ctu13) | The owner recommends bidirectional labeled flows and identifies the CTU server as authoritative. | Download full text records directly, preserve hashes, and explicitly distinguish Botnet/Normal/Background. |

The leading engineering hypothesis is that meaningful temporal comparison requires
correct observation times and learnable state before additional architecture
complexity. It is a hypothesis about this implementation, not evidence that a
particular published method is inferior.

## What this pilot isolates

- B: three-layer, 128-wide adapted attention GNN with a two-class event-node head.
- T: one explicit numeric-age representation; all other inputs stay fixed.
- M: persistent endpoint state using a fixed average update.
- R given M: replace the fixed update with a learned GRU.
- H: reservoir training replay with documented stale memory snapshots.

The T/M pair and R replacement can be estimated; R without M is undefined here.
F/G/L/N alternatives, capacity matching, explicit temporal message-passing changes,
unseen-capture testing and the complete factorial/reverse study remain future work.

The new implementation passes strict-past graph tests, future perturbation tests,
evaluation batching checks and GRU-gradient checks. The old implementations remain
preserved with their original known failures. No historical scores are reused.
