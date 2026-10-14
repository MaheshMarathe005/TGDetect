# Paper-to-experiment audit

## ORTHRUS (Jiang et al., USENIX Security 2025)

Read supplied prepublication PDF; final bibliographic record:
https://www.usenix.org/conference/usenixsecurity25/presentation/jiang-baoxiang
Pages 7173–7192. Local pinned implementation is in `research/orthrus_official`.

Contribution: conservative node-level attack ground truth, dynamic temporal
neighbor GraphTransformer encoder with Word2Vec/type features, relation
reconstruction, and backward/forward dependency tracing for attribution. It
compares six DARPA E3/E5 capture/dataset combinations, not just CADETS E3. It
distinguishes ORTHRUS-ano from the full reconstruction system. Our detector-only
comparison must not be labelled a reproduction of ORTHRUS-full or its QoA.

Paper Table 4 reports E3-CADETS ORTHRUS-full TP=25, FP=23, FN=43, and ORTHRUS-ano
TP=10, FP=0, FN=58. These are **published numbers**, not our experiment. Our
five-epoch calibrated result has TP=17, FP=9687. The discrepancy requires clear
protocol disclosure; it cannot be used to claim the paper's system is weak.
The released CADETS command uses 20 epochs and top-30 clustering. Our shared
calibration is a different operating rule. Both common-protocol and released-rule
diagnostics are needed. Its success against KAIROS is not transitive evidence for
our model against KAIROS.

## KAIROS (Cheng et al., IEEE S&P 2024)

Read supplied preprint. DOI 10.1109/SP54263.2024.00005; author paper
https://arxiv.org/abs/2308.05034; upstream https://github.com/ubc-provenance/kairos.
Local authors-traced module provenance pins commit
`0e0b633beb46a1117c0a6d63be5d2481b59ac0dc`.

Hierarchical feature hashing, TGN memory/GRU, two TransformerConv layers,
relation decoder, rare-node correlated time-window queues, and community-based
attack summaries are separate components. Shared node-level reconstruction
evaluation tests the neural detector, not the native queue/investigation system.
The existing DEDALE adapter cannot be relabelled as a CADETS result. A dedicated
CADETS adapter and direct cloud run are required. Native CADETS code trains 50
epochs; a five-epoch budget-matched comparison alone is insufficient for a broad
state-of-the-art claim.

## Pomsathit (IEEE Access 2025)

Read supplied PDF including rendered page 6 (Table 4).
DOI 10.1109/ACCESS.2025.3639270, volume 13, pages 212973–212982.
Paper specifies three 128-D attention-based GNN layers, ReLU/dropout 0.2, Adam
1e-3/weight decay 1e-5, batch 256, 200 epochs per scenario, reservoir replay of
10% of past samples, and five independent runs. It studies supervised node
classification and continual adaptation on Wazuh/sandbox data.

Its Table 4 reports proposed F1 0.981 ±0.002, static GNN F1 0.955 ±0.003, and
latency 6.7 versus 4.1 ms. These numbers are not comparable to unsupervised
CADETS node detection: data, label access, prevalence, hardware, batch size and
task differ. No public data/code link resolving the exact experiment was found
in the supplied PDF. The repository already correctly treats exact reproduction
as unresolved (`pomsathit_baseline.POMSATHIT_PAPER_SPEC`). Do not invent missing
attention heads, replay loss weight, data splits, or label mappings. Cite its
ideas in related work; explicitly label any local GNN baseline as our adaptation.

## Required evidence and limits

- Independent seeded runs with means and dispersion; count actual runs.
- Same chronological events, labels, scoring unit and validation-only choices.
- Report AP/MCC, TP/FP/FN and alert load, not accuracy alone.
- Give full learned parameter counts and state-memory policy.
- Profile trained inference separately from seeded-weight latency; include batch
  size, synchronization, warmup, hardware, repetitions and preprocessing scope.
- Component ablations and an unchanged full-model comparator.
- Explain pretraining scope, original versus common scoring, and budget mismatch.
- Preserve test reuse disclosure; no claim of an untouched prospective benchmark.
- No claim of high-quality attribution without attack reconstruction evaluation.
- Bound conclusions to the datasets and completed trials actually available.
