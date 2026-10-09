# TG-Detect — Astra From-Scratch Research / Rebuild Prompt

## Mission

Work on the uploaded `TG-Detect` project as a **research-grade, skeptical ML engineer and reproducibility auditor**.

The goal is not to prove that TG-Detect is better. The goal is to determine, from first principles and verifiable evidence, whether a Temporal GNN (TGNN/TGN-style architecture) provides a genuine improvement over the base-paper GNN + rehearsal system, and exactly which component(s) cause any observed change.

The base paper to reproduce is:

**Auttapon Pomsathit, “Adaptive Detection of Advanced Persistent Threats (APT) With Graph Neural Networks and Rehearsal-Based Continual Learning on Wazuh EDR Telemetry,” IEEE Access, 2025, 13, 212973–212982, DOI: 10.1109/ACCESS.2025.3639270.**

A copy of the paper is already inside the project as a PDF. Also verify it against a public copy / DOI / publisher-indexed source and use the web for related-work research.

---

## 0. HARD RESET — DO THIS BEFORE RESEARCH

### Critical rule
Treat **all previous TG-Detect research conclusions, benchmark numbers, TODOs, presentations, experiment logs, and written arguments as untrusted prior hypotheses**. Do not use them as evidence and do not inherit their conclusions.

You may inspect existing source code because it is the object being audited, but do not let comments, old markdown, old benchmark summaries, old plots, or old conclusions decide what is scientifically correct.

### First action: clean the active research workspace
Before changing model architecture or running experiments:

1. Inventory the repository.
2. Identify old research/analysis artifacts and generated outputs.
3. Remove the old research artifacts from the active workspace so they do not bias the work.
4. Preserve the actual executable/source architecture, datasets, configuration needed to run it, and the **base-paper PDF**.
5. Do not silently delete source code just because it contains old comments; only remove files that are clearly prior research notes, result dumps, stale presentations, caches, or disposable experiment scripts.
6. Create a fresh file named:

`ASTRA_RESEARCH_LEDGER.md`

This file becomes the **only authoritative research memory** for this project.

### Known files that must NOT be treated as evidence
Examples include, but are not limited to:

- `RESEARCH_AND_FIXES.md`
- `TODO.md`
- `PLAN.md`
- `PRESENTATION.md`
- `presentation.html`
- old `results/` JSON/log outputs
- `bench_temporal.log`
- `.labtest/research/`
- extracted/copied paper text such as `.labtest/paper.txt`
- `__pycache__/`
- old debug/ablation scripts whose only purpose is reproducing earlier conclusions

If a file is ambiguous, inspect it first and record the decision in `ASTRA_RESEARCH_LEDGER.md`.

Do NOT delete the base-paper PDF.

---

# 1. EXPLAIN THE SCIENTIFIC PROBLEM FIRST

Before writing or changing code, write a very simple explanation in the ledger:

### The core problem
The comparison is currently at risk of being **apples-to-oranges**.

A claim such as:

> “The paper uses a GNN + 10% rehearsal, while TG-Detect uses a TGNN + 10% rehearsal, so TG-Detect should logically beat it.”

is **not sufficient scientific reasoning**.

A stronger model can contain several advantages at once:

- temporal encoding,
- persistent memory,
- different aggregation,
- different feature encoder,
- better loss,
- different normalization,
- different graph construction,
- different labels,
- different train/test split,
- different sampling,
- different threshold selection,
- different leakage exposure,
- more parameters,
- more training compute,
- different replay behavior,
- extra context/features.

If multiple things change together, a performance difference cannot be attributed specifically to temporal graph modeling.

The research question therefore becomes:

> **Under the same task, same information, same data split, same training budget, same rehearsal policy, and same evaluation protocol, does adding temporal state/time-aware message passing to the paper-faithful GNN improve the result?**

Only after answering that question should we test larger TGNN designs.

---

# 2. REBUILD THE BASE-PAPER SPECIFICATION FROM THE PAPER ITSELF

Read the entire local PDF, not only its abstract.

Also search the web for:

1. the DOI/publication record;
2. author or project code, if publicly available;
3. supplementary material, implementation notes, datasets, or repositories directly attributable to this paper;
4. later papers that cite or implement the exact method;
5. original foundational papers for any method the paper names.

Do not replace the paper specification with a later paper's interpretation.

Create:

`BASELINE_PAPER_SPEC.md`

with a table containing:

| Item | What the paper explicitly states | Exact source/page/equation | Implementable exactly? | Missing/ambiguous detail |
|---|---|---|---|---|

At minimum capture:

- task definition;
- graph definition `G=(V,E,X)`;
- node types;
- edge semantics;
- node feature semantics;
- label definition;
- graph/window construction;
- GNN depth;
- hidden dimension;
- activation;
- dropout;
- aggregation;
- attention details;
- output head;
- output target (node classification vs event/edge classification);
- softmax vs sigmoid;
- loss;
- rehearsal buffer;
- buffer size;
- reservoir sampling;
- replay sampling;
- VAE/generative replay statement, if applicable;
- rehearsal weight `lambda`;
- optimizer;
- learning rate;
- weight decay;
- batch size;
- epochs;
- number of seeds;
- hardware;
- datasets;
- train/validation/test procedure;
- metrics;
- threshold selection;
- reported resource measurements.

### Important ambiguity rule
The paper describes a generic GNN update:

`h_v^(l+1) = sigma(W^(l) * AGGREGATE(h_v^(l) U {h_u^(l): u in N(v)}))`

and says the aggregation can be attention-based, but it does **not** fully specify every implementation detail needed to reconstruct a unique neural network.

Do not invent missing details and call the result “exact reproduction”.

Mark every missing item as:

- `EXPLICIT`
- `DERIVED`
- `AMBIGUOUS`
- `NOT REPORTED`

If an external author repository resolves an ambiguity, cite it as external implementation evidence, not as a statement from the paper.

---

# 3. BUILD A PAPER-FAITHFUL REFERENCE IMPLEMENTATION FIRST

Create one canonical implementation, separate from the TGNN code.

Suggested location/name:

`pomsathit_baseline.py`

or an equivalent dedicated baseline module.

### Hard constraint
Do **not** improve it.
Do **not** add temporal memory.
Do **not** add GRU memory.
Do **not** add time encoding.
Do **not** add contrastive learning.
Do **not** add domain-adversarial training.
Do **not** add custom focal loss unless the paper explicitly uses it.
Do **not** add extra normalization or residual blocks unless the paper explicitly specifies them.
Do **not** change the number of layers/hidden dimension.
Do **not** add identity embeddings merely because they help.
Do **not** add engineered temporal features if the goal is to test temporal architecture.

The first baseline must implement only the paper specification.

### Baseline acceptance test
The baseline must have a machine-readable architecture manifest, for example:

`BASELINE_ARCHITECTURE.json`

containing:

- parameter count;
- tensor shapes at every layer;
- input dimensions;
- output dimensions;
- all trainable modules;
- aggregation equation;
- activation locations;
- dropout locations;
- optimizer settings;
- rehearsal settings;
- random seed policy.

Write a tiny unit test that checks every item.

---

# 4. IMPORTANT: AUDIT THE EXISTING “PROMETHEUS/POM SATHIT” IMPLEMENTATION

The project currently has more than one purported implementation of the literature baseline.

Audit every one.

In particular inspect:

- `modal_common.py` → `PomsathitGCN`
- `modal_bench_real.py` → `PomsathitGCN`
- `modal_train_v16.py` → `V13_GNN_Rehearsal`
- any other baseline aliases or old model factories.

Do NOT assume they are equivalent.

Produce a side-by-side table:

| Component | Paper | Existing implementation A | Existing implementation B | Exact? | Consequence |
|---|---|---|---|---|---|

Pay special attention to:

- 3 layers vs 2 layers;
- 128 hidden vs 64 hidden;
- node classification vs edge/event classification;
- softmax vs binary sigmoid;
- node features vs event/netflow features;
- attention aggregation implementation;
- dropout;
- weight decay;
- rehearsal implementation;
- reservoir sampling;
- graph construction;
- initialization;
- target labels;
- temporal information sneaking into a supposedly static baseline.

### Current project clue to verify, not to assume
The repository appears to contain a `V13_GNN_Rehearsal` implementation using a smaller architecture than the paper, while another `PomsathitGCN` is closer to the stated 3-layer/128-hidden configuration. Treat this as an audit target and verify every detail from code rather than trusting this note.

---

# 5. SEPARATE “PAPER-EXACT” FROM “PAPER-ADAPTED TO TG-DETECT DATA”

This is mandatory.

There are two possible outcomes:

### A. Exact reproduction possible
If the paper's original data/task definition and implementation details are available, reproduce it as closely as possible.

### B. Exact reproduction impossible
If the paper does not provide enough information or its Wazuh dataset/task cannot be recreated, do NOT fake an exact reproduction.

Instead create two clearly labeled systems:

1. `POMSATHIT_PAPER_SPEC`
   - faithful to everything explicitly specified;
   - missing details documented.

2. `POMSATHIT_ADAPTED`
   - adapted to TG-Detect's dataset/task;
   - every adaptation explicitly documented.

Never call the adapted model “exact”.

---

# 6. DEFINE THE FAIR TGNN EXPERIMENT

Only after the baseline passes its audit, compare it to the temporal model.

The first TGNN must be a **controlled superset** of the baseline.

Start with:

`Paper GNN`

then add exactly one temporal mechanism at a time:

`Paper GNN + time encoding`

then:

`Paper GNN + causal temporal state`

then:

`Paper GNN + node memory`

then:

`Paper GNN + temporal message passing`

then the complete TGNN.

The first controlled TGNN experiment must keep constant:

- same data;
- same labels;
- same input features;
- same graph construction;
- same batch policy;
- same optimizer;
- same learning rate;
- same weight decay;
- same dropout;
- same training budget;
- same rehearsal policy;
- same random seeds;
- same evaluation code;
- same thresholding protocol.

Only the temporal mechanism should change.

---

# 7. DO NOT GIVE TGNN FREE TEMPORAL INFORMATION THROUGH FEATURES

This is one of the most important controls.

If the baseline receives precomputed features such as:

- inter-arrival time;
- pair inter-arrival mean;
- inter-arrival variance/CV;
- event frequency over future/large windows;
- temporal burst scores;
- “history” counters that effectively encode the time stream;

then the static baseline may already be receiving the temporal information the TGNN is supposed to model.

Conversely, removing temporal features from one model but not the other can create another unfair comparison.

Create explicit feature sets:

- `STATIC_FAIR`
- `TEMPORAL_ONLY_ADDITION`
- `FULL`
- `LEAKAGE_STRESS`

and document exactly what each contains.

Run raw-feature controls such as a simple Logistic Regression and a strong tree baseline whenever practical. If raw features alone nearly solve the task, architecture-level claims become much weaker.

---

# 8. REHEARSAL MUST BE CONTROLLED SEPARATELY

The “10% rehearsal” statement does NOT prove the TGNN is better.

Run the following complete factorial matrix:

1. GNN, no rehearsal
2. GNN, 10% rehearsal
3. TGNN, no rehearsal
4. TGNN, 10% rehearsal

Then isolate the temporal additions:

5. GNN + time encoding only
6. GNN + memory only
7. GNN + temporal message passing only
8. GNN + time + memory
9. full TGNN

If the paper's rehearsal method and the project's rehearsal method differ, make that difference explicit and run an additional matched-rehearsal control.

Do not silently mix reservoir sampling with generative replay.

The paper text should be checked carefully because it mentions both a rehearsal buffer updated by reservoir sampling and replayed samples generated from a lightweight VAE. Resolve this from the complete paper or attributable implementation if possible; otherwise report the ambiguity instead of inventing a mechanism.

---

# 9. AUDIT DATA LEAKAGE BEFORE BELIEVING ANY NUMBER

Build a leakage checklist and test it.

Check for:

- future timestamps accidentally included in features;
- features computed using the full dataset before splitting;
- entity IDs memorized across train/test;
- scenario IDs;
- malware-family identity leakage;
- source-domain leakage;
- labels influencing memory state;
- future graph edges included in earlier snapshots;
- normalization statistics fitted on test data;
- test-set threshold tuning;
- test data used during rehearsal;
- replay buffer crossing evaluation boundaries.

For every feature/state channel state whether it is:

`strictly causal`, `static`, `future-dependent`, `split-dependent`, or `unknown`.

A strictly causal detector must only use information available up to the current event timestamp.

---

# 10. RESEARCH WHAT CAN REPLACE EACH COMPONENT

After the exact baseline is established, do a fresh web literature sweep.

Research at least:

- GCN / GAT / GraphSAGE / GIN variants;
- TGAT;
- TGN;
- DyRep;
- temporal attention / temporal message-passing alternatives;
- memory-update alternatives;
- time-encoding alternatives;
- class-balanced replay;
- reservoir replay;
- herding / representative replay;
- generative replay;
- online concept-drift methods;
- graph-level vs event-level vs node-level prediction;
- provenance / endpoint / APT temporal-graph methods;
- recent APT temporal-graph papers, including CONTINUUM and other post-2024 work found through search.

The purpose is NOT to copy the newest method blindly.

For each candidate component create:

| Existing component | Candidate alternative | What changes mathematically | Why it might help | What new confound it creates | Extra compute/memory | Test needed |
|---|---|---|---|---|---|---|

Examples of mechanisms that may be relevant to investigate:

- TGAT-style temporal attention and learned functional time encoding;
- TGN-style persistent memory + message/update functions;
- DyRep-style dynamic interaction modeling;
- temporal point-process or inter-event modeling;
- pair-level memory for directional beacon/C2 behavior;
- heterogeneous node/edge types;
- drift-aware memory resets or adaptation;
- class-balanced or uncertainty-aware rehearsal.

Do NOT label any alternative “better” until the controlled experiment supports it.

---

# 11. ARCHITECTURE AUDIT — EVERY FUNCTION, EVERY COMPONENT

For both the paper-faithful GNN and TGNN, trace every function from raw input to output.

Make a dataflow diagram in text and, where useful, a Mermaid diagram.

For every function/module answer:

1. What tensor enters?
2. What tensor leaves?
3. What dimensions change?
4. What information is introduced?
5. Is the information causal?
6. Is it learnable?
7. Can it leak labels/future information?
8. What paper claim justifies it?
9. What experiment isolates its contribution?
10. What would happen if it were removed?
11. What would happen if it were replaced?
12. What memory/latency cost does it add?

Include small functions and utility functions when they affect the data path, state, sampling, or evaluation.

---

# 12. DO NOT TRUST THE CURRENT TGNN JUST BECAUSE IT IS MORE COMPLEX

Audit the current TGNN for:

- unnecessary components;
- duplicate encoders;
- redundant memory;
- unstable time encodings;
- hidden dimension mismatches;
- dead features;
- detached tensors that accidentally block learning;
- accidental information sharing between source and destination;
- memory update order bugs;
- incorrect `dt` definitions;
- batch-order dependence;
- evaluation-time state mutation;
- train/test memory contamination;
- hash collision effects;
- node vs pair memory mismatch;
- feature normalization mismatch;
- inconsistent loss definitions;
- threshold tuning mistakes;
- benchmark code paths that are not actually using the architecture they claim to use.

Run static inspection and small synthetic tests for these.

---

# 12A. MANDATORY FALSIFICATION / COMPONENT-CONTRIBUTION STUDY

This section is **mandatory**. The goal is not merely to compare “paper GNN vs full TGNN”. The goal is to experimentally determine the contribution of **every major change TG-Detect introduced** and to determine whether the final gain survives when each contribution is isolated and recombined.

Treat the following as independent candidate components. Do not assume the list is complete: first audit the code and expand this list with every architecture/data/training modification that can plausibly affect performance.

Initial required component set:

1. `F` = improved / engineered features
2. `G` = changed / improved graph construction
3. `T` = temporal encoding / time representation
4. `M` = persistent memory / temporal memory mechanism
5. `R` = GRU / recurrent update mechanism (if distinct from `M` in code)
6. `L` = changed loss function / loss weighting
7. `N` = changed normalization / scaling / normalization layer
8. `H` = rehearsal / replay mechanism, including the 10% rehearsal setting

If the implementation reveals additional independent contributors, add them as separate factors instead of hiding them inside another factor.

## Required experimental philosophy

The experiment must answer four different questions:

### Question A — What does each component do by itself?

Start from the same canonical base model and add exactly one component:

- `B` = canonical paper-faithful baseline
- `B+F`
- `B+G`
- `B+T`
- `B+M`
- `B+R`
- `B+L`
- `B+N`
- `B+H`

Every single-factor experiment must keep all other choices identical to `B`.

### Question B — What happens with pairs?

Run **every pairwise combination** of the independent components where computationally feasible.

For 8 initial factors this means all 28 pairwise combinations:

`B+F+G`, `B+F+T`, `B+F+M`, `B+F+R`, `B+F+L`, `B+F+N`, `B+F+H`,
`B+G+T`, `B+G+M`, `B+G+R`, `B+G+L`, `B+G+N`, `B+G+H`,
`B+T+M`, `B+T+R`, `B+T+L`, `B+T+N`, `B+T+H`,
`B+M+R`, `B+M+L`, `B+M+N`, `B+M+H`,
`B+R+L`, `B+R+N`, `B+R+H`,
`B+L+N`, `B+L+H`,
`B+N+H`.

Do not select only the pairs that are expected to help. The pair list must be generated mechanically from the factor list so that negative or neutral interactions are also visible.

### Question C — What happens when the proposed stack is combined?

Run the exact intended full TG-Detect configuration:

`B+F+G+T+M+R+L+N+H`

with no hidden extras.

Then produce a reverse ablation by removing one component at a time from the full stack:

`FULL-F`, `FULL-G`, `FULL-T`, `FULL-M`, `FULL-R`, `FULL-L`, `FULL-N`, `FULL-H`.

This is required because a component can be weak alone but important in combination, or strong alone but redundant once another component exists.

### Question D — What happens under stronger falsification?

When compute permits, enumerate **all non-empty subsets** of the independent factors. For 8 factors there are 255 possible subsets.

Do NOT automatically run all 255 at high seed count if this is prohibitively expensive. Instead:

1. run all 8 single factors;
2. run all 28 pairs;
3. run the full stack;
4. run all single-removal full-stack ablations;
5. rank components/pairs by effect size and uncertainty;
6. then run higher-order subsets that are scientifically necessary to resolve suspected interactions.

The ledger must explicitly record which subsets were run and which were not, and why.

## Quantify contribution, not just raw score

For every factor `X`, calculate at minimum:

`main_effect(X) = score(B+X) - score(B)`

For every pair `(X,Y)`, calculate at minimum:

`pair_interaction(X,Y) = score(B+X+Y) - score(B+X) - score(B+Y) + score(B)`

For the final stack, compare:

- `score(B)`;
- each single-factor model;
- each pairwise model;
- `FULL`;
- every `FULL-X` model.

Repeat this for F1 and PR-AUC, and where meaningful also recall, precision, FPR, latency, memory, and parameter count.

Do not call a component “important” solely because its raw model has the highest score. Use its isolated effect, reverse-ablation effect, uncertainty, and interactions.

## Required falsification tests

For the temporal components especially, include negative controls:

- shuffled timestamps while preserving all non-temporal data;
- randomly permuted temporal order within an evaluation group where causality allows the test;
- zeroed / constant time encodings;
- frozen memory;
- memory reset at every event/window;
- GRU replaced by identity/no-op;
- normalization replaced by the baseline normalization;
- improved loss replaced with the baseline loss;
- improved features removed;
- improved graph replaced by the baseline graph;
- rehearsal disabled;
- rehearsal shuffled or intentionally mismatched where appropriate;
- synthetic toy streams where temporal ordering is deliberately the only predictive signal.

A temporal method that shows the same improvement when temporal order is destroyed should be treated as suspicious until the cause is explained.

## Distinguish “component effect” from “implementation effect”

For each factor identify the smallest code-level change that activates it.

For example:

- `T`: only the time encoder changes;
- `M`: only persistent state changes;
- `R`: only recurrent/GRU state update changes;
- `L`: only the loss changes;
- `N`: only normalization changes;
- `H`: only rehearsal/replay changes.

If two mechanisms are mathematically inseparable in the current implementation, document that fact and create the closest possible decomposition rather than pretending they are independent.

Do not allow a factor labeled “better features” to secretly change graph construction, labels, split, normalization, or model depth. The same rule applies to every other factor.

## Keep data and compute fair

Unless an experiment is explicitly an efficiency experiment, keep constant:

- dataset version;
- train/validation/test split;
- sample ordering policy;
- input label definitions;
- evaluation code;
- threshold selection;
- number of training epochs / stopping rule;
- seed set;
- optimizer family;
- learning-rate policy;
- batch policy;
- data-loader behavior;
- precision/mixed-precision policy;
- maximum training compute budget.

If a component inherently changes parameter count, record that fact and report both raw performance and compute-normalized performance.

## Required experiment manifest

Create a machine-readable manifest such as:

`ABLATION_MATRIX.json`

Each row/config must specify:

- experiment ID;
- active factors;
- inactive factors;
- model class/hash;
- feature-set ID;
- graph-builder ID;
- loss ID;
- normalization ID;
- rehearsal ID;
- temporal encoder ID;
- memory ID;
- GRU ID;
- seed;
- dataset version;
- split ID;
- compute budget;
- Modal GPU type;
- status.

The experiment runner must fail loudly if an experiment claims one factor is disabled but its code path still activates it.

---

# 12B. ENVIRONMENT, MODEL-PROFILE, AND COMPUTE VALIDATION

Before expensive experiments, verify the complete execution environment locally and on Modal.

## Local environment audit

Record:

- OS;
- Python version;
- package manager;
- PyTorch version;
- CUDA runtime version;
- CUDA toolkit version if installed;
- cuDNN version if relevant;
- PyG / DGL / graph library version;
- NumPy / SciPy / scikit-learn versions;
- compiler/runtime versions that can affect extensions;
- Git commit or repository hash;
- dataset hashes;
- random-seed configuration.

Run a small smoke test locally before using cloud GPU time.

## Model profile

For every baseline, single-factor, pair, and full configuration record:

- total parameters;
- trainable parameters;
- parameter breakdown by module;
- tensor shapes;
- approximate FLOPs/MACs if tooling is available;
- peak GPU memory during a representative forward/backward pass;
- batch throughput;
- event throughput where applicable;
- wall-clock training time;
- persistent memory size;
- rehearsal-buffer size;
- model checkpoint size.

Do not infer that a score improvement is meaningful without understanding whether the model also became much larger or more computationally expensive.

## Modal execution requirement

Use **Modal** for GPU-heavy experiments after the environment has passed local smoke tests. Do not silently fall back to the local laptop for expensive training runs.

Create a reproducible Modal runner, for example:

`modal_experiments.py`

or the repository's equivalent.

The Modal runner must:

1. build a reproducible container image;
2. pin or record package versions;
3. verify Python/PyTorch/CUDA versions inside the container;
4. print `nvidia-smi`/GPU identity and available VRAM;
5. run a CUDA smoke test;
6. run one tiny training step;
7. verify deterministic seed behavior as far as the selected libraries permit;
8. persist logs/results/checkpoints to a durable location;
9. save the exact experiment manifest with the result;
10. fail before a long run if the expected GPU/runtime is unavailable.

Modal supports GPU resource requests on Functions and supports persistent Volumes for shared/persistent files; verify the current supported GPU/resource syntax from the current Modal documentation before implementing the runner. citeturn666954search4turn666954search2

## GPU selection policy

Do NOT hard-code a specific GPU merely because it was available previously.

At launch time:

1. inspect current Modal GPU availability/options;
2. select a GPU with enough VRAM for the largest expected experiment;
3. record the selected GPU type in the experiment manifest;
4. use a consistent GPU class for experiments that are being compared for efficiency;
5. if a different GPU must be used, do not compare wall-clock time directly without correction/qualification.

Modal's current documentation supports explicit GPU requests and cloud/region options, and interactive shells can also request GPUs; use the current syntax rather than copying an obsolete command. citeturn666954search11turn666954search9

## Separate cheap and expensive experiments

Run locally or on low-cost CPU/GPU resources:

- unit tests;
- tensor-shape checks;
- tiny synthetic datasets;
- leakage tests;
- deterministic tests;
- one-batch sanity checks.

Run on Modal GPU:

- full training;
- all seed repetitions;
- all pairwise ablations;
- full-stack/reverse ablations;
- large-data robustness tests;
- hyperparameter sweeps when required;
- memory/throughput profiling at realistic scale.

Use checkpointing so an interrupted cloud job does not invalidate completed experiments.

## Cloud reproducibility

Every Modal experiment must store:

- experiment ID;
- git commit/hash;
- container/image identifier;
- Modal GPU type;
- CPU/RAM allocation;
- dataset checksum;
- config checksum;
- seed;
- start/end time;
- result artifact path;
- stdout/stderr log path;
- model checkpoint path when applicable.

Modal execution is an implementation detail, not a source of scientific advantage. Do not treat a faster GPU as a model improvement.

---

# 13. EXPERIMENT DESIGN

Use a staged protocol.

### Stage A — correctness
- unit tests;
- deterministic toy graph;
- hand-computed message passing sanity check;
- one-event temporal causality test;
- memory update test;
- replay buffer test;
- zero-history test;
- new-node test;
- shuffled-time negative control.

### Stage B — paper baseline
- paper-spec architecture;
- exact stated hyperparameters;
- paper-style rehearsal;
- 5 independent seeds if computationally feasible;
- mean ± standard deviation.

### Stage C — controlled TGNN ablations
- one temporal feature/mechanism at a time;
- same compute budget;
- same data;
- same split;
- same rehearsal.

### Stage D — robustness
Use multiple evaluation views where data permits:

- chronological holdout;
- unseen capture/session;
- unseen family or scenario where labels permit;
- new nodes;
- drifted distribution;
- C2/beacon-heavy subset;
- class imbalance.

### Stage E — efficiency
Report:

- parameter count;
- GPU memory;
- CPU/RAM memory where relevant;
- events/sec;
- latency per event/batch;
- replay storage;
- persistent memory storage;
- training time.

---

# 14. EVALUATION RULES

Use metrics appropriate to imbalanced detection:

- precision;
- recall;
- F1;
- PR-AUC;
- ROC-AUC;
- false-positive rate / false alerts per unit time where possible;
- latency;
- memory overhead.

Do not tune a test threshold using the test labels.

Prefer a threshold selected on validation data, fixed before test evaluation.

Report confidence intervals / standard deviation across independent seeds where feasible.

Do not declare superiority based on one lucky seed.

---

# 15. REQUIRED FINAL TABLES

Create these tables in `ASTRA_RESEARCH_LEDGER.md`:

### Table A — Paper specification
Exact vs ambiguous vs unavailable details.

### Table B — Existing-code audit
Paper vs current baseline vs current TGNN.

### Table C — Controlled ablations
Each component and its isolated contribution.

### Table D — Alternative components
Candidate replacement, rationale, cost, and required experiment.

### Table E — Leakage audit
Feature/state, causality, possible leakage, test result.

### Table F — Final evidence table
Do not use a “winner” column. Use:

- model;
- exact configuration;
- seed count;
- test protocol;
- F1;
- PR-AUC;
- ROC-AUC;
- FPR;
- latency;
- memory;
- parameter count;
- confidence interval / SD;
- notes.

---

# 16. CLAIM DISCIPLINE

Use these words carefully:

- `paper-faithful` = matches all explicitly specified paper details;
- `paper-adapted` = changes something because the project uses a different task/data format;
- `controlled` = only the named component changes;
- `ablation` = component removed from a fixed parent architecture;
- `exploratory` = useful idea not yet fairly validated;
- `verified` = supported by tests/experiments with recorded configuration.

Never write:

“TGNN is better because it is more advanced.”

Never write:

“Our model beats the paper, therefore the architecture is superior.”

Instead write exactly what was measured and under which protocol.

---

# 17. REQUIRED RESEARCH LOG FORMAT

`ASTRA_RESEARCH_LEDGER.md` must always contain:

```text
Date:
Current phase:
Question:
Hypothesis:
Evidence:
Experiment/config:
Result:
Interpretation:
Confounds:
Decision:
Next action:
Status: TODO | RUNNING | VERIFIED | BLOCKED
```

Every code change must have a corresponding ledger entry.

Every experiment must record:

- commit/hash if available;
- file/config version;
- seed;
- dataset version;
- split;
- feature set;
- model;
- hyperparameters;
- rehearsal setting;
- threshold protocol;
- metrics.

---

# 18. STOP CONDITIONS

Stop and document the problem rather than pushing forward when:

- the paper's task cannot be reconstructed;
- an experiment depends on unavailable data;
- a comparison changes multiple variables at once;
- test data influences training;
- future information enters the feature/state path;
- an implementation claim cannot be verified;
- an external paper is being treated as if it were the base paper;
- a reported improvement disappears under a fair control;
- a baseline implementation is still ambiguous.

A blocked, correctly documented experiment is better than a misleading number.

---

# 19. REQUIRED OUTPUTS BEFORE YOU START OPTIMIZING TGNN

Do not begin adding “better” modules until all of these exist:

1. `BASELINE_PAPER_SPEC.md`
2. `ASTRA_RESEARCH_LEDGER.md`
3. canonical paper-faithful baseline implementation;
4. architecture manifest;
5. unit tests for baseline;
6. side-by-side audit of all old baseline implementations;
7. causal/leakage audit;
8. controlled experiment harness;
9. initial web literature review;
10. explicit list of unresolved paper ambiguities.

Then run the baseline first.

Only after the baseline is validated, start the TGNN comparison.

---

# 20. FIRST SESSION DELIVERABLE

For this first session, do NOT attempt to finish the entire TGNN.

The first session must finish with:

- workspace cleaned;
- paper independently re-read;
- paper specification extracted;
- current code architecture mapped;
- all duplicated baseline implementations identified;
- canonical baseline skeleton created;
- exact-vs-adapted boundary documented;
- fresh literature sweep started;
- research ledger populated with what is DONE / TODO / BLOCKED;
- no unverified claim that TG-Detect beats the base paper.

At the end, print a concise status report:

```text
CLEANUP:         PASS/FAIL
PAPER SPEC:      PASS/FAIL/BLOCKED
BASELINE CODE:   PASS/FAIL/BLOCKED
ARCHITECTURE:    PASS/FAIL/BLOCKED
LEAKAGE AUDIT:   PASS/FAIL/BLOCKED
WEB RESEARCH:    PASS/FAIL/BLOCKED
TGNN TEST:       NOT STARTED / RUNNING / DONE

Verified facts:
Unresolved ambiguities:
Files changed:
Files removed:
Files created:
Next exact action:
```

## Final instruction

Do the work as a reproducibility study, not as a confirmation exercise.

Start from the paper, start from the raw project code, verify every assumption, and make the smallest scientifically defensible change at each step.
