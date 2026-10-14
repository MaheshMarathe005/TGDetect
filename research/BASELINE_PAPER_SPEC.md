# Independent baseline paper specification

Date: 2026-09-23. Status: extraction complete; exact reproduction blocked.

Primary source: the preserved local ten-page PDF, independently extracted with `pdftotext -layout`; all pages read, with Tables 1–7 and Figures 1–2 inspected visually. PDF file hashes are in `ENVIRONMENT_AUDIT.json`. The two local copies have identical extracted text but different bytes.

Publication: Auttapon Pomsathit, *Adaptive Detection of Advanced Persistent Threats (APT) With Graph Neural Networks and Rehearsal-Based Continual Learning on Wazuh EDR Telemetry*, IEEE Access 13, 212973–212982 (2025), [DOI](https://doi.org/10.1109/ACCESS.2025.3639270). Title, author, DOI and pagination independently agree with the [public full-text copy](https://htcr.ncsa.or.th/ws/FileUpload/member_results/4ed57aacc9e23ae93ef48a497a0768a3.pdf). Direct DOI access failed in the browsing tool. The PDF states publication on December 2, 2025; do not use the inconsistent January date in some secondary indexes.

`EXPLICIT` means directly stated; `DERIVED` means a disclosed inference; `AMBIGUOUS` means partly specified or inconsistent; `NOT REPORTED` means absent from this complete PDF. Page references below use printed pagination, followed by PDF page where useful. These are paper statements, not independently reproduced results.

| Item | Paper statement / status | Exact source | Implementable exactly? | Missing or ambiguous detail |
|---|---|---|---|---|
| Task | EXPLICIT: adaptive APT detection from Wazuh plus sandbox traces | §III.A/E/F, pp.212974–75 | No, original corpus absent | Labeling and collection protocol |
| Graph | EXPLICIT: G=(V,E,X), X has one row per node | §III.B, Eqs.1–2, p.212975 (PDF 3) | Partially | Construction algorithm |
| Node types | EXPLICIT examples: processes, files, IPs; registry/network entities also discussed | §III.B; §IV.D/Fig.2, pp.212975,212979–80 | Partially | Complete schema and identity scope |
| Edge semantics | EXPLICIT interactions; figure includes connect/spawn/modify | §III.B; Fig.2 | Partially | Direction, multiplicity and relation encoders |
| Features | EXPLICIT examples: command lines, hashes, IPs | Eq.2 discussion | No | Vocabulary, encoding, dimension, fitting scope |
| Labels | AMBIGUOUS: malicious/benign descriptions and tactic summaries | Table 1; §III.C; Table 6 | No | Node label assignment, multi-class mapping, graph-to-node mapping |
| Windows / graph batches | NOT REPORTED: no window construction or temporal boundaries | Full PDF | No | Window width, overlap, event availability time |
| GNN depth | EXPLICIT: 3 layers | §III.G.1(a), p.212976 (PDF 4) | Yes | Which PyG operator |
| Hidden width | EXPLICIT: 128 | §III.G.1(a) | Yes | Input dimension missing |
| Activation | EXPLICIT: ReLU | §III.G.1(b); Eq.3 | Partially | Exact locations around aggregation |
| Dropout | EXPLICIT: 0.2 | §III.G.1(b) | Partially | Feature, attention, layer/output placement unspecified |
| Aggregation | EXPLICIT: attention neighborhood aggregation in training specification; Eq.3 is generic | §III.G.1(f); Eq.3 | No | Score, head count, self-loop and edge handling |
| Update | EXPLICIT generic transform of own/neighborhood aggregate | Eq.3, p.212975 | Partially | No unique executable attention operator |
| Output head | EXPLICIT: linear projection with bias, then softmax | Eq.4 | Partially | Number of classes |
| Prediction target | EXPLICIT: node classification | §III.C, Eq.4 discussion | Yes as target type | Not edge/event classification |
| Softmax vs sigmoid | EXPLICIT softmax across target classes | Eq.4 | Yes after class definition | Binary class count is not an explicit implementation spec |
| Per-sample loss | NOT REPORTED: new/rehearsal losses named, no CE/focal formula | Eq.5 | No | Criterion, reduction, class weights |
| Combined objective | EXPLICIT: new-data loss + lambda times rehearsal loss | Eq.5 | Partially | Lambda value |
| Replay storage | EXPLICIT: 10% of past samples, reservoir sampling | §III.G.1(e) | Partially | Sample unit (node/graph/event), capacity schedule, dedup across epochs |
| Replay sampling | NOT REPORTED: no replay count/distribution per update | §III.D/G | No | Mixture ratio, loss normalization, scenario boundaries |
| VAE | EXPLICIT: lightweight VAE belongs to **Generative Replay IDS comparator** | §III.G.2(d), p.212976; §III.H.1(d), p.212977 | Not part of proposed GNN | VAE architecture missing only for that separate comparator |
| Lambda | NOT REPORTED numerical value | Eq.5 defines its role | No | Do not silently set 1 and call exact |
| Optimizer | EXPLICIT Adam | §III.G.1(c) | Partially | Betas/epsilon and scheduler not reported |
| Learning rate | EXPLICIT 0.001 | §III.G.1(c) | Yes | — |
| Weight decay | EXPLICIT 0.00001 | §III.G.1(c) | Yes | — |
| Batch size | EXPLICIT 256 | §III.G.1(d) | Partially | Unit / neighbor sampler |
| Epochs | EXPLICIT 200 per scenario | §III.G.1(d) | Partially | Scenario order, checkpoint selection |
| Seeds | EXPLICIT five independent repetitions with mean and SD | §III.G.1; §IV.A | Partially | Seed values, determinism configuration |
| Software | EXPLICIT PyTorch Geometric | §III.G | Partially | PyG/PyTorch/Python/CUDA versions absent |
| Hardware | EXPLICIT Tesla V100, 32 GB, shared across comparisons | §III.G | Partially | CPU/RAM, precision mode, timing procedure |
| Dataset size | EXPLICIT benign: 12,500 graphs / 1.2M nodes / 3.5M edges; APT: 8,300 / 0.9M / 2.7M | Table 1, p.212976 | No | Data release, raw logs, event IDs and hashes absent |
| Split | NOT REPORTED train/validation/test fractions, identities, chronological rule | Full PDF; §III.E describes learning strategies only | No | Leakage-safe reconstruction impossible from paper alone |
| Metrics | EXPLICIT precision, recall, F1, ROC-AUC, latency, replay storage overhead | §III.I/Table 3, pp.212977–78 | Partially | Aggregation, threshold and measurement details |
| Threshold | NOT REPORTED | Full PDF | No | Validation procedure and operating point |
| Resources | EXPLICIT Table 4 proposed model: 6.7 ms; replay overhead 10%; static GNN 4.1 ms | Table 3 defines latency per batch; Table 4, p.212978 | No | Synchronization, warmup, hardware utilization, overhead denominator |
| Parameters / FLOPs | NOT REPORTED | Full PDF | No | Cannot infer counts without missing dimensions/operator |
| Normalization/residuals/initialization | NOT REPORTED | Full PDF | No | Do not import TG-Detect choices into an exact reconstruction |

## Corrections and internal reporting limits

The reset prompt raised the possibility that reservoir and VAE replay were mixed. Reading §III.G.2 and §III.H resolves it: the VAE is a separate comparator. The canonical proposed model must not silently gain a VAE.

Table 4 reports proposed ROC-AUC 0.98, while Table 5 reports 0.988 for the full GNN plus rehearsal. This discrepancy is recorded without choosing a preferred value. Table 4's F1 is 0.981 ± 0.002; Fig.1 is a schematic-looking two-line PR plot without underlying predictions. Neither plot nor rounded aggregate metrics reconstructs a test set or threshold. Table 1 describes graph populations while Eq.4 classifies nodes; the labeling bridge is missing. These observations are limitations, not accusations of misconduct.

## Exact versus adapted boundary

- `POMSATHIT_PAPER_SPEC`: a specification and blocked model skeleton. It has no invented attention formula, input dimension, loss or replay weight. Parameter counts and executable layer manifests remain null until resolved.
- `POMSATHIT_ADAPTED`: reserved for a separately documented implementation on a verified corpus. CTU-13 flow labels, 22 flow features, edge classification, graph snapshots and CE/lambda choices would all be **adaptations**. They cannot reproduce the reported Wazuh numbers.
- Existing project aliases are legacy candidates under audit, not canonical implementations. Matching width/depth alone does not establish paper faithfulness.

## External implementation search

Searches on 2026-09-23 used the exact title/DOI plus code, GitHub, dataset, supplementary and implementation terms; also `site:github.com "Pomsathit" "Wazuh"`. No directly attributable implementation, original data release or supplement was identified in returned results. This is a limited search result, not proof none exists. A later [2026 paper citing Pomsathit](https://www.mdpi.com/2073-431X/15/8/501) was located; its citation does not resolve this paper's implementation and is not substituted for it.

Next: choose and record an explicit adapted protocol, or obtain attributable author artifacts before attempting exact reproduction. Do not start the temporal comparison while these gates remain open.
