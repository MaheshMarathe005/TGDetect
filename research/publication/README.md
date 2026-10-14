Run these commands from `research/`. Manuscript assets are in
`../research paper/publication/`; the code in this directory builds them.

# TGNN publication study

Author: **Mahesh Marathe**, A. P. Shah Institute of Technology, Thane, India.

`../../research paper/publication/paper.tex` is the editable IEEE conference manuscript; `../../research paper/publication/paper.pdf` is its
compiled research draft. It is not submitted or peer reviewed. Tables and plots
come from measured artifacts, and must be regenerated after pending jobs finish.

## Evidence and scope

- `protocol.json`: expanded study registered after inspecting the initial seed-0
  test results; includes the retrospective-test disclosure and all timeout caps.
- `LITERATURE_AUDIT.md`: supplied ORTHRUS, KAIROS, and Wazuh paper audit.
- `measured_results.json` / `.csv`: completed-run measurements only.
- `experiment_status.json`: reservations and observed job outcomes; not a bill.
- `figures/`: PDF/SVG/PNG architecture, comparison, learning and PR figures.
- `../artifacts/publication/`: raw results, CPU diagnostics and budget ledgers.
- `../artifacts/v3_vs_orthrus/`: original five-epoch seed-0 results and the
  earlier seeded-weight, three-batch-size latency benchmark.

The candidate is **V3-AttnShort**: retain active-window CNN and affine endpoint
memory; replace the short-memory CNN pass with attention-only readout. V3-CNN
remains the unchanged comparator. Candidate selection was based on previous
latency observations before its accuracy results were inspected. This does not
make the overall development process an untouched-test evaluation.

The initial shared-calibration result favors V3-CNN over ORTHRUS in AP/F1, but
absolute precision is very low and latency is worse. The released-style raw-max,
top-30 clustering diagnostic reverses that seed-0 F1 ordering. A broad claim that
the model beats published ORTHRUS is therefore unsupported. Direct KAIROS neural
tests do not reproduce native attack-window queues or investigation graphs.
GraphSAGE is our disclosed baseline, not the Wazuh supervised continual learner.

## Reproduce

Run from the repository root with the project environment and authenticated
Modal profile. Commands incur cloud costs; they are listed for reproducibility,
not as authorization to repeat the existing study.

```bash
python -m pytest -q test_sliding_v3.py
modal run modal_publication_study.py --wave acceptance
modal run modal_publication_study.py --wave prepare
modal run modal_publication_study.py --wave core
modal run modal_publication_study.py --wave kairos
# After trained seed-zero checkpoints are complete:
modal run modal_publication_study.py --wave latency
modal run modal_publication_analysis.py
python -m publication.report
modal run modal_publication_pdf.py
```

The study refuses to overwrite an existing run directory. Existing ledger keys
are not launched again. CPU diagnostics skip existing local outputs. Individual
GPU failures/timeouts stay in the ledgers, with no automatic retry. The three
disjoint GPU ledgers must be aggregated; their limits are not independent
allowances to multiply by the number of apps.

## Compute and money

The registered jobs reserve **27,900 seconds = 7.75 total GPU-hours**, leaving
900 seconds below the authorized eight-hour ceiling. Parallel L4 workers share
that total. Eight concurrent workers do not mean eight hours per worker.

At the Modal rates checked on 2026-10-10 (L4 $0.000222/s, CPU $0.0000131/core/s,
RAM $0.00000222/GiB/s), the timeout allowance at eight requested CPU cores and
32 GiB per GPU worker estimates $6.19 GPU + $2.92 CPU + $1.98 RAM = **$11.10**.
Source: <https://modal.com/pricing>. This is an estimate, not a provider-enforced
dollar cap. Actual billing includes startup, idle, and usage above requests;
CPU-only preparation/analysis/LaTeX compilation are additional. GPU worker
scaledown is two seconds. No extra training or paid GPU retry is authorized by
this document. The user reports a $15 remaining balance.

## Interpretation limits

1. Same prepared CADETS events and scored-node labels; different feature
   encoders, pretraining scope, and mini-batch memory semantics.
2. Five-epoch comparisons are budget matched, not equal to native released
   training budgets (ORTHRUS 20, KAIROS 50). Incomplete longer runs remain gaps.
3. Validation node-window tail calibration is not a guarantee on multi-day
   node-level false positives. No test-selected deployment threshold is used.
4. Four labeled IDs are not scored; metrics cover 68 malicious observed nodes.
5. No second dataset, full attribution/QoA evaluation, native Wazuh reproduction,
   prospective deployment trial, or complete end-to-end latency measurement.
6. Frozen encoder lookup uses prepared identities. Online unseen-identity costs
   and production ingestion/storage costs have not been established.

The supplied copyrighted PDFs are not included in a redistribution package.
Extracted local text and the pinned authors' code are for source auditing;
respect upstream licenses when publishing an artifact repository.
