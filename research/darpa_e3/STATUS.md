# DARPA TC E3 pilot status

CADETS E3 raw acquisition and chronological class-coverage gate PASS.
THEIA E3 raw acquisition and binary Avro scan COMPLETE; class-coverage gate FAIL:
zero matches to the published node labels in 44,023,627 scanned Event records.
All three sampled partitions have zero positives. No THEIA models were trained.
This could reflect the selected capture's coverage or a mapping mismatch;
it does not establish that the full THEIA dataset contains no attacks.

CADETS sample: 100,000 label-blind sampled events from 12,915,596 Event records;
1,473,556 records lacked the two required endpoints and were excluded.
Train: 60,000 (4,791 positive associations); validation: 20,000 (1,431);
test: 20,000 (1,236). Positive means incident to a published ground-truth node.
This does not verify maliciousness of each event; shared service processes can
participate in legitimate activity before or after compromise.

Matched training: existing GNN_H and TGNN_Q_A_C, fresh 66-input adapters,
20 epochs, seeds 0/1/2, batch 256, Algorithm-R 10% reservoir with 26 replay
events. Four strictly earlier incident events form each sampled graph window.
Both architectures keep their original model components. This is a small
adapted benchmark, not the paper's own dataset or a full DARPA evaluation.

Local tests: 26 passed. GPU training has begun on L4:
https://modal.com/apps/prathampansare25/main/ap-dFOxjYvmmMd8e2F3ziIJkW

Volume: tg-detect-darpa-e3-pilot-v1. Outputs: /results/cadets_e3 and
/results/theia_e3. Download with ~/main_env/bin/python sync_darpa_pilot.py.
The downloader verifies checkpoint hashes, validation thresholds and saved test
metrics. All six CADETS checkpoints passed these checks locally.

## Completed CADETS E3 pilot results

Mean ± sample standard deviation over three seeds, 20 epochs each:

| Model | Test F1 | Precision | Recall | FPR | PR-AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| GNN + rehearsal | 95.55 ± 0.76% | 95.76 ± 2.08% | 95.36 ± 0.54% | 0.281 ± 0.142% | 98.34 ± 0.25% | 99.07 ± 0.10% |
| TGNN + GRU + attention + CNN + rehearsal | 94.61 ± 0.39% | 93.87 ± 0.11% | 95.36 ± 0.68% | 0.410 ± 0.005% | 97.57 ± 0.11% | 98.98 ± 0.07% |

| Seed | GNN F1 | TGNN hybrid F1 |
|---|---:|---:|
| 0 | 96.4183% | 94.1603% |
| 1 | 95.0583% | 94.8338% |
| 2 | 95.1691% | 94.8297% |

GNN leads mean F1 by 0.94 percentage points in this bounded association-label
pilot. Both have the same mean recall; the hybrid produces more false positives.
This does not establish performance on verified event-level attack labels or
unseen attack scenarios. Full-resolution context, larger training budgets and
independent scenario coverage have not been tested here.

After saving all six CADETS runs, the initial app failed when reading THEIA's
audit file from its earlier Volume view. Completed results were unaffected.
The runner now reloads the Volume between datasets and reports pending
preprocessing instead of crashing on a missing audit. Completed studies return
their verified summary before rebuilding data or training. A CPU-only finalizer
records CADETS completion and THEIA's failed class-coverage gate.
