# Next GAT experiment: evidence status and frozen protocol

The full local corrected CADETS stream has been structurally audited. No GPU
training was launched. The requested documented download → chmod → exec chains
and attack-family mapping are unavailable locally and have not been verified
from a public source. The prepared relation vocabulary omits permission-change
operations. This does not prove the raw release has no such events.

The official DARPA release identifies an external ground-truth report:
https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md
The external Drive folder could not be retrieved through the browsing tool.
The report itself has not been read or linked to prepared event indices.
Existing Nginx Backdoor scenario labels must not be presented as independent
malware families.

## Completed structural audit

`slice_audit.json` contains every chronological 32,768-event segment, source
hashes, timestamps, full degree histograms and four role histograms. Each split
starts empty; each candidate retains the exact preceding split history.
Selection uses endpoint structure only, without labels or model scores.
Corrected V3 balanced endpoint windows are replayed, including repeated nodes
within batches and self-edge destination suppression.

A screening threshold requires both sd and ds to appear for at least 1% of
queries and contribute at least 100 nonself edges in a segment. This is a
preliminary support threshold, not proof of adequate statistical power or
attack coverage. Roles can overlap. Counts are structural, not semantic
transfer labels.

| Split | Segments screened | Pass support screen |
|---|---:|---:|
| Training | 108 | 101 |
| Validation | 4 | 3 |
| Test | 59 | 55 |

Highest-ranked candidates, zero-based split-relative half-open indices:

| Split | Start | End | Queries with sd | Queries with ds |
|---|---:|---:|---:|---:|
| Training | 1,671,168 | 1,703,936 | 39.81% | 34.84% |
| Validation | 0 | 32,768 | 7.03% | 7.02% |
| Test | 327,680 | 360,448 | 39.70% | 33.59% |

These are candidates, not adopted dataset slices. The test candidate has not
been checked for attack-chain evidence. Rank is min(sd hit rate, ds hit rate),
which discourages one-sided role support. A selected slice must be re-audited
under the actual runner's declared cold-reset or causal warmup policy; it must
not silently inherit this audit's full-prefix context. No labeled test scores
were used, but a density-conditioned benchmark has restricted generalizability.

## Experiment protocol awaiting evidence

1. Verify externally documented download, permission change and execution
   steps on a shared artifact/dependency chain. Record source URL/file hash,
   timestamps, raw UUIDs, mappings into the prepared test region and whether
   each operation survives preparation. Do not use shared endpoints as proof.
   If chmod must be modeled directly, restore raw events and regenerate data
   before training; all comparison arms must use the same regenerated data.
2. Require source-backed attack-family labels. Until then, report recall by
   available attack scenario and explicitly mark family recall unavailable.
   Include positive counts, TP/FN counts and uncertainty, with an unavailable
   result for zero-support groups. Do not invent labels from paths or commands.
3. First compare old ×1, old ×2, role4 ×1, role4 ×2 using paired seeds and
   identical common weights, frozen original encoder, CNN, memory, scorer,
   optimizer, batches and splits. Separate two-layer weights are trainable;
   tied layers are not part of this first depth comparison.
4. Early stopping monitors benign validation predictive loss only. Proposed
   safety ceiling 100 epochs, minimum 10 epochs, patience 10, improvement
   threshold 0.0001. Save the best checkpoint and record the stop reason.
   Reaching the ceiling must be reported as unconverged. Keep threshold
   calibration data separate from checkpoint-selection validation data.
5. After depth comparisons, compare 8 versus 4 retained history events for
   both two-layer operators. This is a window-composition factor, with all
   other state/history capacities and tensor shapes held fixed. Pad excluded
   history slots and use the same balanced endpoint policy. This changes the
   active window, not just an attention mask. Re-audit support in both windows.
6. Primary reporting: per-group minority recall at the shared calibration
   policy, its positive support and false-positive rate. Also report AP, F1,
   full-model latency with untimed diagnostics, peak VRAM, epochs, best epoch
   and runtime. Provide paired-seed differences and uncertainty. Several
   seeds cannot compensate for too few independent attacks.

## Reproduce the audit

From the repository root:

```bash
PYTHONPATH=research python3 -m gat_study.slice_audit \
  --prepared datasets/application/cadets_v3/prepared
PYTHONPATH=research python3 -m unittest research/test_gat_slice_audit.py
```

The reference test passed and checks exact structural diagnostics for batch
sizes 1, 3 and 16, self-edges, and history continuity across segment boundaries.
The new study runner and Colab training notebook are not yet implemented;
this directory records the completed audit and the evidence needed to make
that training experiment satisfy the requested protocol.
