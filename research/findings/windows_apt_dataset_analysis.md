# Windows-APT 2025 dataset analysis and L4 runtime estimate

Status: pre-training feasibility analysis, 26 September 2026.

## Dataset measured locally

The canonical source is `combined.csv`. The 16 period CSVs sum to the same 102,011 CSV records, so the period files and `combined.csv` must not be concatenated together. `wc -l` reports about 1.6 million physical lines because many CSV fields contain embedded newlines; a CSV-aware reader reports 102,011 records.

| Item | Measurement |
|---|---:|
| Full dataset directory | 480,866,795 bytes (458.59 MiB) |
| Period CSVs | 232,878,507 bytes (222.09 MiB) |
| `combined.csv` | 247,934,817 bytes (236.45 MiB) |
| Combined records | 102,011 |
| Columns in `combined.csv` | 377 |
| Time span | 28 Oct 2024 08:51:44 to 22 Dec 2024 16:33:38 |
| Days containing records | 40 over a 55.32-day calendar span |
| Agents | 5 |
| Event IDs | 86 non-empty values |
| Unique Wazuh rule IDs | 153 |
| Individual ATT&CK techniques | 45 |
| Individual ATT&CK tactics | 10 |
| SHA-256 checks | `combined.csv`, `scenario_manifest.csv`, and `validation_summary.csv` match the supplied checksums |

The five agents are `window-agent3` (56,177 records), `wazuh-agent` (23,772), `agent1` (17,665), `DESKTOP-DS4FBF4` (3,435), and `Maryam` (962). `windows_eventchannel` accounts for 90,601 records; the remainder includes registry/SCA and Linux-style Wazuh telemetry.

The event rows do not contain an event-level label or `Scenario_ID`. MITRE fields are present on 63,619 rows (62.36%) and absent on 38,392 rows. Treating “MITRE field present” as a binary label would be only a weak annotation proxy, not attack ground truth, and the MITRE/rule columns must be excluded from model inputs to prevent label leakage. The supplied scenario manifest has 36 scenarios and 360 validated runs, but the logs do not link rows to those scenario IDs. The provided filtering script confirms that no scenario column is available in the logs.

## Required adapter before training

The current loader (`causal_data.py`) expects network-flow fields such as `Dur`, `TotPkts`, `TotBytes`, `SrcBytes`, `Proto`, and `Dport`. The Windows-APT CSV has none of that fixed flow schema, so it cannot be passed directly to the current models. A valid experiment needs:

1. a causal event ordering from `_source.@timestamp`;
2. a graph rule defining source and destination nodes (for example agent, process/image, file, and network entities);
3. a fixed numeric feature vector for each event, with text/hash/rule fields handled separately;
4. an explicit label protocol, preferably attack-window or scenario labels, and a split by time/scenario; and
5. exclusion of MITRE/rule fields from features if they are used to make weak labels.

For reference only, the tie-safe chronological split gives 61,206 train, 20,404 validation, and 20,401 test records. Under the weak MITRE-presence proxy, the positive fractions are 66.72%, 54.59%, and 57.07%, respectively. These are not proposed ground-truth labels.

## L4 estimate using the verified project benchmark

The estimate uses the project’s measured NVIDIA L4 run: 200 epochs, batch size 256, reservoir rehearsal with 26 replay examples per step, and the same five-position causal ego representation. The existing 16,384-event benchmark has 9,830 training events and 7,800 optimizer steps per run. The full Windows-APT record count would have 61,206 training events and 48,000 steps per run, a 6.1538x step-count increase.

The two requested arms map to the verified implementation as follows:

| Requested model | Project arm | Parameters |
|---|---|---:|
| base paper GNN + rehearsal | `GNN_H` | 43,394 |
| TGNN + GRU + attention + CNN + rehearsal | `TGNN_Q_A_C` | 290,434 |

Measured production timings on the L4 benchmark (16 independent workers sharing one L4) were 131.73 s for `GNN_H` and 370.98 s for `TGNN_Q_A_C` on the 16,384-event run. Linear scaling by the 48,000/7,800 step ratio gives:

| One 200-epoch seed on all 102,011 records | Conservative 16-worker contention | Two-worker L4 estimate* |
|---|---:|---:|
| `GNN_H` | 810.7 s (13.5 min) | 602.3 s (10.0 min) |
| `TGNN_Q_A_C` | 2,283.0 s (38.0 min) | 1,696.1 s (28.3 min) |

\*The two-worker column applies the measured calibration ratio: 25.23 aggregate steps/s at two workers versus 18.74 steps/s per worker when 16 workers shared the card. It is a planning estimate, not a completed Windows-APT training run.

If one seed of each model runs concurrently, the pair is therefore about 28–32 minutes of GPU training after preprocessing; sequential execution is about 38 minutes under the two-worker estimate. For five seeds per model, two-worker scheduling is approximately 1.9 hours when each model family is run in parallel waves, or 2.4 hours when one GNN seed is paired with one hybrid seed per wave. Add roughly 1–5 minutes for CSV parsing, graph construction, and tensor staging; upload time for the 236 MiB canonical CSV is separate and network-dependent.

The estimates scale linearly with the chosen epoch count. For example, 100 epochs is approximately half of the numbers above. The actual Windows-APT run can be slower if graph construction creates larger ego windows, if text features are embedded, or if all raw strings are retained in memory. The raw CSV itself should remain on CPU; the numeric graph tensors are small relative to the 24 GiB L4.

## Decision before launching training

The compute budget is feasible on one L4. The user-authorized weak-label run is complete: GNN_H test F1 is 98.28 ± 0.41%, and TGNN_Q_A_C is 98.97 ± 0.07% over five seeds. The paired hybrid gain is +0.69 ± 0.39 F1 points. Feature-only logistic and histogram-gradient-boosting controls score 99.04% and 99.08% F1, so the deep architecture does not establish an advantage on this annotation target. These values cannot be reported as attack-detection ground truth. A scenario/time-linked label adapter and an untouched scenario-held-out test split are still required for an attack-detection claim.

## Execution-stage corrections (26 September 2026)

The staged source audit verified all 19 supplied checksums and exact equality of
the combined/period event-ID multisets. The original rule-ID count subtracted
one despite no blank ID: the corrected count is 153. Tie-safe chronological
boundaries yield 61,206/20,404/20,401 rows. The actual approved adapter has 72
features, hence 43,394 GNN and 290,434 hybrid parameters. Actual ten-worker L4
training took 783.50 seconds per GNN seed and 1,947.54 seconds per hybrid seed;
mean training utilization was 97.63% with 5,631 MiB maximum recorded GPU memory.
Earlier timing values are provisional: production contention was variable, and
applying hybrid-only concurrency calibration to the GNN was not a validated speed
estimate. The new run measured both architectures on the prepared Windows tensors
and completed the independent saved-artifact audit.
See ../windows_apt/PLAN.md for the registered, user-authorized weak-label study.
