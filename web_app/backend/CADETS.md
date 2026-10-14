> Research scripts and regression tests referenced in this guide remain local.
> The `research/` folder is temporarily excluded from GitHub. Supply its V3
> implementation and supporting packages for replay, preparation and upload inference.

# CADETS V3 investigation workspace

The default Next.js page now runs a separate CADETS workspace. The older Sysmon
workspace is retained at `/legacy`; its dataset/checkpoints are separate and are
not substitutes for the CADETS model.

## Run

From the repository root, using the environment with Torch, FastAPI, NumPy,
scikit-learn and Modal installed:

```sh
python -m web_app.backend.scripts.download_cadets
python -m web_app.backend.scripts.prepare_cadets
python -m uvicorn web_app.backend.backend:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd web_app/frontend
npm run build
npm run start
```

Open http://localhost:3000. Downloads read existing Modal Volume files; they do
not start GPU workloads. Once downloaded/indexed, normal operation is local.
The current machine already has the complete dataset, checkpoint and index.

## Data and integrity

`datasets/application/cadets_v3/` contains the original five-epoch, seed-0 V3 `best.pt`,
token bank, prepared event memmaps, validation/test edge scores, entity attributes,
recorded result and a generated SQLite investigation index. These large artifacts
remain gitignored. The prepared corpus contains 3,526,541 training, 105,481
validation and 1,905,942 test events (5,537,964 total).

The setup command checks the checkpoint SHA-256, node-attribute SHA-256, manifest
SHA-256 and every file in the prepared manifest. It verifies that saved score
endpoints/times/relations match each prepared event and reproduces the original
calibration and node metrics. Index publication uses an atomic rename. If model
source files differ from the checkpoint's source hashes, live replay fails closed.
Restart the backend after rebuilding/replacing an index: explanation caches are
bounded and immutable for one server lifetime.

## Investigation and explanations

`/api/cadets/summary`, `/nodes`, `/events` expose the original result, paginated
alert queue and full indexed corpus. `/events/{split}/{index}` resolves the exact
endpoint attributes from the offline identity bank. Original relation IDs follow
the pinned ORTHRUS CADETS preprocessing mapping.

`/nodes/{id}/explanation` reproduces the exact alert aggregation: relation-specific
validation calibration, top-three incident evidence per 900-second window,
`max(largest, sum(top3)/sqrt(3))`, then the maximum window score. It identifies the
actual contributing events and both scoring channels. This is evidence attribution;
it does not claim causal feature importance, SHAP values or model counterfactuals.

`/nodes/{id}/chain` reconstructs bounded temporal reachability backward and forward
from the strongest contributing event. Expansion respects arrival times in each
direction. It keeps recorded event arrows, reports truncation, and returns sorted
observed events. Shared endpoints can be explored in either direction, so this is
an investigative candidate chain, not a verified information-flow graph. Ground
truth is never used to seed, expand or rank chains. Benchmark node membership is
an optional contextual annotation, not an event attack label. No MITRE tactics
are invented. JSON exports preserve these limitations.

## Actual checkpoint inference

`POST /api/cadets/replay` accepts `{"split":"test","batches":1}` (1–4 batches).
It loads the verified original model, strictly loads its weights and precomputes
the frozen FP32 encoder lookup. Each request resets all model state, replays the
chronological split prefix in the original 1,024-event batches and compares losses
against saved GPU evaluation scores. Concurrent replay is rejected with 409 to
bound CPU/memory use. It reports relation CE and pair anomaly separately for a
sample of events. Timings are local CPU measurements, with setup reported
separately; they are not the Modal L4 benchmark or per-event arrival latency.

All normal browsing uses the verified saved evaluation evidence, avoiding
replaying millions of events for every query. Training events have no saved
inference scores and are explicitly shown as unscored. Validation rows are for
inspection, not independent validation AP/F1 claims.

## Verification

```sh
python -m pytest research/test_cadets_dashboard.py -q
cd web_app/frontend
npm run build
```

Browser checks require local servers and the installed Playwright Chromium;
screenshots, exported chain and browser check results go to
`artifacts/cadets_dashboard_review/`.

The V3 checkpoint's test precision is only 0.4194%, recall 35.2941%, F1 0.8289%,
AP 0.7797% and ROC AUC 89.1762%. UI integration does not improve these model
metrics. The interface exposes the false-alert burden rather than presenting ROC
AUC or imbalance-dominated accuracy as proof of a deployment-ready detector.

## Measured application latency

Five warm local HTTP samples per route, on this machine (not GPU inference):

| Route | Before | After |
| --- | ---: | ---: |
| Test events, 20-row page | 138.1 ms | 4.7 ms |
| Entity-filtered page, node 218987 | 683.4 ms with full-row UNION | 6.0 ms |

The first change reads verified split counts from metadata. The second adds
covering endpoint/page indexes, counts incident rows without materializing full
events, deduplicates self interactions, and fetches only the requested event IDs.
Event data, model scores, alert thresholds and detection metrics are unchanged.
Hidden event/chain views also defer fetching until opened. Measurements are small
local samples, not a production latency guarantee. Raw samples and verification
outputs are in `artifacts/cadets_dashboard_review/`.

Current verification: seven CADETS integration tests pass; production frontend
build passes; nine browser acceptance flows pass with no JavaScript errors.
The separate legacy suite has one pass and ten failures because the old Sysmon
matched checkpoints/results are absent locally (zero registry models). Its UI
route is retained, but its model workflows need those independent artifacts.

## Graph redesign and explicit attack-chain construction

The graph now uses a navy/violet investigation canvas, labeled process/file/
network entity cards, a connected six-entity focus view, expandable context,
bundled repeated observations, direction arrows, zoom controls, keyboard
selection and draggable cards. A bundle with multiple relations is labeled as
an event count; inspecting it selects its strongest evidence event rather than
pretending all events have the same relation.

The attack-chain builder calls `GET /api/cadets/nodes/{id}/construction` with a
selected incident event, before/after direction, time bound and maximum events.
Its bounded beam search extends paths only when consecutive directed endpoints
match and `(timestamp,event_index)` increases strictly. It does not add repeated
entities. Paths are ranked by evidence sum, which is explicitly a ranking aid,
not a calibrated chain probability. The search reports pruning and never accepts
benchmark labels as a search/ranking input. If no extension is found, it displays
the single real event, without inventing attack stages.

The UI supports candidate-path selection, forward replay, reverse backtracing,
pause, speed controls, manual stepping, a time scrubber and automatic horizontal
tracking of the active event. Each step explains the actual endpoint/time link.
Reduced-motion settings disable autoplay and decorative animation; manual
tracing remains available. Node explanations additionally visualize the evidence
score against the real validation threshold.

Design reference: [Elastic visual event analyzer](https://www.elastic.co/docs/solutions/security/investigate/visual-event-analyzer),
particularly readable entity relationships, navigation and contextual event
inspection. The local implementation uses SVG and React; it does not require an
Elastic deployment or an added graph-engine dependency.

Verification while the backend server is stopped:

```sh
python -m pytest research/test_cadets_dashboard.py -q
```

The browser test intercepts CADETS requests and obtains actual responses through
FastAPI TestClient against the real local dataset. It does not start a backend
HTTP server or synthesize telemetry. Screenshots and the constructed-path export
are saved in `artifacts/cadets_dashboard_redesign/`.

### Uploaded-log detection graphs

Open **Detection graph** in the sidebar. Upload JSON arrays, JSONL/NDJSON, or CSV endpoint-event records. The upload is normalized, sorted chronologically with stable ties, tokenized with the trained encoder settings, scored by the checksum-verified V3 checkpoint in 1,024-event batches from fresh state, and indexed in a separate persistent SQLite graph. The original validation calibration and threshold determine predictions; supplied scores, alerts and ground-truth labels are ignored. Each event retains its own observed attributes, preventing later attributes from leaking into earlier model input. Original benchmark data remains separate.

Required event fields: `time` or `timestamp`, `relation`, and `src`/`dst` (or source/destination/target aliases). Endpoints can be IDs/IP strings or objects containing `id`, `type`, `remote_ip`, `host`, `path`, and/or `cmd_line`. Flat CSV fields such as `src_ip`, `dst_ip`, `src_host`, `dst_host`, and endpoint type/path/command fields are supported. Relations must be one of the ten trained event types (CONNECT, EXECUTE, OPEN, READ, RECVFROM, RECVMSG, SENDMSG, SENDTO, WRITE), optionally prefixed EVENT_, or IDs 0–9. Timestamps accept ISO 8601 or Unix seconds/milliseconds/microseconds/nanoseconds. Unrecognized raw CDM envelopes or arbitrary vendor-specific schemas need conversion to these fields; they are rejected rather than guessed. Uploads are bounded to 50 MB and 100,000 events. One background scoring job runs at a time; progress/errors remain visible, and ready datasets survive reloads.

```json
{"timestamp":"2026-10-11T10:00:00Z","relation":"CONNECT","src":{"id":"process-1","host":"workstation-01","type":"subject","cmd_line":"curl"},"dst":{"id":"network-1","remote_ip":"192.0.2.10","type":"netflow"}}
```

The dataset dropdown includes original CADETS predictions and uploaded datasets. Entity-type options and searchable entity names come from the selected dataset: IPs, hosts, processes, files and other entity names where available. Selecting an IP or host loads its observed connected neighborhoods. Selecting malicious or benign filters **starting entities**, while retaining both classes in connected context. Clicking a red or green node requests its neighborhood from the backend, rather than merely hiding unrelated SVG elements. Choose 1–3 hops or reset focus. Display bounds (24 nodes, 120 edges) are explicit; sample counts are not corpus-wide proportions. Addresses never come from fabricated node metadata.

Red/green encode model predictions, not confirmed attacks. Uploaded logs have no benchmark label unless separately evaluated. V3 was trained on CADETS: predictions on other domains are unvalidated. Host/name fields identify graph entities; only the checkpoint's trained endpoint fields enter the encoder. The legacy Sysmon workspace remains separate.

Backend endpoints: `GET/POST /api/graph-datasets`, dataset status `GET /{id}`, searchable options `GET /{id}/entities`, and connected graph `GET /{id}/graph`. Upload data stays in ignored `datasets/application/graph_uploads/`.

