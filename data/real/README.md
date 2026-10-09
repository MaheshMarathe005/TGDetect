# Real host telemetry

Source: https://github.com/OTRF/Security-Datasets

Three Windows host event captures from controlled malicious attack emulations:
Mimikatz logonpasswords, PsExec LSA secrets dumping, and Empire PsExec lateral movement.
These are real operating-system events from lab hosts, not production incident captures.
Capture-level attack context does not establish that every event is malicious.
No per-event labels have been invented.

Original ZIPs: `otrf/*.zip`. Original unmodified JSONL: `otrf/raw/`.
Dashboard-compatible, chronologically sorted Sysmon: `otrf/upload_ready/*.jsonl`.
All Sysmon events with a UTC timestamp and host are retained; other channels remain in raw files.
Flat source fields are mapped into host/event/winlog objects; GUID spelling is normalized.
Sysmon UtcTime is preferred over collector @timestamp; all times are normalized to UTC.
Each entire upload file passed validation; the first 100 events of each passed checkpoint inference.
This checks compatibility, not detection accuracy on an independently labeled dataset.
Source URLs, SHA-256 hashes, counts and test results: `otrf/manifest.json`.
Upstream license is retained as `otrf/LICENSE.upstream`.

In the dashboard, choose Datasets, upload a file from `otrf/upload_ready`,
then validate and process it using a unique dataset ID.
