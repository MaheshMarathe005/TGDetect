# Windows-APT 2025 — feature and label audit

Dataset: Windows-APT 2025: A Dataset for APT-Inspired Attack, combined.csv,
102,011 events. Existing task: Wazuh MITRE annotation presence.

Inspection of windows_apt_data.py confirms that MITRE identifiers, tactics,
techniques, rule descriptions and rule IDs are excluded from the 72-dimensional
feature vector. The target uses _source.rule.mitre.id. Inputs use event type/ID,
provider/channel, executable names, protocol/integrity level, four string
lengths, two ports and two port-missing indicators. Agent/process identifiers
construct graph connectivity; timestamps define ordering and relative ages.
The earlier conversational statement of four missing indicators was incorrect.

Direct inclusion of the target field was not found. Proxy learning remains a
hypothesis: telemetry features may predict the Wazuh rules that produce the
annotation. High accuracy alone does not establish leakage. Repeated event
templates, shared hosts and related processes across chronological partitions
could also make generalization easier; these require explicit audits.

Original held-out results: GNN + rehearsal F1 98.278 ± 0.414%; TGNN hybrid
F1 98.967 ± 0.074%; logistic regression 99.042%; histogram gradient boosting
99.077%. These measure annotation prediction, not verified attack detection.

The threshold_calibration directory contains preliminary post-hoc results.
Its implementation needs an additional audit before those results are considered
final: CPU replay changed scores slightly; hash comparison before inference,
global threshold freezing before test access, matching bootstrap selection,
and complete secondary/control reporting were not fully implemented. Preserve
the files as preliminary evidence, not as a fully verified threshold study.

Next leakage checks: quantify duplicate feature templates across splits;
event-ID/provider/channel ablations; host/scenario holdouts where metadata
supports them; and ground-truth labels independent of Wazuh rule annotations.

New DARPA TC experiments must have separate manifests, explicit label provenance
and dataset names. DARPA node-level ground truth must not silently be described
as verified malicious labels for every incident event.
