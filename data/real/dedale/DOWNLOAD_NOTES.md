# Full DEDALE Windows host logs

Publisher: https://dedale.inria.fr/download.html
Dataset DOI: https://doi.org/10.57745/Y5JLDG

The download includes the full four-week Winlogbeat archive, system malicious-event
annotations, benign user activity timelines, and the publisher README. Real host
events were collected in an emulated enterprise environment with an APT scenario;
this is not a production incident capture.

Run `python3 download_dedale_real.py --status` from the repository root to check
progress. Run `python3 download_dedale_real.py` to resume after interruption.
`status.json` reports completion only after every file matches publisher size/MD5.
Original archives remain compressed; archive-member inventories record the
uncompressed sizes. Full extraction may require much more disk space.

For long-term detection evaluation, use the publisher's first two weeks for
training/calibration and last two weeks for testing. Preserve real timestamp gaps,
host and process identities, and join ground truth by host/channel/record ID using
the existing `dedale_data.py` helpers. Keep benign events when measuring false
positives and report any sampling explicitly. Attack annotations establish malicious
activity; inferred chains require corroboration from process-parent and other
causal evidence. Downloading this dataset does not establish model accuracy or
prove reconstructed attack chains.

The 27 GB archive cannot be uploaded through the dashboard's 20 MiB uploader.
Stream chronological events from archive members for offline preparation; do not
load the entire archive into memory or silently shorten the capture for testing.

Include a citation to DEDALE and the publisher link when using or redistributing
the dataset, as required on the publisher's download page.
