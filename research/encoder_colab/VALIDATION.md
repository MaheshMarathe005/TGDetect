# Local validation — 12 October 2026

Environment: Python with PyTorch `2.11.0+cpu`. No CUDA device or `nvidia-smi`
was available. This validation does not establish L4 speed, GPU memory use,
or an accuracy improvement over the original encoder.

## Completed checks

- Nine tests passed with `python -m pytest -q research/encoder_colab/test_experiment.py`.
- The original PackedEncoder and candidate wrapper match outputs and gradients
  when added components are disabled, including repeated identity requests.
- Masking removes the field's value, type, number, and rarity signal.
- Repeated identities with different rarity counts retain different outputs.
- Compact token bags and the static field cache preserve reference outputs.
- Prefix counts exclude the current record; evaluation counts remain frozen.
- Exact input duplicates do not cross splits; service is absent from inputs.
- Strict thresholding, paired bootstrap behavior, IDF selection, and full-list
  permutation/tail-element probes passed.
- All ten training arms, CPU execution variants, and optional distillation ran
  on a small synthetic pipeline fixture.
- All generated notebook code cells compile. Four original encoder source
  files are bundled verbatim; SGNS only changes imports to avoid graph dependencies.

## Real public-data smoke run

Downloaded the 494,021-row KDD Cup 1999 10% archive from scikit-learn's pinned
Figshare mirror. Verified SHA256:

`8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561`

All ten arms completed on 128 benign training, 32 benign validation, 128 benign
calibration, and 128 mixed test records, with one seed and one task epoch. This
was a pipeline smoke test, not a model-selection experiment. No hyperparameters
were selected from its test scores. The maximum original/wrapper and
original/compact embedding discrepancy was `7.152557373046875e-07`.

## Still requires the notebook's GPU run

- Default three-seed, ten-arm study on the configured 12,000-record sample.
- L4 latency, peak allocated VRAM, and BF16/int8 behavior.
- GPU compile and CUDA-graph optional paths, whose failures are recorded.
- Detection gains on another capture or CADETS. KDD is dated and lacks verified
  timestamps, provenance topology, paths, long lists, and >64-field records.

Optional IDF/list probes are functional examples, not integrated detection
results on KDD. Optional distillation is a one-seed feasibility experiment with
additional training; it does not isolate distillation from extra fine-tuning.
