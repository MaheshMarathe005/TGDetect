# Local acceptance

23 tests passed with:

```sh
/home/mahesh/main_env/bin/python -m pytest -q research/test_gat_study.py research/test_tgnn_final.py
```

The suite covers all eight arms, provenance-mask parity, finite padded rows,
future independence, role-specific value-head isolation, query/history degree
histograms, shared initialization outside GAT, exact original two-GAT baseline
output parity, chunked causal state replay, and gradients through full V3.

The experimental trainer also completed one-epoch synthetic full-model runs for
the original two-GAT and role-factored operators, wrote edge scores, calibrated
the detector, computed a predefined minority-subset metric, and completed paired
whole-forward latency replays. This is an integration smoke test, not a measured
detection improvement or a GPU benchmark. Final encoder-only acceptance passes.

Two inherited TorchScript deprecation warnings were observed during the full-model
scan. No new graph warning/failure occurred. Local PyTorch is CPU-only; no paid
GPU training was launched. Real-data degree histograms, minority accuracy, and
full-model GPU timing remain to be measured using the prepared-data runner.
