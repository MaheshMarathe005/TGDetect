# Encoder acceptance

Four tests passed on local CPU PyTorch with:

```sh
/home/mahesh/main_env/bin/python -m pytest -q research/test_tgnn_final.py
```

Original V3 versus TGNN Final outputs and all encoder parameter gradients are
exactly equal on typed, nested, long-value and repeated-identity fixtures with
identical weights. State names/shapes match. Frozen FP32 lookup matches within
floating-point tolerance and clears on entering training. Loading encoder-only
or event_encoder-prefixed V3 weights works. Source manifests verify that only
imports changed. A fresh-process import does not load torch_geometric or the old
full V3 package. These are component acceptance checks, not accuracy/GPU results.
