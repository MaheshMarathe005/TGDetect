# TGNN Final

Current stage: **encoder only**. The selected component is the original corrected
V3 typed key-value `PackedEncoder`. Forward returns `[N,64]` embeddings.
This is the foundation of the new version, not a complete trained detector.

Included: offline TokenWriter/TokenBank, two salted hashes summed from a shared
5,303×64 table by default, value-order adjacent features, kind/numeric branch,
field MLP, four-head learned-query field pooling, per-call identity deduplication,
and opt-in frozen inference lookup. No rarity conditioning is adopted.

The encoder and its three inheritance dependencies are copied into this package.
Only imports are changed to local relative imports. SOURCE_MANIFEST.json records
original and snapshot hashes, so later edits elsewhere cannot silently change
the selected encoder. Imports require only Python, NumPy and PyTorch; importing
TGNN Final does not instantiate GAT/CNN/memory or require torch_geometric.
Unused inherited raw-event helpers are preserved for exact source compatibility;
TGNNFinal.forward consumes prepared integer identities.

```python
# From repo root: PYTHONPATH=research python your_script.py
import torch
from tgnn_final import TokenWriter, TokenBank, TGNNFinal

writer = TokenWriter('/tmp/my_new_token_bank')  # new directory required
identity = writer.add({'attributes': {'path': '/usr/bin/curl', 'type': 'process'}})
writer.close()
bank = TokenBank('/tmp/my_new_token_bank')
model = TGNNFinal(bank)
vectors = model(torch.tensor([identity]))  # [1,64], trainable

# Optional pretrained encoder/V3 model state_dict:
# model.load_encoder_weights(weights)
# GPU: bank.to_device('cuda'); model.to('cuda')
model.eval()
model.precompute_encoder()
with torch.no_grad():
    cached_vectors = model(torch.tensor([identity]))
```

Weights are not bundled. Constructing the module creates initial weights;
load a compatible trained checkpoint or train it before evaluating accuracy.
V3 encoder parameter names/shapes are preserved. Historical whole-model runtime
state and checkpoint signatures are not TGNN Final model configurations.

## Next component: causal provenance GAT

See NEXT_COMPONENT.md. Only the encoder is accepted into this package today.
Projection, time features, window planning, GAT, CNN, memories and scoring remain
pending experiments. A GAT test will need fixed causal graph/window fixtures:
do not invent provenance links from KDD feature similarity.

Run acceptance checks from repository root:

```sh
/home/mahesh/main_env/bin/python -m pytest -q research/test_tgnn_final.py
```
