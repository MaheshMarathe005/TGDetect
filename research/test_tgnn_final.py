"""Acceptance of the selected encoder, with the historical source as reference."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tgnn_final import TGNNFinal, TokenWriter, TokenBank

ROOT = Path(__file__).resolve().parent


def reference_encoder():
    # Import original files with empty package initializers, avoiding the whole
    # historical model's graph/runtime imports.
    spec = importlib.util.spec_from_file_location('tgnn_final_reference_bundle', ROOT/'encoder_colab/build_notebook.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    temp = tempfile.TemporaryDirectory(prefix='tgnn_final_reference_')
    for relative, source in builder.source_bundle().items():
        path = Path(temp.name)/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
    # Separate interpreter namespaces via a unique package and relative imports.
    package = Path(temp.name)/'upstream_reference'; package.mkdir()
    (package/'__init__.py').write_text('')
    mapping = [('orthrus_comparison/universal.py','_universal.py',{}),
               ('sliding_node_mem_upgrade/encoding.py','_encoding_upgrade.py',{'from orthrus_comparison.universal import':'from ._universal import'}),
               ('sliding_node_mem_v2/encoding.py','_encoding_v2.py',{'from sliding_node_mem_upgrade.encoding import':'from ._encoding_upgrade import'}),
               ('sliding_node_mem_v3/tokens.py','encoder.py',{'from orthrus_comparison.universal import':'from ._universal import','from sliding_node_mem_v2.encoding import':'from ._encoding_v2 import'})]
    for src, dst, replacements in mapping:
        source = (ROOT/src).read_text()
        for a,b in replacements.items(): source=source.replace(a,b)
        (package/dst).write_text(source)
    sys.path.insert(0,temp.name)
    from upstream_reference.encoder import PackedEncoder
    return temp, PackedEncoder


def make_bank(path):
    writer = TokenWriter(path)
    events = [
        {'attributes': {'src': {'path':'/usr/bin/curl','type':'process'},'n':12}},
        {'attributes': {'missing':None,'flags':[True,False,7],'ip':'192.168.1.10'}},
        {'type':'2','attributes':{'long_path':'/'.join(str(i) for i in range(60))}},
    ]
    ids = [writer.add(e) for e in events]
    ids += [writer.add(events[0])]
    writer.close()
    return TokenBank(path), torch.tensor(ids)


def test_encoder_outputs_gradients_and_state_names_match_original(tmp_path):
    torch.set_num_threads(2)
    temp, Encoder = reference_encoder()
    bank, ids = make_bank(tmp_path/'tokens')
    torch.manual_seed(9); old=Encoder(bank); final=TGNNFinal(bank)
    final.load_encoder_weights(old.state_dict())
    assert set(final.event_encoder.state_dict()) == set(old.state_dict())
    a,b=old(ids),final(ids)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    weights=torch.randn_like(a)
    (a*weights).sum().backward();(b*weights).sum().backward()
    for name,p in old.named_parameters():
        torch.testing.assert_close(p.grad,dict(final.event_encoder.named_parameters())[name].grad,rtol=0,atol=0)
    assert b.shape == (4,64)
    temp.cleanup()


def test_frozen_lookup_and_training_invalidation(tmp_path):
    bank,ids=make_bank(tmp_path/'tokens');model=TGNNFinal(bank).eval()
    with torch.no_grad():expected=model(ids)
    size=model.precompute_encoder(batch_size=2)
    assert size == bank.config['identities']*64*4
    with torch.no_grad():torch.testing.assert_close(model(ids),expected,rtol=1e-5,atol=1e-6)
    model.train()
    assert model.event_encoder._frozen_table is None
    model(ids).square().mean().backward()
    assert model.event_encoder.tokens.weight.grad is not None


def test_v3_prefixed_weights_and_model_roundtrip(tmp_path):
    bank,ids=make_bank(tmp_path/'tokens');model=TGNNFinal(bank)
    saved={**model.state_dict(),'gat.example_unused_weight':torch.ones(1)}
    restored=TGNNFinal(bank);restored.load_encoder_weights(saved)
    torch.testing.assert_close(model(ids),restored(ids),rtol=0,atol=0)
    assert list(dict(restored.named_children())) == ['event_encoder']


def test_source_manifest_and_standalone_import():
    root=ROOT/'tgnn_final';manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['sources']:
        source=(ROOT/entry['source']).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==entry['source_sha256']
        for a,b in entry['changes'].items():source=source.replace(a,b)
        assert source==(root/entry['snapshot']).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==entry['snapshot_sha256']
    result=subprocess.run([sys.executable,'-c',
        "import sys;from tgnn_final import TGNNFinal;assert 'torch_geometric' not in sys.modules;assert 'sliding_node_mem_v3' not in sys.modules;print('standalone import OK')"],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
