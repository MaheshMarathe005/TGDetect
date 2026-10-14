"""Build a standalone notebook with a pinned snapshot of the real V3 encoder."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SOURCE_FILES = ['orthrus_comparison/universal.py', 'sliding_node_mem_upgrade/encoding.py',
                'sliding_node_mem_v2/encoding.py', 'sliding_node_mem_v3/tokens.py']


def source_bundle():
    files = {p: (RESEARCH/p).read_text() for p in SOURCE_FILES}
    for directory in ('orthrus_comparison', 'sliding_node_mem_upgrade', 'sliding_node_mem_v2', 'sliding_node_mem_v3'):
        files[directory+'/__init__.py'] = ''
    # Only import plumbing changes; the SGNS objective and hash fitting are preserved.
    sgns = (RESEARCH/'sliding_node_mem_v3/pretrain.py').read_text()
    files['encoder_sgns.py'] = sgns.replace('from orthrus_comparison.run import sha256', 'from encoder_hash import sha256').replace('from .tokens import hashes', 'from sliding_node_mem_v3.tokens import hashes')
    files['encoder_hash.py'] = '''import hashlib
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
'''
    files['experiment.py'] = (HERE/'experiment.py').read_text()
    files['optimizations.py'] = (HERE/'optimizations.py').read_text()
    return files


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(True), 'execution_count': None, 'outputs': []}


def build():
    files = source_bundle()
    hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in files.items()}
    cells = [md('''# TGDetect V3: encoder-only L4 experiment

Select **Runtime → Change runtime type → GPU** (L4 when available), then **Run all**.
This notebook carries the actual original V3 encoder source; no repository upload,
API key, or Drive mount is required. It downloads a public KDD99 subset automatically.
The default uses 5,000 benign training, 1,000 benign validation, 2,000 benign
calibration, and 4,000 test connections (20% attacks), ten arms and three seeds.
This sampled prevalence is artificial: AP/precision are not production estimates.
No GAT/CNN/memory is trained. This is a tiny encoder screening study, not evidence
of CADETS performance. Masked scoring needs extra passes and is timed separately.

The baseline preserves the original encoder exactly. The additional service head
is identical across arms; service is hidden from all encoder inputs. All arms share
benign-training SGNS initialization. Exact encoder-input duplicates/conflicts are
removed before splitting; near duplicates and KDD's dated simulated distribution
remain limitations. There are no verified timestamps: do not call this a temporal
split. Training-prefix rarity never counts a record before reading it; evaluation
uses frozen training counts. Source/data/split hashes and raw scores are exported.

Sources: [official dataset](https://kdd.ics.uci.edu/databases/kddcup99/),
[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html),
[CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
'''), code('''# Use Colab's CUDA-enabled PyTorch; do not replace it with a CPU wheel.
import importlib.util, subprocess, sys
missing = [p for p in ['numpy', 'pandas', 'sklearn', 'matplotlib'] if importlib.util.find_spec(p) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *['scikit-learn' if p == 'sklearn' else p for p in missing]])
import torch
assert torch.cuda.is_available(), 'Choose a GPU runtime first.'
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name())
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
print('Free VRAM GiB:', torch.cuda.mem_get_info()[0] / 2**30)
assert torch.cuda.mem_get_info()[0] >= 4*2**30, 'Free GPU memory is low; restart the GPU runtime.'
if 'L4' not in torch.cuda.get_device_name():
    print('This is not an L4. Results will record the actual GPU; do not label them L4 results.')
'''), md('''## Configuration and test matrix

Change the root for each new experiment; existing results are never overwritten.
Keep the split seed fixed across all arms and seeds. Three seeds are the default;
a one-seed preview is debugging only. More seeds and a fresh dataset are needed
before selecting a production model.

| Arm | Rarity FiLM | Field interaction | Extra pretraining |
|---|---|---|---|
| old | no | no | shared SGNS only |
| old_extra_updates | no | no | service reconstruction, update-matched control |
| rarity | yes | no | shared SGNS only |
| interaction | no | yes | shared SGNS only |
| interaction_extra_updates | no | yes | service reconstruction, update-matched control |
| interaction_masked | no | yes | masked fields |
| rarity_interaction | yes | yes | shared SGNS only |
| rarity_interaction_extra_updates | yes | yes | service reconstruction, update-matched control |
| full | yes | yes | masked fields |
| full_contrastive | yes | yes | masked fields + field-dropout contrastive |

The interaction block is one 64D four-head Transformer layer, without field
positions. This interprets the referenced but unspecified “idea 2”. Full and
masked arms have extra pretraining. The extra-update control matches update
count, not FLOPs/time. All arms retain 64D output. No claim of parameter parity.
Compare interaction_masked against interaction_extra_updates, and full against
rarity_interaction_extra_updates, to isolate the pretraining objective with equal
update counts on the same encoder architecture. Contrastive adds compute per step.
Rarity features are prepared on CPU: latency includes FiLM, not online counting.
''')]
    setup = 'from pathlib import Path\nimport sys, json\nBUNDLE = Path("/content/tgdetect_encoder_bundle")\nBUNDLE.mkdir(parents=True, exist_ok=True)\n'
    setup += 'SOURCE_BUNDLE = ' + repr(files) + '\n'
    setup += '''for relative, source in SOURCE_BUNDLE.items():
    path = BUNDLE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
sys.path.insert(0, str(BUNDLE))
'''
    setup += '(BUNDLE / "source_hashes.json").write_text(' + repr(json.dumps(hashes, indent=2)) + ')\n'
    cells += [code(setup), code('''from experiment import Config, ARMS, run
CFG = Config()
# Optional small smoke preview: CFG.train=256; CFG.val=64; CFG.calibration=128
# CFG.test=256; CFG.epochs=1; CFG.seeds=(0,); CFG.latency_repeats=20
print(CFG)
print('Training arms:', list(ARMS))
print('PyTorch allocator budget GiB:', CFG.memory_budget_gib)
'''), md('''## Train and evaluate

Checkpoints use benign validation service CE. Attack labels are never used to
select checkpoints or thresholds. Calibration A fits empirical-CDF score fusion;
calibration B sets thresholds for 1% and 5% benign FPR targets. Finite samples and
distribution shift mean test FPR can differ. Primary comparisons use service NLL
for every arm. Fused service+masked NLL is a separately reported secondary result.
The four masked scoring fields are selected by training entropy only.

Each completed arm saves its checkpoint, scores, metrics, raw timings, and history.
An error is saved and stops the run. Choose a fresh root after an interrupted run.
'''), code('''RUN = run(CFG)
display(RUN['table'])
display(RUN['summary'])
print(json.dumps(RUN['audit'], indent=2))
'''), md('''## Inspect uncertainty and ranking

The intervals below are paired stratified test-record bootstrap AP differences.
They condition on this test sample and do not measure uncertainty across captures.
Seed standard deviations are separate. A tiny improvement or interval overlapping
zero is inconclusive. Test results do not automatically select a winning arm.
Binomial Wilson intervals for precision/recall/FPR are in each result JSON.
The raw rarity-only control shows whether a neural model adds value beyond simple
frequency. The 60-repeat p95/p99 latency estimates are descriptive and provisional.
'''), code('''import pandas as pd
display(pd.read_json(Path(CFG.root)/'results/paired_ap_intervals.json'))
print('Rarity-only control:', (Path(CFG.root)/'results/rarity_only_control.json').read_text())
import matplotlib.pyplot as plt
table = RUN['table']
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, group in table.groupby('arm', sort=False):
    axes[0].scatter([name]*len(group), group.ap, label=name)
    axes[1].scatter(group.encoder_p50_b128, group.ap, label=name)
axes[0].set_ylabel('Test average precision'); axes[0].tick_params(axis='x', rotation=70)
axes[1].set_xlabel('Encoder batch-128 p50 ms'); axes[1].set_ylabel('Test average precision')
axes[1].legend(fontsize=7); plt.tight_layout()
fig.savefig(Path(CFG.root)/'results/comparison.png', dpi=160, bbox_inches='tight')
plt.show()
'''), md('''## Optional execution experiments

These are disabled by default. Enable after the primary study finishes. They
compare the SAME trained checkpoint, and save numerical error plus timings.
BF16/int8 are precision ablations, not guaranteed-equivalent optimizations.
The int8 variant below is a weight-only token-table storage prototype with explicit
dequantization, not a claim of optimized int8 kernels. The FP32 reference remains
resident so its reported storage savings are theoretical unless it is unloaded.

Field reuse is before FiLM and interaction. The frozen field table deduplicates
identical prepared field rows. Frozen event lookup is legal only while the rarity
snapshot is fixed; it is disabled for streaming counts. Cache build time/memory,
actual requests, and a synthetic repeat-heavy stream are reported separately.
Compact bags are an execution prototype, not FlashAttention varlen support.
With only 41 fields, dynamic packing may lose to ordinary padding.
'''), code('''RUN_OPTIONAL_EXECUTION = False
if RUN_OPTIONAL_EXECUTION:
    from optimizations import execution_study
    execution_results = execution_study(RUN)
    display(pd.DataFrame(execution_results))
'''), md('''## Optional distillation

Distill the full encoder's 64D output into the original encoder using benign
training only. The student's output shape and neural architecture stay unchanged.
This pilot's student has no rarity input, so it may fail to reproduce a teacher
whose output depends on counts. Evaluate rather than assume preservation.
'''), code('''RUN_DISTILLATION = False
if RUN_DISTILLATION:
    from optimizations import distill
    distillation_result = distill(RUN)
    print(json.dumps(distillation_result, indent=2))
'''), md('''## Synthetic coverage probes: field truncation and lists

KDD does not exercise long lists, paths, or >64 fields. These probes demonstrate
IDF-priority selection and order-invariant full-list pooling, but do not establish
their accuracy. Lists can encode meaningful order (commands, arguments), so apply
set pooling only to schema-declared unordered fields. Key IDF is fitted on training
documents; rare keys are not automatically security-relevant keys.
'''), code('''from optimizations import schema_probes
print(json.dumps(schema_probes(), indent=2))
'''), md('''## Export everything

Download and share the zip for review. It contains raw per-record scores so metrics
can be independently recomputed. Latency excludes raw-event ingestion, batch wait,
and TGDetect graph/memory work; it cannot establish full-detector latency.
'''), code('''import shutil
shutil.copy(BUNDLE/'source_hashes.json', Path(CFG.root)/'source_hashes.json')
shutil.copytree(BUNDLE, Path(CFG.root)/'source_snapshot', dirs_exist_ok=True)
archive = shutil.make_archive('/content/tgdetect_encoder_results', 'zip', CFG.root)
print('Results:', archive)
from google.colab import files
files.download(archive)
''')]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.11'}, 'colab': {'name': 'TGDetect_Encoder_L4.ipynb'},
                'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'encoder-{i:03d}'
    path = HERE/'TGDetect_Encoder_L4.ipynb'
    path.write_text(json.dumps(notebook, indent=1))
    print(path)
    return notebook


if __name__ == '__main__':
    build()
