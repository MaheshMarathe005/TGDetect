"""Create the focused follow-up without overwriting the first study notebook."""
import hashlib
import json
from pathlib import Path
from build_notebook import source_bundle, md, code

HERE = Path(__file__).resolve().parent


def build():
    files = source_bundle()
    files['rarity_followup.py'] = (HERE/'rarity_followup.py').read_text()
    hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p,s in files.items()}
    setup = '''from pathlib import Path
import sys, json
BUNDLE = Path('/content/tgdetect_rarity_bundle')
BUNDLE.mkdir(parents=True, exist_ok=True)
'''+ 'SOURCE_BUNDLE = '+repr(files)+'\n'
    setup += '''for relative, source in SOURCE_BUNDLE.items():
    path = BUNDLE/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
sys.path.insert(0, str(BUNDLE))
'''
    setup += '(BUNDLE/"source_hashes.json").write_text('+repr(json.dumps(hashes,indent=2))+')\n'
    cells = [md('''# TGDetect: controlled rarity follow-up

Choose a GPU runtime, then Run all. T4 and L4 are accepted; the actual GPU is
recorded. No credentials or repository clone needed. The KDD download is pinned
by SHA256. This is an encoder experiment, not a full TGDetect model evaluation.

The first study found promising rarity AP, large seed variation, and a strong
raw rarity baseline. This study tests causes instead of combining every idea.
All six arms receive identical training records, batches, common initial weights,
optimizer settings and **eight task epochs**. Benign validation selects checkpoints.
There is no masked-field pretraining or contrastive loss in this study.

Default: **6 arms × 5 seeds × 2 new split seeds = 60 runs**. Each split has
1,000 benign reference records, 4,000 benign optimization records, 1,000 benign
validation, 2,000 benign calibration, and 4,000 test (20% attacks). Reference
records never enter optimizer updates. Different random splits may overlap;
they are sensitivity checks within one dataset, not independent capture replications.
Split stratification and ambiguous-input filtering use labels; training and
calibration never use test labels. KDD remains dated and lacks verified timestamps.
'''), code('''import importlib.util, subprocess, sys
packages = {'numpy':'numpy', 'pandas':'pandas', 'sklearn':'scikit-learn'}
missing = [install for module,install in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import torch
assert torch.cuda.is_available(), 'Select Runtime → Change runtime type → GPU.'
print('GPU:',torch.cuda.get_device_name(), 'PyTorch:',torch.__version__)
print('Free GiB:',torch.cuda.mem_get_info()[0]/2**30)
assert torch.cuda.mem_get_info()[0] > 4*2**30, 'Restart runtime to free GPU memory.'
'''),md('''## Controlled matrix

| Arm | Change | Question |
|---|---|---|
| old | Original encoder | Does rarity add value at equal training budget? |
| prefix_log_film | Read-before-write counts, eval uses final counts | Is the original count regime unstable? |
| frozen_log_film | Same disjoint benign reference for all splits | Does matching count regimes help? |
| frozen_surprise_film | Bounded inverse-frequency feature | Does scale/direction make FiLM easier to learn? |
| shuffled_surprise_film | Shuffle each field's rarity across records | Does the count-value association matter? |
| frozen_surprise_interaction | Add one field-interaction layer | Is its gain worth its latency? |

Every rarity arm zeros the synthetic event-type count. This prevents it encoding
training progress. Prefix counts start from the same reference snapshot, read a
task record before inserting it, and are constructed once. Frozen counts never
count an optimization, validation, calibration, or test record.

Bounded feature: `clip((log1p(Nref)-log1p(count))/log1p(Nref),0,1)`.
Higher values mean rarer. This is a normalized novelty feature, not an exact
probability or likelihood. It is affine in log-count before clipping; any gain
may reflect optimization and scale effects rather than new information.
'''),code(setup),code('''from rarity_followup import FollowupConfig, MATRIX, run_followup
CFG = FollowupConfig()
# Fast debug only (not evidence):
# CFG.train=256; CFG.reference_rows=64; CFG.val=64; CFG.calibration=128
# CFG.test=256; CFG.epochs=1; CFG.seeds=(0,); CFG.split_seeds=(20261013,)
print(CFG)
print(MATRIX)
'''),md('''## Train and inspect

Primary scores are raw service NLL. Report AP, F1, test FPR, per-family recall,
macro-family recall, and non-Neptune recall. The majority attack family must not
hide failures on rare attacks. Record-bootstrap intervals condition on each test
sample; seed/split variation is reported separately. Eight epochs are a fixed
screening budget, not an assumed convergence guarantee.

Secondary experiments include raw reference rarity; fixed late-fusion weights
0.25/0.5/1; and a zero-rarity inference counterfactual. Fusion uses calibration A
median/IQR scaling and retains scores beyond its maximum. Calibration B sets
thresholds. Weights are predeclared, not selected using test labels. Do not choose
the best test fusion weight and call it a prospective result.

Rarity count reads are timed on CPU separately. Encoder timing includes FiLM
and prepared inputs, but not count construction, ingestion or batching wait.
The 15-GiB allocator budget excludes CUDA context allocations.
'''),code('''TABLE, SUMMARY = run_followup(CFG)
display(TABLE)
display(SUMMARY)
'''),code('''import pandas as pd
display(pd.read_json(Path(CFG.root)/'paired_comparisons.json'))
for split in CFG.split_seeds:
    print('Count protocol:',split)
    print((Path(CFG.root)/f'split_{split}/count_protocol.json').read_text())
    print('Raw rarity:',(Path(CFG.root)/f'split_{split}/results/raw_rarity.json').read_text())
    print('Raw inverse-count:',(Path(CFG.root)/f'split_{split}/results/raw_inverse_count.json').read_text())
'''),md('''## Interpretation rules fixed before this run

1. Rarity must improve AP and non-Neptune recall consistently enough across seeds
   and splits to justify its cost; an aggregate mean alone is insufficient.
2. If frozen counts beat prefix counts, count regime mismatch is a plausible
   contributor. Reference size/order and differing eval counts also change, so
   this is evidence about the regime, not a complete causal proof.
3. If shuffled rarity performs similarly, frequency association has not shown
   added value. Check optimization or input-distribution effects before adoption.
4. If old + late rarity fusion matches learned FiLM, prefer testing that simpler
   deployment path before adding encoder complexity.
5. If interaction gains come only from the dominant family, retain it as a
   research candidate. Re-test on another dataset/capture before TGDetect adoption.
6. Keep masked pretraining out until its decoder is trained jointly or refitted
   after encoder updates and validated per field.

The raw frequency baseline is cheap but may confuse a novel benign value with an
attack. A future conditional count test should compare rarity within benign
entity-type/protocol groups with global counts, with minimum-support fallback.
That experiment is not silently bundled into this matrix.
'''),code('''import shutil
shutil.copy(BUNDLE/'source_hashes.json',Path(CFG.root)/'source_hashes.json')
shutil.copytree(BUNDLE,Path(CFG.root)/'source_snapshot',dirs_exist_ok=True)
archive=shutil.make_archive('/content/tgdetect_rarity_followup_results','zip',CFG.root)
from google.colab import files
files.download(archive)
''')]
    for i,cell in enumerate(cells):
        cell['id']=f'rarity-{i:03d}'
    notebook={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},'accelerator':'GPU',
        'colab':{'name':'TGDetect_Rarity_Followup.ipynb'}}}
    path=HERE/'TGDetect_Rarity_Followup.ipynb';path.write_text(json.dumps(notebook,indent=1))
    print(path);return notebook


if __name__=='__main__':
    build()
