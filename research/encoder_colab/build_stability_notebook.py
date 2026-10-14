import hashlib
import json
from pathlib import Path
from build_notebook import source_bundle, md, code

HERE=Path(__file__).resolve().parent


def build():
    sources=source_bundle()
    for name in ('rarity_followup.py','rarity_stability.py'):
        sources[name]=(HERE/name).read_text()
    hashes={p:hashlib.sha256(v.encode()).hexdigest() for p,v in sources.items()}
    install='''from pathlib import Path
import sys,json
BUNDLE=Path('/content/tgdetect_stability_bundle');BUNDLE.mkdir(parents=True,exist_ok=True)
'''+ 'SOURCE_BUNDLE='+repr(sources)+'\n'
    install+='''for relative,source in SOURCE_BUNDLE.items():
    p=BUNDLE/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
sys.path.insert(0,str(BUNDLE))
'''
    install+='(BUNDLE/"source_hashes.json").write_text('+repr(json.dumps(hashes,indent=2))+')\n'
    cells=[md('''# TGDetect: training length, fast counts, and rarity robustness

Upload to Colab, choose GPU (T4 or L4), then Run all. Original versus frozen
log-count FiLM only. Dataset download and source code are bundled. No API keys.

Two encoders × five seeds × two new split seeds = 20 training trajectories.
Each trains to 16 epochs; report validation-selected checkpoints available at
8/12/16 epochs. This gives 60 budget evaluations without restarting training.
The first two studies remain historical; this notebook writes a separate root.
Exact input deduplication, benign-only learning/calibration, and disjoint benign
reference counts follow the previous protocol. New random splits may overlap
earlier data and do not constitute independent captures. KDD is dated, simulated,
and not a full TGDetect evaluation. No labeled Back/Warezclient training is added.
'''),code('''import importlib.util,subprocess,sys
packages={'numpy':'numpy','pandas':'pandas','sklearn':'scikit-learn'}
missing=[install for module,install in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import torch
assert torch.cuda.is_available(),'Select GPU runtime.'
print('GPU:',torch.cuda.get_device_name(),'PyTorch:',torch.__version__)
print('Free GiB:',torch.cuda.mem_get_info()[0]/2**30)
assert torch.cuda.mem_get_info()[0] > 4*2**30,'Restart runtime to free GPU memory.'
'''),code(install),md('''## Fixed research questions

1. Does original vs rarity change after longer equal training? Check convergence
   curves and whether checkpoint selection still reaches the budget boundary.
2. Can prehashed vectorized count lookup reduce the measured Python bottleneck?
   Report cold fingerprint preparation, Python hash+lookup, NumPy prehashed lookup,
   device lookup, and host-staging+count+encoder separately. Match CPU/GPU counts.
   The original deployed path does not pay for unnecessary rarity lookup.
3. How does the chosen checkpoint behave under count shifts?

| Condition | Construction | Interpretation |
|---|---|---|
| clean | Frozen 1,000-record benign reference | Main comparison |
| stale_half_reference | Only half the reference retained | Reduced coverage; proxy for stale snapshot, not verified temporal drift |
| duplicate_updates_30pct | Repeat a random 30% of benign reference | Duplicate-ingestion counter noise |
| collision_small_table | Width 2,048 instead of 65,536 | More CMS collision overcounts |
| missing_fields_10pct | Randomly replace 10% field log counts with zero | Missing/count-zero ambiguity; no trained fallback yet |

Stress choices are fixed without test labels. These simulations do not establish
real deployment drift. The existing encoder is not trained on corrupted counts.
Report frozen-threshold and condition-specific benign recalibration separately.
Recalibration assumes benign data from that condition is actually available.
Back, Warezclient, non-Neptune and all per-family recalls remain explicit, with
sample counts in result JSON. Rare-family estimates are uncertain; no test-driven
threshold or checkpoint selection is allowed.
'''),code('''from rarity_stability import StabilityConfig,run_stability
CFG=StabilityConfig()
# Optional debug only:
# CFG.train=256;CFG.reference_rows=64;CFG.val=64;CFG.calibration=128
# CFG.test=256;CFG.seeds=(0,);CFG.split_seeds=(20261015,);CFG.budgets=(1,2)
# CFG.latency_repeats=5;CFG.timing_repeats=3;CFG.bootstrap=10
print(CFG)
TABLE,SUMMARY=run_stability(CFG)
display(TABLE)
display(SUMMARY)
'''),code('''import pandas as pd
display(pd.read_json(Path(CFG.root)/'paired_comparisons.json'))
print('Clean-condition summary:')
display(TABLE[TABLE.condition.eq('clean')].groupby(['arm','budget','threshold_policy'])[
    ['ap','f1','fpr','non_neptune_recall','back_recall','warezclient_recall','deployed_encoder_ms']].agg(['mean','std']))
'''),md('''## Timing limits and decision rules

The device lookup uses the same four BLAKE2 bucket fingerprints as the Python
count-min reference, and takes the minimum across four rows. Hashes are computed
once per distinct canonical field in each preprocessing call. New field strings
still need hashing. The cold fingerprint cost is measured separately.
Device inputs and the frozen count table consume GPU memory. Staging benchmarks
include host-to-device fingerprint/identity copies, but exclude raw-event ingestion,
GAT/CNN/memory, scoring, batch-formation wait and online counter updates. Counts
are frozen: this notebook does not claim a causal streaming update implementation.

Keep the original if rarity gains disappear with adequate training. If counts
improve minority-family recall but increase false alerts, compare operating points
using benign-only thresholds; do not report AP as proof of low-FPR superiority.
If realistic count shifts cause failure, test a separately trained missing-count
fallback/rarity dropout next. If both encoders still miss Back or Warezclient,
investigate missing features or the service-reconstruction objective on development
data before adding more layers. Budget checkpoints are correlated measurements
of the same training trajectory. Record bootstrap intervals are not independent
capture uncertainty. Do not choose the best tested budget and present it as an
untouched prospective evaluation.
'''),code('''import shutil
shutil.copy(BUNDLE/'source_hashes.json',Path(CFG.root)/'source_hashes.json')
shutil.copytree(BUNDLE,Path(CFG.root)/'source_snapshot',dirs_exist_ok=True)
archive=shutil.make_archive('/content/tgdetect_rarity_stability_results','zip',CFG.root)
from google.colab import files
files.download(archive)
''')]
    for i,c in enumerate(cells):c['id']=f'stability-{i:03}'
    notebook={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},'accelerator':'GPU',
        'colab':{'name':'TGDetect_Rarity_Stability.ipynb'}}}
    path=HERE/'TGDetect_Rarity_Stability.ipynb';path.write_text(json.dumps(notebook,indent=1));print(path)
    return notebook


if __name__=='__main__':build()
