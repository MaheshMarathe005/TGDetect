"""Verify completed pilot artifacts and generate a report without rerunning training."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from causal_data import sha256
from causal_tgn import CausalTGNN
from run_causal_pilot import ARMS, SOURCES, metrics, select_threshold, save_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/causal_pilot_v1'


def main():
    torch.set_num_threads(1)
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    data_path=ROOT/'data/ctu13_verified/scenario06_pilot16384.npz'
    meta=manifest['data_manifest']
    assert sha256(data_path)==meta['processed_sha256']
    assert sha256(ROOT/'data/ctu13_verified/scenario06.binetflow')==meta['raw_source']['sha256']
    assert all(sha256(ROOT/p)==manifest['source_hashes'][p] for p in SOURCES)
    runs=[]
    parts=meta['partitions']
    val=slice(parts['validation']['start'],parts['validation']['end'])
    test=slice(parts['test']['start'],parts['test']['end'])
    for arm in ARMS:
        for seed in range(5):
            name=f'{arm}_seed{seed}'
            r=json.loads((OUT/f'{name}.json').read_text())
            assert r['status']=='COMPLETE' and r['source_hashes']==manifest['source_hashes']
            assert sha256(OUT/f'{name}.pt')==r['checkpoint_sha256']
            p=np.load(OUT/f'{name}_predictions.npz',allow_pickle=False)
            threshold=select_threshold(p['y'][val],p['probability'][val])
            assert threshold==r['threshold']
            assert metrics(p['y'][test],p['probability'][test],threshold)==r['test']
            assert int(p['y'][test].sum())==25
            for mode, expected in r['fixed_checkpoint_controls'].items():
                assert metrics(p['y'][test],p[mode][test],threshold)==expected
            runs.append(r)
    controls=json.loads((OUT/'feature_controls.json').read_text())
    for r in controls:
        p=np.load(OUT/f"{r['arm']}_seed{r['seed']}_predictions.npz",allow_pickle=False)
        assert select_threshold(p['y'][val],p['probability'][val])==r['threshold']
        assert metrics(p['y'][test],p['probability'][test],r['threshold'])==r['test']
    matrix_path=OUT/'ablation_matrix.json'
    matrix=json.loads((matrix_path if matrix_path.exists() else ROOT/'ABLATION_MATRIX.json').read_text())
    counts=Counter(r['status'] for r in matrix['rows'])
    assert counts=={'COMPLETE':40,'NOT_RUN':200}
    assert len({tuple(r['active_factors']) for r in matrix['rows'] if r['group']=='pair'})==28
    save_json(matrix_path,matrix)
    source_dir=OUT/'source'; source_dir.mkdir(exist_ok=True)
    for p in SOURCES:
        saved=source_dir/p
        if saved.exists():
            assert sha256(saved)==manifest['source_hashes'][p]
        else:
            saved.write_bytes((ROOT/p).read_bytes())

    # Profile actual loaded checkpoint modules, not only declared dimensions.
    arrays=np.load(data_path,allow_pickle=False)
    x=torch.from_numpy(arrays['x'][arrays['ego'][:4]])
    valid=torch.from_numpy(arrays['valid'][:4]); ages=torch.from_numpy(arrays['ages'][:4])
    profiles={}
    for arm, factors in ARMS.items():
        model=CausalTGNN(10,factors).eval()
        state=torch.load(OUT/f'{arm}_seed0.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['state_dict'])
        seen={}; hooks=[]
        def record(name):
            def hook(module,inputs,output):
                seen.setdefault(name,{'input_shapes':[list(t.shape) for t in inputs if isinstance(t,torch.Tensor)],
                                      'output_shape':list(output.shape)})
            return hook
        for name,module in model.named_modules():
            if name and (name.startswith('layers') or name in {'head','cell','time_encoder'}):
                hooks.append(module.register_forward_hook(record(name)))
        with torch.no_grad():
            model.forward_events(x,valid,ages,arrays['src'][:4],arrays['dst'][:4],arrays['times'][:4])
        for h in hooks: h.remove()
        assert sum(p.numel() for p in model.parameters())==next(r['parameters'] for r in runs if r['arm']==arm)
        profiles[arm]={'parameters':sum(p.numel() for p in model.parameters()),'observed_shapes':seen}
    save_json(OUT/'model_profiles.json',profiles)

    def mean(values): return float(np.mean(values))
    def cell(v): return f"{100*v['mean']:.2f} ± {100*v['sd']:.2f}"
    names={'B':'Adapted GNN','B_T':'GNN + time','B_M':'GNN + fixed memory',
           'B_MR':'GNN + GRU memory','B_TM':'GNN + time + fixed memory',
           'B_TMR':'Pilot TGNN (time + GRU memory)','B_H':'GNN + rehearsal',
           'B_TMRH':'Pilot TGNN + rehearsal','LogisticRegression':'Logistic regression',
           'HistGradientBoosting':'Gradient-boosted trees'}
    lines=['# Controlled TGNN pilot — completed 23 September 2026','',
           'All 40 neural runs (8 configurations × 5 seeds), 10 feature-control fits,',
           'and 50 fixed-checkpoint interventions completed. This is an explicitly',
           'adapted CTU-13 experiment, not a reproduction of the Wazuh paper or a',
           'test of the complete legacy TG-Detect model.','',
           '## Data and protocol','',
           'Official CTU-13 scenario 6: 558,919 raw flows; 16,384 label-independently',
           'sampled flows sorted by completion-time proxy. Train/validation/test:',
           '9,830/3,277/3,277 records, with 91/25/25 Botnet labels. Background is',
           'treated as an uncertain negative. Ten static features; train-only scaling;',
           'same graph, labels, optimizer, three epochs and seeds 0–4. Thresholds',
           'maximize validation F1 and are fixed for test. See [protocol](ADAPTED_PROTOCOL.md)',
           'for operator, replay, source-selection amendments and state scheduling.','',
           'Scenario 11 and the smaller scenario 6 sample failed recorded class-support',
           'gates before real-data training. The final pilot retains only 25 positive',
           'test flows: one missed attack changes recall by four percentage points.','',
           '## Test results','',
           'Values are percentages, mean ± sample standard deviation across five',
           'initialization seeds. PR-AP is average precision, not trapezoidal PR area.',
           'Seed variability is not uncertainty over captures or independent attacks.',
           'The deterministic feature controls repeat identical fits across seeds.','',
           '| Model | F1 | PR-AP | ROC-AUC | FPR | Parameters |',
           '|---|---:|---:|---:|---:|---:|']
    stats=summary['means_and_sample_sd']
    for name in list(ARMS)+['LogisticRegression','HistGradientBoosting']:
        s=stats[name]
        param=str(profiles[name]['parameters']) if name in profiles else '—'
        lines.append(f"| {names[name]} | {cell(s['f1'])} | {cell(s['pr_ap'])} | {cell(s['roc_auc'])} | {cell(s['fpr'])} | {param} |")
    lines += ['', '## Paired component effects','',
              'Differences in percentage points, paired by training seed. These are',
              'pilot effect estimates; the full factorial study remains incomplete.','',
              '| Contrast | ΔF1 | ΔPR-AP |','|---|---:|---:|']
    for name, effects in summary['paired_seed_effects'].items():
        lines.append(f"| {name} | {cell(effects['f1'])} | {cell(effects['pr_ap'])} |")
    lines += ['', '## Fixed-checkpoint falsification','',
              'Weights and original validation thresholds stay fixed. Reset/frozen/no-op',
              'all produce zero state from a fresh start here, so they are related checks,',
              'not three independent confirmations. Shuffled age preserves graph/order',
              'and tests only the numeric-time channel.','',
              '| Parent | Intervention | F1 | PR-AP |','|---|---|---:|---:|']
    for arm, modes in summary['negative_controls_fixed_threshold'].items():
        for mode, s in modes.items():
            lines.append(f"| {arm} | {mode} | {cell(s['f1'])} | {cell(s['pr_ap'])} |")
    lines += ['', '## Resource measurements','',
              'One CPU thread in ~/main_env; no GPU. Inference includes rebuilding',
              'history across all partitions. Latency is amortized batch throughput,',
              'not a measured individual online request. Tensor state bytes exclude',
              'Python objects. Process RSS is cumulative, not comparable per-model peak.','',
              '| Model | Train seconds | Inference events/s | State tensors KiB | Replay KiB | Checkpoint KiB |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rr=[r for r in runs if r['arm']==arm]
        lines.append(f"| {arm} | {mean([r['train_seconds'] for r in rr]):.2f} | {mean([r['inference_events_per_second'] for r in rr]):.0f} | {mean([r['state_tensor_bytes']+r['pending_state_tensor_bytes'] for r in rr])/1024:.1f} | {mean([r['replay_context_bytes'] for r in rr])/1024:.1f} | {mean([r['checkpoint_bytes'] for r in rr])/1024:.1f} |")
    lines += ['', 'The live training budget is 29,490 event visits per run. Rehearsal adds',
              f"{next(r['training_replay_visits'] for r in runs if r['arm']=='B_H'):,} replay visits. Same epochs do not mean equal wall-clock compute.",
              'The TGNN has more parameters; capacity-matched and compute-matched controls',
              'remain outstanding. Full 200-epoch/multi-capture training belongs on Modal.','',
              '## Interpretation and limits','',
              'The time-only model improved F1 by 6.04 percentage points on average',
              '(paired seed SD 3.60), with a positive difference in every seed. Adding',
              'fixed memory reduced F1 by 6.02 points. The pilot TGNN changed F1 by',
              'only +0.53 points (paired SD 3.20), with mixed signs across seeds.',
              'The feature-only tree had the highest measured F1 and AP. These results',
              'do not support selecting the larger TGNN for this task.',
              '',
              'For the TGNN without rehearsal, resetting memory reduces F1 from',
              '84.33% to 25.22% at its original threshold, yet AP increases from',
              '90.34% to 93.56%. This is evidence of a score/threshold shift, not proof',
              'that memory improves ranking. Constant or shuffled ages reduce AP too,',
              'showing sensitivity to the numeric-time channel in this fitted model.',
              'Interventions cause distribution shift and cannot replace retrained controls.',
              '',
              'Persistent memory is not automatically useful merely because it learns',
              'on a toy stream. Added time parameters and learned offsets still require',
              'matched capacity/constant-time training controls. No result establishes',
              'a general TGNN advantage, cross-family robustness, APT attribution, or',
              'reduced catastrophic forgetting.','',
              'The synthetic learning check is separate: across five seeds, the static',
              'model averaged 49.84% accuracy, GRU memory 100%, and resetting that memory',
              '48.59%. Both toy graphs excluded history, and current query inputs were',
              'identical. This verifies the implementation can learn the designed signal.','',
              '## Verification and artifacts','',
              '- Correctness suite: 21 passed; 4 legacy expected failures remain.',
              '- All stored validation thresholds and test/intervention metrics were recomputed from saved predictions.',
              '- Source, raw/processed data and all 40 checkpoint hashes match their manifests.',
              '- Matrix: 40 completed seed rows; 200 explicitly unrun rows; all 28 initial-factor pairs represented.',
              '- Checkpoints, predictions, logs, runtime details, module profiles and JSON summaries: artifacts/causal_pilot_v1/.',
              '- Authoritative research state and next experiment: ASTRA_RESEARCH_LEDGER.md.','']
    (ROOT/'TGNN_PILOT_REPORT.md').write_text('\n'.join(lines))
    audit={'verified_utc':datetime.now(timezone.utc).isoformat(),'neural_runs':len(runs),
           'feature_control_fits':len(controls),'intervention_evaluations':sum(len(r['fixed_checkpoint_controls']) for r in runs),
           'matrix_status':dict(counts),'source_hashes_unchanged':True,'prediction_metric_recomputation':'PASS',
           'checkpoint_hashes':'PASS','data_hashes':'PASS','report_sha256':sha256(ROOT/'TGNN_PILOT_REPORT.md'),
           'synthetic_script_sha256':sha256(ROOT/'temporal_toy.py'),
           'synthetic_results_sha256':sha256(ROOT/'artifacts/temporal_toy.json'),
           'training_run_manifest_utc':manifest['created_utc'],
           'last_result_file_mtime_utc':datetime.fromtimestamp(max((OUT/f"{r['run_id']}.json").stat().st_mtime for r in runs),timezone.utc).isoformat(),
           'time_note':'Per-run train/inference durations measured; final file mtime is artifact completion, not a separately instrumented end timestamp.'}
    save_json(OUT/'verification.json',audit)
    print(json.dumps(audit,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
