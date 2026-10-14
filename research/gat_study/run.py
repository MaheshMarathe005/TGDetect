"""Real prepared provenance experiment with degree audit and full-model timing."""
import argparse
import hashlib
import json
import itertools
from pathlib import Path
import time

import numpy as np
import torch
from .model import StudyV3,ARMS
from .diagnostics import DegreeAudit
from .training import run as train
from sliding_node_mem_v3.tokens import TokenBank
from sliding_node_mem_v3.prepare import read_batches
from sliding_node_mem_v2.runtime import stream_gc
from sliding_node_mem_v2.scoring import read_edges,calibrated_scores


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def build(prepared,operator,device,checkpoint=None):
    root=Path(prepared);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['compatible']:raise ValueError('Corrected V3 preparation required')
    bank=TokenBank(root/'tokens').to_device(device)
    model=StudyV3(manifest['n_nodes'],bank,signatures=manifest['signatures'],
                  num_relations=manifest['relations'],operator=operator).to(device)
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['signature']['prepared_sha256']!=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Checkpoint prepared-data manifest mismatch')
        if saved['signature']['model']!=model.config:raise ValueError('Checkpoint model configuration mismatch')
        model.load_state_dict(saved['model'])
    return model.eval(),manifest


def audit(prepared,output,device='cpu',batches=128,batch=256):
    torch.manual_seed(0);model,_=build(prepared,'gat2',device)
    degree=DegreeAudit();handle=model.gat.register_forward_pre_hook(degree.hook)
    try:
        with torch.no_grad():
            for index,data in enumerate(read_batches(Path(prepared)/'train.bin',batch)):
                if index>=batches:break
                model.forward_batch(data);model.detach_state()
    finally:handle.remove()
    result=degree.result();result['fixture']='real chronological train prefix; seeded weights; diagnostic only'
    save(output,result);return result


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


def minority_metrics(prepared,output,result,nodes):
    manifest=json.loads((Path(prepared)/'manifest.json').read_text())
    # Same benign validation calibration/threshold as the full result.
    selection=manifest['events']['val']//3
    threshold,scores,_=calibrated_scores(read_edges(Path(output)/'val_edges.bin')[selection:],
                                       read_edges(Path(output)/'test_edges.bin'),result['signature']['false_alert_rate'])
    truth=set(json.loads((Path(prepared)/'ground_truth_nodes.json').read_text()))
    minority=set(nodes)
    if not minority or not minority<=truth:raise ValueError('Minority node IDs must be a nonempty subset of ground truth')
    observed=minority & scores.keys();detected={n for n in observed if scores[n]>threshold}
    return {'known_minority_nodes':len(minority),'scored_minority_nodes':len(observed),
            'detected_minority_nodes':len(detected),'recall_scored':len(detected)/max(len(observed),1),
            'coverage_adjusted_recall':len(detected)/len(minority),'unscored':sorted(minority-observed),
            'detected':sorted(detected),'threshold_source':'unchanged benign validation; no minority-label tuning'}


def paired_timing(prepared,root,arms,seeds,device='cuda',batch=256,repetitions=3,warmup=8,measured=64):
    """Fresh model state for each identical chronological prefix replay."""
    samples=list(itertools.islice(read_batches(Path(prepared)/'train.bin',batch),warmup+measured))
    if len(samples)<warmup+measured:raise ValueError('Need enough prefix batches for timing')
    raw=[]
    for seed in seeds:
        for rep in range(repetitions):
            order=list(arms);shift=rep%len(order);order=order[shift:]+order[:shift]
            for arm in order:
                checkpoint=Path(root)/arm/str(seed)/'best.pt'
                torch.manual_seed(seed);model,_=build(prepared,arm,device,checkpoint)
                # Lookup is the original execution optimization, identical in all arms.
                sync(device);start=time.perf_counter();cache=model.event_encoder.precompute();sync(device)
                setup=time.perf_counter()-start
                durations=[]
                if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                with stream_gc(),torch.no_grad():
                    for index,data in enumerate(samples):
                        sync(device);start=time.perf_counter()
                        model.forward_batch(data);model.detach_state();sync(device)
                        if index>=warmup:durations.append(1000*(time.perf_counter()-start))
                raw.append({'arm':arm,'seed':seed,'repetition':rep,'batch':batch,
                            'p50_ms':float(np.median(durations)),'p95_ms':float(np.quantile(durations,.95)),
                            'samples_ms':durations,'lookup_setup_seconds':setup,'lookup_bytes':cache,
                            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                            'scope':'full V3 forward, host input transfer, state update/detach; excludes raw preparation, lookup build, alert scoring and batch wait'})
                del model
                if str(device).startswith('cuda'):torch.cuda.empty_cache()
    save(Path(root)/'paired_full_model_latency.json',raw);return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--batch',type=int,default=256);parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--minority-nodes',help='JSON list of predefined ground-truth minority node IDs; required for training')
    parser.add_argument('--initialization',help='Common original V3 benign SGNS initialization artifact')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    degree=audit(args.prepared,root/'degree_audit.json',args.device,batch=args.batch)
    print(json.dumps(degree,indent=2))
    if args.audit_only:return
    if not args.minority_nodes:parser.error('--minority-nodes required: define primary subset before examining results')
    if not args.initialization:parser.error('--initialization required so every arm uses the same pretraining artifact')
    nodes=json.loads(Path(args.minority_nodes).read_text());results=[]
    truth=set(json.loads((Path(args.prepared)/'ground_truth_nodes.json').read_text()))
    if not isinstance(nodes,list) or not nodes or not set(nodes)<=truth:
        parser.error('Minority node list must be a nonempty subset of dataset ground truth')
    if not Path(args.initialization).is_file():parser.error('Common initialization artifact does not exist')
    save(root/'protocol.json',{'epochs':args.epochs,'batch':args.batch,'seeds':args.seeds,'arms':args.arms,
                             'minority_definition_sha256':hashlib.sha256(Path(args.minority_nodes).read_bytes()).hexdigest(),
                             'primary':'minority recall at unchanged benign-calibrated alert budget',
                             'history_dropout':0,'window_retention':'unchanged existing V3'})
    for seed in args.seeds:
        order=list(args.arms);shift=seed%len(order);order=order[shift:]+order[:shift]
        for arm in order:
            output=root/arm/str(seed)
            result=train(args.prepared,output,epochs=args.epochs,batch=args.batch,seed=seed,device=args.device,
                         initialization=args.initialization,resident_tokens=True,frozen_eval=True,
                         model_options={'operator':arm})
            subset=minority_metrics(args.prepared,output,result,nodes)
            save(output/'minority_metrics.json',subset)
            results.append({'arm':arm,'seed':seed,'metrics':result['metrics'],'minority':subset,
                            'checkpoint_epoch':result['checkpoint_epoch'],'parameters':result['parameter_count']})
            save(root/'summary.json',results)
    paired_timing(args.prepared,root,args.arms,args.seeds,args.device,args.batch)


if __name__=='__main__':main()
