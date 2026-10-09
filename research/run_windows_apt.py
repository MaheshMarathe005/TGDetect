"""Windows-APT weak-label matched study. Independent workers share exactly one L4."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from windows_apt_data import digest,write_json
from windows_apt_models import make_model,ARMS
from temporal_cnn import rehearsal_schedule
from run_causal_pilot import metrics,select_threshold

LABEL_SCOPE='WEAK_LABEL_WAZUH_MITRE_PRESENCE'

def setup(bundle):
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
 meta=json.loads((Path(bundle)/'prepared/data_manifest.json').read_text())
 assert meta['label_scope']==LABEL_SCOPE and meta['user_authorized'] is True
 path=Path(bundle)/'prepared/data.npz';assert digest(path)==meta['data_sha256']
 a=np.load(path,allow_pickle=False);data={k:a[k] for k in a.files}
 for k in ['x','y','ego','valid','ages']:data[k]=torch.from_numpy(data[k]).cuda()
 data['windows']=data['x'][data['ego']];data['train_end']=int(data['train_end']);data['val_end']=int(data['val_end'])
 return data,meta

def forward(model,data,ids):return model.forward_events(data['windows'][ids],data['valid'][ids],data['ages'][ids],None,None,None)[0]
def step(model,opt,data,ids,replay):
 opt.zero_grad(set_to_none=True)
 loss=F.cross_entropy(forward(model,data,ids),data['y'][ids]);total=loss
 if replay is not None:total=total+F.cross_entropy(forward(model,data,replay),data['y'][replay])
 total.backward();opt.step();return loss.detach()

def worker(bundle,out,arm,seed,benchmark=False,slot=0):
 out=Path(out);data,meta=setup(bundle);torch.manual_seed(seed);np.random.seed(seed)
 model=make_model(arm,data['x'].shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
 if benchmark:
  replay=torch.arange(26,device='cuda')
  for _ in range(10):step(model,opt,data,slice(256,512),replay)
  torch.cuda.synchronize();(out/f'{slot}.ready').write_text('ready')
  deadline=time.monotonic()+180
  while not (out/'go').exists():
   if time.monotonic()>deadline:raise TimeoutError('benchmark barrier')
   time.sleep(.05)
  start=time.perf_counter()
  for i in range(100):step(model,opt,data,slice((i%100)*256,(i%100+1)*256),replay)
  torch.cuda.synchronize()
  write_json(out/f'{slot}.json',{'arm':arm,'steps':100,'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated()})
  return
 schedule=np.load(Path(bundle)/f'prepared/replay_seed{seed}.npz',allow_pickle=False)
 rid=torch.from_numpy(schedule['ids']).cuda();sizes=schedule['sizes']
 torch.manual_seed(seed+10000);losses=[];idx=0;visits=0
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 run_id=f'{arm}_seed{seed}'
 for epoch in range(200):
  model.train();total=torch.zeros((),device='cuda')
  for lo in range(0,data['train_end'],256):
   hi=min(lo+256,data['train_end']);n=int(sizes[idx]);replay=rid[idx,:n] if n else None
   total+=step(model,opt,data,slice(lo,hi),replay)*(hi-lo);visits+=n;idx+=1
  losses.append(float(total/data['train_end']));assert np.isfinite(losses[-1])
  if (epoch+1)%5==0:
   write_json(out/f'{run_id}_progress.json',{'run_id':run_id,'epoch':epoch+1,'epochs':200,'loss':losses[-1],'elapsed_seconds':time.perf_counter()-start,'label_scope':LABEL_SCOPE})
  if (epoch+1)%50==0:
   torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'epoch':epoch+1,'label_scope':LABEL_SCOPE},out/f'{run_id}_intermediate.pt')
 torch.cuda.synchronize();train_seconds=time.perf_counter()-start
 checkpoint=out/f'{run_id}.pt';torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'input_dim':data['x'].shape[1],'arm':arm,'seed':seed,'epochs':200,'label_scope':LABEL_SCOPE},checkpoint)
 model.eval();predictions=[];inference_start=time.perf_counter()
 with torch.no_grad():
  for lo in range(0,len(data['y']),256):predictions.append(forward(model,data,slice(lo,lo+256)).softmax(-1)[:,1].cpu().numpy())
 p=np.concatenate(predictions);y=data['y'].cpu().numpy();val=slice(data['train_end'],data['val_end']);test=slice(data['val_end'],None)
 threshold=select_threshold(y[val],p[val]);path=out/f'{run_id}_predictions.npz'
 np.savez_compressed(path,probability=p,y=y,row_ids=data['row_ids'])
 row={'run_id':run_id,'arm':arm,'seed':seed,'status':'COMPLETE','label_scope':LABEL_SCOPE,'epochs':200,'batch_size':256,
  'epoch_losses':losses,'threshold':threshold,'validation':metrics(y[val],p[val],threshold),'test':metrics(y[test],p[test],threshold),
  'parameters':sum(v.numel() for v in model.parameters()),'train_seconds':train_seconds,'inference_seconds':time.perf_counter()-inference_start,
  'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'training_live_visits':data['train_end']*200,'training_replay_visits':visits,
  'reservoir_capacity':data['train_end']//10,'data_sha256':meta['data_sha256'],'bundle_manifest_sha256':digest(Path(bundle)/'bundle_manifest.json'),
  'checkpoint_sha256':digest(checkpoint),'predictions_sha256':digest(path),'gpu':torch.cuda.get_device_name(),
  'timing_note':'Concurrent workers share one L4; includes contention.'}
 verify(row,out,meta);write_json(out/f'{run_id}.json',row)

def verify(row,out,meta):
 out=Path(out);run_id=row['run_id'];a=np.load(out/f'{run_id}_predictions.npz',allow_pickle=False)
 p,y=a['probability'],a['y'];assert len(p)==meta['rows'] and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 v=meta['partitions']['validation'];t=meta['partitions']['test'];val=slice(v['start'],v['end']);test=slice(t['start'],t['end'])
 assert select_threshold(y[val],p[val])==row['threshold']
 for split,sl in [('validation',val),('test',test)]:assert metrics(y[sl],p[sl],row['threshold'])==row[split]
 assert digest(out/f'{run_id}.pt')==row['checkpoint_sha256']
 assert digest(out/f'{run_id}_predictions.npz')==row['predictions_sha256']
 assert row['epochs']==200 and len(row['epoch_losses'])==200 and np.isfinite(row['epoch_losses']).all()
 ck=torch.load(out/f'{run_id}.pt',map_location='cpu',weights_only=True)
 model=make_model(row['arm'],meta['input_dim']);model.load_state_dict(ck['state_dict'],strict=True)
 assert all(torch.isfinite(x).all() for x in ck['state_dict'].values())
 assert row['data_sha256']==meta['data_sha256']
 return True

def summarize(out):
 out=Path(out);rows=[]
 for arm in ARMS:
  for seed in range(5):
   p=out/f'{arm}_seed{seed}.json'
   if p.exists():rows.append(json.loads(p.read_text()))
 summary={'label_scope':LABEL_SCOPE,'completed_runs':len(rows),'expected_runs':10,'arms':{}}
 for arm in ARMS:
  group=[r for r in rows if r['arm']==arm]
  if group:
   summary['arms'][arm]={'seeds':len(group),'parameters':group[0]['parameters'],'test':{k:{'mean':float(np.mean([r['test'][k] for r in group])),'sd':float(np.std([r['test'][k] for r in group],ddof=1)) if len(group)>1 else None} for k in ['f1','pr_ap','roc_auc','precision','recall','fpr']}}
 if len(rows)==10:
  indexed={(r['arm'],r['seed']):r for r in rows}
  summary['paired_differences']={}
  for metric in ['f1','pr_ap','roc_auc','precision','recall','fpr']:
   values=[indexed[(ARMS[1],s)]['test'][metric]-indexed[(ARMS[0],s)]['test'][metric] for s in range(5)]
   summary['paired_differences'][metric]={'values':values,'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))}
 write_json(out/'summary.json',summary);return summary

def orchestrate(bundle,out,commit):
 bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 assert json.loads((bundle/'remote_verification.json').read_text())['status']=='PASS'
 meta=json.loads((bundle/'prepared/data_manifest.json').read_text())
 manifest_hash=digest(bundle/'bundle_manifest.json');mp=out/'run_manifest.json'
 if mp.exists():assert json.loads(mp.read_text())['bundle_manifest_sha256']==manifest_hash
 else:write_json(mp,{'bundle_manifest_sha256':manifest_hash,'label_scope':LABEL_SCOPE,'arms':ARMS,'seeds':list(range(5)),'epochs':200,'gpu':'L4'})
 env=dict(os.environ,TGNN_TEST_DEVICE='cuda:0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 tested=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','test_windows_apt.py','test_temporal_cnn.py'],cwd=bundle/'source',env=env,text=True,capture_output=True)
 (out/'gpu_tests.txt').write_text(tested.stdout+tested.stderr)
 if tested.returncode:raise RuntimeError('CUDA correctness tests failed')
 commit()
 def sample(phase):
  query=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,capture_output=True)
  try:
   util,mem,power=map(float,query.stdout.strip().split(','))
   with (out/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':phase,'utilization':util,'memory_mib':mem,'power_watts':power})+'\n')
  except ValueError:pass
 def start(arm,seed,directory,bench=False,slot=0):
  args=[sys.executable,str(bundle/'source/run_windows_apt.py'),'--bundle',str(bundle),'--out',str(directory),'--arm',arm,'--seed',str(seed)]
  if bench:args+=['--benchmark','--slot',str(slot)]
  log=(directory/f'{arm}_{seed}_{slot}.log').open('w')
  return subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT),log
 measurements=[]
 single=out/('calibration_single_'+str(time.time_ns()));single.mkdir()
 (single/'go').write_text('go')
 records=[]
 for slot,arm in enumerate(ARMS):
  process,handle=start(arm,slot,single,True,slot)
  deadline=time.monotonic()+240
  while process.poll() is None:
   if time.monotonic()>deadline:
    process.terminate();process.wait();handle.close();raise TimeoutError('single calibration')
   sample('calibration');time.sleep(1)
  handle.close();assert process.returncode==0
  records.append(json.loads((single/f'{slot}.json').read_text()))
 measurements.append({'workers':1,'aggregate_steps_per_second':200/sum(r['seconds'] for r in records),'records':records})
 for count in [2,4,8,10]:
  directory=out/f'calibration_{count}';directory.mkdir(exist_ok=True)
  # A fresh directory avoids stale readiness/results on retries.
  directory=directory/str(time.time_ns());directory.mkdir()
  active=[start(ARMS[i%2],i,directory,True,i) for i in range(count)]
  deadline=time.monotonic()+240
  while len(list(directory.glob('*.ready')))<count:
   if any(p.poll() is not None for p,_ in active) or time.monotonic()>deadline:
    for p,h in active:
     if p.poll() is None:p.terminate()
     p.wait();h.close()
    raise RuntimeError('Calibration readiness failed')
   sample('calibration');time.sleep(1)
  (directory/'go').write_text('go')
  while any(p.poll() is None for p,_ in active):
   if time.monotonic()>deadline:raise TimeoutError('calibration')
   sample('calibration');time.sleep(1)
  for p,h in active:
   h.close();assert p.returncode==0
  records=[json.loads((directory/f'{i}.json').read_text()) for i in range(count)]
  throughput=count*100/max(r['seconds'] for r in records)
  measurements.append({'workers':count,'aggregate_steps_per_second':throughput,'records':records})
  write_json(out/'calibration.json',{'measurements':measurements});commit()
 best=max(m['aggregate_steps_per_second'] for m in measurements)
 selected=min(m['workers'] for m in measurements if m['aggregate_steps_per_second']>=.95*best)
 write_json(out/'calibration.json',{'selected_workers':selected,'measurements':measurements,'note':'Balanced GNN/hybrid workloads; single-worker point measures both sequentially.'})
 write_json(out/'gates.json',{'status':'PASS','remote_data':'PASS','cuda_tests':'PASS','calibration':'PASS','selected_workers':selected})
 commit();pending=[]
 for seed in range(5):
  for arm in ARMS:
   path=out/f'{arm}_seed{seed}.json'
   if path.exists():
    row=json.loads(path.read_text());assert row['bundle_manifest_sha256']==manifest_hash;verify(row,out,meta)
   else:pending.append((arm,seed))
 active=[];last_commit=time.monotonic()
 try:
  while pending or active:
   while pending and len(active)<selected:
    arm,seed=pending.pop(0);p,h=start(arm,seed,out);active.append((p,h,arm,seed))
   sample('training')
   done=[v for v in active if v[0].poll() is not None]
   for p,h,arm,seed in done:
    h.close()
    if p.returncode:raise RuntimeError(f'Training failed {arm} seed{seed}')
    active.remove((p,h,arm,seed))
   if done or time.monotonic()-last_commit>30:
    summary=summarize(out);write_json(out/'progress.json',{'phase':'TRAINING','completed_runs':summary['completed_runs'],'active':[f'{a}_seed{s}' for _,_,a,s in active],'label_scope':LABEL_SCOPE,'time':time.time()});commit();last_commit=time.monotonic()
   time.sleep(1)
  summary=summarize(out);assert summary['completed_runs']==10
  for arm in ARMS:
   for seed in range(5):verify(json.loads((out/f'{arm}_seed{seed}.json').read_text()),out,meta)
  write_json(out/'verification.json',{'status':'PASS','runs':10,'label_scope':LABEL_SCOPE})
  write_json(out/'progress.json',{'phase':'COMPLETE','completed_runs':10,'label_scope':LABEL_SCOPE});commit();return summary
 except BaseException as exc:
  for p,h,_,_ in active:
   if p.poll() is None:p.terminate()
   p.wait();h.close()
  write_json(out/'failure.json',{'error':repr(exc),'label_scope':LABEL_SCOPE});commit();raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--benchmark',action='store_true');p.add_argument('--slot',type=int,default=0);a=p.parse_args()
 worker(a.bundle,a.out,a.arm,a.seed,a.benchmark,a.slot)
