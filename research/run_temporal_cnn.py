"""Independent CUDA workers plus a measured, single-L4 experiment scheduler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from run_causal_pilot import select_threshold
from run_tgnn_components import DATA_HASH, Data, digest, scored, utc, verify_result, write_json
from temporal_cnn import CNN_ARMS, make_model, rehearsal_schedule

SOURCES = ["pomsathit_baseline.py", "causal_tgn.py", "causal_data.py", "tgnn_components.py",
           "run_causal_pilot.py", "run_tgnn_components.py", "temporal_cnn.py", "test_temporal_cnn.py",
           "run_temporal_cnn.py", "modal_temporal_cnn.py", "TEMPORAL_CNN_PROTOCOL.md"]


def setup(data_path):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Registered hardware requires L4")
    data = Data(data_path, torch.device("cuda:0"))
    data.windows = data.x[data.ego]
    return data


def forward(model, data, ids):
    return model.forward_events(data.windows[ids], data.valid[ids], data.ages[ids], None, None, None)[0]


def step(model, opt, data, live_ids, replay_ids):
    opt.zero_grad(set_to_none=True)
    live = F.cross_entropy(forward(model,data,live_ids),data.y[live_ids])
    loss = live
    if replay_ids is not None:
        loss = loss + F.cross_entropy(forward(model,data,replay_ids),data.y[replay_ids])
    loss.backward()
    opt.step()
    return live.detach()


def benchmark_worker(data_path, directory, slot):
    directory = Path(directory)
    data = setup(data_path)
    torch.manual_seed(2026+slot)
    model = make_model("TGNN_Q_A_C").cuda().train()
    opt = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.00001)
    replay = torch.arange(26,device="cuda")
    for _ in range(20):
        step(model,opt,data,slice(256,512),replay)
    torch.cuda.synchronize()
    (directory / f"{slot}.ready").write_text("ready")
    deadline = time.monotonic()+180
    while not (directory / "go").exists():
        if time.monotonic()>deadline:
            raise TimeoutError("Benchmark barrier expired")
        time.sleep(.02)
    started = time.perf_counter()
    for i in range(128):
        lo = (i%30)*256
        step(model,opt,data,slice(lo,lo+256),replay)
    torch.cuda.synchronize()
    write_json(directory / f"{slot}.json", {"seconds": time.perf_counter()-started, "steps":128,
               "peak_allocated_bytes":torch.cuda.max_memory_allocated()})


def train_worker(data_path, output_path, arm, seed):
    output = Path(output_path)
    manifest = json.loads((output / "run_manifest.json").read_text())
    data = setup(data_path)
    schedule = np.load(output / f"replay_seed{seed}.npz",allow_pickle=False)
    replay_ids = torch.from_numpy(schedule["ids"]).cuda()
    replay_sizes = schedule["sizes"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arm).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.00001)
    torch.manual_seed(seed+10000)
    losses, index, replay_visits = [],0,0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    run_id = f"{arm}_seed{seed}"
    for epoch in range(200):
        model.train()
        total = torch.zeros((),device="cuda")
        for lo in range(0,data.train_end,256):
            hi = min(lo+256,data.train_end)
            size = int(replay_sizes[index])
            ids = replay_ids[index,:size] if size else None
            total += step(model,optimizer,data,slice(lo,hi),ids)*(hi-lo)
            replay_visits += size
            index += 1
        losses.append(float(total/data.train_end))
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"Nonfinite loss in {run_id}")
        if (epoch+1)%25 == 0:
            progress = {"run_id":run_id,"epoch":epoch+1,"epochs":200,"loss":losses[-1],
                        "updated_utc":utc(),"elapsed_seconds":time.perf_counter()-start}
            write_json(output / f"{run_id}_progress.json",progress)
            print(json.dumps(progress),flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    model.eval()
    checkpoint = output / f"{run_id}.pt"
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "arm":arm,"factors":CNN_ARMS[arm],"seed":seed,"epochs":200},checkpoint)
    start = time.perf_counter()
    predictions=[]
    with torch.no_grad():
        for lo in range(0,data.n,256):
            predictions.append(forward(model,data,slice(lo,min(lo+256,data.n))).softmax(-1)[:,1].cpu().numpy())
    torch.cuda.synchronize()
    inference_seconds=time.perf_counter()-start
    probability=np.concatenate(predictions)
    val,test=slice(data.train_end,data.val_end),slice(data.val_end,data.n)
    threshold=select_threshold(data.labels[val],probability[val])
    path=output / f"{run_id}_predictions.npz"
    np.savez_compressed(path,probability=probability,y=data.labels,raw_row_id=data.row_ids)
    row={"run_id":run_id,"arm":arm,"seed":seed,"factors":list(CNN_ARMS[arm]),"status":"COMPLETE",
         "epochs":200,"batch_size":256,"epoch_losses":losses,"threshold":threshold,
         "validation":scored(data.labels[val],probability[val],threshold),
         "test":scored(data.labels[test],probability[test],threshold),
         "parameters":sum(p.numel() for p in model.parameters()),
         "train_seconds":train_seconds,"inference_seconds":inference_seconds,
         "peak_gpu_memory_bytes":peak,"training_live_visits":data.train_end*200,
         "training_replay_visits":replay_visits,"reservoir_capacity":983,"completed_utc":utc(),
         "gpu":torch.cuda.get_device_name(0),"data_sha256":DATA_HASH,"source_hashes":manifest["source_hashes"],
         "checkpoint_sha256":digest(checkpoint),"predictions_sha256":digest(path),
         "timing_note":"Concurrent workers share one L4; timings include contention."}
    verify_result(output,row,data)
    row["artifact_verification"]="PASS"
    write_json(output / f"{run_id}.json",row)
    print(json.dumps({"phase":"RESULT","run_id":run_id,"test":row["test"]}),flush=True)


def summarize(output):
    rows=[]
    groups={}
    for arm in CNN_ARMS:
        group=[]
        for seed in range(5):
            path=output / f"{arm}_seed{seed}.json"
            if path.exists():
                group.append(json.loads(path.read_text()))
        rows.extend(group)
        if not group:
            continue
        stats={"seeds":len(group),"parameters":group[0]["parameters"]}
        for split in ("validation","test"):
            stats[split]={field:{"mean":float(np.mean([r[split][field] for r in group])),
                                 "sd":float(np.std([r[split][field] for r in group],ddof=1)) if len(group)>1 else None}
                          for field in ("f1","pr_ap","precision","recall","fpr","roc_auc","accuracy","balanced_accuracy")}
        groups[arm]=stats
    indexed={(r["arm"],r["seed"]):r for r in rows}
    pairs={}
    for baseline,cnn in (("TGNN","TGNN_C"),("TGNN_Q","TGNN_Q_C"),("TGNN_Q_A","TGNN_Q_A_C"),("GNN_H","CNN_H")):
        seeds=[i for i in range(5) if (baseline,i) in indexed and (cnn,i) in indexed]
        if seeds:
            pairs[f"{cnn} minus {baseline}"]={}
            for field in ("f1","pr_ap","precision","recall","fpr"):
                values=[indexed[(cnn,i)]["test"][field]-indexed[(baseline,i)]["test"][field] for i in seeds]
                pairs[f"{cnn} minus {baseline}"][field]={"seed_differences":values,"mean":float(np.mean(values)),
                    "sd":float(np.std(values,ddof=1)) if len(values)>1 else None}
    summary={"completed_runs":len(rows),"expected_runs":40,"arms":groups,"paired_test_differences":pairs,
             "claim_limit":"Previously inspected test with 25 positives; new-capture confirmation required."}
    complete=[a for a,g in groups.items() if g["seeds"]==5]
    if complete:
        vmax=max(groups[a]["validation"]["f1"]["mean"] for a in complete)
        summary["validation_f1_maximizers"]=[a for a in complete if groups[a]["validation"]["f1"]["mean"]==vmax]
        summary["highest_observed_test_f1_exploratory"]=max(complete,key=lambda a:groups[a]["test"]["f1"]["mean"])
    write_json(output/"summary.json",summary)
    lines=["# Temporal CNN study", "",f"Verified completed runs: {len(rows)}/40. Five seeds and 200 epochs per arm.","",
           "Q = local history GRU; A = additional attention; C = causal 1D temporal CNN. All arms include rehearsal.","",
           "| Model | Seeds | F1 mean ± SD (%) | PR-AP mean ± SD (%) | Parameters |","|---|---:|---:|---:|---:|"]
    for arm,g in groups.items():
        def fmt(field):
            m=g["test"][field]
            return f"{100*m['mean']:.2f} ± {100*m['sd']:.2f}" if m["sd"] is not None else f"{100*m['mean']:.2f} (one seed)"
        lines.append(f"| {arm} | {g['seeds']} | {fmt('f1')} | {fmt('pr_ap')} | {g['parameters']} |")
    lines.extend(["",summary["claim_limit"],"", "Full metrics, thresholds, losses, prediction and checkpoint hashes are in per-run JSONs. GPU telemetry and concurrency calibration are saved separately.",""])
    (output/"TEMPORAL_CNN_REPORT.md").write_text("\n".join(lines))
    return summary


def gpu_sample():
    try:
        result=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,power.draw","--format=csv,noheader,nounits"],text=True,timeout=10)
        values=[float(s.strip()) for s in result.strip().split(",")]
        return dict(zip(("gpu_utilization_percent","memory_used_mib","memory_total_mib","power_watts"),values))
    except (subprocess.SubprocessError, ValueError):
        return {"measurement_error":True}


def run_study(data_path,output_path,commit=lambda:None):
    output=Path(output_path)
    output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parent
    env=dict(os.environ,TGNN_TEST_DEVICE="cuda:0",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    preflight=subprocess.run([sys.executable,"-m","pytest","-q","test_temporal_cnn.py"],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(preflight.stdout,flush=True)
    (output/"gpu_preflight.txt").write_text(preflight.stdout)
    if preflight.returncode:
        commit()
        raise RuntimeError("CNN GPU correctness gate failed")
    data=Data(data_path,torch.device("cpu"))
    sources={name:digest(root/name) for name in SOURCES}
    manifest_path=output/"run_manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest["source_hashes"]!=sources or manifest["data_sha256"]!=DATA_HASH:
            raise ValueError("Resume rejected: source/data mismatch")
    else:
        manifest={"created_utc":utc(),"source_hashes":sources,"data_sha256":DATA_HASH,"arms":CNN_ARMS,"seeds":list(range(5)),
                  "epochs":200,"batch_size":256,"gpu":"NVIDIA L4","torch":torch.__version__,"cuda":torch.version.cuda,
                  "python":sys.version,"gpu_preflight":"PASS","profile":"prathampansare25"}
        write_json(manifest_path,manifest)
        (output/"source").mkdir(exist_ok=True)
        for name in SOURCES:
            shutil.copy2(root/name,output/"source"/name)
    for seed in range(5):
        path=output/f"replay_seed{seed}.npz"
        if not path.exists():
            ids,sizes=rehearsal_schedule(seed=seed)
            np.savez_compressed(path,ids=ids,sizes=sizes)
    commit()
    telemetry=[]
    def sample(phase):
        row={"utc":utc(),"phase":phase,**gpu_sample()}
        telemetry.append(row)
        with (output/"gpu_telemetry.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")
    def start(arguments,log):
        handle=log.open("w")
        p=subprocess.Popen([sys.executable,str(root/"run_temporal_cnn.py"),"--data",data_path,*arguments],env=env,stdout=handle,stderr=subprocess.STDOUT)
        return p,handle
    calibration_path=output/"concurrency_calibration.json"
    if calibration_path.exists():
        calibration=json.loads(calibration_path.read_text())
    else:
        measurements=[]
        regressions=0
        for workers in (1,2,4,8,16):
            directory=Path(f"/tmp/tgnn_cnn_calibration_{workers}_{time.time_ns()}")
            directory.mkdir()
            processes=[start(["--benchmark",str(directory),"--slot",str(i)],output/f"calibration_{workers}_{i}.log") for i in range(workers)]
            deadline=time.monotonic()+240
            while len(list(directory.glob("*.ready")))<workers and time.monotonic()<deadline and all(p.poll() is None for p,_ in processes):
                sample(f"calibration_{workers}_warmup")
                time.sleep(1)
            ready=len(list(directory.glob("*.ready")))==workers
            if ready:
                (directory/"go").write_text("go")
            deadline=time.monotonic()+180
            while ready and any(p.poll() is None for p,_ in processes) and time.monotonic()<deadline:
                sample(f"calibration_{workers}_timed")
                time.sleep(1)
            for p,handle in processes:
                if p.poll() is None:
                    p.terminate()
                p.wait(timeout=30)
                handle.close()
            good=ready and all(p.returncode==0 for p,_ in processes)
            if not good:
                measurements.append({"workers":workers,"status":"FAILED","reason":"Inspect calibration worker logs; no training result used."})
                break
            records=[json.loads((directory/f"{i}.json").read_text()) for i in range(workers)]
            throughput=workers*128/max(r["seconds"] for r in records)
            measurements.append({"workers":workers,"status":"PASS","aggregate_steps_per_second":throughput,"worker_measurements":records})
            print(json.dumps({"phase":"calibration","workers":workers,"steps_per_second":throughput}),flush=True)
            prior=[m["aggregate_steps_per_second"] for m in measurements[:-1] if m["status"]=="PASS"]
            regressions=regressions+1 if prior and throughput<.95*max(prior) else 0
            if regressions>=2:
                break
        good=[m for m in measurements if m["status"]=="PASS"]
        if not good:
            raise RuntimeError("No safe GPU concurrency setting passed")
        best=max(m["aggregate_steps_per_second"] for m in good)
        selected=min(m["workers"] for m in good if m["aggregate_steps_per_second"]>=.95*best)
        calibration={"selected_workers":selected,"measurements":measurements,"criterion":"Best aggregate step throughput, within 5% choose fewer processes"}
        write_json(calibration_path,calibration)
        commit()
    selected=calibration["selected_workers"]
    print(json.dumps({"phase":"selected_concurrency","workers":selected}),flush=True)
    pending=[]
    for seed in range(5):
        # Mix anchors and new CNNs early rather than running all controls first.
        for arm in ("CNN_H","TGNN_C","TGNN_Q_C","TGNN_Q_A_C","GNN_H","TGNN","TGNN_Q","TGNN_Q_A"):
            path=output/f"{arm}_seed{seed}.json"
            if path.exists():
                row=json.loads(path.read_text())
                assert row["source_hashes"]==sources
                verify_result(output,row,data)
            else:
                pending.append((arm,seed))
    active=[]
    last_commit=time.monotonic()
    while pending or active:
        while pending and len(active)<selected:
            arm,seed=pending.pop(0)
            p,handle=start(["--output",str(output),"--arm",arm,"--seed",str(seed)],output/f"{arm}_seed{seed}.log")
            active.append((p,handle,arm,seed))
        sample("production")
        finished=[item for item in active if item[0].poll() is not None]
        for p,handle,arm,seed in finished:
            handle.close()
            if p.returncode!=0:
                for other,h,_,_ in active:
                    if other.poll() is None:
                        other.terminate()
                        other.wait(timeout=30)
                    if not h.closed:
                        h.close()
                write_json(output/"progress.json",{"phase":"FAILED","run_id":f"{arm}_seed{seed}","updated_utc":utc()})
                commit()
                raise RuntimeError(f"Training failed: {arm} seed {seed}; see durable log")
            row=json.loads((output/f"{arm}_seed{seed}.json").read_text())
            verify_result(output,row,data)
            active.remove((p,handle,arm,seed))
            print(json.dumps({"phase":"result","run_id":row["run_id"],"test":row["test"]}),flush=True)
        if finished or time.monotonic()-last_commit>=30:
            summary=summarize(output)
            write_json(output/"progress.json",{"phase":"TRAINING","completed_runs":summary["completed_runs"],"expected_runs":40,
                       "active_runs":[f"{a}_seed{s}" for _,_,a,s in active],"workers":selected,"updated_utc":utc()})
            commit()
            last_commit=time.monotonic()
        time.sleep(1)
    summary=summarize(output)
    assert summary["completed_runs"]==40
    for arm in CNN_ARMS:
        for seed in range(5):
            verify_result(output,json.loads((output/f"{arm}_seed{seed}.json").read_text()),data)
    production=[r for r in telemetry if r["phase"]=="production" and "gpu_utilization_percent" in r]
    utilization={"samples":len(production),"selected_workers":selected}
    for key in ("gpu_utilization_percent","memory_used_mib","power_watts"):
        values=[r[key] for r in production]
        utilization[key]={"mean":float(np.mean(values)),"max":float(np.max(values))} if values else None
    write_json(output/"gpu_utilization_summary.json",utilization)
    write_json(output/"verification.json",{"status":"PASS","runs":40,"source_hashes":sources,"data_sha256":DATA_HASH,
               "saved_metrics_and_checkpoints":"PASS","gpu_tests":"PASS","verified_utc":utc()})
    write_json(output/"progress.json",{"phase":"COMPLETE","completed_runs":40,"expected_runs":40,"updated_utc":utc()})
    commit()
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--output")
    parser.add_argument("--arm",choices=list(CNN_ARMS))
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--benchmark")
    parser.add_argument("--slot",type=int,default=0)
    args=parser.parse_args()
    if args.benchmark:
        benchmark_worker(args.data,args.benchmark,args.slot)
    else:
        train_worker(args.data,args.output,args.arm,args.seed)
