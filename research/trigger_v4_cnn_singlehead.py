"""Trigger the deployed function once; retain its call ID and budget reservation."""
import json
import time
from pathlib import Path
import modal


def main():
    directory=Path(__file__).resolve().parent/'artifacts/sliding_v4'
    path=directory/'ledger.json';ledger=json.loads(path.read_text())
    job='V4_CNN_SingleHead_s0_e5';timeout=3600
    if job in ledger['jobs']:raise RuntimeError('Already reserved; refusing duplicate trigger')
    if any(j['status'] in ('running','reserved') for j in ledger['jobs'].values()):
        raise RuntimeError('Another study job remains active')
    reserved=sum(j.get('settled_upper_bound_seconds',j['timeout']) for j in ledger['jobs'].values())
    if reserved+timeout>ledger['gpu_timeout_limit']:raise RuntimeError('Existing GPU-hour allowance exceeded')
    ledger['jobs'][job]=dict(status='reserved',timeout=timeout,started=time.time(),
        app='tg-detect-v4-cnn-singlehead',trigger='deployed Function.from_name().spawn()')
    path.write_text(json.dumps(ledger,indent=2))
    try:
        function=modal.Function.from_name('tg-detect-v4-cnn-singlehead','train')
        call=function.spawn()
        ledger['jobs'][job].update(status='running',call_id=call.object_id)
        path.write_text(json.dumps(ledger,indent=2))
        print('Triggered deployed function:',call.object_id,flush=True)
        print('Accounted upper-bound GPU seconds:',reserved+timeout,'/',ledger['gpu_timeout_limit'],flush=True)
        result=call.get()
        (directory/(job+'.json')).write_text(json.dumps(result,indent=2))
        ledger['jobs'][job]['status']='complete'
        print('Training and comparison complete',flush=True)
    except Exception as exc:
        ledger['jobs'][job].update(status='failed_or_timed_out',error=repr(exc));raise
    finally:path.write_text(json.dumps(ledger,indent=2))


if __name__=='__main__':main()
