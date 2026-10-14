"""Smoke-test and time the isolated sliding upgrade on the authorized Modal profile."""
from pathlib import Path
import modal
from modal_sliding_node_mem import image as base_image

app = modal.App('tg-detect-sliding-upgrade-smoke')
volume = modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1')
root = Path(__file__).parent
image = (base_image.add_local_dir(root/'sliding_node_mem_upgrade', '/root/research/sliding_node_mem_upgrade', ignore=['__pycache__'])
         .add_local_file(root/'test_sliding_upgrade.py', '/root/research/test_sliding_upgrade.py'))

@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=3600, volumes={'/study':volume})
def smoke():
    import json, subprocess, sys
    import torch
    from sliding_node_mem_upgrade.smoke import benchmark
    volume.reload()
    tests = subprocess.run([sys.executable,'-m','pytest','-q','/root/research/test_sliding_upgrade.py'],capture_output=True,text=True)
    if tests.returncode: raise RuntimeError(tests.stdout + tests.stderr)
    print(tests.stdout,flush=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    result = benchmark('/study/prepared/manifest.json', '/study/results/TGNN_Sliding_Node_Mem/resume.pt', 'cuda')
    result['tests']=tests.stdout.strip()
    dest=Path('/study/upgrade_smoke');dest.mkdir(exist_ok=True)
    (dest/'result.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.local_entrypoint()
def main():
    import json
    result=smoke.remote()
    dest=root/'artifacts/sliding_node_mem_upgrade';dest.mkdir(exist_ok=True)
    (dest/'smoke_gpu.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
