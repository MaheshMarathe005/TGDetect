"""CPU-only evaluation of interrupted full-CADETS training checkpoints."""
from pathlib import Path
import json
import modal
app=modal.App('tg-detect-orthrus-checkpoint-evaluation')
volume=modal.Volume.from_name('tg-detect-orthrus-e3-v1')
image=modal.Image.debian_slim()
if modal.is_local():
    from modal_orthrus_study import study_image
    image=study_image.add_local_file(Path(__file__).parent/'evaluate_orthrus_checkpoints.py',
                                    '/root/research/evaluate_orthrus_checkpoints.py')

@app.function(image=image,cpu=8,memory=65536,timeout=86400,
              volumes={'/study':volume},max_containers=3)
def evaluate(arm):
    from evaluate_orthrus_checkpoints import evaluate as evaluate_weights
    volume.reload()
    out=Path('/study/checkpoint_comparison/results')
    try:
        result=evaluate_weights('/study/prepared/manifest.json',
                                f'/study/results/{arm}/resume.pt',out)
        volume.commit()
        from orthrus_comparison.tracing import run_tracing
        traced=run_tracing('/study/prepared/manifest.json',str(out/arm/'result.json'),
                           workers=8,on_checkpoint=volume.commit)
        return dict(arm=arm,epoch=result['checkpoint_epoch'],
                    detection=result['released_top30_kmeans'],tracing=traced['released_last_window'])
    finally:volume.commit()

@app.function(image=image,cpu=1,memory=2048,timeout=86400,volumes={'/study':volume})
def pipeline():
    import traceback
    root=Path('/study/checkpoint_comparison');root.mkdir(exist_ok=True)
    status=root/'status.json'
    try:
        status.write_text(json.dumps(dict(status='evaluating_saved_checkpoints',device='cpu')));volume.commit()
        arms=['ORTHRUS_Reference','TGNN_Node_Mem_V3_Reconstruction','TGNN_Node_Mem_V4_KV']
        results=list(evaluate.map(arms))
        volume.reload();status.write_text(json.dumps(dict(status='complete',results=results),indent=2));volume.commit()
        return results
    except Exception:
        status.write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2));volume.commit()
        raise
