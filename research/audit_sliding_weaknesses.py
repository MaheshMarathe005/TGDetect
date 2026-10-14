"""Reproduce structural weaknesses; synthetic probes are not accuracy results.

Run: PYTHONPATH=research python research/audit_sliding_weaknesses.py
"""
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem
from sliding_node_mem_upgrade.scoring import NodeEvidence
from sliding_node_mem_upgrade.windows import plan_windows


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    findings = {"environment": {"torch": torch.__version__, "device": "cpu"},
                "scope": "Structural synthetic probes, not trained-checkpoint detection results"}
    model = TGNNSlidingNodeMem(8, num_relations=3, buckets=256).eval()
    event = {"attributes": {"src": {"type": "process", "path": "/bin/a"},
                             "dst": {"type": "file", "path": "/tmp/x"}}}

    # A detached first write cannot receive credit from a later batch's loss.
    logits, _ = model.forward_events([event], [0], [1], [0.], [0])
    F.cross_entropy(logits, torch.tensor([0])).backward()
    model.detach_state()
    model.zero_grad(set_to_none=True)
    logits, _ = model.forward_events([event], [0], [1], [1.], [1])
    F.cross_entropy(logits, torch.tensor([1])).backward()
    def grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
    findings["cross_batch_task_credit"] = {
        "write_src_gradient_l1": grad_norm(model.write_src),
        "write_dst_gradient_l1": grad_norm(model.write_dst),
        "admission_gradient_l1": grad_norm(model.admission),
        "read_src_gradient_l1": grad_norm(model.read_src),
        "interpretation": "Single-event batches isolate delayed credit: stored memory is read, but its producing write gets no task gradient."}
    assert grad_norm(model.write_src) == 0 and grad_norm(model.read_src) > 0

    # Both endpoints compete for a single newest-eight history budget.
    history = {0: deque(range(1, 9)), 1: deque([0])}
    planned = plan_windows(history, [0], [1], 9, 9)[0].tolist()
    findings["endpoint_starvation"] = {"chosen_history": planned, "quiet_endpoint_event_kept": 0 in planned}
    assert 0 not in planned

    # A fixed top-half per query does not impose a half-size lifetime budget.
    model = TGNNSlidingNodeMem(8, num_relations=3, buckets=256,
                               rare_count=0, novelty_threshold=3).eval()
    with torch.no_grad():
        model.importance_query.weight.zero_()
        _, aux = model.forward_events([event]*64, [0]*64, [1]*64,
                                      list(map(float, range(64))), [0]*64)
    retained = sorted({i for q in model.short.values() for i, _, _ in q})
    findings["sticky_selection"] = {
        "exceptional_events": int(aux["rare"].sum()),
        "per_full_window_selected": int(aux["selected"][-1].sum()),
        "last_eight_evicted_ids": list(range(48, 56)),
        "retained_evicted_ids": retained,
        "interpretation": "With tied attention and exceptions disabled, all last eight evictions survive, despite selecting five of nine slots per query."}
    assert retained == list(range(48, 56))

    # Tanh compression cannot reproduce target coordinates outside [-1,1].
    with torch.no_grad():
        target = model.event_encoder([dict(event, type="0")])
    findings["compression_target_range"] = {
        "target_max_abs": float(target.abs().max()),
        "fraction_outside_tanh_range": float((target.abs() > 1).float().mean()),
        "minimum_coordinatewise_mse_at_fixed_target": float((target-target.clamp(-1, 1)).square().mean())}
    assert findings["compression_target_range"]["minimum_coordinatewise_mse_at_fixed_target"] > 0

    # Linear positional weights still identify distinct token permutations.
    with torch.no_grad():
        order_tokens = model.event_encoder([
            {"attributes": {"cmd_line": "a b b a"}},
            {"attributes": {"cmd_line": "b a a b"}}])
    findings["ordered_pooling_collision"] = {
        "commands": ["a b b a", "b a a b"],
        "max_embedding_difference": float((order_tokens[0]-order_tokens[1]).abs().max()),
        "interpretation": "Each token's positional weights have identical sums; linear weighted pooling cannot encode these different orders."}
    assert torch.allclose(order_tokens[0], order_tokens[1], atol=1e-6, rtol=1e-6)

    # Benign later events can erase an earlier alert through the degree penalty.
    evidence = NodeEvidence()
    evidence.update([0], [1], [12.])
    before = evidence.scores()[0]
    evidence.update([0]*63, [1]*63, [0.]*63)
    after = evidence.scores()[0]
    findings["degree_penalty_alert_erasure"] = {"score_before": before, "score_after_63_zero_score_events": after}
    assert before == 4 and after == 0

    # One cold entity keeps a single token but pins its entire batch allocation.
    model = TGNNSlidingNodeMem(8, num_relations=3, buckets=256).eval()
    with torch.no_grad():
        model.forward_events([event]*64, [0]+[2]*63, [1]+[3]*63,
                             list(map(float, range(64))), [0]*64)
        model.detach_state()
        model.forward_events([event]*64, [2]*64, [3]*64,
                             list(map(float, range(64, 128))), [0]*64)
        model.detach_state()
    cold = model.records[0].compressed
    refs = {i:r.compressed for i,r in model.records.items()}
    for queue in model.short.values():
        for i, _, ref in queue: refs[i] = ref
    same_block = [i for i,r in refs.items() if r.tensor.data_ptr() == cold.tensor.data_ptr()]
    findings["batch_storage_pinning"] = {
        "live_rows_in_cold_block": len(same_block), "allocated_rows": cold.tensor.shape[0],
        "storage_bytes": cold.tensor.untyped_storage().nbytes(),
        "necessary_live_row_bytes": len(same_block)*64*4}
    assert len(same_block) == 1 and cold.tensor.shape[0] == 64
    findings["memory_floor_half_lives_events"] = {
        "fast": math.log(.5)/math.log(1-.05*.5),
        "slow": math.log(.5)/math.log(1-.05*.005)}

    results = {}
    for arm in ("ORTHRUS_Reference", "TGNN_Sliding_Node_Mem_Upgrade", "TGNN_Sliding_ORTHRUS_Encoder"):
        path = root/"artifacts/three_arm_five_epochs"/(arm+".json")
        data = json.loads(path.read_text())
        mismatches = [p for p,h in data["signature"]["source_hashes"].items()
                      if not (root/"research"/p).exists() or hashlib.sha256((root/"research"/p).read_bytes()).hexdigest() != h]
        results[arm] = {k:data[k] for k in ("training_seconds", "evaluation_seconds", "cuda_peak_allocated_bytes",
                                          "released_top30_kmeans", "threshold_only", "unseen_ground_truth")}
        results[arm]["source_mismatches"] = mismatches
        if "calibrated" in data: results[arm]["calibrated"] = data["calibrated"]
    findings["saved_five_epoch_results"] = results
    counts = results["ORTHRUS_Reference"]["threshold_only"]
    findings["all_benign_accuracy"] = (counts["tn"]+counts["fp"])/sum(counts[k] for k in ("tn","fp","fn","tp"))
    out = root/"artifacts/sliding_model_audit/probes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2)+"\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
