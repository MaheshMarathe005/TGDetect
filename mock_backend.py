"""
TGDetect UI-only mock backend.

Purpose:
- Run the existing Next.js frontend without downloading the CADETS bundle.
- Serve a realistic dummy detection graph for UI/UX work.
- No checkpoint, SciPy, scikit-learn, torch-geometric, or real dataset required.

Run from the TGDetect repository root:
    python -m uvicorn mock_backend:app --host 127.0.0.1 --port 8000 --reload
"""

from collections import deque
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

app = FastAPI(title="TGDetect Mock UI Backend", version="0.1.0")

THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Dummy entities
# Shape matches dashboard/frontend/src/components/cadets/DetectionGraph.tsx
# ---------------------------------------------------------------------------

NODES = [
    {
        "id": 1, "score": 0.94, "alert": True, "benchmark_member": None,
        "ip": None, "prediction": "malicious",
        "attributes": {
            "type": "subject", "name": "powershell.exe",
            "cmd_line": "powershell.exe -enc SQBFAFgA...",
            "host": "FIN-WS-07"
        }
    },
    {
        "id": 2, "score": 0.91, "alert": True, "benchmark_member": None,
        "ip": "185.220.101.45", "prediction": "malicious",
        "attributes": {
            "type": "netflow", "name": "185.220.101.45",
            "remote_ip": "185.220.101.45", "host": "external"
        }
    },
    {
        "id": 3, "score": 0.88, "alert": True, "benchmark_member": None,
        "ip": None, "prediction": "malicious",
        "attributes": {
            "type": "file", "name": "stage.ps1",
            "path": "C:\\Users\\Public\\stage.ps1", "host": "FIN-WS-07"
        }
    },
    {
        "id": 4, "score": 0.84, "alert": True, "benchmark_member": None,
        "ip": None, "prediction": "malicious",
        "attributes": {
            "type": "subject", "name": "rundll32.exe",
            "cmd_line": "rundll32.exe C:\\ProgramData\\cache.dll,Start",
            "host": "FIN-WS-07"
        }
    },
    {
        "id": 5, "score": 0.79, "alert": True, "benchmark_member": None,
        "ip": None, "prediction": "malicious",
        "attributes": {
            "type": "file", "name": "cache.dll",
            "path": "C:\\ProgramData\\cache.dll", "host": "FIN-WS-07"
        }
    },
    {
        "id": 6, "score": 0.73, "alert": True, "benchmark_member": None,
        "ip": "45.77.54.21", "prediction": "malicious",
        "attributes": {
            "type": "netflow", "name": "45.77.54.21",
            "remote_ip": "45.77.54.21", "host": "external"
        }
    },
    {
        "id": 7, "score": 0.69, "alert": True, "benchmark_member": None,
        "ip": None, "prediction": "malicious",
        "attributes": {
            "type": "subject", "name": "cmd.exe",
            "cmd_line": "cmd.exe /c whoami && ipconfig /all",
            "host": "FIN-WS-07"
        }
    },
    {
        "id": 8, "score": 0.63, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "host", "name": "FIN-WS-07",
            "host": "FIN-WS-07", "hostname": "FIN-WS-07"
        }
    },
    {
        "id": 9, "score": 0.56, "alert": False, "benchmark_member": None,
        "ip": "10.10.20.14", "prediction": "benign",
        "attributes": {
            "type": "netflow", "name": "10.10.20.14",
            "remote_ip": "10.10.20.14", "host": "FIN-WS-07"
        }
    },
    {
        "id": 10, "score": 0.48, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "subject", "name": "chrome.exe",
            "cmd_line": "chrome.exe --type=renderer",
            "host": "FIN-WS-07"
        }
    },
    {
        "id": 11, "score": 0.42, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "file", "name": "Quarterly_Report.xlsx",
            "path": "C:\\Users\\analyst\\Documents\\Quarterly_Report.xlsx",
            "host": "FIN-WS-07"
        }
    },
    {
        "id": 12, "score": 0.36, "alert": False, "benchmark_member": None,
        "ip": "142.250.183.110", "prediction": "benign",
        "attributes": {
            "type": "netflow", "name": "142.250.183.110",
            "remote_ip": "142.250.183.110", "host": "external"
        }
    },
    {
        "id": 13, "score": 0.31, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "subject", "name": "explorer.exe",
            "cmd_line": "C:\\Windows\\explorer.exe",
            "host": "FIN-WS-07"
        }
    },
    {
        "id": 14, "score": 0.27, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "file", "name": "ntdll.dll",
            "path": "C:\\Windows\\System32\\ntdll.dll", "host": "FIN-WS-07"
        }
    },
    {
        "id": 15, "score": 0.22, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "host", "name": "DC-01",
            "host": "DC-01", "hostname": "DC-01"
        }
    },
    {
        "id": 16, "score": 0.18, "alert": False, "benchmark_member": None,
        "ip": "10.10.10.5", "prediction": "benign",
        "attributes": {
            "type": "netflow", "name": "10.10.10.5",
            "remote_ip": "10.10.10.5", "host": "DC-01"
        }
    },
    {
        "id": 17, "score": 0.15, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "subject", "name": "svchost.exe",
            "cmd_line": "C:\\Windows\\System32\\svchost.exe -k netsvcs",
            "host": "DC-01"
        }
    },
    {
        "id": 18, "score": 0.12, "alert": False, "benchmark_member": None,
        "ip": None, "prediction": "benign",
        "attributes": {
            "type": "file", "name": "policy.dat",
            "path": "C:\\Windows\\System32\\GroupPolicy\\policy.dat",
            "host": "DC-01"
        }
    },
]

BASE_TIME = 1713484800.0  # illustrative timestamp

_raw_edges = [
    (1, 3, "WRITE", 0.96),
    (1, 2, "CONNECT", 0.95),
    (3, 4, "EXECUTE", 0.92),
    (4, 5, "READ", 0.89),
    (4, 6, "CONNECT", 0.87),
    (4, 7, "EXECUTE", 0.83),
    (7, 8, "OPEN", 0.76),
    (7, 9, "SENDTO", 0.73),
    (8, 15, "CONNECT", 0.61),
    (8, 16, "SENDMSG", 0.55),
    (10, 12, "CONNECT", 0.34),
    (10, 11, "READ", 0.28),
    (13, 11, "OPEN", 0.24),
    (13, 14, "READ", 0.20),
    (15, 17, "EXECUTE", 0.19),
    (17, 18, "READ", 0.16),
    (9, 15, "CONNECT", 0.52),
    (1, 8, "EXECUTE", 0.78),
    (5, 6, "SENDTO", 0.81),
    (2, 1, "RECVFROM", 0.90),
    (6, 4, "RECVMSG", 0.82),
    (8, 13, "EXECUTE", 0.32),
    (15, 16, "SENDTO", 0.18),
    (17, 15, "OPEN", 0.14),
]

EVENTS = [
    {
        "id": f"mock:{i}",
        "split": "mock",
        "idx": i,
        "src": src,
        "dst": dst,
        "time": BASE_TIME + i * 42,
        "relation": 0,
        "relation_name": relation,
        "loss": evidence,
        "evidence": evidence,
    }
    for i, (src, dst, relation, evidence) in enumerate(_raw_edges)
]

DATASET = {
    "id": "cadets",
    "name": "Mock CADETS UI dataset",
    "status": "ready",
    "stage": "UI fixture",
    "progress": 1,
    "events": len(EVENTS),
    "nodes": len(NODES),
    "source": "Embedded mock data",
    "scoring": "Illustrative scores only — not model output",
}

def node_by_id(node_id: int):
    return next((n for n in NODES if n["id"] == node_id), None)

def node_label(node):
    a = node["attributes"]
    return str(
        node.get("ip")
        or a.get("cmd_line")
        or a.get("path")
        or a.get("host")
        or a.get("hostname")
        or a.get("name")
        or f"entity #{node['id']}"
    )

def field_value(node, kind: str):
    a = node["attributes"]
    if kind == "ip":
        return node.get("ip") or a.get("remote_ip")
    if kind == "host":
        return a.get("host") or a.get("hostname")
    if kind == "process":
        return (a.get("cmd_line") or a.get("name")) if a.get("type") in ("subject", "process") else None
    if kind == "file":
        return (a.get("path") or a.get("name")) if a.get("type") == "file" else None
    return a.get("name") or node_label(node)

def prediction_match(node, prediction: str):
    p = (prediction or "all").lower()
    if p in ("all", ""):
        return True
    if p in ("malicious", "alert", "alerts", "predicted_malicious"):
        return bool(node["alert"])
    if p in ("benign", "normal", "predicted_benign"):
        return not bool(node["alert"])
    return True

def bfs_subset(seed_ids, hops: int):
    if not seed_ids:
        return {n["id"] for n in NODES}
    neighbors = {}
    for e in EVENTS:
        neighbors.setdefault(e["src"], set()).add(e["dst"])
        neighbors.setdefault(e["dst"], set()).add(e["src"])
    seen = set(seed_ids)
    q = deque((sid, 0) for sid in seed_ids)
    while q:
        current, depth = q.popleft()
        if depth >= hops:
            continue
        for nxt in neighbors.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))
    return seen

# ---------------------------------------------------------------------------
# General / status
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "mock-ui", "dataset": DATASET["name"]}

# ---------------------------------------------------------------------------
# Detection Graph API
# ---------------------------------------------------------------------------

@app.get("/api/graph-datasets")
def graph_datasets():
    return {"items": [DATASET]}

@app.get("/api/graph-datasets/{dataset}")
def graph_dataset_status(dataset: str):
    if dataset != "cadets":
        raise HTTPException(404, "Mock dataset not found")
    return DATASET

@app.get("/api/graph-datasets/{dataset}/entities")
def graph_entities(
    dataset: str,
    kind: str = Query("ip"),
    q: str = Query(""),
    prediction: str = Query("all"),
):
    if dataset != "cadets":
        raise HTTPException(404, "Mock dataset not found")

    available = ["ip", "host", "process", "file", "entity"]
    if kind not in available:
        kind = "entity"

    query = q.strip().lower()
    counts = {}
    for n in NODES:
        if not prediction_match(n, prediction):
            continue
        value = field_value(n, kind)
        if not value:
            continue
        value = str(value)
        if query and query not in value.lower():
            continue
        record = counts.setdefault(value, {"label": value, "count": 0, "alert": 0})
        record["count"] += 1
        record["alert"] += int(n["alert"])

    items = sorted(counts.values(), key=lambda x: (-x["alert"], -x["count"], x["label"]))[:80]
    return {
        "items": items,
        "available_kinds": available,
        "truncated": len(counts) > 80,
    }

@app.get("/api/graph-datasets/{dataset}/graph")
def graph(
    dataset: str,
    kind: str = Query("ip"),
    value: str = Query(""),
    prediction: str = Query("all"),
    limit: int = Query(24, ge=1, le=100),
    hops: int = Query(2, ge=0, le=6),
    node: int | None = Query(None),
):
    if dataset != "cadets":
        raise HTTPException(404, "Mock dataset not found")

    seed_ids = set()
    if node is not None and node_by_id(node):
        seed_ids.add(node)

    if value:
        target = value.lower()
        for n in NODES:
            v = field_value(n, kind)
            if v is not None and target in str(v).lower():
                seed_ids.add(n["id"])

    allowed_ids = bfs_subset(seed_ids, hops)

    filtered = [
        n for n in NODES
        if n["id"] in allowed_ids and prediction_match(n, prediction)
    ]

    # Keep the requested anchor even if a prediction filter would otherwise hide it.
    if node is not None:
        anchor = node_by_id(node)
        if anchor and all(n["id"] != node for n in filtered):
            filtered.append(anchor)

    filtered = sorted(filtered, key=lambda n: (-float(n["score"]), n["id"]))[:limit]
    ids = {n["id"] for n in filtered}
    events = [e for e in EVENTS if e["src"] in ids and e["dst"] in ids]

    return {
        "dataset": DATASET,
        "nodes": filtered,
        "events": events,
        "counts": {
            "predicted_malicious": sum(1 for n in filtered if n["alert"]),
            "predicted_benign": sum(1 for n in filtered if not n["alert"]),
        },
        "threshold": THRESHOLD,
        "scope": "Mock UI fixture — values are illustrative, not model predictions.",
        "ip_scope": "Private and example external IPs for visualization only.",
        "truncated": len(filtered) >= limit,
        "seeds": sorted(seed_ids),
    }

@app.post("/api/graph-datasets", status_code=503)
async def mock_upload_disabled(request: Request):
    raise HTTPException(
        503,
        "Upload scoring is disabled in mock UI mode. Use the built-in Mock CADETS UI dataset."
    )

# ---------------------------------------------------------------------------
# Minimal CADETS endpoints so the initial Command Center does not show HTTP 500
# ---------------------------------------------------------------------------

@app.get("/api/cadets/summary")
def cadets_summary():
    return {
        "dataset": "Mock CADETS UI dataset",
        "model": "TGNN V3 — UI mock mode",
        "total_events": 125000,
        "events": {"train": 90000, "validation": 15000, "test": 20000},
        "scored_nodes": 1842,
        "alert_nodes": sum(1 for n in NODES if n["alert"]),
        "metrics": {
            "roc_auc": 0.891762,
            "pr_ap": 0.8421,
            "precision": 0.80,
            "recall": 0.76,
            "f1": 0.7795,
        },
        "threshold": THRESHOLD,
        "epochs": 20,
        "batch_size": 1024,
        "seed": 0,
        "parameter_count": 487522,
        "checkpoint_epoch": 18,
        "checkpoint_sha256": "mock-ui-only",
        "validation_ce": 0.41,
        "test_ce": 0.44,
        "limitations": [
            "UI mock data only.",
            "Scores and metrics are illustrative and must not be reported as experimental results.",
        ],
        "unseen_ground_truth": [],
    }

@app.get("/api/cadets/nodes")
def cadets_nodes(
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    alerts_only: bool = Query(True),
):
    rows = [n for n in NODES if n["alert"]] if alerts_only else list(NODES)
    rows = sorted(rows, key=lambda n: (-n["score"], n["id"]))
    return {"total": len(rows), "items": rows[offset:offset + limit], "offset": offset}

@app.get("/api/cadets/events")
def cadets_events(
    split: str = Query("test"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    node: int | None = Query(None),
):
    rows = EVENTS
    if node is not None:
        rows = [e for e in rows if node in (e["src"], e["dst"])]
    return {"total": len(rows), "items": rows[offset:offset + limit], "offset": offset}

@app.get("/api/cadets/events/{split}/{idx}")
def cadets_event(split: str, idx: int):
    event = next((e for e in EVENTS if e["idx"] == idx), None)
    if not event:
        raise HTTPException(404, "Mock event not found")
    src = node_by_id(event["src"])
    dst = node_by_id(event["dst"])
    return {
        **event,
        "src_attributes": dict(src["attributes"]) if src else {},
        "dst_attributes": dict(dst["attributes"]) if dst else {},
        "explanation": (
            "UI fixture: this event is illustrative and exists only to exercise "
            "the Event Explorer and investigation interface."
        ),
    }

@app.get("/api/cadets/nodes/{node_id}/explanation")
def cadets_explanation(node_id: int):
    n = node_by_id(node_id)
    if not n:
        raise HTTPException(404, "Mock node not found")
    related = [e for e in EVENTS if node_id in (e["src"], e["dst"])]
    related = sorted(related, key=lambda e: e["evidence"], reverse=True)[:8]
    return {
        "node": n,
        "window_start": related[0]["time"] if related else BASE_TIME,
        "contributors": related,
        "reconstructed_score": n["score"],
        "single_event_channel": n["score"] * 0.58,
        "repeated_event_channel": n["score"] * 0.42,
        "threshold": THRESHOLD,
        "rule": "Illustrative UI-only evidence aggregation",
        "meaning": "Dummy explanation for interface development.",
        "limitations": "Not generated by the trained TGDetect model.",
    }

@app.get("/api/cadets/nodes/{node_id}/chain")
def cadets_chain(
    node_id: int,
    hops: int = Query(2, ge=1, le=5),
    limit: int = Query(60, ge=1, le=200),
):
    if not node_by_id(node_id):
        raise HTTPException(404, "Mock node not found")
    ids = bfs_subset({node_id}, hops)
    nodes = [n for n in NODES if n["id"] in ids][:limit]
    node_ids = {n["id"] for n in nodes}
    events = [e for e in EVENTS if e["src"] in node_ids and e["dst"] in node_ids][:limit]
    return {
        "nodes": nodes,
        "events": events,
        "truncated": False,
        "interpretation": "UI-only neighborhood around the selected entity.",
        "strategy": "Mock breadth-first neighborhood",
        "labels_used": False,
    }

class ReplayRequest(BaseModel):
    split: str = "test"
    batches: int = 1

@app.post("/api/cadets/replay")
def cadets_replay(payload: ReplayRequest):
    return {
        "events": min(len(EVENTS), payload.batches * 8),
        "device": "mock",
        "setup_seconds": 0.01,
        "batch_ms": [2.4 for _ in range(payload.batches)],
        "max_absolute_error_vs_saved_gpu_scores": 0.0,
        "scope": "UI mock replay only.",
        "scores": [
            {
                "idx": e["idx"],
                "relation_ce": round(e["evidence"] * 0.7, 4),
                "pair_anomaly": round(e["evidence"] * 0.3, 4),
                "combined_loss": round(e["evidence"], 4),
            }
            for e in EVENTS[: min(len(EVENTS), payload.batches * 8)]
        ],
    }

# === TGDETECT MOCK CONSTRUCTION ENDPOINT ===
@app.get("/api/cadets/nodes/{node_id}/construction")
def cadets_construction(
    node_id: int,
    anchor_index: int = Query(0, ge=0),
    direction: str = Query("both"),
    max_steps: int = Query(5, ge=1, le=12),
):
    if not node_by_id(node_id):
        raise HTTPException(404, "Mock node not found")

    related = sorted(
        [e for e in EVENTS if node_id in (e["src"], e["dst"])],
        key=lambda e: (e["time"], e["idx"]),
    )
    anchor = next((e for e in EVENTS if e["idx"] == anchor_index), None)
    if anchor is None or node_id not in (anchor["src"], anchor["dst"]):
        anchor = related[0] if related else EVENTS[0]

    candidates = sorted(EVENTS, key=lambda e: (e["time"], e["idx"]))
    chosen = [anchor]
    entity_ids = {anchor["src"], anchor["dst"]}

    if direction in ("backward", "both"):
        for e in reversed([x for x in candidates if x["time"] <= anchor["time"]]):
            if e["id"] == anchor["id"]:
                continue
            if e["src"] in entity_ids or e["dst"] in entity_ids:
                chosen.insert(0, e)
                entity_ids.update((e["src"], e["dst"]))
                if len(chosen) >= max_steps:
                    break

    if len(chosen) < max_steps and direction in ("forward", "both"):
        for e in [x for x in candidates if x["time"] >= anchor["time"]]:
            if e["id"] == anchor["id"] or e in chosen:
                continue
            if e["src"] in entity_ids or e["dst"] in entity_ids:
                chosen.append(e)
                entity_ids.update((e["src"], e["dst"]))
                if len(chosen) >= max_steps:
                    break

    chosen = sorted(chosen, key=lambda e: (e["time"], e["idx"]))
    anchor_step = next(
        (i for i, e in enumerate(chosen) if e["id"] == anchor["id"]),
        0,
    )
    nodes = [n for n in NODES if n["id"] in entity_ids]

    paths = [
        {
            "id": "mock-primary-path",
            "events": chosen,
            "evidence_sum": float(sum(e.get("evidence") or 0 for e in chosen)),
            "anchor_step": anchor_step,
        }
    ]

    alternate = [
        e for e in related
        if e["id"] != anchor["id"]
    ][: max(1, min(3, max_steps - 1))]
    if alternate:
        alt = sorted([anchor, *alternate], key=lambda e: (e["time"], e["idx"]))
        paths.append(
            {
                "id": "mock-local-path",
                "events": alt,
                "evidence_sum": float(sum(e.get("evidence") or 0 for e in alt)),
                "anchor_step": next(
                    (i for i, e in enumerate(alt) if e["id"] == anchor["id"]),
                    0,
                ),
            }
        )

    return {
        "anchor": anchor,
        "paths": paths,
        "nodes": nodes,
        "search_pruned": False,
        "labels_used": False,
        "ranking": "Illustrative evidence sum for UI development only.",
        "rule": "Events share an endpoint and remain in timestamp order.",
        "interpretation": (
            "Mock candidate paths are for interface testing and are not causal "
            "claims or trained-model output."
        ),
    }
