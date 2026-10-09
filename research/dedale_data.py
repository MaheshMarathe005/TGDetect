"""Build a labeled, time-ordered DEDALE Windows-host dataset (winlogbeat/Sysmon) + causal
ego-graphs on the Modal Volume, for the GNN-vs-TGNN component study.

Design (all recorded in audit.json):
  * Modeling window = [min,max] @timestamp of the class-1 malicious client events, i.e. the
    active-attack period. Detecting the ~9.5k malicious events among same-host benign traffic.
  * Hosts = the victim clients (CLIENT1, CLIENT2) where both classes exist, so host identity
    is not discriminative (it is used only for graph connectivity, never as a feature).
  * Label = 1 iff the event matches a class-1 malicious record (external join; the label is
    NOT derived from any feature). MITRE/rule fields are absent from the raw logs.
  * Benign events in-window are reservoir-subsampled to neg_per_pos * positives; ALL positives
    are kept. Subsampling is identical for both models, so the paired comparison stays fair.
  * Chronological 60/20/20 split; every split is asserted to contain both classes.
Runs on Modal (the 27 GB winlogbeat zip lives on the Volume). No GPU, no training here.

  # Persistent deployment (recommended for long jobs):
  modal deploy dedale_data.py --name tg-detect-dedale-data

  # Launch the deployed Function asynchronously; this returns immediately.
  python dedale_data.py --spawn --neg-per-pos 15 --out-name dedale_host

  # Alternative interactive run (terminal lifetime matters unless detached):
  python -m modal run dedale_data.py --neg-per-pos 25
"""
import hashlib
import math
import modal

APP_NAME = "tg-detect-dedale-data"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("tg-detect-dedale-v1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4")
         .add_local_python_source("causal_data"))

WINLOG = "/vol/raw/system_logs_winlogbeat.zip"
LABELS = "/vol/raw/system_logs_labels.zip"
LABEL_MEMBER = "system_labels/clients_1_and_2/malicious_events_class_1.jsonl"
OUT = "/vol/prepared/dedale_host"
CKPT = "/vol/prepared/dedale_host/_prepare_ckpt.pkl"
FEAT_DIM = 72
HASH_BUCKETS = 64

# Modal resource tuning for the current run:
#   CPU:    16 cores (8 -> 16)
#   Memory: 65536 MB = 64 GiB
#
# This job is a streaming ZIP/JSON preparation pipeline, not a minibatch
# training loop. There is therefore no model batch_size to increase here.
# Keep the negative sampling ratio unchanged for experiment comparability.
MODAL_CPU = 16
MODAL_MEMORY_MB = 65536
CHECKPOINT_EVERY_MEMBERS = 24
DEFAULT_NEG_PER_POS = 15


def _host(e):
    w = e.get("winlog") or {}
    h = (e.get("agent") or {}).get("name") or (e.get("host") or {}).get("name") or w.get("computer_name") or ""
    return str(h).split(".")[0].upper()


def event_key(e):
    """Primary join key: (host, channel, record_id). Falls back to (host, ts, code, sequence)
    when record_id is absent, so labels still match if the export lacks record ids."""
    w = e.get("winlog") or {}
    ev = e.get("event") or {}
    rid = w.get("record_id")
    if rid not in (None, ""):
        return (_host(e), str(w.get("channel") or ""), "rid", str(rid))
    return (_host(e), "ts", str(e.get("@timestamp") or ""), str(ev.get("code") or w.get("event_id") or ""),
            str(ev.get("sequence") or w.get("record_id") or ""))


def _bucket(token):
    h = hashlib.sha256(token.encode("utf-8", "replace")).digest()
    return int.from_bytes(h[:8], "big") % HASH_BUCKETS, (1.0 if h[8] & 1 else -1.0)


def _hour(ts):
    try:
        return int(ts[11:13])
    except (TypeError, ValueError, IndexError):
        return 0


def features(e):
    import numpy as np
    x = np.zeros(FEAT_DIM, dtype=np.float32)
    w = e.get("winlog") or {}
    ed = w.get("event_data") or {}
    ev = e.get("event") or {}
    proc = e.get("process") or {}
    tokens = [("code", str(ev.get("code") or w.get("event_id") or "")),
              ("prov", str(ev.get("provider") or w.get("provider_name") or "")),
              ("act", str(ev.get("action") or "")), ("chan", str(w.get("channel") or "")),
              ("mod", str(ev.get("module") or "")), ("cat", ",".join(ev.get("category") or [])),
              ("img", str(ed.get("Image") or proc.get("executable") or "")),
              ("pimg", str(ed.get("ParentImage") or "")),
              ("ofn", str(ed.get("OriginalFileName") or proc.get("name") or "")),
              ("tgt", str(ed.get("TargetFilename") or ed.get("TargetObject") or "")),
              ("usr", str(ed.get("User") or (e.get("user") or {}).get("name") or ""))]
    for name, val in tokens:
        if val:
            b, s = _bucket(name + "=" + val)
            x[b] += s
    hour = _hour(e.get("@timestamp") or "")
    x[64] = np.log1p(len(str(e.get("message") or "")))
    try:
        x[65] = np.log1p(max(float(ev.get("code") or 0), 0))
    except (TypeError, ValueError):
        x[65] = 0.0
    x[66] = math.sin(2 * math.pi * hour / 24); x[67] = math.cos(2 * math.pi * hour / 24)
    x[68] = 1.0 if ed else 0.0
    x[69] = np.log1p(len(ed))
    x[70] = np.log1p(len(str(ed.get("CommandLine") or "")))
    x[71] = np.log1p(len(str(ed.get("ParentCommandLine") or "")))
    return x


def node_ids(e):
    """Two endpoints per event for the causal graph: the acting process and its host.
    Recent events sharing either node form the temporal ego-history."""
    w = e.get("winlog") or {}
    ed = w.get("event_data") or {}
    host = _host(e)
    proc = ed.get("ProcessGuid") or ed.get("ProcessId") or (e.get("process") or {}).get("pid") \
        or (e.get("event") or {}).get("provider") or "proc"
    return f"{host}|proc|{proc}", f"{host}|host"

def _epoch(ts):
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _iter_events(zf, member, must_contain=None):
    """Yield parsed JSON events from a (bz2/gz/plain) jsonl member. When must_contain is given
    (a tuple of substrings), lines lacking all of them are skipped BEFORE json.loads -- a lossless
    fast path since a host's events always carry its name verbatim, and non-host lines are dropped
    downstream by the _host filter anyway."""
    import bz2, gzip, io, json
    with zf.open(member) as fh:
        if member.endswith(".gz"):
            stream = gzip.GzipFile(fileobj=fh)
        elif member.endswith(".bz2"):
            stream = bz2.BZ2File(fh)
        else:
            stream = fh
        for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
            if must_contain and not any(tok in line for tok in must_contain):
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@app.function(image=image, cpu=MODAL_CPU, memory=MODAL_MEMORY_MB, timeout=32768, volumes={"/vol": volume})
def prepare(neg_per_pos: int = 15, hosts: str = "CLIENT1,CLIENT2", seed: int = 0,
            out_name: str = "dedale_host"):
    """neg_per_pos<=0 keeps EVERY in-window benign (the full unsubsampled dataset); those rows
    are streamed to per-member shard files so a preemption resumes from a member cursor instead
    of rescanning. neg_per_pos>0 keeps the reservoir-subsampled study build (unchanged)."""
    import json, pickle, random, zipfile
    from pathlib import Path
    import numpy as np
    from causal_data import causal_egos, fit_scaler
    volume.reload()
    host_set = {h.strip().upper() for h in hosts.split(",") if h.strip()}
    out_dir = Path(f"/vol/prepared/{out_name}"); keep_all = neg_per_pos <= 0
    ckpt_path = str(out_dir / "_prepare_ckpt.pkl"); shard_dir = out_dir / "shards"
    cursor_path = out_dir / "_full_cursor.json"

    # 1) malicious class-1 keys + active-attack window from the label file
    label_keys = set(); lab_ts = []; lab_hosts = {}
    with zipfile.ZipFile(LABELS) as z:
        for e in _iter_events(z, LABEL_MEMBER):
            label_keys.add(event_key(e)); t = _epoch(e.get("@timestamp"))
            if t is not None:
                lab_ts.append(t)
            lab_hosts[_host(e)] = lab_hosts.get(_host(e), 0) + 1
    ts_min, ts_max = min(lab_ts), max(lab_ts)

    # allowed member dates = label-window days ± 1 (filenames embed YYYY-MM-DD); the per-event
    # timestamp filter below is still authoritative, this just skips clearly out-of-window files.
    import re
    from datetime import datetime, timedelta, timezone
    lo_d = (datetime.fromtimestamp(ts_min, tz=timezone.utc) - timedelta(days=1)).date()
    hi_d = (datetime.fromtimestamp(ts_max, tz=timezone.utc) + timedelta(days=1)).date()
    datef = re.compile(r"(\d{4}-\d{2}-\d{2})")
    def in_window_member(name):
        m = datef.search(name)
        if not m:
            return True                      # unparseable name: scan it to be safe
        try:
            return lo_d <= datetime.strptime(m.group(1), "%Y-%m-%d").date() <= hi_d
        except ValueError:
            return True

    # 2) stream winlogbeat; keep every in-window positive. Subsample mode reservoir-samples
    # benign (checkpointing reservoir+RNG); keep_all mode appends EVERY benign and streams each
    # member's rows to a shard file, tracking a tiny member cursor so preemption resumes cheaply.
    cap = 0 if keep_all else neg_per_pos * len(label_keys)
    rng = random.Random(seed)
    pos, neg = [], []
    seen_neg = scanned = in_window = 0
    with zipfile.ZipFile(WINLOG) as z:
        members = [n for n in z.namelist() if not n.endswith("/") and z.getinfo(n).file_size > 0
                   and in_window_member(n)]
        sig = repr((neg_per_pos, tuple(sorted(host_set)), seed, len(label_keys), cap, len(members)))
        start_mi = 0
        if keep_all:
            shard_dir.mkdir(parents=True, exist_ok=True)
            if cursor_path.exists():
                cur = json.loads(cursor_path.read_text())
                if cur.get("sig") == sig:
                    start_mi = cur["mi"] + 1; seen_neg = cur["seen_neg"]
                    scanned = cur["scanned"]; in_window = cur["in_window"]
                    print(f"resuming keep_all at member {start_mi}/{len(members)} "
                          f"seen_neg={seen_neg} scanned={scanned}", flush=True)
        elif Path(ckpt_path).exists():
            try:
                st = pickle.loads(Path(ckpt_path).read_bytes())
                if st.get("sig") == sig:
                    start_mi = st["mi"] + 1
                    pos = st["pos"]; neg = st["neg"]
                    seen_neg = st["seen_neg"]; scanned = st["scanned"]; in_window = st["in_window"]
                    rng.setstate(st["rng"])
                    print(f"resuming at member {start_mi}/{len(members)} pos={len(pos)} "
                          f"seen_neg={seen_neg} kept_neg={len(neg)}", flush=True)
                else:
                    print("checkpoint sig mismatch; starting full scan", flush=True)
            except Exception as ex:
                print(f"checkpoint load failed ({ex}); starting full scan", flush=True)
        print(f"scanning {len(members)} members; window {lo_d}..{hi_d}; keep_all={keep_all} "
              f"label_keys={len(label_keys)} cap={cap} start_mi={start_mi}", flush=True)
        host_tokens = tuple(host_set)
        def save_ckpt(mi):
            tmp = Path(ckpt_path + ".tmp"); tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(pickle.dumps({"sig": sig, "mi": mi, "pos": pos, "neg": neg,
                "seen_neg": seen_neg, "scanned": scanned, "in_window": in_window,
                "rng": rng.getstate()}, protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(ckpt_path); volume.commit()
        def save_cursor(mi):
            cursor_path.write_text(json.dumps({"sig": sig, "mi": mi, "seen_neg": seen_neg,
                "scanned": scanned, "in_window": in_window})); volume.commit()
        for mi, member in enumerate(members):
            if mi < start_mi:
                continue
            mrows = []
            for e in _iter_events(z, member, host_tokens):
                scanned += 1
                if _host(e) not in host_set:
                    continue
                t = _epoch(e.get("@timestamp"))
                if t is None or not (ts_min <= t <= ts_max):
                    continue
                in_window += 1
                s, d = node_ids(e); k = event_key(e)
                row = (t, s, d, features(e), "|".join(map(str, k)))
                if k in label_keys:
                    mrows.append((row, 1)) if keep_all else pos.append(row)
                else:
                    seen_neg += 1
                    if keep_all:
                        mrows.append((row, 0))
                    elif len(neg) < cap:
                        neg.append(row)
                    elif cap and rng.randrange(seen_neg) < cap:
                        neg[rng.randrange(cap)] = row
            if keep_all:
                with open(shard_dir / f"part_{mi:05d}.pkl", "wb") as fh:
                    pickle.dump(mrows, fh, protocol=pickle.HIGHEST_PROTOCOL)
            if (mi + 1) % CHECKPOINT_EVERY_MEMBERS == 0 or mi + 1 == len(members):
                print(f"  member {mi+1}/{len(members)} in_window={in_window} "
                      f"pos={len(pos)} seen_neg={seen_neg} kept_neg={len(neg)}", flush=True)
                (save_cursor if keep_all else save_ckpt)(mi)
        if keep_all:
            pos, neg = [], []
            for mi in range(len(members)):
                sp = shard_dir / f"part_{mi:05d}.pkl"
                if sp.exists():
                    with open(sp, "rb") as fh:
                        for row, lab in pickle.load(fh):
                            (pos if lab else neg).append(row)
            print(f"loaded shards: pos={len(pos)} neg={len(neg)}", flush=True)
    report = _assemble(pos, neg, label_keys, lab_hosts, ts_min, ts_max, neg_per_pos, hosts,
                       scanned, in_window, seen_neg, seed, out_dir, out_name)
    for p in (Path(ckpt_path), cursor_path):
        if p.exists():
            p.unlink()
    volume.commit()
    return report

def _assemble(pos, neg, label_keys, lab_hosts, ts_min, ts_max, neg_per_pos, hosts,
              scanned, in_window, seen_neg, seed, out_dir=None, out_name="dedale_host"):
    """Turn collected (t,src,dst,feat,ident) rows into a time-ordered, split, scaled npz +
    causal ego-graph, asserting both classes land in every chronological partition."""
    import hashlib, json
    from datetime import datetime, timezone
    from pathlib import Path
    import numpy as np
    from causal_data import causal_egos, fit_scaler
    if not pos:
        raise ValueError("no in-window positives matched label keys; verify event_key join")
    tagged = [(r, 1) for r in pos] + [(r, 0) for r in neg]
    tagged.sort(key=lambda rl: rl[0][0])                      # chronological by observation time
    rows = [r for r, _ in tagged]
    n = len(rows)
    nodes = {}
    def nid(tok):
        i = nodes.get(tok)
        if i is None:
            i = len(nodes); nodes[tok] = i
        return i
    times = np.array([r[0] for r in rows], dtype=np.float64)
    src = np.array([nid(r[1]) for r in rows], dtype=np.int64)
    dst = np.array([nid(r[2]) for r in rows], dtype=np.int64)
    x = np.stack([r[3] for r in rows]).astype(np.float32)
    ids = np.array([r[4] for r in rows])
    y = np.array([lab for _, lab in tagged], dtype=np.int64)

    ends = [int(n * 0.6), int(n * 0.8)]                       # keep equal-timestamp events together
    for k in range(2):
        while ends[k] < n and times[ends[k]] == times[ends[k] - 1]:
            ends[k] += 1
    train_end, val_end = ends
    if not 0 < train_end < val_end < n:
        raise ValueError(f"bad split boundaries {train_end},{val_end} for n={n}")
    for name, lo, hi in [("train", 0, train_end), ("validation", train_end, val_end), ("test", val_end, n)]:
        if len(np.unique(y[lo:hi])) != 2:
            raise ValueError(f"{name} split lacks both classes; widen window or raise neg_per_pos")

    mean, std = fit_scaler(x, train_end)
    x = ((x - mean) / std).astype(np.float32)
    ego, valid, ages = causal_egos(src, dst, times, 4)
    assert np.isfinite(x).all() and np.isfinite(ages).all()
    out = Path(out_dir) if out_dir is not None else Path(OUT); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "prepared.npz", x=x, y=y, ego=ego, valid=valid, ages=ages,
                        train_end=train_end, val_end=val_end, ids=ids, times=times,
                        src=src, dst=dst, n_nodes=np.int64(len(nodes)))

    def iso(t):
        return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
    def span(lo, hi):
        return {"count": int(hi - lo), "positives": int(y[lo:hi].sum()),
                "negatives": int((y[lo:hi] == 0).sum()),
                "start_time": iso(float(times[lo])), "end_time": iso(float(times[hi - 1]))}
    audit = {"dataset": f"{out_name}_winlogbeat_class1",
             "substrate": "winlogbeat/Sysmon Windows host events", "hosts": hosts,
             "host_label_distribution": lab_hosts,
             "subsampling": ("keep_all_benign" if neg_per_pos <= 0 else f"reservoir_{neg_per_pos}x"),
             "modeling_window_epoch": [ts_min, ts_max], "modeling_window_iso": [iso(ts_min), iso(ts_max)],
             "label_keys_total": len(label_keys), "positives_matched": len(pos),
             "match_rate": round(len(pos) / max(len(label_keys), 1), 4),
             "negatives_kept": len(neg), "neg_per_pos_requested": neg_per_pos,
             "neg_per_pos_effective": round(len(neg) / max(len(pos), 1), 3),
             "benign_in_window_seen": seen_neg, "events_scanned": scanned,
             "events_in_window_on_hosts": in_window, "n": n, "nodes": len(nodes),
             "feat_dim": int(x.shape[1]), "train_end": train_end, "val_end": val_end,
             "partitions": {"train": {"end": train_end, **span(0, train_end)},
                            "validation": {"end": val_end, **span(train_end, val_end)},
                            "test": {"end": n, **span(val_end, n)}},
             "split_id": "active_window_chronological_60_20_20_v1", "seed": seed, "neighbors": 4,
             "label_join": "event_key=(host,channel,record_id), fallback (host,ts,code,seq); external file, never a feature",
             "feature_note": "72-d: 64 hashed categorical buckets + 8 numeric; host identity is graph-only, not a feature",
             "training_gate": True}
    audit["prepared_sha256"] = hashlib.sha256((out / "prepared.npz").read_bytes()).hexdigest()
    (out / "audit.json").write_text(json.dumps(audit, indent=2)); volume.commit()
    return {k: audit[k] for k in ("dataset", "positives_matched", "match_rate", "negatives_kept",
            "neg_per_pos_effective", "n", "nodes", "partitions", "modeling_window_iso",
            "prepared_sha256")}


@app.local_entrypoint()
def main(neg_per_pos: int = DEFAULT_NEG_PER_POS, hosts: str = "CLIENT1,CLIENT2", seed: int = 0,
         out_name: str = "dedale_host"):
    """One-shot interactive launcher (use the deployed --spawn path for terminal-independent jobs).
    Use --neg-per-pos 0 --out-name dedale_host_full to build the full unsubsampled dataset."""
    import json
    from pathlib import Path
    report = prepare.remote(neg_per_pos=neg_per_pos, hosts=hosts, seed=seed, out_name=out_name)
    out = Path("artifacts/dedale_v1"); out.mkdir(parents=True, exist_ok=True)
    (out / f"prepare_report_{out_name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def spawn_deployed_prepare(neg_per_pos: int = DEFAULT_NEG_PER_POS,
                           hosts: str = "CLIENT1,CLIENT2", seed: int = 0,
                           out_name: str = "dedale_host") -> str:
    """Spawn the already-deployed prepare Function and return its persistent FunctionCall ID.

    Unlike `prepare.remote()` from `modal run`, this looks up the deployed Function by name and
    uses Modal's server-side FunctionCall API. The local Python process only submits the job and
    can then exit; the remote Function continues independently.
    """
    fn = modal.Function.from_name(APP_NAME, "prepare")
    call = fn.spawn(neg_per_pos=neg_per_pos, hosts=hosts, seed=seed, out_name=out_name)
    print(f"spawned deployed prepare: {APP_NAME}/prepare")
    print(f"function_call_id={call.object_id}")
    print(f"dashboard={call.get_dashboard_url()}")
    return call.object_id


if __name__ == "__main__":
    # Plain-Python launcher for a deployed Function. This process exits after submission.
    import argparse
    parser = argparse.ArgumentParser(description="Spawn the deployed DEDALE prepare job.")
    parser.add_argument("--spawn", action="store_true", help="spawn the deployed prepare Function")
    parser.add_argument("--neg-per-pos", type=int, default=DEFAULT_NEG_PER_POS)
    parser.add_argument("--hosts", default="CLIENT1,CLIENT2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", default="dedale_host")
    args = parser.parse_args()
    if not args.spawn:
        parser.error("Use --spawn for the terminal-independent deployed-job launcher.")
    spawn_deployed_prepare(args.neg_per_pos, args.hosts, args.seed, args.out_name)
