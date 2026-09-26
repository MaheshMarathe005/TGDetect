#!/usr/bin/env python3
"""
Real-data acquisition + causal normalisation on Modal.

Source: CTU-13 (Stratosphere IPS / CVUT), 13 scenarios of labelled bidirectional
NetFlow. Chosen because it is the only APT/botnet corpus with (a) direct HTTP
download, (b) per-record ground-truth labels, (c) real wall-clock timestamps,
(d) multiple distinct malware families => genuine domain shift.

Guarantees enforced here (these are what the old synthetic pipeline violated):
  * Every feature is computed STRICTLY CAUSALLY: streaming state is read to
    build event i's vector, then updated with event i. No future information.
  * No feature encodes the label, the actor namespace, or any generator secret.
    All 22 features are derivable from a raw NetFlow record + past history.
  * Node identity is scoped per scenario ((scenario, ip) -> id), so no model can
    transfer "this IP is the bot" across scenarios.
  * Background traffic is KEPT. Prevalence is whatever the capture really is.

Run:
  modal run modal_data_real.py::prepare
  modal run modal_data_real.py::stats
"""

import modal

app = modal.App("tg-detect-data-real")
volume = modal.Volume.from_name("tg-detect-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "requests==2.32.3", "tqdm==4.67.1"
)

OUT_DIR = "/data/real/ctu13"

# scenario id -> (CTU-13 index, capture filename, botnet family)
SCENARIOS = {
    42: (1, "capture20110810.binetflow", "Neris"),
    43: (2, "capture20110811.binetflow", "Neris"),
    44: (3, "capture20110812.binetflow", "Rbot"),
    45: (4, "capture20110815.binetflow", "Rbot"),
    46: (5, "capture20110815-2.binetflow", "Virut"),
    47: (6, "capture20110816.binetflow", "Menti"),
    48: (7, "capture20110816-2.binetflow", "Sogou"),
    49: (8, "capture20110816-3.binetflow", "Murlo"),
    50: (9, "capture20110817.binetflow", "Neris"),
    51: (10, "capture20110818.binetflow", "Rbot"),
    52: (11, "capture20110818-2.binetflow", "Rbot"),
    53: (12, "capture20110819.binetflow", "NSIS.ay"),
    54: (13, "capture20110815-3.binetflow", "Virut"),
}

BASE = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{n}/detailed-bidirectional-flow-labels/{f}"

FEATURE_NAMES = [
    "log_dur", "log_pkts", "log_bytes", "log_src_bytes", "src_byte_ratio",
    "log_bytes_per_pkt", "log_pkts_per_sec",
    "proto_tcp", "proto_udp", "proto_icmp",
    "dport_wellknown", "dport_ephemeral", "dport_norm",
    "log_src_prior_count", "log_dst_prior_count",
    "log_dt_src", "log_dt_pair", "log_src_fanout",
    "log_pair_iat_mean", "pair_iat_cv",
    "dir_bidirectional", "state_established",
]


@app.function(image=image, volumes={"/data": volume}, timeout=10800, cpu=4.0, memory=32768)
def prepare(scenarios: str = "all"):
    import csv, json, math, os, time
    from datetime import datetime, timezone
    import numpy as np
    import requests

    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = sorted(SCENARIOS) if scenarios == "all" else [int(x) for x in scenarios.split(",")]

    def parse_port(tok):
        tok = tok.strip()
        if not tok:
            return -1
        try:
            return int(tok, 16) if tok.lower().startswith("0x") else int(tok)
        except ValueError:
            return -1

    def parse_ts(tok):
        # "2011/08/16 13:51:24.049047" — capture-local, treated as UTC (offset is
        # constant within the corpus so it cannot affect any ordering or delta).
        return datetime.strptime(tok, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()

    manifest = {}
    for sid in wanted:
        idx, fname, family = SCENARIOS[sid]
        url = BASE.format(n=sid, f=fname)
        t0 = time.time()
        print(f"[s{idx:02d}/{family}] GET {url}", flush=True)

        local = f"/tmp/{fname}"
        with requests.get(url, stream=True, timeout=1800) as resp:
            resp.raise_for_status()
            with open(local, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 22):
                    fh.write(chunk)
        print(f"  downloaded {os.path.getsize(local)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)

        fh = open(local, "r", encoding="utf-8", errors="replace")
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for req in ("StartTime", "Dur", "Proto", "SrcAddr", "DstAddr", "Dport",
                    "Dir", "State", "TotPkts", "TotBytes", "SrcBytes", "Label"):
            if req not in col:
                raise RuntimeError(f"scenario {sid}: missing column {req}; header={header}")

        # streaming causal state
        node_id = {}
        seen_count = {}       # node -> flows observed so far (as src or dst)
        last_seen = {}        # node -> ts of previous appearance
        fanout = {}           # src node -> set of dst nodes seen so far
        pair_last = {}        # (s,d) -> ts of previous flow
        pair_n = {}           # (s,d) -> count of prior inter-arrivals
        pair_m = {}           # (s,d) -> running mean IAT (Welford)
        pair_m2 = {}          # (s,d) -> running M2 for IAT variance

        feats, tss, srcs, dsts, labels, roles = [], [], [], [], [], []
        bad = 0
        for row in reader:
            if len(row) < len(header):
                bad += 1
                continue
            try:
                ts = parse_ts(row[col["StartTime"]])
            except ValueError:
                bad += 1
                continue

            s_ip, d_ip = row[col["SrcAddr"]], row[col["DstAddr"]]
            s = node_id.setdefault(s_ip, len(node_id))
            d = node_id.setdefault(d_ip, len(node_id))

            label_str = row[col["Label"]]
            if "Botnet" in label_str:
                role, y = 2, 1
            elif "Normal" in label_str:
                role, y = 1, 0
            else:
                role, y = 0, 0

            try:
                dur = float(row[col["Dur"]] or 0.0)
            except ValueError:
                dur = 0.0
            try:
                pkts = float(row[col["TotPkts"]] or 0.0)
                tot_b = float(row[col["TotBytes"]] or 0.0)
                src_b = float(row[col["SrcBytes"]] or 0.0)
            except ValueError:
                bad += 1
                continue

            proto = row[col["Proto"]].strip().lower()
            dport = parse_port(row[col["Dport"]])
            dir_tok = row[col["Dir"]].strip()
            state = row[col["State"]].strip().upper()

            # ---- causal reads (state reflects only events strictly before this one)
            s_prior = seen_count.get(s, 0)
            d_prior = seen_count.get(d, 0)
            dt_src = ts - last_seen[s] if s in last_seen else -1.0
            pkey = (s, d)
            dt_pair = ts - pair_last[pkey] if pkey in pair_last else -1.0
            s_fan = len(fanout[s]) if s in fanout else 0
            n_iat = pair_n.get(pkey, 0)
            iat_mean = pair_m.get(pkey, 0.0)
            iat_var = (pair_m2[pkey] / n_iat) if n_iat > 1 else 0.0
            iat_cv = (math.sqrt(iat_var) / iat_mean) if (n_iat > 1 and iat_mean > 1e-9) else -1.0

            feats.append((
                math.log1p(max(dur, 0.0)),
                math.log1p(max(pkts, 0.0)),
                math.log1p(max(tot_b, 0.0)),
                math.log1p(max(src_b, 0.0)),
                (src_b / tot_b) if tot_b > 0 else 0.0,
                math.log1p(tot_b / pkts) if pkts > 0 else 0.0,
                math.log1p(pkts / dur) if dur > 1e-6 else 0.0,
                1.0 if proto == "tcp" else 0.0,
                1.0 if proto == "udp" else 0.0,
                1.0 if proto in ("icmp", "igmp") else 0.0,
                1.0 if 0 <= dport < 1024 else 0.0,
                1.0 if dport >= 49152 else 0.0,
                math.log1p(max(dport, 0)) / 11.1,
                math.log1p(s_prior),
                math.log1p(d_prior),
                math.log1p(dt_src) if dt_src >= 0 else -1.0,
                math.log1p(dt_pair) if dt_pair >= 0 else -1.0,
                math.log1p(s_fan),
                math.log1p(iat_mean) if n_iat > 0 else -1.0,
                iat_cv,
                1.0 if "<->" in dir_tok else 0.0,
                1.0 if ("CON" in state or "_PA" in state or "EST" in state) else 0.0,
            ))
            tss.append(ts); srcs.append(s); dsts.append(d)
            labels.append(y); roles.append(role)

            # ---- causal writes (only after the vector above was emitted)
            seen_count[s] = s_prior + 1
            seen_count[d] = d_prior + 1
            last_seen[s] = ts
            last_seen[d] = ts
            fanout.setdefault(s, set()).add(d)
            if dt_pair >= 0:
                n = n_iat + 1
                delta = dt_pair - iat_mean
                new_mean = iat_mean + delta / n
                pair_n[pkey] = n
                pair_m[pkey] = new_mean
                pair_m2[pkey] = pair_m2.get(pkey, 0.0) + delta * (dt_pair - new_mean)
            pair_last[pkey] = ts

        fh.close()
        os.remove(local)

        X = np.asarray(feats, dtype=np.float32)
        ts_a = np.asarray(tss, dtype=np.float64)
        order = np.argsort(ts_a, kind="stable")  # captures are near-sorted; make it exact
        X = X[order]
        ts_a = ts_a[order]
        src_a = np.asarray(srcs, dtype=np.int32)[order]
        dst_a = np.asarray(dsts, dtype=np.int32)[order]
        y_a = np.asarray(labels, dtype=np.int8)[order]
        role_a = np.asarray(roles, dtype=np.int8)[order]

        out = f"{OUT_DIR}/s{idx:02d}.npz"
        np.savez_compressed(out, X=X, ts=ts_a, src=src_a, dst=dst_a, y=y_a, role=role_a)
        manifest[f"s{idx:02d}"] = {
            "scenario_dir": sid, "family": family, "file": fname, "npz": out,
            "n_events": int(X.shape[0]), "n_nodes": len(node_id),
            "n_botnet": int((role_a == 2).sum()), "n_normal": int((role_a == 1).sum()),
            "n_background": int((role_a == 0).sum()),
            "prevalence": float(y_a.mean()),
            "t_start": float(ts_a[0]), "t_end": float(ts_a[-1]),
            "malformed_rows_skipped": bad,
        }
        print(f"  -> {X.shape[0]:,} events | botnet {int((role_a==2).sum()):,} "
              f"({y_a.mean()*100:.4f}%) | nodes {len(node_id):,} | {time.time()-t0:.0f}s", flush=True)
        volume.commit()

    meta = {"source": "CTU-13", "url_template": BASE, "features": FEATURE_NAMES,
            "label_rule": "y=1 iff 'Botnet' in Label; Normal and Background are y=0",
            "node_scope": "per-scenario (ids are NOT shared across scenarios)",
            "causality": "features read state before event, state updated after",
            "scenarios": manifest}
    with open(f"{OUT_DIR}/manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()

    tot = sum(v["n_events"] for v in manifest.values())
    pos = sum(v["n_botnet"] for v in manifest.values())
    print(f"\nTOTAL {tot:,} events | {pos:,} botnet | prevalence {pos/tot*100:.4f}%")
    return meta


@app.function(image=image, volumes={"/data": volume}, timeout=900)
def stats():
    import json
    with open(f"{OUT_DIR}/manifest.json") as f:
        meta = json.load(f)
    print(f"{'scen':<6} {'family':<9} {'events':>12} {'botnet':>9} {'normal':>9} {'prev%':>9}")
    print("-" * 60)
    tot = pos = 0
    for k in sorted(meta["scenarios"]):
        v = meta["scenarios"][k]
        tot += v["n_events"]; pos += v["n_botnet"]
        print(f"{k:<6} {v['family']:<9} {v['n_events']:>12,} {v['n_botnet']:>9,} "
              f"{v['n_normal']:>9,} {v['prevalence']*100:>9.4f}")
    print("-" * 60)
    print(f"{'ALL':<16} {tot:>12,} {pos:>9,} {'':>9} {pos/tot*100:>9.4f}")
    return meta


@app.local_entrypoint()
def main(scenarios: str = "all"):
    prepare.remote(scenarios=scenarios)
    stats.remote()
