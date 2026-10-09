"""Fetch the DEDALE APT dataset (doi:10.57745/Y5JLDG) directly onto a Modal Volume.

DEDALE is ~3.2 TB across 40 files. The bulk is raw weekly packet captures; the
structured logs used to build/evaluate an IDS are far smaller. Download tiers:
  meta       README + labels + suricata + userbehaviour  (~0.05 GB)
  structured zeek + cicflowmeter + system beats           (~88 GB, no raw pcaps)
  raw        weekly full captures                          (~3.1 TB)
  all        everything
Focused subsets (pass as --tier): zeek (11.8 GB), zeek_min (6.9 GB),
  cicflowmeter (10.3 GB), winlogbeat (27.1 GB), hostlogs (66.3 GB). You can also
  pass a comma-separated list of exact filenames.

Files stream to /vol/raw, are md5-verified against the Dataverse manifest, and the
Volume is committed after each file so a re-run resumes and skips finished files.
Nothing trains here.

  python -m modal run modal_dedale.py --tier meta
  python -m modal run modal_dedale.py --tier zeek_min
  python -m modal run modal_dedale.py::inspect
"""
import modal

BASE = "https://entrepot.recherche.data.gouv.fr"
PID = "doi:10.57745/Y5JLDG"
app = modal.App("tg-detect-dedale")
volume = modal.Volume.from_name("tg-detect-dedale-v1", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.10")

RAW_PREFIXES = ("green_internal_week", "orange_dmz_week", "orange_servers_week",
                "router1_green_week", "router1_orange_week", "router1_red_week")
META = {"README.md", "system_logs_labels.zip", "suricata_alerts.zip", "userbehaviour_logs.zip"}


def classify(name):
    if name in META:
        return "meta"
    if any(name.startswith(p) for p in RAW_PREFIXES):
        return "raw"
    return "structured"


# Named convenience subsets layered on top of the meta/structured/raw tiers, so a
# focused pull is a one-liner. "zeek" = network flow logs whose conn.log carries
# the network labels inline (cleanest provenance, natural src->dst temporal graph).
SUBSETS = {
    "zeek":         lambda n: n.startswith("zeek_"),                       # 11.76 GB, 4 zones
    "zeek_min":     lambda n: n in {"zeek_green_internal.zip", "zeek_orange_servers.zip"},  # 6.95 GB
    "cicflowmeter": lambda n: n.startswith("cicflowmeter_"),              # 10.33 GB
    "winlogbeat":   lambda n: n == "system_logs_winlogbeat.zip",         # 27.12 GB Windows/Sysmon
    "hostlogs":     lambda n: n.startswith("system_logs_") and n not in META,  # 66.35 GB all beats
}


def select(manifest, tier):
    """Resolve `tier` into {filename: meta}. Accepts a tier (meta/structured/raw/all),
    a named SUBSET, or a comma-separated list of exact filenames."""
    names = [s.strip() for s in tier.split(",") if s.strip()]
    if len(names) > 1 or (names and names[0] in manifest):  # explicit filename list
        chosen = set(names)
        return {n: m for n, m in manifest.items() if n in chosen}
    key = names[0] if names else "meta"
    if key in SUBSETS:
        return {n: m for n, m in manifest.items() if SUBSETS[key](n)}
    return {n: m for n, m in manifest.items() if key == "all" or m["tier"] == key}


def load_manifest():
    import json, urllib.request
    url = f"{BASE}/api/datasets/:persistentId/?persistentId={PID}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research", "Accept": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace"))
    files = data["data"]["latestVersion"]["files"]
    out = {}
    for f in files:
        df = f["dataFile"]
        out[df["filename"]] = {"id": df["id"], "size": df.get("filesize", 0),
                               "md5": df.get("md5") or df.get("checksum", {}).get("value"),
                               "tier": classify(df["filename"])}
    return out

def _download(fid, dest, expect_md5, expect_size=0):
    """Stream a datafile to dest, resuming from a prior .part via an HTTP Range request
    when the server honours it (206). If the server ignores Range (200) we restart cleanly."""
    import hashlib, os, urllib.request
    tmp = dest + ".part"
    url = f"{BASE}/api/access/datafile/{fid}"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expect_size and have >= expect_size:  # partial already full-length: redo, md5 will confirm
        have = 0
    headers = {"User-Agent": "Mozilla/5.0 research"}
    if have:
        headers["Range"] = f"bytes={have}-"
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600)
    resuming = have > 0 and getattr(r, "status", 200) == 206
    h = hashlib.md5()
    if resuming:  # server sends only the missing tail; keep + extend the existing bytes
        with open(tmp, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        n = have; w = open(tmp, "ab")
    else:  # fresh download (no partial, or Range ignored): truncate and start over
        n = 0; w = open(tmp, "wb")
    with r, w:
        while True:
            b = r.read(1 << 22)  # 4 MiB
            if not b:
                break
            w.write(b)
            h.update(b)
            n += len(b)
    got = h.hexdigest()
    if expect_md5 and got != expect_md5:
        os.remove(tmp)
        raise ValueError(f"md5 mismatch for {os.path.basename(dest)}: got {got} want {expect_md5}")
    os.replace(tmp, dest)
    return n, got


@app.function(image=image, cpu=4, memory=8192, timeout=21600, volumes={"/vol": volume})
def fetch(tier: str = "meta"):
    """Download one tier of DEDALE onto the Volume, verifying md5 and skipping done files."""
    import hashlib, json, os
    from pathlib import Path
    volume.reload()
    manifest = load_manifest()
    raw = Path("/vol/raw"); raw.mkdir(parents=True, exist_ok=True)
    (Path("/vol") / "manifest.json").write_text(json.dumps(manifest, indent=2)); volume.commit()

    wanted = select(manifest, tier)
    report = {"tier": tier, "files": {}, "downloaded_bytes": 0, "skipped": [], "planned_gb": round(sum(m["size"] for m in wanted.values()) / 1e9, 2)}
    for name in sorted(wanted, key=lambda k: wanted[k]["size"]):
        m = wanted[name]
        dest = raw / name
        if dest.exists():  # resume: keep file if md5 already matches
            h = hashlib.md5()
            with open(dest, "rb") as f:
                for blk in iter(lambda: f.read(1 << 22), b""):
                    h.update(blk)
            if not m["md5"] or h.hexdigest() == m["md5"]:
                report["skipped"].append(name); report["files"][name] = {"status": "present", "bytes": dest.stat().st_size}
                continue
        n, got = _download(m["id"], str(dest), m["md5"], m["size"])
        report["downloaded_bytes"] += n
        report["files"][name] = {"status": "downloaded", "bytes": n, "md5": got}
        volume.commit()  # durable per-file progress
        print(f"  {name:34} {n/1e6:9.1f} MB  md5 OK", flush=True)
    (Path("/vol") / f"fetch_{tier}.json").write_text(json.dumps(report, indent=2)); volume.commit()
    report["downloaded_gb"] = round(report["downloaded_bytes"] / 1e9, 2)
    return report

@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={"/vol": volume})
def inspect():
    """List what is on the Volume and peek into README + label archives to learn the schema."""
    import io, json, zipfile
    from pathlib import Path
    volume.reload()
    raw = Path("/vol/raw")
    present = {p.name: p.stat().st_size for p in sorted(raw.glob("*"))} if raw.exists() else {}
    result = {"present": {k: round(v / 1e6, 2) for k, v in present.items()},
              "present_count": len(present), "readme": None, "archives": {}}
    readme = raw / "README.md"
    if readme.exists():
        result["readme"] = readme.read_text(errors="replace")[:20000]
    for zname in ("system_logs_labels.zip", "suricata_alerts.zip", "userbehaviour_logs.zip"):
        zp = raw / zname
        if not zp.exists():
            continue
        info = {"members": [], "samples": {}}
        with zipfile.ZipFile(zp) as z:
            for m in z.infolist()[:60]:
                info["members"].append({"name": m.filename, "bytes": m.file_size})
            for m in z.infolist():
                if m.is_dir() or m.file_size == 0:
                    continue
                with z.open(m) as fh:
                    head = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read(1500)
                info["samples"][m.filename] = head
                if len(info["samples"]) >= 3:
                    break
        result["archives"][zname] = info
    (Path("/vol") / "inspection.json").write_text(json.dumps(result, indent=2)); volume.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={"/vol": volume})
def peek(target: str = "zeek_green_internal.zip", member_substr: str = "conn"):
    """Crack a zip on the Volume and print the first matching log's schema (Zeek conn.log,
    handling .gz) so we can see the inline label column DEDALE adds to network data."""
    import gzip, zipfile
    from pathlib import Path
    volume.reload()
    zp = Path("/vol/raw") / target
    if not zp.exists():
        print("MISSING", zp); return {"error": "missing", "target": target}
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        print(f"{target}: {len(names)} members. First 40:")
        for n in names[:40]:
            print("   ", n)
        hit = next((n for n in names if member_substr in n.lower() and not n.endswith("/")), None)
        print("FIRST MATCH:", hit, flush=True)
        if not hit:
            return {"members": names[:40], "members_total": len(names)}
        raw = z.read(hit)
    if hit.endswith(".gz"):
        raw = gzip.decompress(raw)
    lines = raw.decode("utf-8", "replace").splitlines()
    print("----- head (25 lines) -----")
    for ln in lines[:25]:
        print(ln)
    fields = next((ln for ln in lines if ln.startswith("#fields")), None)
    print("----- #fields -----"); print(fields, flush=True)
    return {"target": target, "member": hit, "members_total": len(names), "fields": fields}


@app.function(image=image, cpu=2, memory=16384, timeout=7200, volumes={"/vol": volume})
def zeek_label_stats(target: str = "zeek_green_internal.zip"):
    """Scan every conn_labeled.csv in a Zeek zip: tally benign vs malicious, the label /
    attack_step / tactic / technique distributions, and the date span. Feeds the power
    analysis with a REAL positive count instead of a guess. CPU-only, no GPU."""
    import csv, io, json, zipfile
    from collections import Counter
    from pathlib import Path
    volume.reload()
    zp = Path("/vol/raw") / target
    with zipfile.ZipFile(zp) as z:
        labeled = sorted(n for n in z.namelist() if n.endswith("conn_labeled.csv"))
        total = pos = 0
        labelvals, steps, tactics, techs = Counter(), Counter(), Counter(), Counter()
        tmin = tmax = None
        for n in labeled:
            with z.open(n) as fh:
                r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter=";")
                for row in r:
                    total += 1
                    lab = (row.get("label") or "").strip()
                    labelvals[lab] += 1
                    try:
                        t = float(row.get("ts"))
                        tmin = t if tmin is None else min(tmin, t)
                        tmax = t if tmax is None else max(tmax, t)
                    except (TypeError, ValueError):
                        pass
                    if lab not in ("0", ""):
                        pos += 1
                        steps[(row.get("attack_step") or "").strip()] += 1
                        tactics[(row.get("tactic") or "").strip()] += 1
                        techs[(row.get("technique") or "").strip()] += 1
    out = {"target": target, "labeled_files": len(labeled), "total_conns": total,
           "malicious_conns": pos, "benign_conns": total - pos,
           "positive_rate": round(pos / total, 6) if total else 0,
           "date_span_epoch": [tmin, tmax], "label_values": dict(labelvals),
           "attack_step": dict(steps.most_common(25)), "tactic": dict(tactics.most_common(25)),
           "technique": dict(techs.most_common(25))}
    print(json.dumps({k: out[k] for k in ("target", "labeled_files", "total_conns",
          "malicious_conns", "benign_conns", "positive_rate", "label_values")}, indent=2), flush=True)
    (Path("/vol") / f"labelstats_{target}.json").write_text(json.dumps(out, indent=2)); volume.commit()
    return out


@app.function(image=image, cpu=2, memory=16384, timeout=3600, volumes={"/vol": volume})
def wl_peek(sample: int = 4, cap: int = 3000000):
    """Inspect the winlogbeat zip: member layout, one full event, and the candidate join-key
    fields present in BOTH the main logs and the malicious label file, so we can label reliably."""
    import bz2, gzip, io, json, zipfile
    from pathlib import Path
    volume.reload()
    raw = Path("/vol/raw")

    def keyfields(e):
        wlog = e.get("winlog") or {}
        return {"winlog.record_id": wlog.get("record_id"), "winlog.channel": wlog.get("channel"),
                "winlog.computer": wlog.get("computer_name"), "agent.name": (e.get("agent") or {}).get("name"),
                "host.name": (e.get("host") or {}).get("name"), "@timestamp": e.get("@timestamp"),
                "event.code": (e.get("event") or {}).get("code"), "event.provider": (e.get("event") or {}).get("provider")}

    def read_events(z, member, k):
        evs = []
        with z.open(member) as fh:
            if member.endswith(".gz"):
                stream = gzip.GzipFile(fileobj=fh)
            elif member.endswith(".bz2"):
                stream = bz2.BZ2File(fh)
            else:
                stream = fh
            for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(evs) >= k:
                    break
        return evs

    def read_events_all(z, member):
        with z.open(member) as fh:
            if member.endswith(".gz"):
                stream = gzip.GzipFile(fileobj=fh)
            elif member.endswith(".bz2"):
                stream = bz2.BZ2File(fh)
            else:
                stream = fh
            for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _host(e):
        w = e.get("winlog") or {}
        h = (e.get("agent") or {}).get("name") or (e.get("host") or {}).get("name") or w.get("computer_name") or ""
        return str(h).split(".")[0].upper()

    def _key(e):  # identical to dedale_data.event_key, replicated (that module isn't mounted here)
        w = e.get("winlog") or {}; ev = e.get("event") or {}
        rid = w.get("record_id")
        if rid not in (None, ""):
            return (_host(e), str(w.get("channel") or ""), "rid", str(rid))
        return (_host(e), "ts", str(e.get("@timestamp") or ""), str(ev.get("code") or w.get("event_id") or ""),
                str(ev.get("sequence") or w.get("record_id") or ""))

    def _epoch(ts):
        from datetime import datetime, timezone
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None

    from collections import Counter
    import re
    result = {}
    with zipfile.ZipFile(raw / "system_logs_labels.zip") as z:
        lm = "system_labels/clients_1_and_2/malicious_events_class_1.jsonl"
        label_keys = set(); lab_hosts = Counter(); lab_chans = Counter(); lab_codes = Counter(); lab_ts = []
        for e in read_events_all(z, lm):
            label_keys.add(_key(e)); lab_hosts[_host(e)] += 1
            lab_chans[str((e.get("winlog") or {}).get("channel") or "")] += 1
            lab_codes[str((e.get("event") or {}).get("code") or (e.get("winlog") or {}).get("event_id") or "")] += 1
            t = _epoch(e.get("@timestamp"))
            if t is not None:
                lab_ts.append(t)
        result["label_keyfields"] = [keyfields(e) for e in read_events(z, lm, sample)]
        result["label_count"] = len(label_keys); result["label_hosts"] = dict(lab_hosts)
        result["label_channels"] = dict(lab_chans.most_common(20))
        result["label_codes"] = dict(lab_codes.most_common(20))
    ts_min, ts_max = min(lab_ts), max(lab_ts)
    from datetime import datetime, timezone
    win_dates = {datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat() for t in lab_ts}
    result["label_window_iso"] = [datetime.fromtimestamp(ts_min, tz=timezone.utc).isoformat(),
                                  datetime.fromtimestamp(ts_max, tz=timezone.utc).isoformat()]
    result["label_window_dates"] = sorted(win_dates)

    with zipfile.ZipFile(raw / "system_logs_winlogbeat.zip") as z:
        names = [n for n in z.namelist() if not n.endswith("/") and z.getinfo(n).file_size > 0 and n.endswith(".bz2")]
        result["member_count"] = len(names)
        datef = re.compile(r"(\d{4}-\d{2}-\d{2})")
        def member_date(n):
            m = datef.search(n)
            return m.group(1) if m else None
        in_window = sorted(n for n in names if member_date(n) in win_dates)
        result["members_in_window"] = len(in_window); result["members_total"] = len(names)
        result["in_window_head"] = in_window[:6]
        dm = in_window[0] if in_window else names[0]
        result["data_member"] = dm
        import itertools
        evs = list(itertools.islice(read_events_all(z, dm), sample))
        result["main_keyfields"] = [keyfields(e) for e in evs]
        # join probe restricted to attack-window members (where the labeled events live)
        hosts_seen, benign_ch, pos_ch, matches, scanned, client_events = Counter(), Counter(), Counter(), 0, 0, 0
        stop = False
        for mi, member in enumerate(in_window):
            for e in read_events_all(z, member):
                scanned += 1
                if _host(e) in ("CLIENT1", "CLIENT2"):
                    client_events += 1; hosts_seen[_host(e)] += 1
                    ch = str((e.get("winlog") or {}).get("channel") or "")
                    t = _epoch(e.get("@timestamp"))
                    if t is not None and ts_min <= t <= ts_max:
                        if _key(e) in label_keys:
                            matches += 1; pos_ch[ch] += 1
                        else:
                            benign_ch[ch] += 1
                if cap and scanned >= cap:
                    stop = True; break
            print(f"probe member {mi+1}/{len(in_window)} scanned={scanned} client={client_events} matches={matches}", flush=True)
            if stop:
                break
        result["join_probe"] = {"scanned": scanned, "client_events": client_events,
                                "capped": bool(cap and scanned >= cap),
                                "matches_found": matches, "match_ceiling": len(label_keys),
                                "match_rate_of_ceiling": round(matches / max(len(label_keys), 1), 4),
                                "client_hosts": dict(hosts_seen),
                                "positive_channels": dict(pos_ch.most_common(20)),
                                "benign_channels": dict(benign_ch.most_common(20))}
    (Path("/vol") / "wl_peek.json").write_text(json.dumps(result, indent=2, default=str)); volume.commit()
    print(json.dumps({k: result[k] for k in ("member_count", "members_in_window", "members_total",
                      "label_count", "label_hosts", "label_channels", "label_codes",
                      "label_window_iso", "main_keyfields", "join_probe")},
                     indent=2, default=str), flush=True)
    return result


@app.local_entrypoint()
def main(tier: str = "meta"):
    import json
    from pathlib import Path
    report = fetch.remote(tier)
    out = Path("artifacts/dedale_v1"); out.mkdir(parents=True, exist_ok=True)
    (out / f"fetch_{tier}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
