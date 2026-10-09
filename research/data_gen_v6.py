#!/usr/bin/env python3
"""
TG-Detect v10 Formal OOD Generator (Modal)
──────────────────────────────────────────────────────────────────────
Generates 1,000,000+ events spanning a 14-day continuous timeline.
Includes formalized structural topological mutations:
- Standard APT (0 mutations)
- Moderate OOD (+1 topological mutation/insertion)
- Extreme OOD (>=3 topological mutations/shuffling)

Writes directly to Modal Volume at /data/v6_ood_formal.jsonl
"""

import modal
import os
import time
import random
import math
import json

app = modal.App("tg-detect-datagen-v6")
volume = modal.Volume.from_name("tg-detect-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "tqdm")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def generate_v6_data(target_events: int = 1_000_000, days: int = 14):
    output_path = "/data/v6_ood_formal.jsonl"
    print(f"Generating {target_events} formal OOD events over {days} days...")
    
    start_ts = 1700000000.0
    end_ts = start_ts + (days * 86400.0)
    current_ts = start_ts
    time_step = (days * 86400.0) / target_events
    
    datasets = ["darpa", "unsw", "lanl"]
    
    active_apts = []
    test_split_ts = start_ts + (days * 0.7 * 86400.0)
    
    def spawn_apt(ts):
        duration = random.uniform(2 * 86400, 5 * 86400)
        domain = random.choice(datasets)
        
        ood_type = "Standard"
        if ts > test_split_ts:
            r = random.random()
            if r > 0.66: ood_type = "Extreme"
            elif r > 0.33: ood_type = "Moderate"
            
        stages = 3
        if ood_type == "Moderate": stages = 4
        elif ood_type == "Extreme": stages = 3
            
        return {
            "start_ts": ts,
            "end_ts": ts + duration,
            "domain": domain,
            "stage": 0,
            "total_stages": stages,
            "actor": f"apt_actor_{random.randint(1000, 9999)}",
            "target": f"internal_target_{random.randint(100, 999)}",
            "next_action_ts": ts + random.uniform(10, 3600),
            "ood_type": ood_type,
            "chain_id": f"chain_{random.randint(10000, 99999)}"
        }

    events_written = 0
    start_time = time.time()
    last_apt_event_id = {}
    
    with open(output_path, "w") as f:
        while events_written < target_events:
            if random.random() < (200.0 / target_events): # Spawn about 200 APT chains
                active_apts.append(spawn_apt(current_ts))
            
            is_malicious = False
            event = None
            causal_parent = None
            
            for apt in active_apts:
                if current_ts >= apt["next_action_ts"] and apt["stage"] < apt["total_stages"]:
                    is_malicious = True
                    
                    if apt["ood_type"] == "Standard":
                        tactics_map = {0: ["Initial_Access"], 1: ["Lateral_Movement"], 2: ["Exfiltration"]}
                    elif apt["ood_type"] == "Moderate":
                        # +1 mutation: Insertion of noise/discovery step
                        tactics_map = {0: ["Initial_Access"], 1: ["Discovery"], 2: ["Lateral_Movement"], 3: ["Exfiltration"]}
                    else:
                        # Extreme OOD: Complete topological scrambling (>= 3 mutations)
                        tactics_map = {0: ["Discovery"], 1: ["Impact"], 2: ["Lateral_Movement"]}
                    
                    tactics = tactics_map[apt["stage"]]
                    event_id = f"evt_{events_written}"
                    causal_parent = last_apt_event_id.get(apt["actor"], None)
                    last_apt_event_id[apt["actor"]] = event_id
                    
                    event = {
                        "event_id": event_id,
                        "ts": current_ts,
                        "src_id": apt["actor"],
                        "dst_id": apt["target"],
                        "label": 1,
                        "source_tag": f"{apt['domain']}_log",
                        "tactics": tactics,
                        "apt_stage": apt["stage"],
                        "chain_id": apt["chain_id"],
                        "ood_type": apt["ood_type"],
                        "causal_parent": causal_parent,
                        "attrs": {
                            "frequency": random.uniform(0.1, 0.5),
                            "temporal_burst": random.uniform(0.0, 0.1),
                            "rarity": random.uniform(0.8, 1.0)
                        }
                    }
                    
                    apt["stage"] += 1
                    if apt["stage"] < apt["total_stages"]:
                        apt["next_action_ts"] = current_ts + random.uniform(3600, 24 * 3600)
                        if apt["stage"] % 2 != 0:
                            apt["actor"] = apt["target"]
                            apt["target"] = f"critical_server_{random.randint(10, 99)}"
                        else:
                            apt["actor"] = apt["target"]
                            apt["target"] = f"external_ip_{random.randint(100, 999)}"
                    break
            
            active_apts = [a for a in active_apts if a["stage"] < a["total_stages"]]
            
            if not is_malicious:
                event_id = f"evt_{events_written}"
                ds = random.choice(datasets)
                event = {
                    "event_id": event_id,
                    "ts": current_ts,
                    "src_id": f"user_{random.randint(1, 5000)}",
                    "dst_id": f"sys_{random.randint(1, 5000)}",
                    "label": 0,
                    "source_tag": f"{ds}_background",
                    "tactics": [],
                    "apt_stage": -1,
                    "ood_type": "None",
                    "causal_parent": None,
                    "attrs": {
                        "frequency": random.uniform(5.0, 100.0),
                        "temporal_burst": random.uniform(0.5, 1.0),
                        "rarity": random.uniform(0.0, 0.2)
                    }
                }
            
            f.write(json.dumps(event) + "\n")
            events_written += 1
            current_ts += time_step * random.uniform(0.1, 2.0)
            
            if events_written % 100000 == 0:
                print(f"  Generated {events_written}/{target_events} formal OOD events...")

    volume.commit()
    print(f"✅ Generated {events_written} events in {time.time() - start_time:.2f}s.")
    print(f"✅ Saved directly to Modal volume at {output_path}")

@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  TG-Detect v10 Formal OOD Generator")
    print("=" * 60)
    generate_v6_data.remote(target_events=1_000_000, days=14)
