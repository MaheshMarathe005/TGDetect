#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows-APT 2025 helper: filter scenarios from scenario_manifest.csv and (optionally) filter logs.

Examples:
  # Filter scenario list only:
  python parse_attack_mapping_and_filter.py \
    --scenario-manifest scenario_manifest.csv \
    --technique T1059.003 \
    --out-scenarios filtered_scenarios.csv

  # Filter logs too (requires Scenario_ID or Scenrario_ID column in logs):
  python parse_attack_mapping_and_filter.py \
    --scenario-manifest scenario_manifest.csv \
    --logs combined.csv \
    --tactic "Discovery" \
    --out-logs subset_logs.csv \
    --out-scenarios filtered_scenarios.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Set

import pandas as pd


def normalize_list_field(value: str) -> Set[str]:
    """
    Parses fields like:
      - "Defense Evasion; Discovery; Execution"
      - "T1007; T1016; T1016.001"
    Returns a set of stripped tokens.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    s = str(value).strip()
    if not s:
        return set()
    parts = [p.strip() for p in s.split(";")]
    return {p for p in parts if p}


def pick_scenario_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Scenario_ID", "Scenrario_ID", "scenario_id", "scenrario_id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-manifest", required=True, help="Path to scenario_manifest.csv")
    ap.add_argument("--logs", default=None, help="Optional: path to logs CSV (e.g., combined.csv)")
    ap.add_argument("--tactic", default=None, help="Filter by a MITRE tactic name (exact match token)")
    ap.add_argument("--technique", default=None, help="Filter by a MITRE technique ID (exact match token)")
    ap.add_argument("--group", default=None, help="Filter by a MITRE group name (substring match)")
    ap.add_argument("--out-scenarios", required=True, help="Output CSV for filtered scenarios")
    ap.add_argument("--out-logs", default=None, help="Optional: output CSV for filtered logs")
    args = ap.parse_args()

    manifest_path = Path(args.scenario_manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"[error] scenario-manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    scen = pd.read_csv(manifest_path, low_memory=False, on_bad_lines="skip")

    # Column names in your snippet: Scenrario_ID, Scenario_Name, Mitre_Groups, Simiulated Tactics, Simulated Techniques
    # Keep it robust to variants.
    scenario_id_col = pick_scenario_id_col(scen) or "Scenrario_ID"
    tactic_col = None
    tech_col = None
    group_col = None

    for c in scen.columns:
        cl = c.lower()
        if "tactic" in cl and tactic_col is None:
            tactic_col = c
        if "technique" in cl and tech_col is None:
            tech_col = c
        if "group" in cl and group_col is None:
            group_col = c

    if scenario_id_col not in scen.columns:
        print(f"[error] Could not find scenario id column in manifest. Columns: {list(scen.columns)}", file=sys.stderr)
        return 3

    # Filtering
    mask = pd.Series([True] * len(scen))
    if args.tactic:
        if not tactic_col:
            print("[error] Tactic column not found in manifest.", file=sys.stderr)
            return 4
        wanted = args.tactic.strip()
        mask = mask & scen[tactic_col].fillna("").apply(lambda v: wanted in normalize_list_field(v))

    if args.technique:
        if not tech_col:
            print("[error] Technique column not found in manifest.", file=sys.stderr)
            return 5
        wanted = args.technique.strip()
        mask = mask & scen[tech_col].fillna("").apply(lambda v: wanted in normalize_list_field(v))

    if args.group:
        if not group_col:
            print("[error] Group column not found in manifest.", file=sys.stderr)
            return 6
        wanted = args.group.strip().lower()
        mask = mask & scen[group_col].fillna("").astype(str).str.lower().str.contains(wanted, na=False)

    filtered_scen = scen.loc[mask].copy()
    out_scen = Path(args.out_scenarios).expanduser().resolve()
    out_scen.parent.mkdir(parents=True, exist_ok=True)
    filtered_scen.to_csv(out_scen, index=False)
    print(f"[ok] Filtered scenarios: {len(filtered_scen)} -> {out_scen}", file=sys.stderr)

    # Optionally filter logs
    if args.logs:
        logs_path = Path(args.logs).expanduser().resolve()
        if not logs_path.exists():
            print(f"[error] logs not found: {logs_path}", file=sys.stderr)
            return 7
        logs = pd.read_csv(logs_path, low_memory=False, on_bad_lines="skip")

        log_scenario_col = pick_scenario_id_col(logs)
        if not log_scenario_col:
            print(
                "[warn] No Scenario_ID/Scenrario_ID column found in logs. "
                "Saved filtered scenarios only. If you want log filtering, include scenario id in logs.",
                file=sys.stderr,
            )
            return 0

        allowed = set(filtered_scen[scenario_id_col].astype(str).tolist())
        subset = logs[logs[log_scenario_col].astype(str).isin(allowed)].copy()

        if args.out_logs:
            out_logs = Path(args.out_logs).expanduser().resolve()
            out_logs.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(out_logs, index=False)
            print(f"[ok] Filtered logs: {len(subset):,} -> {out_logs}", file=sys.stderr)
        else:
            print("[warn] --out-logs not provided, skipping writing filtered logs.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
