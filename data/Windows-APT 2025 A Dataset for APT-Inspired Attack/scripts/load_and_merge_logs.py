#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows-APT 2025 helper: load CSV logs and merge per-period files.

Examples:
  python load_and_merge_logs.py --data-dir ./ --out merged.csv
  python load_and_merge_logs.py --data-dir ./ --include combined.csv --out merged.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_EXCLUDES = {"combined.csv", "scenario_manifest.csv", "validation_summary.csv", "checksums.sha256"}


def find_csv_files(data_dir: Path, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> List[Path]:
    exclude_set = set(exclude or [])
    include_set = set(include or [])

    csvs = []
    for p in sorted(data_dir.glob("*.csv")):
        if p.name in exclude_set:
            continue
        if include_set and p.name not in include_set:
            continue
        csvs.append(p)
    return csvs


def safe_read_csv(path: Path, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Reads a CSV safely for large files.
    Notes:
      - low_memory=False reduces dtype guessing issues
      - on_bad_lines='skip' avoids failing on rare malformed lines
    """
    if chunksize:
        chunks = []
        for chunk in pd.read_csv(
            path,
            low_memory=False,
            on_bad_lines="skip",
            chunksize=chunksize,
        ):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    return pd.read_csv(path, low_memory=False, on_bad_lines="skip")


def merge_csvs(paths: List[Path], chunksize: Optional[int] = None) -> pd.DataFrame:
    frames = []
    for p in paths:
        print(f"[merge] Reading: {p.name}", file=sys.stderr)
        df = safe_read_csv(p, chunksize=chunksize)
        df["source_file"] = p.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing Windows-APT CSV files")
    ap.add_argument("--out", required=True, help="Output CSV path for merged logs")
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional: explicit CSV filenames to include (example: 1-11-November.csv 12-13-December.csv)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDES),
        help=f"Optional: CSV filenames to exclude (default: {sorted(DEFAULT_EXCLUDES)})",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Optional: read CSVs in chunks (useful for large files). Example: 200000",
    )

    args = ap.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not data_dir.exists():
        print(f"[error] data-dir not found: {data_dir}", file=sys.stderr)
        return 2

    chunksize = args.chunksize if args.chunksize and args.chunksize > 0 else None

    csv_files = find_csv_files(data_dir, include=args.include, exclude=args.exclude)
    if not csv_files:
        print("[error] No CSV files found with the given include/exclude filters.", file=sys.stderr)
        return 3

    merged = merge_csvs(csv_files, chunksize=chunksize)
    print(f"[merge] Rows: {len(merged):,} | Cols: {len(merged.columns):,}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[ok] Wrote merged CSV: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
