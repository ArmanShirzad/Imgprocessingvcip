#!/usr/bin/env python3
"""
2026-05-31_AS_unified_model_csv_merger.py
-----------------------------------------
Concatenate per-model rate-task CSVs into single unified CSVs (one per model).

Use case: each finetuned/calibration model has its data split across multiple
CSVs (original ICIP-era 8 cells + our 2026-05-31 completion 10 cells). The
Phase 3 plot script can accept multiple --csv per model, but for clean paper
artifacts + Table 3 we want ONE CSV per model with all 18 cells.

Also: optionally backfill avg_mbps from a bitrate lookup CSV (e.g., output of
the bitrate-hunt that wrote a per-(codec,crf) mean Mbps lookup).

Output format = matches final_evaluation_results_Dataset504_all-Crop.csv:
  folder, case_id, dice_avg, hd95_avg, dice_class_1, avg_mbps, bits_per_voxel

Run locally after both Cal1 scoring + d504 fill-in + Cal1 fill-in are done:
  python3 12-5-2026/scripts/2026-05-31_AS_unified_model_csv_merger.py \\
      --model baseline:paper_figures/30janplots-eval/appended/final_evaluation_resultsWithHD95_APPENDED.csv \\
      --model baseline:/tmp/2026-05-31_baseline_completion.csv \\
      --model d504:paper_figures/comparison_result/final_evaluation_results_Dataset504_all-Crop.csv \\
      --model d504:/tmp/2026-05-31_d504_completion.csv \\
      --model cal1:/tmp/2026-05-31_cal1_8cells.csv \\
      --model cal1:/tmp/2026-05-31_cal1_completion.csv \\
      --bitrate_lookup /tmp/2026-05-31_bitrate_lookup.csv \\
      --out_dir /tmp/vcip_phase3/unified

--bitrate_lookup is optional. If provided, expects schema:
  codec, crf, mean_mbps
  libx265, 18, 1.234
  ...
Rows with missing avg_mbps get filled from this lookup.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_bitrate_lookup(path: Path) -> dict[tuple[str, int], float]:
    lookup = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                codec = r["codec"].strip()
                crf = int(r["crf"])
                mbps = float(r["mean_mbps"])
                lookup[(codec, crf)] = mbps
            except (KeyError, ValueError):
                continue
    return lookup


def parse_folder(folder: str) -> tuple[str, int] | None:
    if "_crf" not in folder:
        return None
    codec, crf_s = folder.rsplit("_crf", 1)
    try:
        return codec, int(crf_s)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="model_name:path/to.csv (repeatable; multiple files per model concatenate)")
    ap.add_argument("--bitrate_lookup", type=Path, default=None,
                    help="optional CSV with columns codec,crf,mean_mbps")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bitrate_lookup = load_bitrate_lookup(args.bitrate_lookup) if args.bitrate_lookup else {}
    if bitrate_lookup:
        print(f"Loaded {len(bitrate_lookup)} bitrate entries from {args.bitrate_lookup}")

    # Group input files by model
    by_model: dict[str, list[Path]] = defaultdict(list)
    for spec in args.model:
        if ":" not in spec:
            raise SystemExit(f"--model expects model_name:path  (got {spec})")
        name, p = spec.split(":", 1)
        by_model[name].append(Path(p))

    # For each model: concatenate, dedupe, optionally backfill avg_mbps, write
    for model, paths in by_model.items():
        print(f"\n=== {model} ===")
        rows: dict[tuple[str, str], dict] = {}  # (folder, case_id) -> row
        for p in paths:
            if not p.is_file():
                print(f"  WARN: {p} missing — skipping")
                continue
            n_added = 0
            with open(p) as f:
                for r in csv.DictReader(f):
                    folder = r.get("folder", "").strip()
                    case_id = r.get("case_id", "").strip()
                    if not folder or not case_id:
                        continue
                    key = (folder, case_id)
                    # Later file wins on conflict; warn on duplicates
                    if key in rows:
                        prev = rows[key]
                        if prev["dice_avg"] != r.get("dice_avg", ""):
                            pass  # silent; later wins
                    rows[key] = {
                        "folder": folder,
                        "case_id": case_id,
                        "dice_avg": r.get("dice_avg", ""),
                        "hd95_avg": r.get("hd95_avg", ""),
                        "dice_class_1": r.get("dice_class_1", r.get("dice_avg", "")),
                        "avg_mbps": r.get("avg_mbps", "").strip(),
                        "bits_per_voxel": r.get("bits_per_voxel", "").strip(),
                    }
                    n_added += 1
            print(f"  {p.name}: {n_added} rows ingested")

        # Backfill avg_mbps from lookup
        if bitrate_lookup:
            n_filled = 0
            for key, row in rows.items():
                if not row["avg_mbps"]:
                    cell = parse_folder(row["folder"])
                    if cell and cell in bitrate_lookup:
                        row["avg_mbps"] = f"{bitrate_lookup[cell]:.6f}"
                        n_filled += 1
            print(f"  backfilled avg_mbps in {n_filled} rows from bitrate_lookup")

        # Per-cell summary
        cells = defaultdict(int)
        for key in rows:
            cells[key[0]] += 1
        print(f"  cells covered: {len(cells)}")
        for k in sorted(cells.keys()):
            print(f"    {k:25s} n={cells[k]}")

        out_path = args.out_dir / f"2026-05-31_unified_{model}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["folder", "case_id", "dice_avg", "hd95_avg",
                                              "dice_class_1", "avg_mbps", "bits_per_voxel"])
            w.writeheader()
            for key in sorted(rows.keys()):
                w.writerow(rows[key])
        print(f"  wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
