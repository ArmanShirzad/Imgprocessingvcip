#!/usr/bin/env python3
"""
2026-05-31_AS_build_bitrate_lookup.py
-------------------------------------
Build a per-(codec, CRF) mean bitrate lookup table from compression_stats.csv
(the codec_stress_test_parallel.py log that captured per-case avg_mbps during
encoding).

Output schema (matches what unified_model_csv_merger.py expects for
--bitrate_lookup):
  codec, crf, n_cases, mean_mbps, std_mbps, min_mbps, max_mbps

Run locally (after scp'ing compression_stats.csv down) OR on Willi:

  # On Willi (use the local-to-Willi path):
  python3 /scratch/shirzarm/vcip/scripts/2026-05-31_AS_build_bitrate_lookup.py \\
      --in_csv /scratch/shirzarm/phase3_codecs/compression_stats.csv \\
      --out_csv /scratch/shirzarm/vcip/refs/2026-05-31_bitrate_lookup.csv

Then scp the lookup down + feed it to the merger via --bitrate_lookup.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, required=True,
                    help="compression_stats.csv from codec_stress_test_parallel.py")
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)

    with open(args.in_csv) as f:
        for r in csv.DictReader(f):
            try:
                codec = r["codec"].strip()
                crf = int(r["crf"])
                mbps = float(r["avg_mbps"])
            except (KeyError, ValueError):
                continue
            if mbps <= 0:
                continue
            by_cell[(codec, crf)].append(mbps)

    print(f"Loaded bitrate samples for {len(by_cell)} (codec, crf) cells from {args.in_csv}\n")

    # Sort: codec then CRF
    keys = sorted(by_cell.keys())

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "crf", "n_cases", "mean_mbps", "std_mbps", "min_mbps", "max_mbps"])
        for codec, crf in keys:
            vals = by_cell[(codec, crf)]
            n = len(vals)
            mean = sum(vals) / n
            if n > 1:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                std = var ** 0.5
            else:
                std = 0.0
            w.writerow([codec, crf, n, f"{mean:.6f}", f"{std:.6f}",
                        f"{min(vals):.6f}", f"{max(vals):.6f}"])
            print(f"  {codec:12s} CRF {crf:2d}  n={n:4d}  mean={mean:.4f} Mbps  std={std:.4f}  range=[{min(vals):.4f},{max(vals):.4f}]")

    print(f"\nWrote {args.out_csv}")
    print(f"Cells covered: {len(keys)}")

    # Highlight whether the 10 cells we needed are present
    needed = [("libx265", crf) for crf in [18, 23, 28, 33, 38, 43]] + \
             [("libx264", crf) for crf in [33, 38]] + \
             [("libsvtav1", crf) for crf in [33, 38]]
    print(f"\nCoverage of the 10 cells we need for the rate-task plot:")
    for cell in needed:
        if cell in by_cell:
            v = by_cell[cell]
            print(f"  {cell[0]:12s} CRF {cell[1]:2d}  ✓ n={len(v)}  mean={sum(v)/len(v):.4f} Mbps")
        else:
            print(f"  {cell[0]:12s} CRF {cell[1]:2d}  ✗ MISSING")


if __name__ == "__main__":
    main()
