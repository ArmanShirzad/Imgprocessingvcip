#!/usr/bin/env python3
"""
2026-05-31_AS_phase3_bdrate.py
------------------------------
VCIP 2026 Phase 3 — BD-Rate matrix per model.

BD-Rate (Bjøntegaard-Delta Rate) measures the average bitrate savings between
two rate-distortion curves at matched quality. Negative BD-Rate vs a reference
codec = bitrate savings. Per Alireza's Phase 3, we extend the original Table 2
from H.264-vs-AV1 only to the full 3-codec pairwise matrix.

Implementation: standard 3rd-degree polynomial fit on (log-bitrate, Dice) and
integrate the difference over the overlapping bitrate range. Quality axis is
Dice (and optionally HD95).

Inputs:
  --summary  /tmp/vcip_phase3/2026-05-31_phase3_per_cell_summary.csv
             (output of 2026-05-31_AS_phase3_rate_task_plot.py — has columns
              model, codec, crf, dice_mean, hd95_mean, mbps_mean)

Output:
  --out_csv  BD-Rate matrix per model: rows = codec pairs, cols = (Dice, HD95)
  --out_md   Same matrix as a Markdown table for the paper draft
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np


def bd_rate_piecewise(curve_a: list[tuple[float, float]],
                      curve_b: list[tuple[float, float]]) -> float | None:
    """BD-Rate of curve_b relative to curve_a (negative = curve_b saves bitrate).

    Each curve = [(bitrate_Mbps, quality), ...]. Quality is Dice (higher better)
    or for HD95 negate the y-values before passing.

    Standard Bjøntegaard implementation:
      1. Fit cubic polynomial through (log10(rate), quality)
      2. Find overlap quality range
      3. Integrate horizontal distance (rate_b - rate_a) over overlap
      4. Express as percent change vs rate_a
    Returns None if curves don't overlap in quality range.
    """
    if len(curve_a) < 4 or len(curve_b) < 4:
        return None
    a = sorted([(r, q) for r, q in curve_a if r is not None and r > 0])
    b = sorted([(r, q) for r, q in curve_b if r is not None and r > 0])
    if len(a) < 4 or len(b) < 4:
        return None

    log_ra = np.log10([r for r, _ in a])
    qa = np.array([q for _, q in a])
    log_rb = np.log10([r for r, _ in b])
    qb = np.array([q for _, q in b])

    # Quality overlap range
    qmin = max(qa.min(), qb.min())
    qmax = min(qa.max(), qb.max())
    if qmax <= qmin:
        return None

    # Cubic poly fit: quality -> log(rate)  (Bjøntegaard's inverse form)
    pa = np.polyfit(qa, log_ra, 3)
    pb = np.polyfit(qb, log_rb, 3)

    # Integrate (poly_b - poly_a) over [qmin, qmax]
    # NumPy 2.0+ renamed np.trapz -> np.trapezoid
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    samples = np.linspace(qmin, qmax, 200)
    int_a = trapz_fn(np.polyval(pa, samples), samples)
    int_b = trapz_fn(np.polyval(pb, samples), samples)
    avg_diff = (int_b - int_a) / (qmax - qmin)

    # avg_diff is in log10(rate) units; convert to percent
    return (10 ** avg_diff - 1) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    args = ap.parse_args()

    # Load summary
    by_model_codec: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    with open(args.summary) as f:
        for r in csv.DictReader(f):
            try:
                m = r["model"]
                c = r["codec"]
                d = float(r["dice_mean"])
                h = float(r["hd95_mean"])
                mb = float(r["mbps_mean"]) if r["mbps_mean"] else None
            except (ValueError, KeyError):
                continue
            if mb is None or mb <= 0:
                continue
            by_model_codec[(m, c)].append((mb, d, h))

    codecs = sorted({c for _, c in by_model_codec})
    models = sorted({m for m, _ in by_model_codec})
    print(f"Loaded models: {models}")
    print(f"Loaded codecs: {codecs}")

    # Pairwise BD-Rate per model
    out_rows = []
    for model in models:
        print(f"\n=== Model: {model} ===")
        for ca, cb in itertools.combinations(codecs, 2):
            pa = by_model_codec.get((model, ca), [])
            pb = by_model_codec.get((model, cb), [])
            if len(pa) < 4 or len(pb) < 4:
                print(f"  {ca:12s} vs {cb:12s}  insufficient points (need >=4, got {len(pa)}/{len(pb)})")
                continue
            curve_a_dice = [(r, d) for r, d, _ in pa]
            curve_b_dice = [(r, d) for r, d, _ in pb]
            bd_dice = bd_rate_piecewise(curve_a_dice, curve_b_dice)

            # For HD95 (lower=better) negate
            curve_a_hd = [(r, -h) for r, _, h in pa]
            curve_b_hd = [(r, -h) for r, _, h in pb]
            bd_hd = bd_rate_piecewise(curve_a_hd, curve_b_hd)

            line = f"  {cb:12s} vs {ca:12s}  BD-Rate(Dice)={bd_dice:+.2f}%" if bd_dice is not None else f"  {cb:12s} vs {ca:12s}  BD-Rate(Dice)=N/A"
            if bd_hd is not None:
                line += f"  BD-Rate(HD95)={bd_hd:+.2f}%"
            print(line)
            out_rows.append({"model": model, "codec_b": cb, "codec_a": ca,
                             "bdrate_dice_pct": f"{bd_dice:.4f}" if bd_dice is not None else "",
                             "bdrate_hd95_pct": f"{bd_hd:.4f}" if bd_hd is not None else ""})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "codec_b", "codec_a", "bdrate_dice_pct", "bdrate_hd95_pct"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {args.out_csv}")

    # Markdown table for paper draft
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("# BD-Rate matrix per model\n\n")
        f.write("Negative BD-Rate = codec_b saves bitrate vs codec_a at matched quality.\n\n")
        for model in models:
            f.write(f"## Model: {model}\n\n")
            f.write("| codec_b vs codec_a | BD-Rate (Dice) % | BD-Rate (HD95) % |\n")
            f.write("|---|---|---|\n")
            for r in out_rows:
                if r["model"] != model:
                    continue
                f.write(f"| {r['codec_b']} vs {r['codec_a']} | {r['bdrate_dice_pct']} | {r['bdrate_hd95_pct']} |\n")
            f.write("\n")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
