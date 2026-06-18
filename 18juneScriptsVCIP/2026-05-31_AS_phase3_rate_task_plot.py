#!/usr/bin/env python3
"""
2026-05-31_AS_phase3_rate_task_plot.py
--------------------------------------
VCIP 2026 Phase 3 — rate-task curves.

Reads multiple per-case CSVs (one per model: baseline / 504 / Cal1 / LOCO_A/B/C
when ready) and produces:
  - Dice vs CRF curves (per codec, per model)
  - Dice vs bitrate curves (per codec, per model) IF avg_mbps is populated
    OR falls back to a bitrate-lookup table derived from the canonical
    APPENDED.csv where avg_mbps is populated
  - HD95 vs bitrate
  - Per-codec safe bitrate threshold (Dice >= 0.84, per Alireza Phase 3 spec)

Per-cell summary stats (mean, 5th-pct) printed + saved.

This script is the local Python replacement for the submitted-paper plot
pipeline (plot_paper_figures_v2.py + plot_baseline_vs_finetuned.py), updated
for VCIP's 3 codecs + multiple finetuned models.

Run locally:
  python3 12-5-2026/scripts/2026-05-31_AS_phase3_rate_task_plot.py \\
      --csv baseline:paper_figures/30janplots-eval/appended/final_evaluation_resultsWithHD95_APPENDED.csv \\
      --csv baseline_new:/path/to/2026-05-31_baseline_completion.csv \\
      --csv d504:paper_figures/comparison_result/final_evaluation_results_Dataset504_all-Crop.csv \\
      --csv d504_new:/path/to/2026-05-31_d504_completion.csv \\
      --csv cal1:/path/to/2026-05-31_cal1_8cells.csv \\
      --csv cal1_new:/path/to/2026-05-31_cal1_completion.csv \\
      --out_dir /tmp/vcip_phase3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Color per codec to match the submitted paper's Fig 2 style
CODEC_COLORS = {
    "libx264":   "#1f77b4",  # blue (H.264)
    "libx265":   "#2ca02c",  # green (HEVC)
    "libsvtav1": "#ff7f0e",  # orange (AV1)
}

# Line style per model
MODEL_STYLES = {
    "baseline":     {"linestyle": "-",  "marker": "o", "alpha": 1.0,  "linewidth": 2.0},
    "d504":         {"linestyle": "--", "marker": "s", "alpha": 0.85, "linewidth": 1.6},
    "cal1":         {"linestyle": ":",  "marker": "^", "alpha": 0.85, "linewidth": 1.6},
    "loco_a":       {"linestyle": "-.", "marker": "D", "alpha": 0.85, "linewidth": 1.6},
    "loco_b":       {"linestyle": "-.", "marker": "v", "alpha": 0.85, "linewidth": 1.6},
    "loco_c":       {"linestyle": "-.", "marker": "P", "alpha": 0.85, "linewidth": 1.6},
}

CODEC_DISPLAY = {
    "libx264":   "H.264 (AVC)",
    "libx265":   "HEVC",
    "libsvtav1": "AV1",
}


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def parse_cell(folder: str) -> tuple[str, int] | None:
    """libx264_crf28 -> ('libx264', 28)"""
    if "_crf" not in folder:
        return None
    codec, crf_s = folder.rsplit("_crf", 1)
    try:
        return codec, int(crf_s)
    except ValueError:
        return None


def group_by_cell(rows: list[dict]) -> dict[tuple[str, int], dict[str, list[float]]]:
    """Return {(codec, crf): {'dice':[...], 'hd95':[...], 'mbps':[...] }}."""
    out = defaultdict(lambda: {"dice": [], "hd95": [], "mbps": []})
    for r in rows:
        cell = parse_cell(r.get("folder", ""))
        if cell is None:
            continue
        for col_csv, col_out in [("dice_avg", "dice"), ("hd95_avg", "hd95"), ("avg_mbps", "mbps")]:
            v = r.get(col_csv, "")
            if v == "" or v is None:
                continue
            try:
                out[cell][col_out].append(float(v))
            except ValueError:
                pass
    return out


def build_bitrate_lookup(*sources: dict[tuple[str, int], dict[str, list[float]]]) -> dict[tuple[str, int], float]:
    """Merge sources to build {(codec, crf): mean_mbps} using whichever source
    has avg_mbps populated. Used to fill bitrate for cells where the
    completion CSV left avg_mbps blank."""
    lookup = {}
    for src in sources:
        for cell, cols in src.items():
            if cell in lookup:
                continue
            mbps = cols.get("mbps", [])
            if mbps:
                lookup[cell] = float(np.mean(mbps))
    return lookup


def per_cell_summary(rows: list[dict]) -> list[dict]:
    g = group_by_cell(rows)
    out = []
    for (codec, crf), cols in sorted(g.items()):
        d = cols["dice"]
        h = cols["hd95"]
        if not d:
            continue
        out.append({
            "codec": codec, "crf": crf, "n": len(d),
            "dice_mean": float(np.mean(d)),
            "dice_5pct": float(np.percentile(d, 5)),
            "hd95_mean": float(np.mean(h)) if h else float("nan"),
            "mbps_mean": float(np.mean(cols["mbps"])) if cols["mbps"] else float("nan"),
        })
    return out


def find_safe_bitrate_threshold(curve: list[dict], dice_threshold: float = 0.84) -> tuple[float, int] | None:
    """Given a list of {crf, dice_mean, mbps_mean} sorted by bitrate ascending,
    find the smallest bitrate at which dice_mean >= threshold. Returns (mbps, crf)
    or None if curve never crosses."""
    sorted_curve = sorted(curve, key=lambda c: c.get("mbps_mean") or 1e9)
    for pt in sorted_curve:
        if pt["dice_mean"] >= dice_threshold:
            return pt.get("mbps_mean"), pt["crf"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="model_name:path/to.csv  (use multiple --csv to combine "
                         "multiple files per model — e.g. baseline + baseline_new)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--dice_threshold", type=float, default=0.84)
    ap.add_argument("--clean_baseline_dice", type=float, default=0.8669,
                    help="Reference horizontal line on the Dice plot.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse --csv args: combine rows per model
    model_rows: dict[str, list[dict]] = defaultdict(list)
    for spec in args.csv:
        if ":" not in spec:
            raise SystemExit(f"--csv expects model_name:path  (got {spec})")
        name, p = spec.split(":", 1)
        # Strip _new suffix when combining
        model = name.replace("_new", "")
        path = Path(p)
        if not path.is_file():
            print(f"WARN: {path} not found, skipping")
            continue
        rows = load_csv(path)
        model_rows[model].extend(rows)
        print(f"loaded {len(rows)} rows from {p} (model: {model})")

    if "baseline" not in model_rows:
        raise SystemExit("ERROR: at least one --csv baseline:... is required to build the bitrate lookup")

    # Build (codec, crf) -> mean_mbps lookup from baseline (which has avg_mbps populated)
    baseline_cells = group_by_cell(model_rows["baseline"])
    bitrate_lookup = build_bitrate_lookup(baseline_cells)
    # Also try other models as fallback
    for m, rows in model_rows.items():
        bitrate_lookup.update({k: v for k, v in build_bitrate_lookup(group_by_cell(rows)).items() if k not in bitrate_lookup})

    print(f"\nBitrate lookup built for {len(bitrate_lookup)} (codec, crf) cells")

    # Build per-cell summary per model + fill missing bitrate from lookup
    summaries: dict[str, list[dict]] = {}
    for model, rows in model_rows.items():
        summary = per_cell_summary(rows)
        for s in summary:
            if not np.isfinite(s["mbps_mean"]):
                s["mbps_mean"] = bitrate_lookup.get((s["codec"], s["crf"]), float("nan"))
        summaries[model] = summary

    # Write combined per-cell summary CSV
    out_csv = args.out_dir / "2026-05-31_phase3_per_cell_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "codec", "crf", "n", "dice_mean", "dice_5pct", "hd95_mean", "mbps_mean"])
        for model in sorted(summaries.keys()):
            for s in sorted(summaries[model], key=lambda x: (x["codec"], x["crf"])):
                w.writerow([model, s["codec"], s["crf"], s["n"],
                            f"{s['dice_mean']:.4f}", f"{s['dice_5pct']:.4f}",
                            f"{s['hd95_mean']:.4f}", f"{s['mbps_mean']:.4f}" if np.isfinite(s["mbps_mean"]) else ""])
    print(f"\nWrote per-cell summary to {out_csv}")

    # --- Plot Dice vs bitrate, one subplot per codec
    codecs = sorted({s["codec"] for m in summaries.values() for s in m})
    fig, axes = plt.subplots(1, len(codecs), figsize=(5 * len(codecs), 4.5), sharey=True)
    if len(codecs) == 1:
        axes = [axes]
    for ax, codec in zip(axes, codecs):
        for model, summary in sorted(summaries.items()):
            pts = [s for s in summary if s["codec"] == codec and np.isfinite(s["mbps_mean"])]
            if not pts:
                continue
            pts.sort(key=lambda x: x["mbps_mean"])
            xs = [p["mbps_mean"] for p in pts]
            ys = [p["dice_mean"] for p in pts]
            style = MODEL_STYLES.get(model, {"linestyle": "-", "marker": "x", "alpha": 0.7, "linewidth": 1.2})
            ax.plot(xs, ys, color=CODEC_COLORS[codec], label=model, **style)
        ax.axhline(args.clean_baseline_dice, color="green", linestyle="--", alpha=0.5,
                   label=f"clean baseline ({args.clean_baseline_dice:.3f})")
        ax.axhline(args.dice_threshold, color="red", linestyle=":", alpha=0.4,
                   label=f"Dice = {args.dice_threshold:.2f} threshold")
        ax.set_xscale("log")
        ax.set_xlabel("Bitrate (Mbps, log scale)")
        ax.set_ylabel("Dice")
        ax.set_title(CODEC_DISPLAY.get(codec, codec))
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Rate-task curve: Dice vs bitrate per codec, per model")
    fig.tight_layout()
    out_dice = args.out_dir / "2026-05-31_phase3_dice_vs_bitrate.png"
    fig.savefig(out_dice, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dice}")

    # --- Plot HD95 vs bitrate
    fig, axes = plt.subplots(1, len(codecs), figsize=(5 * len(codecs), 4.5), sharey=True)
    if len(codecs) == 1:
        axes = [axes]
    for ax, codec in zip(axes, codecs):
        for model, summary in sorted(summaries.items()):
            pts = [s for s in summary if s["codec"] == codec and np.isfinite(s["mbps_mean"]) and np.isfinite(s["hd95_mean"])]
            if not pts:
                continue
            pts.sort(key=lambda x: x["mbps_mean"])
            xs = [p["mbps_mean"] for p in pts]
            ys = [p["hd95_mean"] for p in pts]
            style = MODEL_STYLES.get(model, {"linestyle": "-", "marker": "x", "alpha": 0.7, "linewidth": 1.2})
            ax.plot(xs, ys, color=CODEC_COLORS[codec], label=model, **style)
        ax.set_xscale("log")
        ax.set_xlabel("Bitrate (Mbps, log scale)")
        ax.set_ylabel("HD95 (mm)")
        ax.set_title(CODEC_DISPLAY.get(codec, codec))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Rate-task curve: HD95 vs bitrate per codec, per model")
    fig.tight_layout()
    out_hd95 = args.out_dir / "2026-05-31_phase3_hd95_vs_bitrate.png"
    fig.savefig(out_hd95, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_hd95}")

    # --- Safe bitrate threshold per (model, codec)
    print(f"\nSafe bitrate threshold (minimum bitrate at which mean Dice >= {args.dice_threshold}):")
    threshold_rows = []
    for model in sorted(summaries.keys()):
        for codec in codecs:
            sub = [s for s in summaries[model] if s["codec"] == codec]
            if not sub:
                continue
            res = find_safe_bitrate_threshold(sub, args.dice_threshold)
            if res is None:
                print(f"  {model:12s} {codec:12s}  NEVER crosses {args.dice_threshold} (max dice = {max(s['dice_mean'] for s in sub):.4f})")
                threshold_rows.append({"model": model, "codec": codec, "safe_mbps": "", "safe_crf": "", "max_dice": max(s["dice_mean"] for s in sub)})
            else:
                mbps, crf = res
                mbps_s = f"{mbps:.4f}" if mbps is not None and np.isfinite(mbps) else "?"
                print(f"  {model:12s} {codec:12s}  {mbps_s} Mbps @ CRF {crf}")
                threshold_rows.append({"model": model, "codec": codec, "safe_mbps": mbps_s, "safe_crf": crf, "max_dice": ""})

    out_thr = args.out_dir / "2026-05-31_phase3_safe_bitrate_thresholds.csv"
    with open(out_thr, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "codec", "safe_mbps", "safe_crf", "max_dice"])
        w.writeheader()
        w.writerows(threshold_rows)
    print(f"Wrote {out_thr}")


if __name__ == "__main__":
    main()
