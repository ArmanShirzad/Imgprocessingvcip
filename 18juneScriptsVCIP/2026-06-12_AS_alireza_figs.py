#!/usr/bin/env python3
"""
2026-06-12_AS_alireza_figs.py
-----------------------------
Generate the two figures Alireza requested (2026-06-12 email):

CHANGE 1 -> fig_radar_2026-06-12.png
  Side-by-side radar blocks: left = Dice, right = HD95. Each block has 3 panels
  (one per codec H.264/HEVC/AV1). Spokes = CRF {18,23,28,33,38,43}. One polygon
  per model {Baseline, LOCO-A, LOCO-B, LOCO-C}; held-out codec marked with a star
  in the legend and a ring marker on that panel. Vertex values annotated.
  Replaces Table IV (LOCO matrix).

CHANGE 2 -> fig_ratetask3_2026-06-12.png
  3 panels (one per codec). Dual y-axis: Dice left (0.5-1.0), HD95 right INVERTED
  (lower=better plotted upward). x = achieved bitrate (Mbps, log scale). Lines:
  Baseline / Cal1 / Two-stage for Dice (solid/dashed/dotted) and HD95 (lighter).

Inputs (local repo):
  vcip_phase3/6model/2026-06-10_6model_per_cell_summary.csv
  vcip_phase3/2026-06-01_two_stage_cal1.csv
Outputs: vcip_latex/<name>.png
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "vcip_phase3/6model/2026-06-10_6model_per_cell_summary.csv"
TWOSTAGE = ROOT / "vcip_phase3/2026-06-01_two_stage_cal1.csv"
OUT = ROOT / "vcip_latex"

CRFS = [18, 23, 28, 33, 38, 43]
CODECS = ["libx264", "libx265", "libsvtav1"]
CLABEL = {"libx264": "H.264", "libx265": "HEVC", "libsvtav1": "AV1"}
# which LOCO model holds out which codec
HELDOUT = {"libx264": "loco_A", "libx265": "loco_B", "libsvtav1": "loco_C"}
LOCO_LABEL = {"loco_A": "LOCO-A (hold H.264)", "loco_B": "LOCO-B (hold HEVC)",
              "loco_C": "LOCO-C (hold AV1)"}
RADAR_MODELS = ["baseline", "loco_A", "loco_B", "loco_C"]
MCOLOR = {"baseline": "#1f3b8c", "loco_A": "#c0392b", "loco_B": "#27ae60", "loco_C": "#8e44ad"}
MSTYLE = {"baseline": "-", "loco_A": "--", "loco_C": ":", "loco_B": "-."}


def load_summary():
    df = pd.read_csv(SUMMARY)
    df["crf"] = df["crf"].astype(int)
    return df


def cell(df, model, codec, crf, col):
    r = df[(df.model == model) & (df.codec == codec) & (df.crf == crf)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ----------------------------------------------------------------- radar
def radar(df, metric, axes, title_suffix):
    """metric in {'dice_mean','hd95_mean'}; draw 3 codec panels onto axes list."""
    N = len(CRFS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    for ax, codec in zip(axes, CODECS):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"CRF {c}" for c in CRFS], fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        for model in RADAR_MODELS:
            vals = [cell(df, model, codec, c, metric) for c in CRFS]
            vals += vals[:1]
            held = (HELDOUT[codec] == model)
            ax.plot(angles, vals, color=MCOLOR[model], linestyle=MSTYLE[model],
                    linewidth=2.0 if held else 1.3, marker="*" if held else "o",
                    markersize=9 if held else 3, zorder=3 if held else 2)
            if held:  # annotate held-out vertices
                for ang, v in zip(angles[:-1], vals[:-1]):
                    ax.annotate(f"{v:.3f}", (ang, v), fontsize=5.5,
                                color=MCOLOR[model], ha="center", va="bottom")
        ax.set_title(f"{CLABEL[codec]}", fontsize=9, pad=12)
    axes[0].set_ylabel(title_suffix, fontsize=9, labelpad=22)


def make_radar(df):
    fig, axes = plt.subplots(1, 6, figsize=(14.5, 2.05),
                             subplot_kw=dict(polar=True))
    # left block Dice (panels 0-2), right block HD95 (panels 3-5)
    dice_ax = list(axes[:3])
    hd_ax = list(axes[3:])
    # dice radial range tight around 0.6-0.87
    for ax in dice_ax:
        ax.set_ylim(0.60, 0.88)
    radar(df, "dice_mean", dice_ax, "Dice")
    for ax in hd_ax:
        ax.set_ylim(3.5, 12)
    radar(df, "hd95_mean", hd_ax, "HD95 (mm)")
    # shared legend
    handles = [plt.Line2D([0], [0], color=MCOLOR[m], linestyle=MSTYLE[m],
               marker="*" if m.startswith("loco") else "o", markersize=7,
               label={"baseline": "Baseline"}.get(m, LOCO_LABEL.get(m, m)))
               for m in RADAR_MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Dice (left block)            HD95 (right block)", fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    out = OUT / "fig_radar_2026-06-12.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print("wrote", out)


# ------------------------------------------------------- two-stage cell means
def load_twostage():
    ts = pd.read_csv(TWOSTAGE)
    rows = []
    for folder, g in ts.groupby("folder"):
        m = re.match(r"(libx264|libx265|libsvtav1)_crf(\d+)_twostage", folder)
        if not m:
            continue
        rows.append({"codec": m.group(1), "crf": int(m.group(2)),
                     "dice": g["dice_avg"].mean(), "hd95": g["hd95_avg"].mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- rate-task 3 panel
def make_ratetask(df):
    ts = load_twostage()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.15), sharey=True)
    for ax, codec in zip(axes, CODECS):
        ax2 = ax.twinx()
        # x = baseline mean bitrate per crf (matched operating point)
        xb = [cell(df, "baseline", codec, c, "mbps_mean") for c in CRFS]
        # Dice (left axis)
        ax.plot(xb, [cell(df, "baseline", codec, c, "dice_mean") for c in CRFS],
                "-o", color="#1f77b4", lw=1.8, ms=4, label="Baseline")
        ax.plot(xb, [cell(df, "cal1", codec, c, "dice_mean") for c in CRFS],
                "--s", color="#d62728", lw=1.6, ms=4, label="Cal1")
        # HD95 (right axis, inverted)
        ax2.plot(xb, [cell(df, "baseline", codec, c, "hd95_mean") for c in CRFS],
                 "-", color="#7fbfe0", lw=1.4, alpha=0.9)
        ax2.plot(xb, [cell(df, "cal1", codec, c, "hd95_mean") for c in CRFS],
                 "--", color="#e88", lw=1.4, alpha=0.9)
        # two-stage where available
        tsd = ts[ts.codec == codec].sort_values("crf")
        if len(tsd):
            xt = [cell(df, "baseline", codec, c, "mbps_mean") for c in tsd.crf]
            ax.plot(xt, tsd.dice.values, ":^", color="#2ca02c", lw=1.8, ms=5,
                    label="Two-stage")
            ax2.plot(xt, tsd.hd95.values, ":", color="#9bd49b", lw=1.4, alpha=0.9)
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.0)
        ax2.set_ylim(14, 3.5)  # inverted: lower HD95 plots higher
        ax.set_title(CLABEL[codec], fontsize=10)
        ax.set_xlabel("Achieved bitrate (Mbps, log)", fontsize=8)
        ax.axhline(0.84, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
        if codec == CODECS[0]:
            ax.set_ylabel("Dice", fontsize=9)
        if codec == CODECS[-1]:
            ax2.set_ylabel("HD95 (mm, inverted)", fontsize=9)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    out = OUT / "fig_ratetask3_2026-06-12.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    df = load_summary()
    make_radar(df)
    make_ratetask(df)
