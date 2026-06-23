#!/usr/bin/env python3
"""Figure 2 FINAL (clean, authoritative): per-codec rate-task.
Dice left (Baseline solid, Cal1 dashed); HD95 right axis inverted, OPAQUE
dash-dot (baseline blue, cal1 red); libaom-av1 cross-encoder overlay on the AV1
panel only (green dotted). NO two-stage (removed from the VCIP submission).
Output: vcip_latex/fig_ratetask3_final.png"""
from pathlib import Path
from collections import defaultdict
import csv, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"vcip_latex/fig_ratetask3_final.png"
CRFS=[18,23,28,33,38,43]; CODECS=["libx264","libx265","libsvtav1"]
CLABEL={"libx264":"H.264","libx265":"HEVC","libsvtav1":"AV1"}

df=pd.read_csv(ROOT/"vcip_phase3/6model/2026-06-10_6model_per_cell_summary.csv")
df["crf"]=df["crf"].astype(int)
def cell(model,codec,crf,col):
    r=df[(df.model==model)&(df.codec==codec)&(df.crf==crf)]
    return float(r[col].iloc[0]) if len(r) else np.nan

# libaom-av1 baseline (50-case subset): mean dice + bitrate per CRF
ad=defaultdict(list)
for r in csv.DictReader(open(ROOT/"vcip_phase3/libaom/2026-06-09_phase5_libaom_scored.csv")):
    ad[int(r['folder'].split('crf')[1])].append(float(r['dice_avg']))
ab=defaultdict(list)
for r in csv.DictReader(open(ROOT/"vcip_phase3/libaom/2026-06-09_phase5_libaom_bitrate.csv")):
    ab[int(r['crf'])].append(float(r['avg_mbps']))
aom=sorted((np.mean(ab[c]),np.mean(ad[c])) for c in ad)

fig,axes=plt.subplots(1,3,figsize=(12.5,2.7),sharey=True)
for ax,codec in zip(axes,CODECS):
    ax2=ax.twinx()
    if codec==CODECS[0]: ax2_first=ax2
    xb=[cell("baseline",codec,c,"mbps_mean") for c in CRFS]
    bl=ax.plot(xb,[cell("baseline",codec,c,"dice_mean") for c in CRFS],"-o",
            color="#1f3b8c",lw=1.9,ms=5,zorder=5,
            label="Baseline: AV1 libsvtav1 (Dice)" if codec=="libsvtav1" else "Baseline (Dice)")
    ax.plot(xb,[cell("cal1",codec,c,"dice_mean") for c in CRFS],"--s",
            color="#c0392b",lw=1.7,ms=4,label="Comp.-aware fine-tune (Dice)")
    ax2.plot(xb,[cell("baseline",codec,c,"hd95_mean") for c in CRFS],"-.",
             color="#2a6db0",lw=1.6,alpha=1.0,label="Baseline (HD95)")
    ax2.plot(xb,[cell("cal1",codec,c,"hd95_mean") for c in CRFS],"-.",
             color="#e08a2a",lw=1.6,alpha=1.0,label="Comp.-aware fine-tune (HD95)")
    if codec=="libsvtav1":
        al=ax.plot([p[0] for p in aom],[p[1] for p in aom],":^",color="#0a8f0a",
                lw=2.4,ms=10,markeredgecolor="white",markeredgewidth=0.7,zorder=6,
                label="AV1 libaom enc. (Dice)")
        ax.legend([bl[0],al[0]],["Baseline, libsvtav1 enc.","Baseline, libaom enc."],
                  loc="lower left",fontsize=6.8,frameon=True,framealpha=0.92,
                  title="AV1 encoders")
    ax.set_xscale("log"); ax.set_ylim(0.5,1.0); ax2.set_ylim(14,3.5)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", ls=":", lw=0.6, color="#bbbbbb", alpha=0.9)
    ax.grid(True, which="minor", ls=":", lw=0.4, color="#dddddd", alpha=0.7)
    ax.set_title(CLABEL[codec],fontsize=10)
    ax.set_xlabel("Achieved bitrate (Mbps, log)",fontsize=8)
    ax.axhline(0.84,color="gray",ls=":",lw=0.8,alpha=0.6)
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    if codec==CODECS[0]: ax.set_ylabel("Dice",fontsize=9)
    if codec==CODECS[-1]: ax2.set_ylabel("HD95 (mm, inverted)",fontsize=9)
# combined legend (Dice lines from ax + HD95 lines from ax2 + libaom) for ALL line types
hD,lD = axes[0].get_legend_handles_labels()        # Baseline (Dice), Comp.-aware fine-tune (Dice)
hH,lH = ax2_first.get_legend_handles_labels()       # Baseline (HD95), Comp.-aware fine-tune (HD95)
# libaom is named in the AV1 panel's own legend; keep it OUT of the global legend
# to avoid double-listing the same blue baseline line.
handles = hD + hH
labels  = lD + lH
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(rect=[0,0.02,1,1])
fig.savefig(OUT,dpi=220,bbox_inches="tight"); print("wrote",OUT)
