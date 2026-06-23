#!/usr/bin/env python3
"""Figure 3 (final): 2-row LOCO radar — top row Dice, bottom row HD95, one
column per codec (H.264/HEVC/AV1). Legible version matching the 2026-06-14
layout. Output: vcip_latex/fig_radar_2row_2026-06-22.png"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SUMMARY=ROOT/"vcip_phase3/6model/2026-06-10_6model_per_cell_summary.csv"
OUT=ROOT/"vcip_latex/fig_radar_2row_2026-06-22.png"
CRFS=[18,23,28,33,38,43]; CODECS=["libx264","libx265","libsvtav1"]
CLABEL={"libx264":"H.264","libx265":"HEVC","libsvtav1":"AV1"}
HELDOUT={"libx264":"loco_A","libx265":"loco_B","libsvtav1":"loco_C"}
LOCO_LABEL={"loco_A":"LOCO-A (held-out: H.264)","loco_B":"LOCO-B (held-out: HEVC)",
            "loco_C":"LOCO-C (held-out: AV1)"}
MODELS=["baseline","loco_A","loco_B","loco_C"]
MCOLOR={"baseline":"#000000","loco_A":"#c0392b","loco_B":"#2060c0","loco_C":"#2a9d2a"}
MSTYLE={"baseline":"-","loco_A":"--","loco_B":"-.","loco_C":":"}

df=pd.read_csv(SUMMARY); df["crf"]=df["crf"].astype(int)
def cell(model,codec,crf,col):
    r=df[(df.model==model)&(df.codec==codec)&(df.crf==crf)]
    return float(r[col].iloc[0]) if len(r) else np.nan

ang=np.linspace(0,2*np.pi,len(CRFS),endpoint=False).tolist(); ang+=ang[:1]
fig,axes=plt.subplots(2,3,figsize=(10.5,7.6),subplot_kw=dict(polar=True),gridspec_kw=dict(wspace=0.0,hspace=0.34))

def panel(ax,codec,metric,rlim):
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([f"CRF {c}" for c in CRFS],fontsize=9)
    ax.set_ylim(*rlim); ax.tick_params(axis="y",labelsize=11)   # bigger ring (radial) numbers
    for m in MODELS:
        vals=[cell(m,codec,c,metric) for c in CRFS]; vals+=vals[:1]
        held=(HELDOUT[codec]==m)
        ax.plot(ang,vals,color=MCOLOR[m],linestyle=MSTYLE[m],
                lw=2.2 if held else 1.4,marker="*" if held else "o",
                markersize=11 if held else 3.5,zorder=4 if held else 2)

for j,codec in enumerate(CODECS):
    panel(axes[0,j],codec,"dice_mean",(0.60,0.90))
    axes[0,j].set_title(f"Dice — {CLABEL[codec]}",fontsize=10,pad=14,fontweight="bold")
    panel(axes[1,j],codec,"hd95_mean",(3.5,16))
    axes[1,j].set_title(f"HD95 (mm) — {CLABEL[codec]}",fontsize=10,pad=14,fontweight="bold")

handles=[plt.Line2D([0],[0],color=MCOLOR[m],linestyle=MSTYLE[m],
         marker="*" if m!="baseline" else "o",markersize=8,
         label=("Baseline" if m=="baseline" else LOCO_LABEL[m])) for m in MODELS]
handles.append(plt.Line2D([0],[0],color="gray",marker="*",linestyle="None",
               markersize=10,label="held-out codec (this panel)"))
fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=8.5,frameon=False,
           bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Leave-One-Codec-Out generalization across the 6-CRF grid",
             fontsize=11,y=1.0)
fig.subplots_adjust(left=0.02,right=0.99,top=0.90,bottom=0.08,wspace=0.0,hspace=0.34)
fig.savefig(OUT,dpi=200,bbox_inches="tight"); print("wrote",OUT)
