#!/usr/bin/env python3
"""Standalone deliverable: AV1 cross-encoder ablation table (libsvtav1 vs
libaom-av1), as a clean PNG + markdown. Data from the 50-case comparison."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
rows=[
    ["18","0.851","0.847","+0.004","2.084","1.533"],
    ["23","0.850","0.845","+0.005","1.503","1.114"],
    ["28","0.849","0.844","+0.005","0.973","0.722"],
    ["33","0.841","0.840","+0.001","0.611","0.463"],
    ["38","0.831","0.836","-0.006","0.379","0.294"],
    ["43","0.832","0.830","+0.002","0.235","0.190"],
]
cols=["CRF","libsvtav1\nDice","libaom\nDice","Δ Dice","libsvtav1\nMbps","libaom\nMbps"]
foot="Max |Δ| matched-CRF = 0.006   |   mean Δ matched-bitrate = +0.0002   (50 calibration cases)"

fig,ax=plt.subplots(figsize=(7.2,2.4)); ax.axis("off")
t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center")
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.7)
for (r,c),cell in t.get_celld().items():
    if r==0:
        cell.set_facecolor("#e7eef7"); cell.set_text_props(fontweight="bold")
    cell.set_edgecolor("#999999")
ax.set_title("AV1 cross-encoder ablation: libsvtav1 vs libaom-av1",
             fontsize=11,fontweight="bold",pad=10)
fig.text(0.5,0.04,foot,ha="center",fontsize=8.5,style="italic")
png=OUT/"table_cross_encoder_2026-06-23.png"
fig.savefig(png,dpi=220,bbox_inches="tight"); print("wrote",png)

md=OUT/"table_cross_encoder_2026-06-23.md"
md.write_text(
"# AV1 cross-encoder ablation: libsvtav1 vs libaom-av1 (50 calibration cases)\n\n"
"| CRF | libsvtav1 Dice | libaom Dice | Δ Dice | libsvtav1 Mbps | libaom Mbps |\n"
"|---|---|---|---|---|---|\n"
+ "".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n" for r in rows)
+ "\nMax |Δ| matched-CRF = 0.006; mean Δ matched-bitrate = +0.0002. "
"The two AV1 encoders agree within 0.006 Dice at every operating point, so the "
"rate-task result is encoder-implementation-independent.\n")
print("wrote",md)
