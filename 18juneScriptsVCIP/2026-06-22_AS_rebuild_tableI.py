#!/usr/bin/env python3
"""Rebuild matched-bitrate Table I (Dice/5pct/SSIM) with the new AV1 cells,
and report the refined AV1 safe-bitrate."""
import csv, numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

base = defaultdict(dict)
for r in csv.DictReader(open(ROOT/'vcip_phase3/6model/2026-06-10_6model_per_cell_summary.csv')):
    if r['model'] != 'baseline':
        continue
    base[r['codec']][int(r['crf'])] = {
        'dice': float(r['dice_mean']), 'd5': float(r['dice_5pct']),
        'mbps': float(r['mbps_mean'])}

ssim = {
    ('libx264',18):0.95685,('libx264',23):0.91807,('libx264',28):0.86591,
    ('libx264',33):0.80360,('libx264',38):0.72670,('libx264',43):0.57746,
    ('libx265',18):0.95531,('libx265',23):0.91971,('libx265',28):0.87011,
    ('libx265',33):0.81108,('libx265',38):0.73287,('libx265',43):0.64272,
    ('libsvtav1',18):0.95748,('libsvtav1',23):0.94569,('libsvtav1',28):0.92827,
    ('libsvtav1',33):0.90727,('libsvtav1',38):0.88283,('libsvtav1',43):0.85733,
}

inf = defaultdict(list)
for r in csv.DictReader(open(ROOT/'vcip_phase3/2026-06-18_step4_av1_inference.csv')):
    inf[int(r['folder'].split('crf')[1])].append(float(r['dice_avg']))
for r in csv.DictReader(open(ROOT/'vcip_phase3/step3_5_av1/2026-06-11_step1_psnr_ssim_per_cell.csv')):
    crf = int(r['crf']); v = np.array(inf[crf])
    base['libsvtav1'][crf] = {'dice': v.mean(), 'd5': float(np.percentile(v,5)),
                              'mbps': float(r['mean_bitrate_mbps'])}
    ssim[('libsvtav1',crf)] = float(r['mean_ssim'])

def closest(codec, ref):
    cells = base[codec]
    crf = min(cells, key=lambda c: abs(cells[c]['mbps']-ref))
    d = cells[crf]
    return crf, d['mbps'], d['dice'], d['d5'], ssim.get((codec,crf), float('nan'))

refs = [2.0, 1.0, 0.6, 0.35, 0.2]
print("=== Matched-bitrate (CRF, Mbps, Dice, 5%, SSIM) per codec ===")
for ref in refs:
    s = f"ref {ref:>4} "
    for c in ['libx264','libx265','libsvtav1']:
        crf,mb,dc,d5,ss = closest(c,ref)
        s += f"| {c[:3]} CRF{crf:>2} {mb:.3f} D{dc:.3f} 5{d5:.3f} S{ss:.3f} "
    print(s)

print("\n=== LaTeX rows (Ref & [CRF Mbps Dice 5% SSIM]x3) ===")
for ref in refs:
    parts = [f"{ref:.2g}"]
    for c in ['libx264','libx265','libsvtav1']:
        crf,mb,dc,d5,ss = closest(c,ref)
        parts += [str(crf), f"{mb:.3f}", f"{dc:.3f}", f"{d5:.3f}", f"{ss:.3f}"]
    print(" & ".join(parts) + r"\\")

# AV1 safe-bitrate refined
ok = [(d['mbps'],crf,d['dice']) for crf,d in base['libsvtav1'].items() if d['dice']>=0.84]
mb,crf,dc = min(ok)
print(f"\nAV1 safe-bitrate refined: {mb:.4f} Mbps @ CRF{crf} (Dice {dc:.4f})  [was 0.611 @ CRF33]")
