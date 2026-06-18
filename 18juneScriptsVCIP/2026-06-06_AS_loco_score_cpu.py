#!/usr/bin/env python3
"""
2026-06-06_AS_loco_score_cpu.py
-------------------------------
Score LOCO codec-eval predictions on CPU (no GPU). Fork of
2026-05-31_AS_baseline_score_cpu.py adapted for the LOCO output layout.

Predictions written by 2026-06-04_AS_loco_codec_eval_mitarb_sbatch.sh live at:
  /scratch/shirzarm/vcip/loco_eval/<EXP>/<cell>/predictions/<case_id>.nii.gz

This scores ONE Exp (A|B|C) across whatever cells have predictions on disk
(handles partial cells — scores all .nii.gz present, warns if < 299). Run it
incrementally as codec eval progresses; re-run when more cells finish.

Output:
  /scratch/shirzarm/vcip/refs/2026-06-06_loco_<EXP>_completion.csv
  Schema matches the unified merger input:
    folder, case_id, dice_avg, hd95_avg, dice_class_1, avg_mbps, bits_per_voxel
  (folder = bare cell name e.g. "libx264_crf18" so the merger groups it correctly)

Run on Willi login shell (no sbatch needed):
  conda activate prostate_phase2
  for E in A B C; do
    python3 /scratch/shirzarm/vcip/scripts/2026-06-06_AS_loco_score_cpu.py \\
        --exp $E \\
        --gt_dir /scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/labelsTr \\
        --out_csv /scratch/shirzarm/vcip/refs/2026-06-06_loco_${E}_completion.csv \\
        --workers 8
  done

Time: ~1.5 s/case in parallel; full 18 cells × 299 ≈ 90 min per Exp on 8 workers.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def score_one_case(args):
    folder, case_id, pred_path, gt_path = args
    try:
        pred_img = sitk.ReadImage(str(pred_path))
        gt_img = sitk.ReadImage(str(gt_path))
        pred = sitk.GetArrayFromImage(pred_img)
        gt = sitk.GetArrayFromImage(gt_img)

        classes = sorted([int(c) for c in np.unique(gt) if c != 0])
        if not classes:
            return (folder, case_id, 1.0 if pred.sum() == 0 else 0.0, 0.0)

        dices, hds = [], []
        for c in classes:
            p_mask = (pred == c).astype(np.uint8)
            g_mask = (gt == c).astype(np.uint8)
            inter = int(np.logical_and(p_mask, g_mask).sum())
            denom = int(p_mask.sum() + g_mask.sum())
            dices.append(1.0 if denom == 0 else 2.0 * inter / denom)

            if p_mask.sum() == 0 and g_mask.sum() == 0:
                hds.append(0.0); continue
            if p_mask.sum() == 0 or g_mask.sum() == 0:
                hds.append(50.0); continue

            gi = sitk.GetImageFromArray(g_mask); gi.CopyInformation(gt_img)
            pi = sitk.GetImageFromArray(p_mask); pi.CopyInformation(pred_img)
            dist_g = sitk.Abs(sitk.SignedMaurerDistanceMap(gi, squaredDistance=False, useImageSpacing=True))
            dist_p = sitk.Abs(sitk.SignedMaurerDistanceMap(pi, squaredDistance=False, useImageSpacing=True))
            cg = sitk.LabelContour(gi); cp = sitk.LabelContour(pi)
            a_pg = sitk.GetArrayFromImage(dist_g)[sitk.GetArrayFromImage(cp) == 1]
            a_gp = sitk.GetArrayFromImage(dist_p)[sitk.GetArrayFromImage(cg) == 1]
            all_d = np.concatenate([a_pg, a_gp])
            hds.append(float(np.percentile(all_d, 95)) if len(all_d) else 0.0)
        return (folder, case_id, float(np.mean(dices)), float(np.mean(hds)))
    except Exception as e:
        print(f"  ERROR {folder}/{case_id}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["A", "B", "C"])
    ap.add_argument("--gt_dir", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--loco_eval_root", type=Path,
                    default=Path("/scratch/shirzarm/vcip/loco_eval"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    exp_root = args.loco_eval_root / args.exp
    if not exp_root.is_dir():
        raise SystemExit(f"No loco_eval dir for Exp {args.exp} at {exp_root}")

    # Discover all cells with a predictions/ dir
    tasks = []
    cell_dirs = sorted(d for d in exp_root.iterdir() if (d / "predictions").is_dir())
    for cdir in cell_dirs:
        cell = cdir.name  # e.g. libx264_crf18
        pdir = cdir / "predictions"
        pred_files = sorted(p for p in pdir.iterdir() if p.name.endswith(".nii.gz"))
        if not pred_files:
            continue
        if len(pred_files) < 299:
            print(f"WARN {cell}: only {len(pred_files)}/299 predictions (partial — scoring what exists)")
        for pred_path in pred_files:
            case_id = pred_path.name.replace(".nii.gz", "")
            gt_path = args.gt_dir / f"{case_id}.nii.gz"
            tasks.append((cell, case_id, pred_path, gt_path))
        print(f"queued {cell}: {len(pred_files)} cases")

    print(f"\nExp {args.exp}: {len(cell_dirs)} cells, {len(tasks)} total cases, {args.workers} workers")
    print(f"Output: {args.out_csv}\n")

    results = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(score_one_case, tasks), 1):
            if res is not None:
                results.append(res)
            if i % 100 == 0:
                print(f"  scored {i}/{len(tasks)}", flush=True)

    print(f"\nScored {len(results)} cases.")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "case_id", "dice_avg", "hd95_avg", "dice_class_1", "avg_mbps", "bits_per_voxel"])
        for folder, case_id, dice, hd in results:
            w.writerow([folder, case_id, f"{dice:.10f}", f"{hd:.10f}", f"{dice:.10f}", "", ""])
    print(f"Wrote {args.out_csv}")

    cells = defaultdict(list)
    for folder, _, dice, _ in results:
        cells[folder].append(dice)
    print(f"\nExp {args.exp} per-cell mean Dice:")
    for k in sorted(cells.keys()):
        v = cells[k]
        print(f"  {k:25s} n={len(v):3d}  mean={sum(v)/len(v):.4f}")


if __name__ == "__main__":
    main()
