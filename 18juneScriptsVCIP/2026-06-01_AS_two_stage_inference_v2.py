#!/usr/bin/env python3
"""
2026-06-01_AS_two_stage_inference_v2.py
---------------------------------------
v2 of the two-stage inference experiment — fixes two issues with v1:

  1. v1 called predict_from_files() inside a per-case loop → 299× worker-pool
     spawn → BrokenPipeError + multiprocessing fragility.
  2. v1 ran on login shell → Willi cluster watchdog SIGTERMed the GPU process.

v2 architecture (batch mode):
  Phase A (CPU): for each case, load Stage-1 prediction, compute bounding
    box (+margin), crop the compressed image, save the crop to a staging
    dir with nnUNet's expected naming convention (case_id_0000.nii.gz).
    Save bbox metadata to a sidecar JSON.

  Phase B (GPU, ONE invocation): call nnUNetv2_predict subprocess on the
    staging dir → fills crops_pred dir with predictions in cropped geometry.

  Phase C (CPU): for each case, read the cropped prediction, paste back into
    the original volume coordinates using the saved bbox, score Dice + HD95
    against GT.

Must run via sbatch (mitarb or student MIG) — NOT login shell.

CLI is the same as v1; an sbatch wrapper at
2026-06-01_AS_two_stage_inference_mitarb_sbatch.sh loops over 8 cells × 2
stage2 models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ------------------------------------------------------------ scoring helpers


def bbox_of_mask(arr: np.ndarray, margin: int = 16) -> tuple[slice, slice, slice] | None:
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return None
    z0, z1 = int(nz[0].min()), int(nz[0].max())
    y0, y1 = int(nz[1].min()), int(nz[1].max())
    x0, x1 = int(nz[2].min()), int(nz[2].max())
    z0 = max(z0 - margin, 0); z1 = min(z1 + margin, arr.shape[0] - 1)
    y0 = max(y0 - margin, 0); y1 = min(y1 + margin, arr.shape[1] - 1)
    x0 = max(x0 - margin, 0); x1 = min(x1 + margin, arr.shape[2] - 1)
    return (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))


def dice_hd95(pred: np.ndarray, gt: np.ndarray, ref_img: sitk.Image) -> tuple[float, float]:
    classes = sorted([int(c) for c in np.unique(gt) if c != 0])
    if not classes:
        return (1.0 if pred.sum() == 0 else 0.0, 0.0)
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
        gi = sitk.GetImageFromArray(g_mask); gi.CopyInformation(ref_img)
        pi = sitk.GetImageFromArray(p_mask); pi.CopyInformation(ref_img)
        dg = sitk.Abs(sitk.SignedMaurerDistanceMap(gi, squaredDistance=False, useImageSpacing=True))
        dp = sitk.Abs(sitk.SignedMaurerDistanceMap(pi, squaredDistance=False, useImageSpacing=True))
        cg = sitk.LabelContour(gi); cp = sitk.LabelContour(pi)
        apg = sitk.GetArrayFromImage(dg)[sitk.GetArrayFromImage(cp) == 1]
        agp = sitk.GetArrayFromImage(dp)[sitk.GetArrayFromImage(cg) == 1]
        allv = np.concatenate([apg, agp])
        hds.append(float(np.percentile(allv, 95)) if len(allv) else 0.0)
    return float(np.mean(dices)), float(np.mean(hds))


# ------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_pred_dir", type=Path, required=True)
    ap.add_argument("--imagesTs_dir", type=Path, required=True)
    ap.add_argument("--stage2_dataset_id", type=int, required=True,
                    help="nnUNet dataset id for the stage-2 model (501 for Cal1, 504 for d504)")
    ap.add_argument("--stage2_trainer", required=True,
                    help="Trainer class name, e.g. nnUNetTrainerCodecAug505_v2_Calibration or nnUNetTrainer")
    ap.add_argument("--stage2_chk", default="checkpoint_final.pth",
                    help="checkpoint filename inside fold_0/ (checkpoint_best.pth for 504)")
    ap.add_argument("--stage2_nnunet_results", type=Path, required=True,
                    help="Path to set as nnUNet_results env var (e.g. /scratch/shirzarm/vcip/d504_ckpt)")
    ap.add_argument("--gt_dir", type=Path, required=True)
    ap.add_argument("--folder_label", required=True)
    ap.add_argument("--margin_voxels", type=int, default=16)
    ap.add_argument("--out_pred_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--staging_dir", type=Path, default=None,
                    help="Where to put crops + stage-2 outputs (default: tempdir)")
    args = ap.parse_args()

    args.out_pred_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.staging_dir is None:
        staging_root = Path(tempfile.mkdtemp(prefix=f"twostage_{args.folder_label}_"))
    else:
        staging_root = args.staging_dir
        staging_root.mkdir(parents=True, exist_ok=True)

    crops_in_dir = staging_root / "crops_in"
    crops_pred_dir = staging_root / "crops_pred"
    crops_in_dir.mkdir(exist_ok=True)
    crops_pred_dir.mkdir(exist_ok=True)
    bbox_json = staging_root / "bboxes.json"

    print(f"Staging dir: {staging_root}")

    # ---------- Phase A: build crops + sidecar metadata ----------
    print("\n=== Phase A: building crops ===")
    stage1_files = sorted(p for p in args.stage1_pred_dir.iterdir() if p.name.endswith(".nii.gz"))
    print(f"Stage-1 predictions found: {len(stage1_files)}")

    bboxes = {}        # case_id -> bbox tuple (for paste-back)
    case_meta = {}     # case_id -> dict(stage1_dice, ref_image_path, fallback)

    for i, stage1_path in enumerate(stage1_files, 1):
        case_id = stage1_path.name.replace(".nii.gz", "")
        image_path = args.imagesTs_dir / f"{case_id}_0000.nii.gz"
        gt_path = args.gt_dir / f"{case_id}.nii.gz"
        if not image_path.exists() or not gt_path.exists():
            continue

        try:
            stage1_img = sitk.ReadImage(str(stage1_path))
            stage1_arr = sitk.GetArrayFromImage(stage1_img)
            image_img = sitk.ReadImage(str(image_path))
            image_arr = sitk.GetArrayFromImage(image_img)
            gt_img = sitk.ReadImage(str(gt_path))
            gt_arr = sitk.GetArrayFromImage(gt_img)

            stage1_dice, _ = dice_hd95(stage1_arr, gt_arr, gt_img)

            bbox = bbox_of_mask(stage1_arr, margin=args.margin_voxels)
            if bbox is None:
                # Empty Stage-1 → mark as fallback (use Stage-1 directly later)
                case_meta[case_id] = {
                    "stage1_dice": stage1_dice,
                    "stage1_pred_path": str(stage1_path),
                    "image_path": str(image_path),
                    "gt_path": str(gt_path),
                    "fallback": True,
                }
                continue

            crop_arr = image_arr[bbox]
            origin = np.array(image_img.GetOrigin())
            spacing = np.array(image_img.GetSpacing())
            direction = np.array(image_img.GetDirection()).reshape(3, 3)
            offset_voxel = np.array([bbox[2].start, bbox[1].start, bbox[0].start])
            new_origin = origin + direction @ (offset_voxel * spacing)

            crop_img = sitk.GetImageFromArray(crop_arr)
            crop_img.SetOrigin(tuple(new_origin))
            crop_img.SetSpacing(tuple(spacing))
            crop_img.SetDirection(image_img.GetDirection())

            sitk.WriteImage(crop_img, str(crops_in_dir / f"{case_id}_0000.nii.gz"))

            bboxes[case_id] = [
                bbox[0].start, bbox[0].stop,
                bbox[1].start, bbox[1].stop,
                bbox[2].start, bbox[2].stop,
            ]
            case_meta[case_id] = {
                "stage1_dice": stage1_dice,
                "stage1_pred_path": str(stage1_path),
                "image_path": str(image_path),
                "gt_path": str(gt_path),
                "fallback": False,
            }
        except Exception as e:
            print(f"  ERROR Phase A {case_id}: {type(e).__name__}: {e}")
            continue

        if i % 50 == 0:
            print(f"  cropped {i}/{len(stage1_files)}")

    with open(bbox_json, "w") as f:
        json.dump({"bboxes": bboxes, "meta": case_meta}, f, indent=2)

    n_cropped = len(bboxes)
    n_fallback = sum(1 for m in case_meta.values() if m["fallback"])
    print(f"Phase A done: {n_cropped} crops written; {n_fallback} fallback (empty Stage-1)")

    # ---------- Phase B: ONE call to nnUNetv2_predict ----------
    print(f"\n=== Phase B: nnUNetv2_predict on {n_cropped} crops ===")
    env = os.environ.copy()
    env["nnUNet_results"] = str(args.stage2_nnunet_results)
    env["nnUNet_compile"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = [
        "nnUNetv2_predict",
        "-i", str(crops_in_dir),
        "-o", str(crops_pred_dir),
        "-d", str(args.stage2_dataset_id),
        "-c", "3d_fullres",
        "-f", "0",
        "-tr", args.stage2_trainer,
        "-p", "nnUNetPlans",
        "--disable_tta",
        "-device", "cuda",
        "-npp", "4",
        "-nps", "4",
        "-step_size", "0.7",
        "--continue_prediction",
        "-chk", args.stage2_chk,
    ]
    print("  CMD:", " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"ERROR: nnUNetv2_predict exited with code {res.returncode}")
        return

    # ---------- Phase C: paste back + score ----------
    print(f"\n=== Phase C: paste back + score ===")
    write_header = not (args.append and args.out_csv.exists())
    mode = "a" if args.append else "w"
    f_csv = open(args.out_csv, mode, newline="")
    w_csv = csv.writer(f_csv)
    if write_header:
        w_csv.writerow(["folder", "case_id", "dice_avg", "hd95_avg",
                        "dice_class_1", "stage1_dice", "stage2_localization_used",
                        "bbox_z0z1y0y1x0x1"])

    n = 0; two_stage_sum = 0.0; stage1_sum = 0.0
    for case_id, meta in case_meta.items():
        try:
            stage1_img = sitk.ReadImage(meta["stage1_pred_path"])
            stage1_arr = sitk.GetArrayFromImage(stage1_img)
            gt_img = sitk.ReadImage(meta["gt_path"])
            gt_arr = sitk.GetArrayFromImage(gt_img)

            if meta["fallback"]:
                final_arr = stage1_arr
                bbox_str = ""
                stage2_used = 0
            else:
                bb = bboxes[case_id]
                bbox = (slice(bb[0], bb[1]), slice(bb[2], bb[3]), slice(bb[4], bb[5]))
                stage2_path = crops_pred_dir / f"{case_id}.nii.gz"
                if not stage2_path.exists():
                    print(f"  WARN: no stage-2 prediction for {case_id}; using stage-1")
                    final_arr = stage1_arr
                    bbox_str = ""
                    stage2_used = 0
                else:
                    stage2_crop_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(stage2_path)))
                    final_arr = np.zeros_like(stage1_arr)
                    if stage2_crop_arr.shape == final_arr[bbox].shape:
                        final_arr[bbox] = stage2_crop_arr
                        bbox_str = f"{bb[0]},{bb[1]},{bb[2]},{bb[3]},{bb[4]},{bb[5]}"
                        stage2_used = 1
                    else:
                        print(f"  WARN: shape mismatch {case_id}; using stage-1")
                        final_arr = stage1_arr
                        bbox_str = ""
                        stage2_used = 0

            final_img = sitk.GetImageFromArray(final_arr)
            final_img.CopyInformation(stage1_img)
            sitk.WriteImage(final_img, str(args.out_pred_dir / f"{case_id}.nii.gz"))

            two_stage_dice, two_stage_hd95 = dice_hd95(final_arr, gt_arr, gt_img)
            w_csv.writerow([args.folder_label, case_id, f"{two_stage_dice:.10f}",
                            f"{two_stage_hd95:.10f}", f"{two_stage_dice:.10f}",
                            f"{meta['stage1_dice']:.10f}", str(stage2_used), bbox_str])

            two_stage_sum += two_stage_dice
            stage1_sum += meta["stage1_dice"]
            n += 1
        except Exception as e:
            print(f"  ERROR Phase C {case_id}: {type(e).__name__}: {e}")
            continue

    f_csv.close()
    print(f"\nDone. Scored {n} cases.")
    print(f"  Two-stage mean Dice:  {two_stage_sum/n:.4f}")
    print(f"  Stage-1   mean Dice:  {stage1_sum/n:.4f}")
    print(f"  Δ two-stage vs stage-1: {(two_stage_sum - stage1_sum)/n:+.4f}")
    print(f"  Output CSV:   {args.out_csv}")
    print(f"  Output preds: {args.out_pred_dir}")
    print(f"  Staging dir kept at: {staging_root}  (delete manually if not debugging)")


if __name__ == "__main__":
    main()
