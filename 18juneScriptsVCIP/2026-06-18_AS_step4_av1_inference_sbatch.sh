#!/bin/bash
# 2026-06-18_AS_step4_av1_inference_sbatch.sh
# ------------------------------------------------
# Baseline rate-task inference across the 5 new AV1 VCIP cells.
# Cells: libsvtav1 CRF 30, 35, 36, 37, 41
#
# What this sbatch does for each missing cell:
#   1. If predictions_fold0_baseline_2026_05_27/ exists with 299 .nii.gz
#      files → SKIP inference, score directly.
#   2. Else → run nnUNetv2_predict (TTA off, baseline ckpt).
#   3. Score every case: compute Dice + HD95 vs GT, append to combined CSV
#
# Output CSV:
#   /scratch/shirzarm/vcip/refs/2026-06-18_step4_av1_inference.csv

#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=vcip_av1_infer
#SBATCH --output=/scratch/shirzarm/vcip/logs/av1_infer_%j.log
#SBATCH --error=/scratch/shirzarm/vcip/logs/av1_infer_%j.err

set -euo pipefail

mkdir -p /scratch/shirzarm/vcip/logs /scratch/shirzarm/vcip/refs

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Started:   $(date)"
echo "Partition: mitarb (full A100 SXM4 40GB)"
echo "Purpose:   baseline inference for 5 new AV1 CRFs"
echo "TTA:       OFF"
echo "=========================================="

source /scratch/shirzarm/.conda/etc/profile.d/conda.sh 2>/dev/null || \
  source $(conda info --base)/etc/profile.d/conda.sh
conda activate prostate_phase2

export nnUNet_compile=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.free --format=csv

LOCAL_BASE=/scratch/shirzarm/vcip/baseline_ckpt/Dataset501_ProstateUS/nnUNetTrainer__nnUNetPlans__3d_fullres
TWEETY_BASE=/scratch/shirzarm/tweety_mp/nnUNet_results/Dataset501_ProstateUS/nnUNetTrainer__nnUNetPlans__3d_fullres

mkdir -p "$LOCAL_BASE"
for f in dataset.json plans.json; do
    if [ ! -f "$LOCAL_BASE/$f" ] && [ -f "$TWEETY_BASE/$f" ]; then
        cp "$TWEETY_BASE/$f" "$LOCAL_BASE/$f"
        echo "Copied $f from Tweety to local."
    fi
done

test -f "$LOCAL_BASE/fold_0/checkpoint_final.pth" || { echo "ERROR: baseline ckpt missing"; exit 1; }
test -f "$LOCAL_BASE/dataset.json"                 || { echo "ERROR: dataset.json missing"; exit 1; }
test -f "$LOCAL_BASE/plans.json"                   || { echo "ERROR: plans.json missing"; exit 1; }

export nnUNet_results=/scratch/shirzarm/vcip/baseline_ckpt
echo "nnUNet_results: $nnUNet_results"

CELLS=(
    "libsvtav1 30"
    "libsvtav1 35"
    "libsvtav1 36"
    "libsvtav1 37"
    "libsvtav1 41"
)

OUT_CSV=/scratch/shirzarm/vcip/refs/2026-06-18_step4_av1_inference.csv

SCORE_PY=$(cat <<'PY'
import argparse, csv, os, sys
import numpy as np
import SimpleITK as sitk

ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir", required=True)
ap.add_argument("--gt_dir", required=True)
ap.add_argument("--folder_label", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--append", action="store_true")
args = ap.parse_args()

write_header = not (args.append and os.path.exists(args.out_csv))
mode = "a" if args.append else "w"

with open(args.out_csv, mode, newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["folder", "case_id", "dice_avg", "hd95_avg", "dice_class_1", "avg_mbps", "bits_per_voxel"])

    preds = sorted(p for p in os.listdir(args.pred_dir) if p.endswith(".nii.gz"))
    print(f"  scoring {args.folder_label}: {len(preds)} predictions")
    for i, pred_name in enumerate(preds, 1):
        case_id = pred_name.replace(".nii.gz", "")
        pred_path = os.path.join(args.pred_dir, pred_name)
        gt_path   = os.path.join(args.gt_dir, f"{case_id}.nii.gz")
        if not os.path.exists(gt_path):
            print(f"    skip {case_id}: GT missing")
            continue
        try:
            pred_img = sitk.ReadImage(pred_path)
            gt_img   = sitk.ReadImage(gt_path)
            pred = sitk.GetArrayFromImage(pred_img)
            gt   = sitk.GetArrayFromImage(gt_img)

            classes = sorted([int(c) for c in np.unique(gt) if c != 0])
            if not classes:
                dice = 1.0 if pred.sum() == 0 else 0.0
                hd95 = 0.0
            else:
                dices, hds = [], []
                for c in classes:
                    p_mask = (pred == c).astype(np.uint8)
                    g_mask = (gt   == c).astype(np.uint8)
                    inter  = int(np.logical_and(p_mask, g_mask).sum())
                    denom  = int(p_mask.sum() + g_mask.sum())
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
                dice = float(np.mean(dices))
                hd95 = float(np.mean(hds))
        except Exception as e:
            print(f"    error {case_id}: {e}")
            continue

        w.writerow([args.folder_label, case_id, dice, hd95, dice, "", ""])
        if i % 50 == 0:
            print(f"    {i}/{len(preds)}")

print(f"  done {args.folder_label}")
PY
)

GT_DIR=/scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/labelsTr

for cell in "${CELLS[@]}"; do
    read -r codec crf <<< "$cell"
    label="${codec}_crf${crf}"

    if   [ -d "/scratch/shirzarm/vcip/phase3_codecs/$label/imagesTs" ]; then
        IMGS="/scratch/shirzarm/vcip/phase3_codecs/$label/imagesTs"
    elif [ -d "/scratch/shirzarm/tweety_mp/phase3_codecs/$label/imagesTs" ]; then
        IMGS="/scratch/shirzarm/tweety_mp/phase3_codecs/$label/imagesTs"
    else
        echo "SKIP $label — imagesTs not found"
        continue
    fi

    n_input=$(ls "$IMGS" 2>/dev/null | grep -c '\.nii\.gz$' || echo 0)
    OUT_DIR="$(dirname "$IMGS")/predictions_fold0_baseline_2026_05_27"

    if [ -d "$OUT_DIR" ] && [ "$(ls "$OUT_DIR" 2>/dev/null | grep -c '\.nii\.gz$')" -ge "$n_input" ] && [ "$n_input" -gt 0 ]; then
        echo "$(date)  SKIP inference for $label  — $n_input predictions already at $OUT_DIR"
    else
        mkdir -p "$OUT_DIR"
        echo "=========================================="
        echo "$(date)  PREDICT  $label   n=$n_input   OUT=$OUT_DIR"
        echo "=========================================="
        nnUNetv2_predict \
            -i "$IMGS" \
            -o "$OUT_DIR" \
            -d 501 \
            -c 3d_fullres \
            -f 0 \
            -tr nnUNetTrainer \
            -p nnUNetPlans \
            --disable_tta \
            -device cuda \
            -npp 6 -nps 6 \
            -step_size 0.7 \
            --continue_prediction \
            -chk checkpoint_final.pth || { echo "ERROR: prediction failed for $label — skipping scoring"; continue; }
    fi

    echo "$(date)  SCORE   $label   pred_dir=$OUT_DIR"
    python3 -c "$SCORE_PY" \
        --pred_dir "$OUT_DIR" \
        --gt_dir   "$GT_DIR" \
        --folder_label "$label" \
        --out_csv  "$OUT_CSV" \
        --append
done

echo "=========================================="
echo "Inference and scoring finished: $(date)"
echo "Combined CSV: $OUT_CSV"
echo "=========================================="
