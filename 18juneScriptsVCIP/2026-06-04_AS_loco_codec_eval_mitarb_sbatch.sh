#!/bin/bash
# 2026-06-04_AS_loco_codec_eval_mitarb_sbatch.sh
# ----------------------------------------------
# Phase C: run a single LOCO checkpoint (Exp A | B | C) against all 18 cells
# (3 codecs × 6 CRFs) on Willi-local imagesTs, producing per-case predictions.
# Submit one job per Exp, ideally with --dependency=afterok:<training_jobid>
# so each codec eval kicks off the moment its training completes.
#
# Reads: /scratch/shirzarm/vcip/phase3_codecs/<cell>/imagesTs/   (Willi local, all 18 cells)
# Reads: /scratch/shirzarm/vcip/nnUNet_results/Dataset505_LOCO_<EXP>/fold_0/checkpoint_final.pth
# Writes: /scratch/shirzarm/vcip/loco_eval/<EXP>/<cell>/predictions/
#
# Throughput: nnUNetv2_predict on full A100 mitarb ≈ 2-3 s/case (TTA off via
# --disable_tta). 18 cells × 299 cases × 2.5 s ≈ 3.7 h per Exp + 18 startup
# overheads × ~20 s ≈ 6 min. Total ~4 h per Exp. --time=08:00:00 is safe pad.
#
# Submit:
#   sbatch --dependency=afterok:1721 --export=EXP=B 2026-06-04_AS_loco_codec_eval_mitarb_sbatch.sh
#   sbatch --dependency=afterok:1722 --export=EXP=C 2026-06-04_AS_loco_codec_eval_mitarb_sbatch.sh
#   sbatch --dependency=afterok:1737 --export=EXP=A 2026-06-04_AS_loco_codec_eval_mitarb_sbatch.sh

#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=vcip_loco_eval
#SBATCH --output=/scratch/shirzarm/vcip/logs/loco_eval_%x_%j.log
#SBATCH --error=/scratch/shirzarm/vcip/logs/loco_eval_%x_%j.err

set -euo pipefail

mkdir -p /scratch/shirzarm/vcip/logs

EXP="${EXP:?Must set --export=EXP=A|B|C}"

case "$EXP" in
    A) HELD_OUT=libx264   ;;
    B) HELD_OUT=libx265   ;;
    C) HELD_OUT=libsvtav1 ;;
    *) echo "ERROR: bad EXP=$EXP"; exit 1 ;;
esac

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Mode:          LOCO codec eval"
echo "Exp:           $EXP (held out: $HELD_OUT)"
echo "Started:       $(date)"
echo "=========================================="

source /scratch/shirzarm/.conda/etc/profile.d/conda.sh 2>/dev/null || \
  source $(conda info --base)/etc/profile.d/conda.sh
conda activate prostate_phase2

export nnUNet_results=/scratch/shirzarm/vcip/nnUNet_results
export nnUNet_compile=0

echo "=== GPU diagnostics ==="
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# Verify the LOCO ckpt exists (snapshotted by the training sbatch)
CKPT_DIR=$nnUNet_results/Dataset505_LOCO_${EXP}/nnUNetTrainerCodecAug505_v2_LOCO__nnUNetPlans__3d_fullres
test -f "$CKPT_DIR/fold_0/checkpoint_final.pth" || {
    echo "ERROR: LOCO ckpt missing at $CKPT_DIR/fold_0/checkpoint_final.pth"
    echo "Training job must have completed and snapshotted first."
    exit 1
}
echo "Found LOCO ckpt for Exp $EXP"

# nnUNetv2_predict expects model dir laid out per nnUNet convention.
# The snapshot has the right structure already. Set nnUNet_results so predict
# resolves Dataset505_LOCO_$EXP/<trainer>/fold_0.

CELLS=(
    "libx264_crf18"   "libx264_crf23"   "libx264_crf28"
    "libx264_crf33"   "libx264_crf38"   "libx264_crf43"
    "libx265_crf18"   "libx265_crf23"   "libx265_crf28"
    "libx265_crf33"   "libx265_crf38"   "libx265_crf43"
    "libsvtav1_crf18" "libsvtav1_crf23" "libsvtav1_crf28"
    "libsvtav1_crf33" "libsvtav1_crf38" "libsvtav1_crf43"
)

OUT_BASE=/scratch/shirzarm/vcip/loco_eval/${EXP}
mkdir -p "$OUT_BASE"

for cell in "${CELLS[@]}"; do
    IMGS_DIR=/scratch/shirzarm/vcip/phase3_codecs/${cell}/imagesTs
    OUT_DIR=${OUT_BASE}/${cell}/predictions

    if [ ! -d "$IMGS_DIR" ]; then
        echo "SKIP $cell — imagesTs missing at $IMGS_DIR (should be Willi-local from 2026-06-02 cp)"
        continue
    fi

    mkdir -p "$OUT_DIR"

    # Skip if all 299 preds already exist (resume safety)
    if [ "$(ls "$OUT_DIR" 2>/dev/null | grep -c '.nii.gz$' || echo 0)" -eq 299 ]; then
        echo "SKIP $cell — 299 preds already exist at $OUT_DIR"
        continue
    fi

    echo "=========================================="
    echo "$(date)  Exp $EXP  cell $cell"
    echo "  imgs: $IMGS_DIR"
    echo "  out:  $OUT_DIR"
    echo "=========================================="

    nnUNetv2_predict \
        -i "$IMGS_DIR" \
        -o "$OUT_DIR" \
        -d Dataset505_LOCO_${EXP} \
        -c 3d_fullres \
        -f 0 \
        -tr nnUNetTrainerCodecAug505_v2_LOCO \
        -p nnUNetPlans \
        --disable_tta \
        -device cuda \
        -npp 4 -nps 4 \
        -step_size 0.7 \
        --continue_prediction \
        -chk checkpoint_final.pth || {
            echo "ERROR predict failed for $cell — continuing to next"
            continue
        }
done

echo "=========================================="
echo "LOCO Exp $EXP codec eval finished: $(date)"
echo "Predictions tree: $OUT_BASE"
echo "Next: score predictions to CSV using 2026-06-04_AS_loco_score_cpu.py"
echo "=========================================="
