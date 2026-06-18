#!/bin/bash
# 2026-06-10_AS_phase5_libaom_sbatch.sh
# v2 of 2026-06-09 — fix: prepend ffaom lib dir to LD_LIBRARY_PATH so the
# ffaom ffmpeg binary can find libaom.so inside a Slurm job (where conda
# activate prostate_phase2 overwrites LD_LIBRARY_PATH with its own libs,
# causing the ffaom binary's dynamic linker to miss libaom.so).
#
# Submit:
#   sbatch /scratch/shirzarm/vcip/scripts/2026-06-10_AS_phase5_libaom_sbatch.sh

#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=vcip_phase5_libaom
#SBATCH --output=/scratch/shirzarm/vcip/logs/phase5_libaom_%j.log
#SBATCH --error=/scratch/shirzarm/vcip/logs/phase5_libaom_%j.err

set -euo pipefail
mkdir -p /scratch/shirzarm/vcip/logs

ROOT=/scratch/shirzarm/vcip/phase5_libaom
IMAGESTR=/scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr
GT_DIR=/scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/labelsTr
SCRIPT=/scratch/shirzarm/vcip/scripts/2026-06-09_AS_phase5_libaom.py
CRFS=(18 23 28 33 38 43)
FFAOM_LIB=/scratch/shirzarm/.conda/envs/ffaom/lib

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Phase 5 libaom-av1 encoder ablation (v2)"
echo "Started:  $(date)"
echo "=========================================="

source /scratch/shirzarm/.conda/etc/profile.d/conda.sh 2>/dev/null || \
  source $(conda info --base)/etc/profile.d/conda.sh
conda activate prostate_phase2

export FFMPEG=/scratch/shirzarm/.conda/envs/ffaom/bin/ffmpeg
test -x "$FFMPEG" || { echo "ERROR: $FFMPEG not found — run: conda create -y -n ffaom -c conda-forge ffmpeg aom"; exit 1; }

# Prepend ffaom's lib dir so the ffmpeg binary finds libaom.so even though
# prostate_phase2 is the active env (its LD_LIBRARY_PATH doesn't include ffaom libs).
export LD_LIBRARY_PATH="${FFAOM_LIB}:${LD_LIBRARY_PATH:-}"

# Confirm libaom-av1 is available before doing anything.
# IMPORTANT: capture ffmpeg output to a variable BEFORE grepping. Under
# `set -o pipefail`, `ffmpeg ... | grep -q` lets grep close the pipe on first
# match -> ffmpeg gets SIGPIPE (exit 141) -> pipefail propagates 141 as the
# pipeline status -> the `|| exit 1` fires even though grep MATCHED. (Same bug
# that silently killed jobs 1710-1717.) Command substitution lets ffmpeg run to
# completion, so no SIGPIPE.
ENC_LIST="$("$FFMPEG" -hide_banner -encoders 2>/dev/null || true)"
if ! printf '%s\n' "$ENC_LIST" | grep -q libaom-av1; then
    echo "ERROR: libaom-av1 not in $FFMPEG encoders"
    echo "Encoders (av1 lines):"
    printf '%s\n' "$ENC_LIST" | grep -i av1 || true
    exit 1
fi
echo "libaom-av1 confirmed OK"

nvidia-smi -L

# --- 1) ENCODE (CPU, resumable — skips already-encoded volumes)
echo "=== STEP 1: encode subset with libaom-av1 ==="
python3 "$SCRIPT" encode --imagestr "$IMAGESTR" --root "$ROOT" --workers 8

# --- 2) BASELINE INFERENCE per CRF (GPU)
export nnUNet_results=/scratch/shirzarm/vcip/baseline_ckpt
export nnUNet_compile=0
echo "=== STEP 2: baseline inference per CRF ==="
for crf in "${CRFS[@]}"; do
    IMGS="$ROOT/libaom_av1_crf${crf}/imagesTs"
    OUT="$ROOT/libaom_av1_crf${crf}/predictions"
    [ -d "$IMGS" ] || { echo "SKIP crf$crf: no imagesTs"; continue; }
    mkdir -p "$OUT"
    n=$(ls "$OUT" 2>/dev/null | grep -c '.nii.gz$' || echo 0)
    have=$(ls "$IMGS" 2>/dev/null | grep -c '_0000.nii.gz$' || echo 0)
    if [ "$n" -ge "$have" ] && [ "$have" -gt 0 ]; then
        echo "SKIP crf$crf: $n preds already (>= $have inputs)"; continue
    fi
    echo "--- $(date) baseline predict libaom_av1_crf$crf ($have inputs) ---"
    nnUNetv2_predict -i "$IMGS" -o "$OUT" \
        -d Dataset501_ProstateUS -c 3d_fullres -f 0 \
        -tr nnUNetTrainer -p nnUNetPlans \
        --disable_tta -device cuda -npp 4 -nps 4 -step_size 0.7 \
        --continue_prediction -chk checkpoint_final.pth || {
            echo "ERROR predict failed crf$crf — continuing"; continue; }
done

# --- 3) SCORE (CPU)
echo "=== STEP 3: score ==="
python3 "$SCRIPT" score --gt_dir "$GT_DIR" --root "$ROOT" --workers 8

echo "=========================================="
echo "Phase 5 libaom-av1 finished: $(date)"
echo "Bitrate: $ROOT/2026-06-09_phase5_libaom_bitrate.csv"
echo "Scored:  $ROOT/2026-06-09_phase5_libaom_scored.csv"
echo "=========================================="
