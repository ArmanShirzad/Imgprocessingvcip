#!/bin/bash
# 2026-06-18_AS_step3_5_psnr_ssim_av1_sbatch.sh
# ---------------------------------------------------------------------------
# VCIP 2026 -- Alireza SSIM-harmonization STEP 3.5 (CPU only, no GPU).
# Measures mean PSNR + SSIM for the 5 new AV1 compressed cells vs originals.
# NO GPU requested: this is pure ffmpeg encode/decode + metric on CPU.
# ---------------------------------------------------------------------------
#SBATCH --job-name=vcip_step3_5_psnrssim
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shirzarm/vcip/logs/step3_5_psnrssim_%j.out
#SBATCH --error=/scratch/shirzarm/vcip/logs/step3_5_psnrssim_%j.err

set -euo pipefail

mkdir -p /scratch/shirzarm/vcip/logs /scratch/shirzarm/vcip/phase1_step3_5

set +u
CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [ ! -f "$CONDA_SH" ]; then
    for b in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.conda" \
             /opt/conda /scratch/shirzarm/miniconda3 /scratch/shirzarm/anaconda3; do
        if [ -f "$b/etc/profile.d/conda.sh" ]; then CONDA_SH="$b/etc/profile.d/conda.sh"; break; fi
    done
fi
if [ ! -f "$CONDA_SH" ]; then
    echo "FATAL: could not locate conda.sh (conda base unknown). PATH=$PATH"; exit 1
fi
echo "sourcing $CONDA_SH"
source "$CONDA_SH"
conda activate prostate_phase2 || { echo "FATAL: conda activate prostate_phase2 failed"; exit 1; }
echo "env active: python=$(which python)"
set -u

ENC_LIST="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
if ! printf '%s\n' "$ENC_LIST" | grep -q "libsvtav1"; then
    echo "ERROR: libsvtav1 not in ffmpeg encoders"; exit 1
fi

python /scratch/shirzarm/vcip/scripts/2026-06-11_AS_step1_psnr_ssim_18cells.py \
    --image_dir   /scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr \
    --splits_json /scratch/shirzarm/nnUNet_preprocessed/Dataset501_ProstateUS/splits_final.json \
    --out_dir     /scratch/shirzarm/vcip/phase1_step3_5 \
    --codecs      libsvtav1 \
    --crfs        30 35 36 37 41 \
    --jobs        8

echo "STEP 3.5 done. Table + CSVs in /scratch/shirzarm/vcip/phase1_step3_5/"
