#!/bin/bash
#SBATCH --job-name=vcip_step1_psnrssim
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shirzarm/vcip/logs/step1_psnrssim_%j.out
#SBATCH --error=/scratch/shirzarm/vcip/logs/step1_psnrssim_%j.err
# ---------------------------------------------------------------------------
# VCIP 2026 -- Alireza SSIM-harmonization STEP 1 (CPU only, no GPU).
# Measures mean PSNR + SSIM for all 18 existing compressed cells vs originals.
# Time padded ~2-3x over the ~2-3h estimate per feedback-slurm-time-padding.
# NO GPU requested: this is pure ffmpeg encode/decode + metric on CPU.
# ---------------------------------------------------------------------------
set -euo pipefail

mkdir -p /scratch/shirzarm/vcip/logs /scratch/shirzarm/vcip/phase1_step1

# ---- activate conda env (robust, self-diagnosing) -------------------------
# NOTE: /scratch/shirzarm/.conda is the ENVS dir, NOT the conda base -- that is
# why job 1858 died at line 21 ("conda.sh: No such file or directory"). Resolve
# the real base via `conda info --base` (proven to work in the libaom job 1831),
# then fall back to common install locations. Whole block runs under `set +u`
# so conda's init scripts don't trip nounset.
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

# verify ffmpeg has the needed encoders WITHOUT a SIGPIPE-prone pipeline
# (feedback-sigpipe-pipefail: capture to a var, then grep the var)
ENC_LIST="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
for enc in libx264 libx265 libsvtav1; do
    if ! printf '%s\n' "$ENC_LIST" | grep -q "$enc"; then
        echo "ERROR: $enc not in ffmpeg encoders"; exit 1
    fi
done
echo "ffmpeg encoders OK (libx264/libx265/libsvtav1)"

python /scratch/shirzarm/vcip/scripts/2026-06-11_AS_step1_psnr_ssim_18cells.py \
    --image_dir   /scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr \
    --splits_json /scratch/shirzarm/nnUNet_preprocessed/Dataset501_ProstateUS/splits_final.json \
    --out_dir     /scratch/shirzarm/vcip/phase1_step1 \
    --jobs 8

echo "STEP 1 done. Table + CSVs in /scratch/shirzarm/vcip/phase1_step1/"
