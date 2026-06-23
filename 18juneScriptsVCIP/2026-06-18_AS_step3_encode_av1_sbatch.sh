#!/bin/bash
# 2026-06-18_AS_step3_encode_av1_sbatch.sh
# ----------------------------------------
# CPU-only sbatch for encoding the 5 new AV1 CRF folders (30, 35, 36, 37, 41).
# No GPU requested — pure ffmpeg + SimpleITK.
#
# Submit:
#   sbatch /scratch/shirzarm/vcip/scripts/2026-06-18_AS_step3_encode_av1_sbatch.sh

#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=vcip_encode_av1
#SBATCH --output=/scratch/shirzarm/vcip/logs/encode_av1_%j.log
#SBATCH --error=/scratch/shirzarm/vcip/logs/encode_av1_%j.err

set -euo pipefail

mkdir -p /scratch/shirzarm/vcip/logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "----------------------------------------"

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
set -u

python /scratch/shirzarm/vcip/scripts/2026-05-24_AS_encode_missing_crfs.py \
    --workers "$SLURM_CPUS_PER_TASK" \
    --out_base /scratch/shirzarm/vcip/phase3_codecs \
    --jobs "libsvtav1:30,libsvtav1:35,libsvtav1:36,libsvtav1:37,libsvtav1:41"

echo "----------------------------------------"
echo "Finished: $(date)"
echo "Outputs live in /scratch/shirzarm/vcip/phase3_codecs/libsvtav1_crf<crf>/imagesTs/"
