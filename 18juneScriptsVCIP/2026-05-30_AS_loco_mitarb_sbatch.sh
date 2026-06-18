#!/bin/bash
# 2026-05-30_AS_loco_mitarb_sbatch.sh
# -----------------------------------
# VCIP 2026 — LOCO Exp A/B/C training, running on the MITARB partition
# (full A100 SXM4 40GB, weekend slot per Alireza's 2026-05-29 email).
#
# Differences from 2026-05-23_AS_loco_sbatch.sh (the student MIG version):
#   - --partition=mitarb (was students)
#   - --account=mitarb   (NEW: required for mitarb access)
#   - --gres=gpu:mitarb:1 (was gpu:student:1)
#   - --mem=40G          (was 32G; mitarb has more)
#   - --time=08:00:00    (mitarb hard cap; student MIG was 16h)
#   - NO inline anti-504 guard call (won't fit in 8h with training, and we've
#     reframed away from clean-Dice ≥ 0.85 as the success metric anyway — per
#     project-vcip-success-metric memory)
#   - NO inline codec eval (separate sbatch handles that:
#     2026-05-30_AS_loco_codec_eval_mitarb_sbatch.sh — runs in ~30 min on A100
#     after training)
#
# Why split training and eval into separate slots:
#   On the full A100, 100-epoch LOCO training takes ~4-5h. Adding the 4h
#   anti-504 guard and/or codec eval would push past the 8h cap. Separate
#   sbatch keeps each job comfortably inside the slot.
#
# Submit one per experiment by setting EXP env var:
#   sbatch --export=EXP=A /scratch/shirzarm/vcip/scripts/2026-05-30_AS_loco_mitarb_sbatch.sh
#   sbatch --export=EXP=B /scratch/shirzarm/vcip/scripts/2026-05-30_AS_loco_mitarb_sbatch.sh
#   sbatch --export=EXP=C /scratch/shirzarm/vcip/scripts/2026-05-30_AS_loco_mitarb_sbatch.sh
#
# Exp A: train libx265 + libsvtav1, hold out libx264
# Exp B: train libx264 + libsvtav1, hold out libx265
# Exp C: train libx264 + libx265,    hold out libsvtav1
#
# After all 3 finish, run the codec-eval sbatch on each Dataset505_LOCO_<EXP>
# checkpoint to produce the rate-task numbers vs baseline + 504.

#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=vcip_loco_m
#SBATCH --output=/scratch/shirzarm/vcip/logs/loco_mitarb_%x_%j.log
#SBATCH --error=/scratch/shirzarm/vcip/logs/loco_mitarb_%x_%j.err

set -euo pipefail

mkdir -p /scratch/shirzarm/vcip/logs

EXP="${EXP:?Must set --export=EXP=A|B|C when submitting}"

declare -A HELD_OUT=(
  ["A"]="libx264"
  ["B"]="libx265"
  ["C"]="libsvtav1"
)
held_out="${HELD_OUT[$EXP]:?Unknown EXP $EXP}"

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    mitarb (full A100 SXM4 40GB)"
echo "Exp:          $EXP (hold out $held_out)"
echo "Started:      $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

source /scratch/shirzarm/.conda/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/.conda/etc/profile.d/conda.sh 2>/dev/null || \
  source $(conda info --base)/etc/profile.d/conda.sh
conda activate prostate_phase2

export nnUNet_results=/scratch/shirzarm/vcip/nnUNet_results
export nnUNet_compile=0
export nnUNet_n_proc_DA=4         # bumped from 2 — more CPU cores on mitarb
export OMP_NUM_THREADS=4
mkdir -p $nnUNet_results

# --- GPU diagnostics (no python probe; that stole CUDA context on MIG slots)
echo "=== GPU diagnostics ==="
echo "SLURM_JOB_GPUS:       ${SLURM_JOB_GPUS:-(unset)}"
echo "CUDA_VISIBLE_DEVICES (from slurm): ${CUDA_VISIBLE_DEVICES:-(unset)}"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv

# NOTE: mitarb GPU is the full A100 SXM4 (not MIG-partitioned), so the shell
# MIG-UUID override that we needed for student MIG slots is NOT needed here.
# slurm sets CUDA_VISIBLE_DEVICES to the right index and we trust it.

# --- Sanity-check the pretrained ckpt
PRETRAINED=/scratch/shirzarm/vcip/baseline_ckpt/Dataset501_ProstateUS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
test -f "$PRETRAINED" || { echo "ERROR: pretrained ckpt missing at $PRETRAINED"; exit 1; }

# --- Training
HELD_OUT_CODEC="$held_out" \
nnUNetv2_train 501 3d_fullres 0 \
    -tr nnUNetTrainerCodecAug505_v2_LOCO \
    -pretrained_weights "$PRETRAINED"

# --- Snapshot to Dataset505_LOCO_<EXP> so other EXPs don't overwrite the dir
src="$nnUNet_results/Dataset501_ProstateUS/nnUNetTrainerCodecAug505_v2_LOCO__nnUNetPlans__3d_fullres"
dst_root="$nnUNet_results/Dataset505_LOCO_${EXP}"
dst="$dst_root/nnUNetTrainerCodecAug505_v2_LOCO__nnUNetPlans__3d_fullres"
mkdir -p "$dst"
cp -r "$src/fold_0" "$dst/fold_0"
cp "$src/dataset.json" "$dst/" 2>/dev/null || true
cp "$src/plans.json"   "$dst/" 2>/dev/null || true
echo "$held_out" > "$dst_root/HELD_OUT_CODEC.txt"
echo "Snapshot saved to $dst"

echo "=========================================="
echo "LOCO Exp $EXP training finished: $(date)"
echo "Next: sbatch --export=EXP=$EXP 2026-05-30_AS_loco_codec_eval_mitarb_sbatch.sh"
echo "=========================================="
