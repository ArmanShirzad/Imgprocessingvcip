harmonization scripts walkthrough

 Step 1 (Mean PSNR / SSIM Table)
Script: 
2026-06-11_AS_step1_psnr_ssim_18cells.py
 (submitted via 
2026-06-11_AS_step1_psnr_ssim_sbatch.sh
)
Description: This script processes the 18 baseline compressed sets (3 codecs x 6 CRFs: 18, 23, 28, 33, 38, 43) against the 299-volume validation set, calculates the average PSNR and SSIM on-the-fly using ffmpeg filters, and outputs the markdown table for Step 1.
2. Step 2 (Linear Interpolation for Matching operating points)
Analysis File: 
2026-06-12_step2_ssim_harmonization.md
Description: The linear interpolation calculations mapping HEVC and AV1 CRFs to match H.264 SSIM targets (anchored on H.264 CRFs 18, 23, 28, 33, 38) were compiled here, identifying the new AV1 CRF targets as 30 and 41.
3. Steps 3 & 4 (Encoding and GPU Inference of New AV1 CRFs)
AV1 Encoding: 
2026-05-24_AS_encode_missing_crfs.py
 (run via 
2026-06-18_AS_step3_encode_av1_sbatch.sh
) compressed the dataset for the new AV1 CRFs (30, 35, 36, 37, 41).
PSNR/SSIM of New CRFs: 
2026-06-11_AS_step1_psnr_ssim_18cells.py
 was run again via 
2026-06-18_AS_step3_5_psnr_ssim_av1_sbatch.sh
 to calculate quality metrics for these new operating points.
GPU Inference & Scoring: The inline script SCORE_PY in 
2026-06-18_AS_step4_av1_inference_sbatch.sh
 executed nnU-Net predictions on the new AV1 sets and computed Dice/HD95 scores.
4. Step 5 (Effect on the Results & Safe-Bitrate Refinement)
Script: 
2026-06-22_AS_rebuild_tableI.py
Description: This script aggregates the original baseline metrics, the new AV1 inference scores, and the new SSIM metrics. It rebuilds Table I (yielding matched-bitrate metrics) and computes the refined AV1 safe-bitrate threshold showing that mean Dice remains 
≥
0.84
≥0.84 down to 0.415 Mbps (CRF 37) (which was previously overstated as 0.611 @ CRF 33).