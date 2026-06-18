#!/usr/bin/env python3
"""
2026-05-31_AS_measure_bitrate_4cells.py
---------------------------------------
Measure the actual bitrate of the 4 (codec, CRF) cells that were encoded
in the 2026-05-24 interactive session but never logged to compression_stats.csv:

  libx264_crf33   libx265_crf33   libsvtav1_crf33   libsvtav1_crf38

Our current rate-task plot uses log-linear INTERPOLATED bitrate for these
4 cells (accurate to ~5%). For the final paper we want measured bitrate.

Approach: for each cell, re-encode each imagesTr volume (or a stratified
subset) with the same ffmpeg parameters used originally, parse the achieved
bitrate from ffmpeg's stderr output. ffmpeg reports bitrate at the end of
encoding like:
  video:1234kB audio:0kB ... muxing overhead: ...
  bitrate=  789.34kbits/s

We compute: file_size_bits / duration_seconds = effective video bitrate.
For consistency with the existing compression_stats.csv we use the SAME
formula: avg_mbps = (file_size_in_bytes * 8 / 1e6) / video_duration_seconds.

Writes one row per (case, cell) into a CSV with the same schema as the
existing compression_stats.csv, so we can simply append.

Runs entirely on Willi login CPU. ~30 sec per encode, 8 workers in parallel.
~50 vols × 4 cells / 8 workers × 30 sec = ~12 min for a 50-vol sample.
For full 299 vols × 4 cells: ~75 min. Still well under one mitarb slot, but
runs on CPU so doesn't need GPU at all.

Usage (sample for quick mean estimate):
  python3 /scratch/shirzarm/vcip/scripts/2026-05-31_AS_measure_bitrate_4cells.py \\
      --imagestr /scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr \\
      --case_subset /scratch/shirzarm/vcip/phase1/calibration_subset_50.txt \\
      --out_csv /scratch/shirzarm/vcip/refs/2026-05-31_measured_bitrate_4cells.csv \\
      --workers 8

(omit --case_subset to encode all 299 per cell.)

After it finishes, update the bitrate lookup:
  scp the CSV down, then re-run 2026-05-31_AS_build_bitrate_lookup.py with
  the appended file as input.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk


FPS = 30
GOP = 300

# Match the existing on-the-fly aug + ICIP-era encode parameters exactly
CELLS = [
    ("libx264",   33),
    ("libx265",   33),
    ("libsvtav1", 33),
    ("libsvtav1", 38),
]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = a.min(); hi = a.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def encode_one(args):
    codec, crf, case_id, nifti_path = args
    try:
        img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(img)  # (Z, H, W)
        u8 = _to_uint8(arr)
        z, h, w = u8.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            u8 = np.pad(u8, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            h, w = u8.shape[1], u8.shape[2]
        duration_s = z / FPS

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            video_path = Path(tf.name)
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "gray",
                "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                "-an", "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
            ]
            if codec == "libx264":
                cmd += ["-g", str(GOP), "-x264-params", "no-scenecut=1"]
            elif codec == "libx265":
                cmd += ["-x265-params", f"keyint={GOP}:min-keyint={GOP}"]
            elif codec == "libsvtav1":
                cmd += ["-preset", "8"]
            cmd.append(str(video_path))

            p = subprocess.run(cmd, input=u8.tobytes(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
            if p.returncode != 0:
                return (codec, crf, case_id, None, None, f"encode err: {p.stderr.decode()[:150]}")

            size_bytes = video_path.stat().st_size
            mbps = (size_bytes * 8) / 1e6 / duration_s
            # bits per voxel
            n_voxels = z * h * w
            bpv = (size_bytes * 8) / n_voxels
            return (codec, crf, case_id, mbps, bpv, None)
        finally:
            try: video_path.unlink(missing_ok=True)
            except: pass
    except Exception as e:
        return (codec, crf, case_id, None, None, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagestr", type=Path, required=True,
                    help="nnUNet_raw/Dataset501_ProstateUS/imagesTr — source volumes")
    ap.add_argument("--case_subset", type=Path, default=None,
                    help="Optional text file with one case_id per line. If omitted, "
                         "encodes ALL 299 cases per cell.")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Determine case list
    if args.case_subset and args.case_subset.exists():
        with open(args.case_subset) as f:
            cases = sorted([l.strip() for l in f if l.strip()])
        print(f"Using {len(cases)} cases from {args.case_subset}")
    else:
        cases = sorted({p.name.replace("_0000.nii.gz", "")
                        for p in args.imagestr.iterdir()
                        if p.name.endswith("_0000.nii.gz")})
        print(f"Using ALL {len(cases)} cases under {args.imagestr}")

    # Build tasks: (codec, crf, case_id, nifti_path)
    tasks = []
    for codec, crf in CELLS:
        for cid in cases:
            nifti = args.imagestr / f"{cid}_0000.nii.gz"
            if nifti.exists():
                tasks.append((codec, crf, cid, nifti))
    print(f"Total encode tasks: {len(tasks)} ({len(CELLS)} cells × {len(cases)} cases)")

    # Write header
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "crf", "case_id", "avg_mbps", "bits_per_voxel", "out_nifti"])

        with Pool(processes=args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(encode_one, tasks), 1):
                codec, crf, cid, mbps, bpv, err = res
                if err:
                    print(f"  ERR  {codec}_crf{crf}/{cid}: {err}")
                    continue
                w.writerow([codec, crf, cid, f"{mbps:.6f}", f"{bpv:.6f}", ""])
                if i % 50 == 0:
                    print(f"  {i}/{len(tasks)}")

    # Per-cell summary
    print(f"\nWrote {args.out_csv}")
    from collections import defaultdict
    by_cell = defaultdict(list)
    with open(args.out_csv) as f:
        for r in csv.DictReader(f):
            by_cell[(r["codec"], int(r["crf"]))].append(float(r["avg_mbps"]))
    print("\nMeasured mean bitrate (Mbps) for the 4 cells:")
    for (codec, crf) in sorted(by_cell.keys()):
        v = by_cell[(codec, crf)]
        print(f"  {codec:12s} CRF {crf:2d}  n={len(v):3d}  mean={sum(v)/len(v):.4f}  std={float(np.std(v, ddof=1)):.4f}  min={min(v):.4f}  max={max(v):.4f}")


if __name__ == "__main__":
    main()
