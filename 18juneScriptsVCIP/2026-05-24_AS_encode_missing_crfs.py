#!/usr/bin/env python3
"""
2026-05-24_AS_encode_missing_crfs.py
------------------------------------
Encode the 4 missing CRF folders for the VCIP 2026 6-op grid {18, 23, 28, 33, 38, 43}.

Per the Tweety inventory (2026-05-20), the following encodes are MISSING:
  - libx264_crf33     (299 vols)
  - libx265_crf33     (299 vols)
  - libsvtav1_crf33   (299 vols)
  - libsvtav1_crf38   (299 vols)

Total = 4 × 299 = 1,196 new compressed volumes. CPU-only (no GPU).

For each (codec, CRF, val_case_id):
  1. Read the original uncompressed NIfTI from nnUNet_raw/Dataset501_ProstateUS/imagesTr/
  2. ffmpeg encode -> .mp4, then decode back to (Z, H, W) uint8
  3. Write the decoded volume as <codec>_crf<crf>/imagesTs/<case_id>_0000.nii.gz
     to the OUTPUT_BASE (Willi local /scratch first; user can rsync to Tweety after).

Val case IDs come from splits_final.json fold 0 val list (299 cases).

Usage (defaults are correct for Willi VCIP layout):
  # Single-process (slow but debuggable)
  python 2026-05-24_AS_encode_missing_crfs.py --workers 1

  # Parallel — 8 workers, ~10-15x faster
  python 2026-05-24_AS_encode_missing_crfs.py --workers 8

  # Just one job
  python 2026-05-24_AS_encode_missing_crfs.py --jobs libx264:33 --workers 8

  # Encode to Tweety directly (slower, but skips later rsync)
  python 2026-05-24_AS_encode_missing_crfs.py --out_base /scratch/shirzarm/tweety_mp/phase3_codecs --workers 8

Submit as sbatch with --cpus-per-task=N and --workers=N (no GPU needed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

FPS = 30
GOP = 300

DEFAULT_JOBS = [
    ("libx264", 33),
    ("libx265", 33),
    ("libsvtav1", 33),
    ("libsvtav1", 38),
]


# ----------------------------------------------------------------- encode/decode

def encode_decode(arr_u8: np.ndarray, codec: str, crf: int) -> np.ndarray:
    """Round-trip a (Z, H, W) uint8 volume through ffmpeg. Returns decoded array."""
    z, h, w = arr_u8.shape
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        arr_u8 = np.pad(arr_u8, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        h, w = arr_u8.shape[1], arr_u8.shape[2]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = Path(tf.name)
    try:
        enc_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
            "-an", "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
        ]
        if codec == "libx264":
            enc_cmd += ["-g", str(GOP), "-x264-params", "no-scenecut=1"]
        elif codec == "libx265":
            enc_cmd += ["-x265-params", f"keyint={GOP}:min-keyint={GOP}"]
        elif codec == "libsvtav1":
            enc_cmd += ["-preset", "8"]
        elif codec == "libaom-av1":
            enc_cmd += ["-b:v", "0", "-cpu-used", "4"]
        enc_cmd.append(str(video_path))

        p = subprocess.run(enc_cmd, input=arr_u8.tobytes(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError(f"encode failed ({codec} crf={crf}): {p.stderr.decode()[:400]}")

        dec_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-an", "-frames:v", str(z),
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        p = subprocess.run(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError(f"decode failed: {p.stderr.decode()[:400]}")
        buf = np.frombuffer(p.stdout, dtype=np.uint8)
        eff_h = h
        eff_w = w
        arr = buf.reshape(z, eff_h, eff_w)
        orig_h = arr.shape[1] - pad_h
        orig_w = arr.shape[2] - pad_w
        return arr[:, :orig_h, :orig_w].copy()
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass


# ------------------------------------------------------------------- per-case

def encode_one_case(case_id: str, src_path: Path, codec: str, crf: int,
                    out_dir: Path, overwrite: bool) -> tuple[str, str, float]:
    """Worker: encode one (codec, crf, case) and write the decoded volume."""
    out_path = out_dir / f"{case_id}_0000.nii.gz"
    if out_path.exists() and not overwrite:
        return (case_id, "SKIP_exists", 0.0)
    if not src_path.exists():
        return (case_id, "FAIL_no_src", 0.0)

    t0 = time.time()
    try:
        img = sitk.ReadImage(str(src_path))
        arr = sitk.GetArrayFromImage(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        decoded = encode_decode(arr, codec, crf)
        out_img = sitk.GetImageFromArray(decoded.astype(np.uint8))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path), useCompression=True)
        return (case_id, "OK", time.time() - t0)
    except Exception as e:
        return (case_id, f"FAIL: {type(e).__name__}: {str(e)[:120]}", time.time() - t0)


# --------------------------------------------------------------- val ID loader

def load_val_ids(splits_json: Path, fold: int = 0) -> list[str]:
    with open(splits_json) as f:
        splits = json.load(f)
    return list(splits[fold]["val"])


# -------------------------------------------------------------------- driver

def parse_jobs(spec: str | None) -> list[tuple[str, int]]:
    if not spec:
        return DEFAULT_JOBS
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        codec, crf = item.split(":")
        out.append((codec.strip(), int(crf.strip())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path,
                    default=Path("/scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr"),
                    help="Uncompressed source NIfTIs ({case_id}_0000.nii.gz)")
    ap.add_argument("--splits_json", type=Path,
                    default=Path("/scratch/shirzarm/nnUNet_preprocessed/Dataset501_ProstateUS/splits_final.json"),
                    help="Used to get fold-0 val IDs")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out_base", type=Path,
                    default=Path("/scratch/shirzarm/vcip/phase3_codecs"),
                    help="Per-codec/CRF imagesTs/ folders are created under here")
    ap.add_argument("--jobs", type=str, default=None,
                    help="Comma-separated codec:crf pairs. Default = the 4 missing encodes.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel ffmpeg workers (CPU-bound). Match --cpus-per-task in sbatch.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-encode even if the output file already exists.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only process this many cases per job (smoke test).")
    args = ap.parse_args()

    val_ids = load_val_ids(args.splits_json, args.fold)
    print(f"[encode] fold {args.fold} val set: {len(val_ids)} cases")
    if args.limit > 0:
        val_ids = val_ids[:args.limit]
        print(f"[encode] LIMIT={args.limit} — only first {len(val_ids)} cases will be encoded")

    jobs = parse_jobs(args.jobs)
    print(f"[encode] jobs: {jobs}")
    print(f"[encode] out_base: {args.out_base}")
    print(f"[encode] workers: {args.workers}")

    for (codec, crf) in jobs:
        out_dir = args.out_base / f"{codec}_crf{crf}" / "imagesTs"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[encode] === {codec}_crf{crf} -> {out_dir} ===")

        tasks = [(cid, args.raw_dir / f"{cid}_0000.nii.gz", codec, crf, out_dir, args.overwrite)
                 for cid in val_ids]

        t0 = time.time()
        ok = skip = fail = 0
        if args.workers <= 1:
            for tk in tasks:
                cid, status, dur = encode_one_case(*tk)
                if status == "OK": ok += 1
                elif status == "SKIP_exists": skip += 1
                else: fail += 1
                if (ok + skip + fail) % 25 == 0:
                    print(f"  ... {ok + skip + fail}/{len(tasks)}  ok={ok} skip={skip} fail={fail}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(encode_one_case, *tk) for tk in tasks]
                done = 0
                for fut in as_completed(futures):
                    cid, status, dur = fut.result()
                    done += 1
                    if status == "OK": ok += 1
                    elif status == "SKIP_exists": skip += 1
                    else: fail += 1
                    if done % 25 == 0:
                        print(f"  ... {done}/{len(tasks)}  ok={ok} skip={skip} fail={fail}", flush=True)
                    if status.startswith("FAIL"):
                        print(f"    ✗ {cid}: {status}", flush=True)

        elapsed = time.time() - t0
        per = elapsed / max(len(tasks), 1)
        print(f"[encode] {codec}_crf{crf} done in {elapsed:.1f}s  ({per:.2f}s/case)  ok={ok} skip={skip} fail={fail}")

    print("\n[encode] all jobs complete.")


if __name__ == "__main__":
    main()
