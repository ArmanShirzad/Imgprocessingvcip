#!/usr/bin/env python3
"""
2026-06-11_AS_step1_psnr_ssim_18cells.py
----------------------------------------
VCIP 2026 -- Alireza SSIM-harmonization roadmap, STEP 1 (CPU only, no GPU).

Measure mean PSNR and SSIM for every one of the 18 existing compressed
validation cells (3 codecs x 6 CRFs) against the uncompressed originals, and
emit the exact table Alireza asked for:

    Codec | CRF | Bitrate (Mbps) | Mean PSNR | Mean SSIM

This is a CPU-only fork of 2026-05-21_AS_quality_metric_sweep_v2.py with the
nnU-Net inference / Dice / HD95 stages REMOVED (Step 1 needs no GPU). The
encode settings are byte-for-byte the canonical phase-3 settings, so the
streams measured here are identical to the compressed sets used for the paper's
rate-task evaluation:
    libx264   : -g 300 -x264-params no-scenecut=1
    libx265   : -x265-params keyint=300:min-keyint=300
    libsvtav1 : -preset 8
    all       : gray rawvideo in, -pix_fmt yuv420p, FPS 30, even-pad with edge

PSNR/SSIM are computed exactly as in Alireza's command (ffmpeg psnr + ssim
filters), per volume, then averaged per cell. Encode is deterministic, so we
re-encode on the fly rather than depending on the on-disk decoded sets; the
per-volume bitrate measured here should match the canonical bitrate lookup
within encoder noise.

Robustness (per repo feedback memories):
  - every ffmpeg subprocess wrapped in try/except; a failed volume is logged
    and skipped, never kills the run (feedback-wrap-codec-subprocess).
  - no `producer | grep` pipelines under pipefail (feedback-sigpipe-pipefail).
  - parallel across volumes with a process pool (--jobs).

Usage (on Willi, CPU node or login):
  python 2026-06-11_AS_step1_psnr_ssim_18cells.py \\
      --image_dir /scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr \\
      --splits_json /scratch/shirzarm/nnUNet_preprocessed/Dataset501_ProstateUS/splits_final.json \\
      --out_dir /scratch/shirzarm/vcip/phase1_step1 \\
      --jobs 8

Resumable: rows already in the per-volume CSV are skipped on restart.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

FPS = 30
GOP = 300
DEFAULT_CODECS = ["libx264", "libx265", "libsvtav1"]
DEFAULT_CRFS = [18, 23, 28, 33, 38, 43]  # the locked 6-op grid (18 cells total)


# ---------------------------------------------------------------- encode/decode

def encode_volume(arr_u8: np.ndarray, codec: str, crf: int, out_path: Path) -> int:
    """Encode a (z,h,w) uint8 volume as a single grayscale video. Returns size in bits."""
    z, h, w = arr_u8.shape
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        arr_u8 = np.pad(arr_u8, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        h, w = arr_u8.shape[1], arr_u8.shape[2]

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
    cmd.append(str(out_path))

    p = subprocess.run(cmd, input=arr_u8.tobytes(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"encode failed ({codec} crf={crf}): {p.stderr.decode()[:400]}")
    return os.path.getsize(out_path) * 8


def decode_volume(video_path: Path, shape_zhw: tuple[int, int, int]) -> np.ndarray:
    z, h, w = shape_zhw
    pad_h, pad_w = h % 2, w % 2
    eff_h, eff_w = h + pad_h, w + pad_w
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-an", "-frames:v", str(z),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"decode failed: {p.stderr.decode()[:400]}")
    arr = np.frombuffer(p.stdout, dtype=np.uint8).reshape(z, eff_h, eff_w)
    return arr[:, :h, :w]


# --------------------------------------------------------- PSNR + SSIM (ffmpeg)

def psnr_ssim(orig_arr: np.ndarray, dec_arr: np.ndarray, tmp: Path) -> tuple[float, float]:
    """Alireza's command, per volume: ffmpeg psnr + ssim filters over the raw frames."""
    z, h, w = orig_arr.shape
    orig_yuv, dec_yuv = tmp / "orig.yuv", tmp / "dec.yuv"
    orig_yuv.write_bytes(orig_arr.tobytes())
    dec_yuv.write_bytes(dec_arr.tobytes())
    # IMPORTANT: the psnr/ssim filters print their "average:"/"All:" summary at
    # AV_LOG_INFO. Using "-loglevel error" SUPPRESSES those lines -> nan. Keep
    # the level at info so the summary is emitted to stderr for parsing.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(FPS), "-i", str(dec_yuv),
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(FPS), "-i", str(orig_yuv),
        "-lavfi", "[0:v][1:v]psnr;[0:v][1:v]ssim", "-f", "null", "-",
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.decode()
    m_psnr = re.search(r"average:([0-9.]+)", out)
    m_ssim = re.search(r"All:([0-9.]+)", out)
    psnr = float(m_psnr.group(1)) if m_psnr else float("nan")
    ssim = float(m_ssim.group(1)) if m_ssim else float("nan")
    return psnr, ssim


# ---------------------------------------------------------------- per-volume job

def process_one(case_id: str, codec: str, crf: int, image_path: str) -> dict | None:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix=f"{case_id}_{codec}_crf{crf}_"))
    try:
        orig_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.uint8)
        bits = encode_volume(orig_arr, codec, crf, tmp / "stream.mp4")
        n_frames = orig_arr.shape[0]
        bitrate_mbps = bits / (n_frames / FPS) / 1e6
        dec_arr = decode_volume(tmp / "stream.mp4", orig_arr.shape)
        psnr, ssim = psnr_ssim(orig_arr, dec_arr, tmp)
        return {
            "case_id": case_id, "codec": codec, "crf": crf,
            "bitrate_mbps": bitrate_mbps, "psnr": psnr, "ssim": ssim,
            "duration_s": time.time() - t0,
        }
    except Exception as e:  # never let one bad volume kill the run; surface why
        import traceback
        return {"_error": f"{type(e).__name__}: {e}",
                "_trace": traceback.format_exc().splitlines()[-3:],
                "case_id": case_id, "codec": codec, "crf": crf}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- case list

def load_val_cases(splits_json: Path | None, image_dir: Path,
                   cases_txt: Path | None) -> list[str]:
    if cases_txt and cases_txt.exists():
        return [c.strip() for c in cases_txt.read_text().splitlines() if c.strip()]
    if splits_json and splits_json.exists():
        folds = json.loads(splits_json.read_text())
        val = folds[0]["val"]  # fold-0 validation = the paper's n=299
        return [str(v) for v in val]
    # fallback: every image in the dir
    ids = sorted({p.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
                  for p in image_dir.glob("*.nii.gz")})
    return ids


def resolve_image(image_dir: Path, cid: str) -> str | None:
    for cand in (image_dir / f"{cid}_0000.nii.gz", image_dir / f"{cid}.nii.gz"):
        if cand.exists():
            return str(cand)
    return None


# ---------------------------------------------------------------- driver

PERVOL_FIELDS = ["case_id", "codec", "crf", "bitrate_mbps", "psnr", "ssim", "duration_s"]


def load_done(pervol_csv: Path) -> set[tuple[str, str, int]]:
    done = set()
    if pervol_csv.exists():
        with pervol_csv.open() as f:
            for r in csv.DictReader(f):
                done.add((r["case_id"], r["codec"], int(r["crf"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=Path,
                    default=Path("/scratch/shirzarm/nnUNet_raw/Dataset501_ProstateUS/imagesTr"))
    ap.add_argument("--splits_json", type=Path,
                    default=Path("/scratch/shirzarm/nnUNet_preprocessed/Dataset501_ProstateUS/splits_final.json"))
    ap.add_argument("--cases_txt", type=Path, default=None,
                    help="optional: explicit case-id list (one per line); overrides splits_json")
    ap.add_argument("--out_dir", type=Path, default=Path("/scratch/shirzarm/vcip/phase1_step1"))
    ap.add_argument("--codecs", nargs="+", default=DEFAULT_CODECS)
    ap.add_argument("--crfs", type=int, nargs="+", default=DEFAULT_CRFS)
    ap.add_argument("--jobs", type=int, default=8, help="parallel encode workers")
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all; else cap (debug)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pervol_csv = args.out_dir / "2026-06-11_step1_psnr_ssim_pervol.csv"
    summary_csv = args.out_dir / "2026-06-11_step1_psnr_ssim_per_cell.csv"
    table_md = args.out_dir / "2026-06-11_step1_psnr_ssim_table.md"

    cases = load_val_cases(args.splits_json, args.image_dir, args.cases_txt)
    if args.max_cases:
        cases = cases[:args.max_cases]
    cases = [c for c in cases if resolve_image(args.image_dir, c)]
    done = load_done(pervol_csv)

    tasks = [(c, codec, crf)
             for codec in args.codecs for crf in args.crfs for c in cases
             if (c, codec, crf) not in done]
    print(f"[step1] {len(cases)} cases x {len(args.codecs)} codecs x {len(args.crfs)} CRFs "
          f"= {len(cases)*len(args.codecs)*len(args.crfs)} rows; {len(tasks)} to do, "
          f"{len(done)} already done; jobs={args.jobs}", flush=True)

    new_file = not pervol_csv.exists() or pervol_csv.stat().st_size == 0
    with pervol_csv.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=PERVOL_FIELDS)
        if new_file:
            wr.writeheader()
        t0 = time.time()
        n_ok = 0
        n_err = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(process_one, c, codec, crf,
                              resolve_image(args.image_dir, c)): (c, codec, crf)
                    for (c, codec, crf) in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                if row is None:
                    continue
                if "_error" in row:
                    n_err += 1
                    if n_err <= 5:  # print the first few in full so we see WHY
                        print(f"  x {row['case_id']} {row['codec']} crf={row['crf']}: "
                              f"{row['_error']}\n      {' | '.join(row['_trace'])}", flush=True)
                    continue
                wr.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                             for k in PERVOL_FIELDS})
                fh.flush()
                n_ok += 1
                if i % 50 == 0:
                    rate = i / (time.time() - t0)
                    eta = (len(tasks) - i) / rate / 60 if rate else 0
                    print(f"  [{i}/{len(tasks)}] ok={n_ok} {rate:.1f} vol/s ETA {eta:.0f} min",
                          flush=True)

    aggregate(pervol_csv, summary_csv, table_md, args.codecs, args.crfs)


def aggregate(pervol_csv: Path, summary_csv: Path, table_md: Path,
              codecs: list[str], crfs: list[int]) -> None:
    """Per-cell mean bitrate / PSNR / SSIM + the Codec|CRF|Bitrate|PSNR|SSIM table."""
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    with pervol_csv.open() as f:
        for r in csv.DictReader(f):
            key = (r["codec"], int(r["crf"]))
            d = cells.setdefault(key, {"bitrate_mbps": [], "psnr": [], "ssim": []})
            for k in d:
                try:
                    v = float(r[k])
                    if not np.isnan(v):
                        d[k].append(v)
                except (ValueError, KeyError):
                    pass

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    rows = []
    for codec in codecs:
        for crf in crfs:
            d = cells.get((codec, crf)) or {"bitrate_mbps": [], "psnr": [], "ssim": []}
            # report each metric independently; n = volumes with a valid PSNR
            rows.append((codec, crf,
                         _mean(d["bitrate_mbps"]),
                         _mean(d["psnr"]),
                         _mean(d["ssim"]),
                         len(d["psnr"])))

    with summary_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["codec", "crf", "mean_bitrate_mbps", "mean_psnr", "mean_ssim", "n"])
        for codec, crf, br, ps, ss, n in rows:
            wr.writerow([codec, crf, f"{br:.6f}", f"{ps:.4f}", f"{ss:.6f}", n])

    label = {"libx264": "H.264", "libx265": "HEVC", "libsvtav1": "AV1"}
    lines = ["| Codec | CRF | Bitrate (Mbps) | Mean PSNR (dB) | Mean SSIM | n |",
             "|---|---|---|---|---|---|"]
    for codec, crf, br, ps, ss, n in rows:
        lines.append(f"| {label.get(codec, codec)} | {crf} | "
                     f"{br:.4f} | {ps:.3f} | {ss:.5f} | {n} |")
    table = "\n".join(lines)
    table_md.write_text(table + "\n")
    print("\n" + table + "\n")
    print(f"[step1] wrote:\n  {pervol_csv}\n  {summary_csv}\n  {table_md}")


if __name__ == "__main__":
    main()
