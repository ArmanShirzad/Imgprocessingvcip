#!/usr/bin/env python3
"""
2026-06-09_AS_phase5_libaom.py
------------------------------
Phase 5 — AV1 encoder ablation (Alireza proposal): does the AV1 *encoder*
(libsvtav1 vs libaom-av1) change the rate-task result? If Δ Dice is small at
matched CRF across operating points, the segmentation-vs-bitrate finding is
implementation-independent.

We already have libsvtav1 on all 299 cases (Phase 3). This script produces the
libaom-av1 counterpart on a stratified subset, then (after baseline inference)
scores it so the two encoders can be compared on the SAME cases at matched CRF.

Two subcommands:
  encode  — for each case in subset, each CRF: libaom-av1 round-trip, write the
            decoded volume as imagesTs/<case>_0000.nii.gz (nnUNet naming, correct
            geometry copied from the source) + record achieved bitrate.
  score   — read baseline predictions per cell, compute Dice/HD95 vs GT, write CSV.

Layout (under --root, default /scratch/shirzarm/vcip/phase5_libaom):
  <root>/libaom_av1_crf<NN>/imagesTs/<case>_0000.nii.gz   (encode output)
  <root>/libaom_av1_crf<NN>/predictions/<case>.nii.gz     (nnUNetv2_predict output)
  <root>/2026-06-09_phase5_libaom_bitrate.csv             (encode)
  <root>/2026-06-09_phase5_libaom_scored.csv              (score)

CRF grid matches Phase 3: {18,23,28,33,38,43}. Subset defaults to the existing
50-vol calibration subset; pass --case_subset to change, or --all for 299.
"""
from __future__ import annotations
import argparse, csv, os, subprocess, tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk

FPS = 30
GOP = 300
CRFS = [18, 23, 28, 33, 38, 43]

# ffmpeg binary — point at the dedicated libaom-enabled build via $FFMPEG.
# prostate_phase2's ffmpeg lacks libaom-av1, so we use a separate `ffaom` env.
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def _to_uint8(arr):
    a = arr.astype(np.float32); lo = a.min(); hi = a.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def encode_decode_libaom(u8, crf):
    """libaom-av1 constant-quality round-trip; returns decoded uint8 (Z,H,W)."""
    z, h, w = u8.shape
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        u8 = np.pad(u8, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        h, w = u8.shape[1], u8.shape[2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        vp = Path(tf.name)
    try:
        enc = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
               "-an", "-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0",
               "-cpu-used", "8", "-row-mt", "1",
               "-g", str(GOP), "-pix_fmt", "yuv420p", str(vp)]
        try:
            p = subprocess.run(enc, input=u8.tobytes(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=180)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"libaom encode timeout (crf={crf})")
        if p.returncode != 0:
            raise RuntimeError(f"libaom encode failed (crf={crf}): {p.stderr.decode()[:200]}")
        size_bytes = vp.stat().st_size
        dec = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(vp), "-an", "-frames:v", str(z),
               "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        try:
            p = subprocess.run(dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"libaom decode timeout (crf={crf})")
        if p.returncode != 0:
            raise RuntimeError(f"libaom decode failed (crf={crf}): {p.stderr.decode()[:200]}")
        arr = np.frombuffer(p.stdout, dtype=np.uint8).reshape(z, h, w)
        arr = arr[:, :h - pad_h, :w - pad_w].copy()
        mbps = (size_bytes * 8) / 1e6 / (z / FPS)
        return arr, mbps
    finally:
        try: vp.unlink(missing_ok=True)
        except Exception: pass


def _encode_one(args):
    case_id, crf, src_nifti, out_nifti = args
    try:
        img = sitk.ReadImage(str(src_nifti))
        u8 = _to_uint8(sitk.GetArrayFromImage(img))
        dec, mbps = encode_decode_libaom(u8, crf)
        out = sitk.GetImageFromArray(dec.astype(np.float32))
        out.CopyInformation(img)          # preserve spacing/origin/direction
        out_nifti.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(out_nifti))
        return (case_id, crf, mbps, None)
    except Exception as e:
        return (case_id, crf, None, f"{type(e).__name__}: {e}")


def cmd_encode(args):
    if args.all or not (args.case_subset and args.case_subset.exists()):
        cases = sorted({p.name.replace("_0000.nii.gz", "")
                        for p in args.imagestr.iterdir() if p.name.endswith("_0000.nii.gz")})
        print(f"Encoding ALL {len(cases)} cases")
    else:
        cases = sorted(l.strip() for l in open(args.case_subset) if l.strip())
        print(f"Encoding {len(cases)} subset cases from {args.case_subset}")

    tasks = []
    for crf in CRFS:
        for cid in cases:
            src = args.imagestr / f"{cid}_0000.nii.gz"
            if not src.exists():
                continue
            out = args.root / f"libaom_av1_crf{crf}" / "imagesTs" / f"{cid}_0000.nii.gz"
            if out.exists():    # resume: skip already-encoded
                continue
            tasks.append((cid, crf, src, out))
    print(f"Encode tasks (after resume-skip): {len(tasks)}")

    rows = []
    args.root.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for i, (cid, crf, mbps, err) in enumerate(pool.imap_unordered(_encode_one, tasks), 1):
            if err:
                print(f"  ERR {cid} crf{crf}: {err}")
            else:
                rows.append((cid, crf, mbps))
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    bcsv = args.root / "2026-06-09_phase5_libaom_bitrate.csv"
    write_header = not bcsv.exists()
    with open(bcsv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["case_id", "crf", "avg_mbps"])
        for cid, crf, mbps in rows:
            w.writerow([cid, crf, f"{mbps:.6f}"])
    print(f"Appended {len(rows)} bitrate rows to {bcsv}")
    per = defaultdict(list)
    for _, crf, mbps in rows:
        per[crf].append(mbps)
    for crf in sorted(per):
        v = per[crf]
        print(f"  libaom_av1 CRF {crf}: n={len(v)} mean_mbps={sum(v)/len(v):.4f}")


# ---- scoring (mirror of loco_score) ----
def _score_one(args):
    folder, case_id, pred_path, gt_path = args
    try:
        pi_img = sitk.ReadImage(str(pred_path)); gi_img = sitk.ReadImage(str(gt_path))
        pred = sitk.GetArrayFromImage(pi_img); gt = sitk.GetArrayFromImage(gi_img)
        classes = sorted(int(c) for c in np.unique(gt) if c != 0)
        if not classes:
            return (folder, case_id, 1.0 if pred.sum() == 0 else 0.0, 0.0)
        ds, hs = [], []
        for c in classes:
            pm = (pred == c).astype(np.uint8); gm = (gt == c).astype(np.uint8)
            inter = int(np.logical_and(pm, gm).sum()); den = int(pm.sum() + gm.sum())
            ds.append(1.0 if den == 0 else 2.0 * inter / den)
            if pm.sum() == 0 and gm.sum() == 0: hs.append(0.0); continue
            if pm.sum() == 0 or gm.sum() == 0:  hs.append(50.0); continue
            gI = sitk.GetImageFromArray(gm); gI.CopyInformation(gi_img)
            pI = sitk.GetImageFromArray(pm); pI.CopyInformation(pi_img)
            dg = sitk.Abs(sitk.SignedMaurerDistanceMap(gI, squaredDistance=False, useImageSpacing=True))
            dp = sitk.Abs(sitk.SignedMaurerDistanceMap(pI, squaredDistance=False, useImageSpacing=True))
            cg = sitk.LabelContour(gI); cp = sitk.LabelContour(pI)
            apg = sitk.GetArrayFromImage(dg)[sitk.GetArrayFromImage(cp) == 1]
            agp = sitk.GetArrayFromImage(dp)[sitk.GetArrayFromImage(cg) == 1]
            allp = np.concatenate([apg, agp])
            hs.append(float(np.percentile(allp, 95)) if len(allp) else 0.0)
        return (folder, case_id, float(np.mean(ds)), float(np.mean(hs)))
    except Exception as e:
        print(f"  ERR score {folder}/{case_id}: {e}")
        return None


def cmd_score(args):
    tasks = []
    for crf in CRFS:
        pdir = args.root / f"libaom_av1_crf{crf}" / "predictions"
        if not pdir.is_dir():
            print(f"SKIP crf{crf}: no predictions dir"); continue
        for pf in sorted(p for p in pdir.iterdir() if p.name.endswith(".nii.gz")):
            cid = pf.name.replace(".nii.gz", "")
            tasks.append((f"libaom_av1_crf{crf}", cid, pf, args.gt_dir / f"{cid}.nii.gz"))
    print(f"Scoring {len(tasks)} predictions")
    res = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_one, tasks), 1):
            if r: res.append(r)
            if i % 100 == 0: print(f"  {i}/{len(tasks)}", flush=True)
    out = args.root / "2026-06-09_phase5_libaom_scored.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "case_id", "dice_avg", "hd95_avg", "dice_class_1", "avg_mbps", "bits_per_voxel"])
        for folder, cid, d, h in res:
            w.writerow([folder, cid, f"{d:.10f}", f"{h:.10f}", f"{d:.10f}", "", ""])
    print(f"Wrote {out}")
    per = defaultdict(list)
    for folder, _, d, _ in res:
        per[folder].append(d)
    for k in sorted(per):
        v = per[k]
        print(f"  {k:22s} n={len(v):3d} mean_dice={sum(v)/len(v):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--imagestr", type=Path, required=True)
    e.add_argument("--case_subset", type=Path, default=Path("/scratch/shirzarm/vcip/phase1/calibration_subset_50.txt"))
    e.add_argument("--all", action="store_true")
    e.add_argument("--root", type=Path, default=Path("/scratch/shirzarm/vcip/phase5_libaom"))
    e.add_argument("--workers", type=int, default=8)
    e.set_defaults(func=cmd_encode)
    s = sub.add_parser("score")
    s.add_argument("--gt_dir", type=Path, required=True)
    s.add_argument("--root", type=Path, default=Path("/scratch/shirzarm/vcip/phase5_libaom"))
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
