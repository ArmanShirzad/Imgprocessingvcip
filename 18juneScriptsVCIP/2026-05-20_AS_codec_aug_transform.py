#!/usr/bin/env python3
"""
2026-05-20_AS_codec_aug_transform.py
------------------------------------
VCIP 2026 — on-the-fly codec augmentation for Phase 4 fine-tuning.

Derives from the smoke-tested `lastphase/codec_aug_transform.py` (Feb 9 2026
smoke run at `/scratch/shirzarm/lastphase/smoke_out_20260209_182805/`).
The original supports H.264 + AV1; this rewrite adds H.265 and locks in the
post-2026-05-20 design choices:

  * 20% CLEAN per batch (anti-catastrophic-forgetting anchor, per Alireza)
  * 3-codec mix (H.264 + H.265 + AV1) — VVC dropped per Alireza 2026-05-20
  * CRF pools drawn from Alireza's 6-op grid {18, 23, 28, 33, 38, 43} plus
    one moderate-heavy slot per codec to stretch the model toward the
    edge of the safe range
  * `--held_out_codec` switch removes one codec from the sampling pool for
    leave-one-out experiments A/B/C

Reference design lives in `12-5-2026/dataset504_catastrophic_forgetting_analysis.md` —
this transform is engineered to invert every Dataset 504 failure cause.

Used by `nnUNetTrainerCodecAug505_v2.py` (Willi-side). API matches the
existing nnU-Net augmentation transform contract: `__call__(self, **data_dict)`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

# ----------------------------------------------------------------- config

FPS = 30
GOP = 300

CLEAN_PROBABILITY = 0.20  # Alireza 2026-05-20

# CRF pools — drawn from {18, 23, 28, 33, 38, 43} per Alireza, with one
# moderate-heavy extra per codec to keep the model honest near the edge.
CRF_POOLS = {
    "libx264":   [18, 23, 28, 33, 38, 43],
    "libx265":   [18, 23, 28, 33, 38, 43],
    "libsvtav1": [18, 23, 28, 33, 38, 43],
}

CODEC_LIST = list(CRF_POOLS.keys())


# ------------------------------------------------------------ encode pass

def _encode_decode(arr_u8: np.ndarray, codec: str, crf: int) -> np.ndarray:
    """Round-trip a (Z, H, W) uint8 volume through ffmpeg. Returns decoded array."""
    z, h, w = arr_u8.shape
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        arr_u8 = np.pad(arr_u8, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        h, w = arr_u8.shape[1], arr_u8.shape[2]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = Path(tf.name)
    try:
        # Encode
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
            "-an", "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
        ]
        if codec == "libx264":
            cmd += ["-g", str(GOP), "-x264-params", "no-scenecut=1"]
        elif codec == "libx265":
            # pools=2 caps x265 thread fanout (default = all node cores,
            # which oversubscribes our 4-CPU slurm cgroup and may have
            # contributed to the 1720 SIGSEGV-style crash).
            cmd += ["-x265-params", f"keyint={GOP}:min-keyint={GOP}:pools=2:numa-pools=2"]
        elif codec == "libsvtav1":
            cmd += ["-preset", "8"]
        cmd.append(str(video_path))

        # timeout=60s prevents an indefinite hang if a codec stalls instead
        # of exiting cleanly. Real encodes take <30s; 60s is a generous cap.
        try:
            p = subprocess.run(cmd, input=arr_u8.tobytes(),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=60)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"encode timed out (60s) ({codec} crf={crf})")
        if p.returncode != 0:
            raise RuntimeError(f"encode failed ({codec} crf={crf}): {p.stderr.decode()[:300]}")

        # Decode
        dec_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-an", "-frames:v", str(z),
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        try:
            p = subprocess.run(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=60)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"decode timed out (60s) ({codec} crf={crf})")
        if p.returncode != 0:
            raise RuntimeError(f"decode failed: {p.stderr.decode()[:300]}")
        buf = np.frombuffer(p.stdout, dtype=np.uint8)
        arr = buf.reshape(z, h, w)
        # un-pad
        orig_h = arr.shape[1] - pad_h
        orig_w = arr.shape[2] - pad_w
        return arr[:, :orig_h, :orig_w].copy()
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass


@lru_cache(maxsize=128)
def _cached_round_trip(arr_bytes: bytes, shape: tuple[int, int, int], codec: str, crf: int) -> bytes:
    """LRU-cached round-trip — identical (vol, codec, crf) returns same bytes."""
    arr = np.frombuffer(arr_bytes, dtype=np.uint8).reshape(shape)
    out = _encode_decode(arr, codec, crf)
    return out.tobytes()


# -------------------------------------------------------- transform class

class CodecAugmentTransform:
    """nnU-Net augmentation transform: round-trips images through random codecs.

    With probability CLEAN_PROBABILITY, the image is left untouched. Otherwise
    a random (codec, CRF) is sampled and the image is encoded -> decoded.

    Args:
        held_out_codec: one of {"libx264", "libx265", "libsvtav1"} or None.
                        If set, that codec is excluded from the sampling pool
                        (used for codec-split leave-one-out experiments).
        seed: rng seed for sampling — useful for reproducibility in smoke tests.
        cache: enable LRU cache. Disable for stress tests of true randomness.
    """

    def __init__(
        self,
        held_out_codec: Optional[str] = None,
        seed: Optional[int] = None,
        cache: bool = True,
    ):
        if held_out_codec and held_out_codec not in CODEC_LIST:
            raise ValueError(
                f"held_out_codec={held_out_codec!r} not in {CODEC_LIST}"
            )
        self.codec_pool = [c for c in CODEC_LIST if c != held_out_codec]
        self.held_out_codec = held_out_codec
        self.rng = np.random.default_rng(seed)
        self.cache = cache

    def _sample(self) -> Optional[tuple[str, int]]:
        if self.rng.random() < CLEAN_PROBABILITY:
            return None  # clean pass-through
        codec = self.rng.choice(self.codec_pool)
        crf = int(self.rng.choice(CRF_POOLS[codec]))
        return codec, crf

    # batchgeneratorsv2 (new nnU-Net) calls transforms with key "image";
    # legacy batchgenerators (v1) used "data". Support both.
    _IMG_KEYS = ("image", "data")

    def _get_img_key(self, data_dict: dict) -> str:
        for k in self._IMG_KEYS:
            if k in data_dict:
                return k
        raise KeyError(
            f"Codec aug expected one of {self._IMG_KEYS} in data_dict; "
            f"got keys={list(data_dict.keys())}"
        )

    def __call__(self, **data_dict):
        """Apply codec round-trip to a single training sample.

        batchgeneratorsv2 invokes transforms one batch element at a time, so
        the image tensor has shape (C, Z, H, W) for 3D_FULLRES or (C, H, W)
        for 2D. There is NO batch axis at this stage. We normalise to uint8
        per-channel via min/max, encode/decode each channel as a video, then
        de-normalise back to the original dtype.
        """
        img_key = self._get_img_key(data_dict)
        img = data_dict[img_key]

        sample = self._sample()
        if sample is None:
            return data_dict  # clean pass-through

        codec, crf = sample

        try:
            arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
        except Exception:
            arr = np.asarray(img)

        # Expand to a leading "batch" axis of size 1 so the (B, C, ...) code
        # path below works uniformly for both batched and per-sample inputs.
        added_batch = arr.ndim == 4 or arr.ndim == 3  # (C,Z,H,W) or (C,H,W)
        if added_batch:
            arr_b = arr[np.newaxis, ...]
        else:
            arr_b = arr  # already has batch axis

        # 2D path: (B, C, H, W) — codec_aug runs frame-by-frame which is
        # nonsensical for 2D nnU-Net (one slice = one frame). For now we
        # skip aug on anything that isn't 3D. 3D_FULLRES is the actual scope.
        if arr_b.ndim < 5:
            return data_dict

        u8 = self._to_uint8(arr_b)
        processed = np.empty_like(u8)
        for b in range(u8.shape[0]):
            for c in range(u8.shape[1]):
                vol = u8[b, c]  # (Z, H, W)
                shape = vol.shape
                try:
                    if self.cache:
                        out_bytes = _cached_round_trip(vol.tobytes(), shape, codec, crf)
                        processed[b, c] = np.frombuffer(out_bytes, dtype=np.uint8).reshape(shape)
                    else:
                        processed[b, c] = _encode_decode(vol, codec, crf)
                except (RuntimeError, ValueError, OSError) as e:
                    # Job 1720 (2026-06-04): libx265 crashed on a specific shape mid-train,
                    # raised RuntimeError → killed dataloader worker → nnUNet aborted.
                    # Over 50k+ encodes per training run even a 0.01% codec failure rate is
                    # fatal. Fall back to clean (the sample becomes an effective clean-pass
                    # for that iteration) so training keeps going.
                    print(f"[codec_aug] WARN encode fail ({codec} crf={crf} shape={shape}): "
                          f"{type(e).__name__}: {str(e)[:120]} — using clean fallback", flush=True)
                    processed[b, c] = vol

        out_arr = self._from_uint8(processed, arr_b)
        if added_batch:
            out_arr = out_arr[0]  # squeeze the synthetic batch axis

        # Preserve the original tensor type (torch.Tensor stays a torch.Tensor)
        if hasattr(img, "cpu"):
            import torch
            data_dict[img_key] = torch.from_numpy(out_arr).to(img.device, dtype=img.dtype)
        else:
            data_dict[img_key] = out_arr

        data_dict["codec_aug_meta"] = {"codec": codec, "crf": crf}
        return data_dict

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        """Map float input to uint8 [0, 255] using per-(B,C) min/max."""
        a = arr.astype(np.float32)
        lo = a.min(axis=tuple(range(2, a.ndim)), keepdims=True)
        hi = a.max(axis=tuple(range(2, a.ndim)), keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        out = ((a - lo) / denom * 255.0).clip(0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _from_uint8(u8: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Reverse _to_uint8 using ref's per-(B,C) min/max."""
        a = ref.astype(np.float32)
        lo = a.min(axis=tuple(range(2, a.ndim)), keepdims=True)
        hi = a.max(axis=tuple(range(2, a.ndim)), keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        out = (u8.astype(np.float32) / 255.0) * denom + lo
        return out.astype(ref.dtype)


# ---------------------------------------------------------- self-test

if __name__ == "__main__":
    # Quick local check (no nnU-Net) — confirms sample distribution
    t = CodecAugmentTransform(seed=42)
    counts = {"clean": 0}
    for c in CODEC_LIST:
        counts[c] = 0
    n = 1000
    for _ in range(n):
        s = t._sample()
        if s is None:
            counts["clean"] += 1
        else:
            counts[s[0]] += 1
    print(f"Distribution over {n} samples (target: clean ~{CLEAN_PROBABILITY:.0%}, "
          f"each codec ~{(1-CLEAN_PROBABILITY)/len(CODEC_LIST):.1%}):")
    for k, v in counts.items():
        print(f"  {k:>10s}: {v:>4d} ({v/n*100:.1f}%)")
