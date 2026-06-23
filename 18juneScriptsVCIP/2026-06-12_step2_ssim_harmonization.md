# Step 2 — SSIM harmonization analysis (VCIP, Alireza roadmap)

Inputs: Step 1 PSNR/SSIM table (full 299-vol set, job after 1858 fix).
Targets: H.264 SSIM at CRF {18, 23, 28, 33, 38} (CRF 43 dropped per Alireza).

## H.264 SSIM anchors
| H.264 CRF | SSIM | Bitrate (Mbps) |
|---|---|---|
| 18 | 0.9568 | 2.020 |
| 23 | 0.9181 | 0.850 |
| 28 | 0.8659 | 0.347 |
| 33 | 0.8036 | 0.149 |
| 38 | 0.7267 | 0.069 |

## CRF needed to match each H.264 SSIM target (linear interp in CRF-SSIM)
| H.264 target SSIM | HEVC CRF | AV1 CRF | Note |
|---|---|---|---|
| 0.9568 (CRF18) | 18.0 | 18.3 | both at grid top |
| 0.9181 (CRF23) | 23.2 | **30.4** | AV1 needs NEW CRF ~30 |
| 0.8659 (CRF28) | 28.4 | **41.3** | AV1 needs NEW CRF ~41 |
| 0.8036 (CRF33) | 33.5 | 53.5 | AV1 UNREACHABLE (past CRF43) |
| 0.7267 (CRF38) | 38.3 | 68.6 | AV1 UNREACHABLE (past CRF43) |

## Findings
1. **HEVC tracks H.264 SSIM almost exactly at the same CRF.** Every match lands
   within ~0.5 CRF of the existing grid point. HEVC needs **essentially no new
   encodes** to harmonize; at most a precise CRF 34 for the 0.8036 target.
2. **AV1 is far stronger per CRF (~7-13 CRF "ahead").** To match H.264's SSIM it
   needs HIGHER CRFs: ~30 for the 0.918 target and ~41 for the 0.866 target.
   These are the genuinely new AV1 encodes.
3. **AV1 cannot reach H.264's two lowest-quality targets (0.804, 0.727) within
   CRF <= 43.** Matching them would require AV1 CRF ~53 and ~68 -- exactly the
   regime Alireza flagged as "quality degrades significantly after CRF 43."
   So SSIM harmonization is only meaningful over the upper three distortion
   levels for AV1; the cliff region cannot be harmonized and should stay on the
   native-CRF comparison. (This actually reinforces the paper's existing finding
   that AV1 degrades gracefully and does not exhibit H.264's low-bitrate cliff.)

## Recommended new CRFs to encode (Step 3)
- **AV1 (new): 30, 41** for SSIM-matching to H.264 CRF23 / CRF28, **plus
  Alireza's explicit 35, 36, 37** to close the 0.379->0.611 Mbps gap in Table III.
  -> AV1 new set = {30, 35, 36, 37, 41} (5 encodes x 299 vols).
- **HEVC (new): optionally 34** for a precise 0.8036 match; otherwise the
  existing grid already harmonizes within SSIM noise. -> HEVC new = {34} or none.
- This is ~5 new AV1 + 0-1 new HEVC, consistent with Alireza's "~3-4 new CRF
  per codec" for AV1 (he over-estimated HEVC; its intra-coding tracks H.264).

## What to send Alireza now
The Step 1 table + this mapping. Decision he owns: (a) confirm the AV1 new-CRF
set {30,35,36,37,41}; (b) accept that AV1's two lowest H.264 targets are
out-of-range and the harmonized comparison covers the top three levels only;
(c) the dropped CRF-43 anchor means the harmonized axis does not cover the
H.264 cliff -- the native-CRF rate-task curves (current Fig) still carry that.
