"""
Score-CAM parity reference for the shipped skin-disease pipeline.

Companion to parity_check.py (which covers the VAE + CNN stages). This file
covers Stage 3 — Score-CAM — and exists for ONE reason: to decisively separate
an on-device explainability bug from expected model behaviour.

It loads the SAME .tflite files the app ships (Flutter/assets/models) and
replicates every POST-MODEL arithmetic step of ai_service.dart `_runScoreCam`
exactly (validated bit-for-bit by scorecam_check/validate_mirror.py):
  * every resize is `dart_resize_linear`, an exact replica of package:image
    4.8.0 `copyResize(interpolation: linear)` — corner-aligned source mapping
    (dst * src/dst) with the bilinear result TRUNCATED into uint8. cv2.resize
    INTER_LINEAR is NOT equivalent (half-pixel-centre mapping + rounding); at
    the 10x10 -> 300x300 mask upsample the conventions differ on ~99% of pixels
    by a mean of 37/255, which visibly shifts the heatmap. This was the single
    biggest parity bug in the previous version of this file.
  * [0,255] input scale (EfficientNet rescales internally)
  * top-K = 10 channels by mean |activation|
  * per-channel min-max normalise -> .round().clamp -> uint8 -> linear
    upsample (normalise-BEFORE-upsample, matching the Dart code, not the
    textbook order)
  * masked forward passes batched in chunks of 4, exactly the app's camBatch
    invoke pattern (batched == sequential, verified off-device)
  * scores -> softmax weights -> weighted sum of the recovered raw upsampled
    maps accumulated through float32 like the app's Float32List, ReLU,
    min-max to [0,1]
  * Dart `_jetColor` + 60/40 blend using Dart's .round() (half away from
    zero — NOT Python/NumPy round-half-to-even) and JPEG quality 90
  * predicted class for the rationale = the app's REAL prediction path
    (`_runCnnEnsemble`): B2@260 + B3@300, 4-flip TTA in one batched invoke
    per backbone, ensemble mean, `_tineaPriorScale` calibration
  * the same centroid/spread/coverage rationale text as `_heatmapRationale`

What this CANNOT match bit-for-bit (measured, not assumed — see
scorecam_check/xnnpack_equiv.py and decisive_heatmap_diff.py):
  * TFLite KERNELS. By default this runs the XNNPACK delegate (the runtime's
    fast CPU path). The app ships withOUT XNNPACK — but only because the
    delegate SEGFAULTS on some arm64 Android SoCs, a crash workaround, NOT a
    correctness fix. Measured on a real image, XNNPACK vs the reference kernels
    on THIS machine give: identical predicted class, calibrated probs equal to
    ~2e-3, and per-class heatmaps correlated at Pearson 0.99 (max local diff
    0.13, RMS 0.03) — i.e. the same WHERE-did-it-look story, which is all this
    diagnostic needs. Pass --faithful to instead run the reference kernels
    (the device's exact algorithm; ~12-18x slower) when you want the closest
    achievable numbers.
  * CPU ARCHITECTURE. x86 here vs arm64 on device differ in the last float
    ulps even on identical kernels, so cross-device probs match to ~1e-5 and
    overlays are visually identical, never bit-identical.
  * JPEG DECODING. cv2 (libjpeg-turbo) and package:image's pure-Dart decoder
    differ by up to a few LSB per pixel — feed both sides a PNG for the
    strictest byte-level comparison.

Bottom line: treat this as a STRUCTURAL parity oracle. A real integration bug
(wrong resize, wrong scale, wrong class) yields a grossly different heatmap
(low correlation / wrong location); a 0.99-correlated overlay with the same
prediction and rationale means the on-device Score-CAM is behaving correctly
and any "looks off" complaint is the model attending to the wrong region.

How to use:
  1. Run the app on one disease image; from the device logs read:
         [ScoreCAM] done — featShape=...
     and open the Score-CAM overlay + read the rationale shown in Results.
  2. Run:  python score_cam.py path/to/the_same_image.jpg   [--faithful]
  3. Compare the saved <image>_scorecam_<class>.jpg overlays and the printed
     rationale to the device (see the structural-vs-bitwise note above).

Constants below MUST match ai_service.dart.
"""
import math
import os
import sys
import time

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        OpResolverType = tf.lite.experimental.OpResolverType

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
CLASSES = ["Acne", "Eczema", "Tinea"]      # alphabetical = training order

# --- must match ai_service.dart ---
B2_W, B2_H = 260, 260                       # _b2W, _b2H
B3_W, B3_H = 300, 300                       # _b3W, _b3H
TOP_K = 10                                  # _topK
MAX_W = 1280                                # Dart downsample cap (mirrors `original`)
CAM_BATCH = 4                               # camBatch — masked passes per invoke
USE_TTA = True                              # _useTta
TINEA_PRIOR_SCALE = 1.0                     # _tineaPriorScale


def _interp(name, xnnpack=True):
    """Build an interpreter with min(4, cores) threads — the app's thread count.

    xnnpack=True (default) uses the runtime's fast XNNPACK CPU path: ~12-18x
    faster, and measured to leave the predicted class, calibrated probs (~2e-3)
    and per-class heatmaps (Pearson 0.99) effectively unchanged vs the device.

    xnnpack=False forces BUILTIN_WITHOUT_DEFAULT_DELEGATES — the reference
    kernels, which is the EXACT path the app runs (it omits XNNPACK to dodge an
    arm64 segfault, not for correctness). Slower, but the closest achievable
    numbers to the device. Falls back to the default resolver on runtimes
    lacking the kwarg."""
    path = os.path.join(MODELS, name)
    threads = min(4, os.cpu_count() or 1)
    if xnnpack:
        return _allocated(Interpreter(model_path=path, num_threads=threads))
    try:
        it = Interpreter(
            model_path=path, num_threads=threads,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except (TypeError, ValueError):
        it = Interpreter(model_path=path, num_threads=threads)
    return _allocated(it)


def _allocated(it):
    it.allocate_tensors()
    return it


# Default to the fast path so importers (e.g. xai_eval.py) get the speedup;
# main() rebuilds these as reference kernels when --faithful is passed.
FEAT = _interp("b3_feature_extractor.tflite")
B2 = _interp("cnn_b2_model.tflite")
B3 = _interp("cnn_b3_model.tflite")


def use_reference_kernels():
    """Rebuild the module interpreters on the app's exact no-XNNPACK path."""
    global FEAT, B2, B3
    FEAT = _interp("b3_feature_extractor.tflite", xnnpack=False)
    B2 = _interp("cnn_b2_model.tflite", xnnpack=False)
    B3 = _interp("cnn_b3_model.tflite", xnnpack=False)


# -----------------------------------------------------------------------------
# Dart arithmetic mirrors
# -----------------------------------------------------------------------------
def dart_round(x):
    """Dart num.round() for the non-negative values used here: halves round
    AWAY from zero. Python round() / np.round() round half to EVEN, and the
    naive floor(x + 0.5) rounds up floats just below a .5 boundary, so neither
    is a faithful substitute."""
    x = np.asarray(x, np.float64)
    f = np.floor(x)
    return np.where(x - f >= 0.5, f + 1.0, f)


def dart_resize_linear(src, dst_w, dst_h):
    """Exact replica of package:image 4.8.0 copyResize(interpolation: linear)
    (lib/src/transform/copy_resize.dart) on a uint8 (H,W) or (H,W,3) array —
    the resize used by EVERY stage of the Dart pipeline.

    Differs from cv2.resize INTER_LINEAR in both convention and storage:
      * source position = dst * (src/dst), corner-aligned with truncated index
        (cv2 uses the half-pixel-centre convention (dst+0.5)*src/dst - 0.5)
      * the bilinear value is stored with .toInt() truncation
        (PixelUint8.setRgba), not rounded
    The bilinear weights use the same factored expression as Dart's `_linear`
    helper, evaluated in float64, so outputs are bit-identical to the app.
    """
    src_h, src_w = src.shape[:2]
    if dst_w == src_w and dst_h == src_h:
        return src.copy()                       # copyResize returns src.clone()
    s = src.astype(np.float64)
    fy = np.arange(dst_h, dtype=np.float64) * (src_h / dst_h)
    iy = fy.astype(np.int64)                    # fy.toInt() — truncation
    ky = fy - iy
    ny = np.minimum(iy + 1, src_h - 1)
    fx = np.arange(dst_w, dtype=np.float64) * (src_w / dst_w)
    ix = fx.astype(np.int64)
    kx = fx - ix
    nx = np.minimum(ix + 1, src_w - 1)
    icc = s[np.ix_(iy, ix)]
    inc = s[np.ix_(iy, nx)]
    icn = s[np.ix_(ny, ix)]
    inn = s[np.ix_(ny, nx)]
    if s.ndim == 3:
        kx, ky = kx[None, :, None], ky[:, None, None]
    else:
        kx, ky = kx[None, :], ky[:, None]
    out = icc + kx * (inc - icc + ky * (icc + inn - icn - inc)) + ky * (icn - icc)
    return out.astype(np.uint8)                 # setRgba .toInt() — truncation


def load_rgb(path):
    """Mirror the app's `original`: decode + bakeOrientation (cv2.imread applies
    EXIF orientation by default, same effect) + copyResize(width: 1280, linear)
    when wider. Dart derives height as (width * srcH/srcW).round()."""
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"Could not read image: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # uint8, like _toUint8Rgb guarantees
    h, w = img.shape[:2]
    if w > MAX_W:
        new_h = int(dart_round(MAX_W * (h / w)))
        img = dart_resize_linear(img, MAX_W, new_h)
    return img


# -----------------------------------------------------------------------------
# Interpreter helpers
# -----------------------------------------------------------------------------
def _invoke(interp, batch):
    """Run a (N,H,W,3) float32 batch, resizing the input tensor when the batch
    shape changes (mirrors resizeInputTensor + allocateTensors in the app)."""
    inp = interp.get_input_details()[0]
    if list(inp["shape"]) != list(batch.shape):
        interp.resize_tensor_input(inp["index"], list(batch.shape))
        interp.allocate_tensors()
    interp.set_tensor(inp["index"], np.ascontiguousarray(batch, np.float32))
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"])


def _run(interp, x):
    """Single-sample helper (kept for xai_eval.py): (1,H,W,3) in, row 0 out."""
    return _invoke(interp, np.asarray(x, np.float32))[0]


# -----------------------------------------------------------------------------
# App prediction path (mirrors `_runCnnEnsemble` + `_calibrateProbs` + `_argmax`)
# -----------------------------------------------------------------------------
def app_predict(original_rgb):
    """The class the APP would predict — B2@260 + B3@300, each averaged over the
    4 flip TTA views in ONE batched invoke, ensemble mean, Tinea prior scale.
    The old single-plain-B3 pass here could disagree with the app on borderline
    images, mis-targeting the rationale."""
    per_backbone = []
    for interp, size in ((B2, B2_W), (B3, B3_W)):
        resized = dart_resize_linear(original_rgb, size, size)
        views = [resized]
        if USE_TTA:                            # identity, horizontal, vertical, both
            views += [resized[:, ::-1], resized[::-1], resized[::-1, ::-1]]
        out = _invoke(interp, np.stack(views).astype(np.float32))  # (n, 3)
        # Dart accumulates each view's float32 probs into doubles, then * 1/n.
        sums = [0.0] * len(CLASSES)
        for k in range(len(views)):
            for i in range(len(CLASSES)):
                sums[i] += float(out[k, i])
        inv = 1.0 / len(views)
        per_backbone.append([v * inv for v in sums])
    raw = [(per_backbone[0][i] + per_backbone[1][i]) / 2.0
           for i in range(len(CLASSES))]
    w = [raw[0], raw[1], raw[2] * TINEA_PRIOR_SCALE]
    total = w[0] + w[1] + w[2]
    cal = [v / total for v in w]
    pred = 0
    for i in range(1, len(cal)):               # first max, like _argmax
        if cal[i] > cal[pred]:
            pred = i
    return cal, pred


# -----------------------------------------------------------------------------
# Score-CAM (mirrors `_runScoreCam` + `_assembleHeatmap`)
# -----------------------------------------------------------------------------
def run_score_cam(original_rgb):
    """
    Args:
        original_rgb : (H, W, 3) uint8 RGB, already downsampled to <=MAX_W.
    Returns:
        heatmaps     : dict class -> (B3_H, B3_W) float32 in [0,1]
        feat_shape   : (featH, featW, featC)
    """
    img300_u8 = dart_resize_linear(original_rgb, B3_W, B3_H)
    img_b3 = img300_u8.astype(np.float32)       # imgBuf — [0,255] integer floats
    img64 = img_b3.astype(np.float64)

    # ── Extract feature maps (featH, featW, featC) ──
    features = _invoke(FEAT, img_b3[np.newaxis])[0]
    feat_h, feat_w, feat_c = features.shape

    # ── Select top-K channels by mean |activation| ──
    # (np sum order differs from Dart's sequential loop only in the last ulp;
    # neither side's sort is stable under exact ties, which don't occur on
    # real float means)
    means = np.abs(features.astype(np.float64)).sum(axis=(0, 1)) / (feat_h * feat_w)
    top_k_idx = np.argsort(-means, kind="stable")[:TOP_K]

    raw_upsampled = np.empty((TOP_K, B3_H, B3_W), np.float32)
    masked = np.empty((TOP_K, B3_H, B3_W, 3), np.float32)
    for ki, ch in enumerate(top_k_idx):
        raw_chan = features[:, :, ch]           # (featH, featW) float32
        c_min, c_max = float(raw_chan.min()), float(raw_chan.max())
        rng = (c_max - c_min) if abs(c_max - c_min) > 1e-8 else 1.0

        # normalise-BEFORE-upsample: ((v - cMin) / range * 255).round().clamp
        # -> uint8 -> copyResize linear (matches Dart; round is half-away)
        chan_u8 = np.clip(
            dart_round((raw_chan.astype(np.float64) - c_min) / rng * 255),
            0, 255).astype(np.uint8)
        mask = dart_resize_linear(chan_u8, B3_W, B3_H).astype(np.float64) / 255.0

        # recover approx raw upsampled value; the app stores both this and the
        # masked image into Float32Lists, hence the float32 casts
        raw_upsampled[ki] = (mask * rng + c_min).astype(np.float32)
        masked[ki] = (img64 * mask[:, :, np.newaxis]).astype(np.float32)

    # ── Masked forward passes, batched like the app's camBatch chunks ──
    scores = np.empty((TOP_K, len(CLASSES)), np.float64)
    for base in range(0, TOP_K, CAM_BATCH):
        chunk = masked[base:base + CAM_BATCH]
        out = _invoke(B3, chunk)
        scores[base:base + chunk.shape[0]] = out[:chunk.shape[0]].astype(np.float64)

    # ── Assemble one heatmap per class ──
    heatmaps = {}
    for cls, name in enumerate(CLASSES):
        s = scores[:, cls]
        m = float(s.max())
        e = [math.exp(float(v) - m) for v in s]
        tot = 0.0
        for v in e:                             # sequential, like .reduce(+)
            tot += v
        wgt = [v / tot for v in e]              # softmax weights

        # The app accumulates into a Float32List — the running sum is rounded
        # to float32 after EVERY channel. Replicate that stepped precision.
        hm = np.zeros((B3_H, B3_W), np.float32)
        for ki in range(TOP_K):
            hm = (hm.astype(np.float64)
                  + wgt[ki] * raw_upsampled[ki].astype(np.float64)
                  ).astype(np.float32)
        np.maximum(hm, 0, out=hm)               # ReLU
        h_min, h_max = float(hm.min()), float(hm.max())
        h_rng = (h_max - h_min) if (h_max - h_min) > 1e-8 else 1.0
        heatmaps[name] = ((hm.astype(np.float64) - h_min) / h_rng).astype(np.float32)
    return heatmaps, (feat_h, feat_w, feat_c)


# -----------------------------------------------------------------------------
# Overlay + rationale (mirror `_jetColor`, `_saveHeatmapOverlay`, `_heatmapRationale`)
# -----------------------------------------------------------------------------
def jet_rgb_u8(t):
    """Vectorised Dart `_jetColor` (NOT cv2 COLORMAP_JET): float map in, uint8
    RGB out, with Dart's half-away .round()."""
    t = np.clip(t.astype(np.float64), 0.0, 1.0)
    r = np.empty_like(t)
    g = np.empty_like(t)
    b = np.empty_like(t)
    m0 = t < 0.125
    m1 = (t >= 0.125) & (t < 0.375)
    m2 = (t >= 0.375) & (t < 0.625)
    m3 = (t >= 0.625) & (t < 0.875)
    m4 = t >= 0.875
    r[m0], g[m0], b[m0] = 0.0, 0.0, 0.0
    b[m0] = 0.5 + t[m0] * 4
    r[m1], b[m1] = 0.0, 1.0
    g[m1] = (t[m1] - 0.125) * 4
    g[m2] = 1.0
    r[m2] = (t[m2] - 0.375) * 4
    b[m2] = 1.0 - (t[m2] - 0.375) * 4
    r[m3], b[m3] = 1.0, 0.0
    g[m3] = 1.0 - (t[m3] - 0.625) * 4
    g[m4], b[m4] = 0.0, 0.0
    r[m4] = 1.0 - (t[m4] - 0.875) * 4
    rgb = np.stack([r, g, b], axis=-1) * 255
    return np.clip(dart_round(rgb), 0, 255).astype(np.uint8)


def save_overlay(heatmap, original_rgb, out_path):
    """Jet-colour the [0,1] heatmap, copyResize-linear it to the photo's size,
    60/40 blend with Dart rounding, write JPEG quality 90 (the app's value)."""
    jet = jet_rgb_u8(heatmap)                                   # (300,300,3)
    h, w = original_rgb.shape[:2]
    jet_up = dart_resize_linear(jet, w, h).astype(np.float64)
    blend = original_rgb.astype(np.float64) * 0.6 + jet_up * 0.4
    overlay = np.clip(dart_round(blend), 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def rationale(heatmap, class_name):
    """Mirror `_heatmapRationale`: centroid (location), spread (focal vs
    diffuse), coverage. Wording thresholds identical; coverage % uses Dart
    rounding."""
    n = B3_W * B3_H
    ys, xs = np.mgrid[0:B3_H, 0:B3_W]
    v = heatmap.astype(np.float64)
    s = float(v.sum())
    high_count = int((v > 0.5).sum())
    if s < 1e-6:
        return f"The attention map for {class_name} was diffuse with no dominant region."
    cx = float((v * xs).sum() / s) / B3_W
    cy = float((v * ys).sum() / s) / B3_H
    ccx, ccy = cx * B3_W, cy * B3_H
    var_sum = float((v * ((xs - ccx) ** 2 + (ys - ccy) ** 2)).sum())
    spread = math.sqrt(var_sum / s) / math.sqrt(B3_W * B3_W + B3_H * B3_H)
    coverage = high_count / n

    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "central")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    loc = "the centre of the image" if (vert == "central" and horiz == "centre") \
        else f"the {vert}-{horiz} region"
    focal = "a single focal area" if spread < 0.22 \
        else ("a moderately spread area" if spread < 0.32 else "a diffuse area")
    return (f"For {class_name}, the model concentrated on {focal} in {loc}, with "
            f"high-attention pixels covering about {int(dart_round(coverage * 100))}% of the image.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    faithful = "--faithful" in sys.argv
    if not args:
        sys.exit("usage: python score_cam.py <image_path> [--faithful]")
    path = args[0]
    kernels = "reference (no XNNPACK — the device's exact path)" if faithful \
        else "XNNPACK (fast; heatmap Pearson 0.99 vs reference)"
    if faithful:
        use_reference_kernels()
    print(f"kernels: {kernels}")
    t0 = time.perf_counter()
    img = load_rgb(path)
    print(f"image: {path}  ({img.shape[1]}x{img.shape[0]} after downsample)\n")

    # Predicted class via the app's real path (ensemble + TTA + calibration),
    # so the rationale targets the same class the app would show.
    t1 = time.perf_counter()
    probs, pred_idx = app_predict(img)
    pred = CLASSES[pred_idx]
    t2 = time.perf_counter()
    print("[CNN] " + " ".join(f"{c}={probs[i]:.3f}" for i, c in enumerate(CLASSES))
          + f"  -> target={pred}  (B2+B3 ensemble, TTA, calibrated — the app's path)\n")

    heatmaps, (fh, fw, fc) = run_score_cam(img)
    t3 = time.perf_counter()
    print(f"[ScoreCAM] featShape={fh}x{fw}x{fc}  topK={TOP_K}\n")

    base = os.path.splitext(path)[0]
    for name in CLASSES:
        out = f"{base}_scorecam_{name}.jpg"
        save_overlay(heatmaps[name], img, out)
        tag = "  <- predicted (rationale target)" if name == pred else ""
        print(f"  saved {out}{tag}")
    t4 = time.perf_counter()

    print("\nRATIONALE (predicted class):")
    print("  " + rationale(heatmaps[pred], pred))
    print(f"\n[TIMING] load={t1 - t0:.2f}s cnn={t2 - t1:.2f}s "
          f"scorecam={t3 - t2:.2f}s overlays={t4 - t3:.2f}s "
          f"total={t4 - t0:.2f}s")


if __name__ == "__main__":
    main()
