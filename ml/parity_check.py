"""
Dart <-> Python parity check on a SINGLE image.

Purpose: decisively separate "Flutter integration bug" from "model/data
problem". It loads the SAME .tflite files the app ships (Flutter/assets/models)
and mirrors ai_service.dart's CNN path exactly, printing probabilities in the
same format as the app's developer.log lines so you can eyeball-compare.

How to use:
  1. Run the app on one image and read these lines from the device logs:
         [CNN-B2] Acne=.. Eczema=.. Tinea=..
         [CNN-B3] Acne=.. Eczema=.. Tinea=..
         [CNN]    Acne=.. Eczema=.. Tinea=.. -> <diagnosis> (..%)
  2. Run:  python parity_check.py path/to/the_same_image.jpg
  3. Compare. If the two B2/B3 vectors agree to ~1-2%, the Flutter
     integration is CORRECT and any field errors are model/data (domain
     shift / dataset leakage), not code. If they diverge a lot, the
     preprocessing or wiring differs and that's the bug to chase.

Constants below MUST match ai_service.dart.
"""
import os
import sys
import numpy as np
import cv2

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
CLASSES = ["Acne", "Eczema", "Tinea"]          # alphabetical = training order

# --- must match ai_service.dart ---
B2, B3 = (260, 260), (300, 300)
MAX_W = 1280                                    # Dart downsample cap
USE_TTA = True                                  # _useTta
TINEA_PRIOR_SCALE = 1.0                         # _tineaPriorScale (no-op while 1.0)
MIN_CNN_CONFIDENCE = 0.40                       # _minCnnConfidence
MIN_CNN_MARGIN = 0.10                           # _minCnnMargin
# VAE gate
PATCH, STRIDE = 64, 64
ANOM_THRESH, ANOM_RATIO = 0.004, 0.20

b2 = Interpreter(model_path=os.path.join(MODELS, "cnn_b2_model.tflite")); b2.allocate_tensors()
b3 = Interpreter(model_path=os.path.join(MODELS, "cnn_b3_model.tflite")); b3.allocate_tensors()
_vae_path = os.path.join(MODELS, "vae_model.tflite")
vae = Interpreter(model_path=_vae_path); vae.allocate_tensors()


def _run(interp, x):
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(i["index"], x.astype(np.float32)); interp.invoke()
    return interp.get_tensor(o["index"])[0]


def _tta_views(x):
    # identity, horizontal flip, vertical flip, both — matches ai_service.dart.
    if not USE_TTA:
        return [x]
    return [x, x[:, ::-1], x[::-1, :], x[::-1, ::-1]]


def _cnn_one(interp, img, size):
    r = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)   # [0,255]
    return np.mean([_run(interp, v[np.newaxis].astype(np.float32))
                    for v in _tta_views(r)], axis=0)


def load_rgb(path):
    """Mirror the Dart `original`: RGB, downsample to <=MAX_W (linear)."""
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"Could not read image: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[1] > MAX_W:
        h = int(img.shape[0] * MAX_W / img.shape[1])
        img = cv2.resize(img, (MAX_W, h), interpolation=cv2.INTER_LINEAR)
    return img


def fmt(p):
    return " ".join(f"{c}={p[i]:.3f}" for i, c in enumerate(CLASSES))


def calibrate(p):
    """Mirror _calibrateProbs: scale Tinea prior then renormalise."""
    w = np.array([p[0], p[1], p[2] * TINEA_PRIOR_SCALE], dtype=np.float64)
    return w / w.sum()


def vae_ratio(img):
    h, w, _ = img.shape
    if h < PATCH or w < PATCH:
        return 0.0
    anom = total = 0
    for y in range(0, h - PATCH + 1, STRIDE):
        for x in range(0, w - PATCH + 1, STRIDE):
            patch = img[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0
            if float(np.ravel(_run(vae, patch[np.newaxis]))[0]) > ANOM_THRESH:
                anom += 1
            total += 1
    return anom / total if total else 0.0


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python parity_check.py <image_path>")
    path = sys.argv[1]
    img = load_rgb(path)
    print(f"image: {path}  ({img.shape[1]}x{img.shape[0]} after downsample)\n")

    # --- VAE gate ---
    ratio = vae_ratio(img)
    gate = "ANOMALY" if ratio > ANOM_RATIO else "NORMAL"
    print(f"[VAE] ratio={ratio:.3f} threshold={ANOM_RATIO} decision={gate}")
    if gate == "NORMAL":
        print("  -> pipeline would stop here and report 'No Disease Detected'.")

    # --- CNN ensemble (always shown so you can compare even if VAE gates) ---
    p2 = _cnn_one(b2, img, B2)
    p3 = _cnn_one(b3, img, B3)
    print(f"[CNN-B2] {fmt(p2)}")
    print(f"[CNN-B3] {fmt(p3)}")

    raw = (np.asarray(p2) + np.asarray(p3)) / 2.0
    cal = calibrate(raw)
    pred = int(np.argmax(cal))
    conf = float(cal[pred])
    s = np.sort(cal)
    margin = float(s[-1] - s[-2])
    print(f"[CNN] {fmt(cal)} -> {CLASSES[pred]} ({conf * 100:.1f}%)  margin={margin * 100:.1f}%")

    uncertain = conf < MIN_CNN_CONFIDENCE or margin < MIN_CNN_MARGIN
    if gate == "NORMAL":
        decision = "No Disease Detected (VAE gate)"
    elif uncertain:
        decision = "No Disease Detected (CNN uncertain)"
    else:
        decision = f"{CLASSES[pred]} ({conf * 100:.1f}%)"
    print(f"\nFINAL DECISION: {decision}")


if __name__ == "__main__":
    main()
