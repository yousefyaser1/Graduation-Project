"""
Device-equivalent pipeline verification.

Mirrors the Flutter `AIService` exactly (no TTA, single forward pass):
  - VAE gate: 64x64 patches, stride 32, input normalised to [0,1],
    MSE > 0.008 = anomalous patch; ratio > 0.20 = ANOMALY.
  - CNN ensemble: resize to 260/300 (bilinear), input kept in [0,255]
    (EfficientNet rescales internally), 50/50 average of softmax outputs.

Run:
  python verify_pipeline.py [N_PER_CLASS_CNN] [N_PER_CLASS_VAE]

Reports CNN confusion matrix + accuracy, and how often the VAE gate
labels a *diseased* image as NORMAL (a false "No Disease Detected").
"""
import os
import sys
import glob
import random
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
TEST = os.path.join(ROOT, "New_Augmented_Dataset", "test")
CLASSES = ["Acne", "Eczema", "Tinea"]          # alphabetical = training order
FOLDERS = {"Acne": "acne", "Eczema": "eczema", "Tinea": "tinea"}

PATCH, STRIDE = 64, 32
ANOM_THRESH, ANOM_RATIO = 0.004, 0.15   # must match ai_service.dart
B2, B3 = (260, 260), (300, 300)
MAX_W = 1280                                    # Dart downsample cap

vae = Interpreter(model_path=os.path.join(MODELS, "vae_model.tflite")); vae.allocate_tensors()
b2 = Interpreter(model_path=os.path.join(MODELS, "cnn_b2_model.tflite")); b2.allocate_tensors()
b3 = Interpreter(model_path=os.path.join(MODELS, "cnn_b3_model.tflite")); b3.allocate_tensors()


def _run(interp, x):
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(i["index"], x.astype(np.float32)); interp.invoke()
    return interp.get_tensor(o["index"])[0]


def load_rgb(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if img.shape[1] > MAX_W:                    # mirror Dart downsample
        h = int(img.shape[0] * MAX_W / img.shape[1])
        img = cv2.resize(img, (MAX_W, h), interpolation=cv2.INTER_LINEAR)
    return img


def cnn_predict(img):
    p2 = _run(b2, cv2.resize(img, B2)[np.newaxis].astype(np.float32))   # [0,255]
    p3 = _run(b3, cv2.resize(img, B3)[np.newaxis].astype(np.float32))
    return (p2 + p3) / 2.0


def vae_ratio(img):
    h, w, _ = img.shape
    anom = total = 0
    for y in range(0, h - PATCH, STRIDE):
        for x in range(0, w - PATCH, STRIDE):
            patch = img[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0
            if float(np.ravel(_run(vae, patch[np.newaxis]))[0]) > ANOM_THRESH:
                anom += 1
            total += 1
    return anom / total if total else 0.0


def sample(cls, n):
    files = sorted(glob.glob(os.path.join(TEST, FOLDERS[cls], "*")))
    random.Random(42).shuffle(files)
    return files[:n]


def main():
    n_cnn = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_vae = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    print(f"=== CNN ensemble (device-equivalent, no TTA) — {n_cnn}/class ===")
    cm = np.zeros((3, 3), int)
    for ci, cls in enumerate(CLASSES):
        for f in sample(cls, n_cnn):
            cm[ci, int(np.argmax(cnn_predict(load_rgb(f))))] += 1
    print(f"{'true/pred':>10}" + "".join(f"{c:>9}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>10}" + "".join(f"{v:>9}" for v in cm[i]))
    acc = cm.trace() / cm.sum() * 100
    print(f"Overall accuracy: {acc:.1f}%")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {cm[i, i] / cm[i].sum() * 100:.1f}%  ({cm[i, i]}/{cm[i].sum()})")

    print(f"\n=== VAE gate on diseased images — {n_vae}/class ===")
    print("(every test image IS diseased, so 'NORMAL' here = false negative)")
    for cls in CLASSES:
        ratios = [vae_ratio(load_rgb(f)) for f in sample(cls, n_vae)]
        gated_normal = sum(r <= ANOM_RATIO for r in ratios)
        print(f"  {cls}: mean ratio={np.mean(ratios):.3f} "
              f"min={min(ratios):.3f} max={max(ratios):.3f} | "
              f"falsely NORMAL: {gated_normal}/{len(ratios)}")


if __name__ == "__main__":
    main()
