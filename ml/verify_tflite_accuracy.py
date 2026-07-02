"""
Quick accuracy check for the CNN models.

Run this BEFORE re-converting to see whether the issue is:
  (a) the Keras models themselves (bad training)
  (b) the TFLite conversion (quantization degradation)

Usage:
    python verify_tflite_accuracy.py
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
KERAS_B2   = r"C:\Users\A\Downloads\ensemble_b2_model.keras"
KERAS_B3   = r"C:\Users\A\Downloads\b3_model.keras"
TFLITE_B2  = r"C:\Users\A\Graduation-Project\Flutter\assets\models\cnn_b2_model.tflite"
TFLITE_B3  = r"C:\Users\A\Graduation-Project\Flutter\assets\models\cnn_b3_model.tflite"
TEST_DIR   = r"C:\Users\A\Graduation-Project\New_Augmented_Dataset\test"

CLASSES    = ['acne', 'eczema', 'tinea']   # alphabetical — matches training order
B2_SIZE    = (260, 260)
B3_SIZE    = (300, 300)
MAX_IMAGES = 30   # images per class to evaluate (keep fast)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_images(test_dir, class_name, size, max_n):
    """Return list of float32 [0,255] arrays resized to `size`."""
    folder = os.path.join(test_dir, class_name)
    paths  = glob.glob(os.path.join(folder, "*.jpg"))[:max_n]
    imgs   = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, size)
        imgs.append(img)
    return imgs


def eval_keras(model, test_dir, classes, size):
    print(f"\n  Evaluating Keras model (input size {size})...")
    correct, total = 0, 0
    per_class = {c: [0, 0] for c in classes}   # [correct, total]

    for true_idx, cls in enumerate(classes):
        imgs = load_images(test_dir, cls, size, MAX_IMAGES)
        for img in imgs:
            inp  = img[np.newaxis]          # (1, H, W, 3) — [0,255], Rescaling inside
            prob = model(inp, training=False).numpy()[0]
            pred = int(np.argmax(prob))
            per_class[cls][1] += 1
            total += 1
            if pred == true_idx:
                per_class[cls][0] += 1
                correct += 1

    print(f"  Overall accuracy : {correct}/{total} = {correct/total*100:.1f}%")
    for cls in classes:
        c, t = per_class[cls]
        print(f"    {cls:8s}: {c}/{t} = {c/t*100:.1f}%")
    return correct / total


def eval_tflite(tflite_path, test_dir, classes, size):
    print(f"\n  Evaluating TFLite: {os.path.basename(tflite_path)} (input {size})...")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    # Log input tensor dtype to help diagnose type mismatches
    in_dtype = interp.get_input_details()[0]['dtype']
    print(f"  Input tensor dtype: {in_dtype}")

    correct, total = 0, 0
    per_class = {c: [0, 0] for c in classes}

    for true_idx, cls in enumerate(classes):
        imgs = load_images(test_dir, cls, size, MAX_IMAGES)
        for img in imgs:
            inp = img[np.newaxis]   # [0,255] float32 — EfficientNet rescales internally
            interp.set_tensor(in_idx, inp)
            interp.invoke()
            prob = interp.get_tensor(out_idx)[0]
            pred = int(np.argmax(prob))
            per_class[cls][1] += 1
            total += 1
            if pred == true_idx:
                per_class[cls][0] += 1
                correct += 1

    print(f"  Overall accuracy : {correct}/{total} = {correct/total*100:.1f}%")
    for cls in classes:
        c, t = per_class[cls]
        print(f"    {cls:8s}: {c}/{t} = {c/t*100:.1f}%")
    return correct / total


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CNN Model Accuracy Verification")
    print("=" * 60)

    # ── Keras models ──────────────────────────────────────────────────────
    print("\n[1] Keras models (ground truth for this evaluation)")
    if os.path.exists(KERAS_B2):
        b2_keras = tf.keras.models.load_model(KERAS_B2)
        acc_b2_keras = eval_keras(b2_keras, TEST_DIR, CLASSES, B2_SIZE)
    else:
        print(f"  SKIP — {KERAS_B2} not found")
        acc_b2_keras = None

    if os.path.exists(KERAS_B3):
        b3_keras = tf.keras.models.load_model(KERAS_B3)
        acc_b3_keras = eval_keras(b3_keras, TEST_DIR, CLASSES, B3_SIZE)
    else:
        print(f"  SKIP — {KERAS_B3} not found")
        acc_b3_keras = None

    # ── Current TFLite models (quantized) ─────────────────────────────────
    print("\n[2] Current TFLite models (may have quantization degradation)")
    acc_b2_tflite = eval_tflite(TFLITE_B2, TEST_DIR, CLASSES, B2_SIZE)
    acc_b3_tflite = eval_tflite(TFLITE_B3, TEST_DIR, CLASSES, B3_SIZE)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if acc_b2_keras is not None:
        drop = (acc_b2_keras - acc_b2_tflite) * 100
        print(f"B2  Keras: {acc_b2_keras*100:.1f}%  |  TFLite: {acc_b2_tflite*100:.1f}%  |  drop: {drop:.1f}pp")
    if acc_b3_keras is not None:
        drop = (acc_b3_keras - acc_b3_tflite) * 100
        print(f"B3  Keras: {acc_b3_keras*100:.1f}%  |  TFLite: {acc_b3_tflite*100:.1f}%  |  drop: {drop:.1f}pp")

    print("\nInterpretation:")
    print("  • Large accuracy drop between Keras and TFLite → quantization is the culprit")
    print("    Fix: re-run cnn_tflite_conversion.py (quantization line was already removed)")
    print("  • Low Keras accuracy (≤40%) → model trained poorly; retrain needed")
    print("  • High TFLite accuracy but still wrong in Flutter → preprocessing mismatch")
