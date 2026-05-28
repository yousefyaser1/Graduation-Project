"""
VAE anomaly detection evaluation — runs locally.

Uses the patch-based sliding window approach matching the deployed Flutter app:
  - PATCH_SIZE = 64, STRIDE = 32
  - Per-patch anomaly threshold: 0.008
  - Image anomaly threshold: anomaly_ratio > 0.20

Inputs:
  - vae_model.tflite  (already in project directory)
  - New_Augmented_Dataset/val/  (acne, eczema, tinea) — 325 images
  - NORMAL_DIR: point to a folder of normal skin images if available

Outputs (saved to vae_eval_output/):
  - anomaly_ratio_histogram.png
  - per_class_anomaly_ratios.csv
  - vae_evaluation_summary.txt
"""

import os
import csv
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(SCRIPT_DIR, "vae_model.tflite")
VAL_DIR     = os.path.join(SCRIPT_DIR, "New_Augmented_Dataset", "val")
NORMAL_DIR  = None   # set to a folder of normal skin images if available, e.g.:
                     # NORMAL_DIR = r"C:\Users\A\Downloads\normal_skin_images"
OUT_DIR     = os.path.join(SCRIPT_DIR, "vae_eval_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Thresholds (matching Flutter app / score_cam.py) ─────────────────────────
PATCH_SIZE        = 64
STRIDE            = 32
ANOMALY_THRESHOLD = 0.008   # per-patch MSE threshold
ANOMALY_RATIO     = 0.20    # fraction of anomalous patches to flag image

# ── Load TFLite model ─────────────────────────────────────────────────────────
print("Loading VAE TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
inp_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]
print(f"Input shape:  {inp_detail['shape']}")
print(f"Output shape: {out_detail['shape']}")

# ── Sliding window inference on a single image ────────────────────────────────
def evaluate_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    h, w, _ = img_resized.shape

    anomaly_count = 0
    patch_count   = 0
    patch_errors  = []

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = img_resized[y:y+PATCH_SIZE, x:x+PATCH_SIZE].astype(np.float32) / 255.0
            patch = patch[np.newaxis]   # (1, 64, 64, 3)

            interpreter.set_tensor(inp_detail['index'], patch)
            interpreter.invoke()
            error = float(interpreter.get_tensor(out_detail['index'])[0])

            patch_errors.append(error)
            patch_count += 1
            if error > ANOMALY_THRESHOLD:
                anomaly_count += 1

    ratio = anomaly_count / patch_count if patch_count > 0 else 0.0
    return ratio, patch_errors

# ── Run over a class folder ───────────────────────────────────────────────────
def evaluate_folder(folder_path, label, expected_anomaly=True):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in exts]

    ratios = []
    correct = 0

    for i, fname in enumerate(files):
        path = os.path.join(folder_path, fname)
        ratio, _ = evaluate_image(path)
        if ratio is None:
            continue
        is_anomaly = ratio > ANOMALY_RATIO
        ratios.append(ratio)
        if is_anomaly == expected_anomaly:
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{label}] {i+1}/{len(files)}  last ratio={ratio:.3f}")

    acc = correct / len(ratios) * 100 if ratios else 0.0
    metric = "TPR" if expected_anomaly else "TNR"
    print(f"\n  {label}: {len(ratios)} images | {metric} = {correct}/{len(ratios)} = {acc:.2f}%")
    return ratios, acc

# ── Evaluate diseased classes (expect ANOMALY) ────────────────────────────────
all_diseased_ratios = []
class_results = {}

diseased_classes = ['acne', 'eczema', 'tinea']
print("\n=== Diseased images (expect: ANOMALY) ===")
for cls in diseased_classes:
    cls_dir = os.path.join(VAL_DIR, cls)
    if not os.path.isdir(cls_dir):
        print(f"  Skipping {cls} — folder not found: {cls_dir}")
        continue
    ratios, tpr = evaluate_folder(cls_dir, cls.capitalize(), expected_anomaly=True)
    class_results[cls] = {'ratios': ratios, 'rate': tpr, 'metric': 'TPR', 'n': len(ratios)}
    all_diseased_ratios.extend(ratios)

# ── Evaluate normal images if available (expect NORMAL) ───────────────────────
normal_ratios = []
tnr = None

if NORMAL_DIR and os.path.isdir(NORMAL_DIR):
    print("\n=== Normal images (expect: NORMAL) ===")
    normal_ratios, tnr = evaluate_folder(NORMAL_DIR, "Normal", expected_anomaly=False)
    fpr = 100.0 - tnr
    class_results['normal'] = {'ratios': normal_ratios, 'rate': tnr, 'metric': 'TNR', 'n': len(normal_ratios)}
    print(f"  FPR (normal wrongly forwarded to CNN) = {fpr:.2f}%")
else:
    print("\n[INFO] NORMAL_DIR not set — skipping FPR evaluation.")
    print("       Set NORMAL_DIR at the top of this script to compute FPR/TNR.")

# ── Summary ───────────────────────────────────────────────────────────────────
total_diseased = sum(r['n'] for k, r in class_results.items() if k != 'normal')
total_correct  = sum(int(r['n'] * r['rate'] / 100) for k, r in class_results.items() if k != 'normal')
overall_tpr = total_correct / total_diseased * 100 if total_diseased else 0.0

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"Patch threshold   : {ANOMALY_THRESHOLD}")
print(f"Ratio threshold   : {ANOMALY_RATIO}")
print(f"Overall TPR       : {total_correct}/{total_diseased} = {overall_tpr:.2f}%")
if tnr is not None:
    print(f"Overall TNR       : {tnr:.2f}%")
    print(f"Overall FPR       : {100-tnr:.2f}%")
for cls, r in class_results.items():
    print(f"  {cls.capitalize():<10}: {r['metric']}={r['rate']:.2f}%  n={r['n']}")

# ── Histogram ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = {'acne': '#e74c3c', 'eczema': '#3498db', 'tinea': '#2ecc71', 'normal': '#95a5a6'}

for cls, r in class_results.items():
    ax.hist(r['ratios'], bins=25, alpha=0.6, label=cls.capitalize(),
            color=colors.get(cls, '#7f8c8d'), edgecolor='white', linewidth=0.4)

ax.axvline(ANOMALY_RATIO, color='black', linestyle='--', linewidth=1.5,
           label=f'Decision threshold ({ANOMALY_RATIO})')
ax.set_xlabel("Anomaly Ratio (fraction of anomalous patches)", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title("VAE Anomaly Ratio Distribution — Validation Set", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
hist_path = os.path.join(OUT_DIR, "anomaly_ratio_histogram.png")
plt.savefig(hist_path, dpi=200)
plt.close()
print(f"\nSaved: {hist_path}")

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "per_class_anomaly_ratios.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['class', 'anomaly_ratio'])
    for cls, r in class_results.items():
        for ratio in r['ratios']:
            w.writerow([cls, f"{ratio:.4f}"])
print(f"Saved: {csv_path}")

# ── Text summary ──────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "vae_evaluation_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"VAE Evaluation Summary\n{'='*40}\n")
    f.write(f"Patch threshold   : {ANOMALY_THRESHOLD}\n")
    f.write(f"Ratio threshold   : {ANOMALY_RATIO}\n")
    f.write(f"Overall TPR       : {overall_tpr:.2f}%  ({total_correct}/{total_diseased})\n")
    if tnr is not None:
        f.write(f"Overall TNR       : {tnr:.2f}%\n")
        f.write(f"Overall FPR       : {100-tnr:.2f}%\n")
    f.write("\nPer-class:\n")
    for cls, r in class_results.items():
        f.write(f"  {cls.capitalize():<10}: {r['metric']}={r['rate']:.2f}%  n={r['n']}\n")
print(f"Saved: {summary_path}")

print("\n=== DONE ===")
