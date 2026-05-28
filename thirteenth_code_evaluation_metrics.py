"""
Evaluation script — generates all metrics for the Testing & Results chapter.
Runs the best configuration: B2+B3 50/50 ensemble + 20-step TTA on the VAL set.
Outputs:
  - confusion_matrix.png
  - roc_curves.png
  - metrics_summary.csv
  - per_class_accuracy.csv
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import csv

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/kaggle/New_Augmented_Dataset"
VAL_DIR  = os.path.join(BASE_DIR, "val")
B2_MODEL = "/kaggle/input/datasets/youssefyasser0123/skin-b2-model/ensemble_b2_model.keras"
B3_MODEL = "/kaggle/input/datasets/youssefyasser0123/b3-model-keras/b3_model.keras"

IMG_B2   = (260, 260)
IMG_B3   = (300, 300)
BATCH    = 32
TTA_STEPS = 20
AUTOTUNE  = tf.data.AUTOTUNE
OUT_DIR   = "/kaggle/working"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading B2 model...")
b2 = tf.keras.models.load_model(B2_MODEL)
print("Loading B3 model...")
b3 = tf.keras.models.load_model(B3_MODEL)

# ── Val datasets ──────────────────────────────────────────────────────────────
val_b2_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH, label_mode='categorical')
val_b3_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B3,
    batch_size=BATCH, label_mode='categorical')

CLASS_NAMES = val_b2_raw.class_names
print(f"Classes: {CLASS_NAMES}")
print(f"Val samples: {sum(1 for _ in val_b2_raw.unbatch())}")

val_b2 = val_b2_raw.cache().prefetch(AUTOTUNE)
val_b3 = val_b3_raw.cache().prefetch(AUTOTUNE)

# True labels
all_labels = np.concatenate([y.numpy() for _, y in val_b2])
y_true = np.argmax(all_labels, axis=1)

# Per-class sample counts
for i, c in enumerate(CLASS_NAMES):
    print(f"  {c}: {np.sum(y_true == i)} samples")

# ── TTA augmenters ────────────────────────────────────────────────────────────
aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])

def tta_predict(model, dataset, n_steps):
    accum = None
    for step in range(n_steps):
        preds = np.concatenate([
            model(aug(imgs, training=True), training=False).numpy()
            for imgs, _ in dataset
        ])
        accum = preds if accum is None else accum + preds
        print(f"  TTA step {step+1}/{n_steps}", end="\r", flush=True)
    print()
    return accum / n_steps

# ── Run TTA inference ─────────────────────────────────────────────────────────
print(f"\nRunning {TTA_STEPS}-step TTA on B2...")
probs_b2 = tta_predict(b2, val_b2, TTA_STEPS)
print(f"Running {TTA_STEPS}-step TTA on B3...")
probs_b3 = tta_predict(b3, val_b3, TTA_STEPS)

# 50/50 ensemble
probs_ensemble = 0.5 * probs_b2 + 0.5 * probs_b3
y_pred = np.argmax(probs_ensemble, axis=1)

overall_acc = np.mean(y_pred == y_true) * 100
print(f"\nEnsemble B2+B3 (50/50) + TTA-{TTA_STEPS}: {overall_acc:.2f}%")

# ── Classification Report ──────────────────────────────────────────────────────
print("\n--- Classification Report ---")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print(report)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
print("\n--- Confusion Matrix ---")
print(f"{'':>12}" + "".join(f"{c:>12}" for c in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:>12}" + "".join(f"{v:>12}" for v in row))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 14, "weight": "bold"}
)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label", fontsize=13)
ax.set_title("Confusion Matrix — B2+B3 Ensemble + TTA (Validation Set)", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"\nSaved: {cm_path}")

# ── ROC Curves ────────────────────────────────────────────────────────────────
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

fig, ax = plt.subplots(figsize=(7, 6))
for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs_ensemble[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls.capitalize()} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curves — B2+B3 Ensemble + TTA (Validation Set)", fontsize=13)
ax.legend(loc="lower right", fontsize=12)
plt.tight_layout()
roc_path = os.path.join(OUT_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"Saved: {roc_path}")

# ── Per-class accuracy ────────────────────────────────────────────────────────
per_class = []
for i, cls in enumerate(CLASS_NAMES):
    mask = y_true == i
    acc_i = np.mean(y_pred[mask] == y_true[mask]) * 100
    tp = cm[i, i]
    total = cm[i].sum()
    per_class.append((cls, acc_i, int(tp), int(total)))
    print(f"  {cls}: {acc_i:.2f}%  ({tp}/{total})")

# ── Save CSV summaries ────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "per_class_accuracy.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Class", "Accuracy (%)", "Correct", "Total"])
    for row in per_class:
        w.writerow(row)
print(f"Saved: {csv_path}")

cm_csv = os.path.join(OUT_DIR, "confusion_matrix.csv")
with open(cm_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([""] + CLASS_NAMES)
    for i, row in enumerate(cm):
        w.writerow([CLASS_NAMES[i]] + list(row))
print(f"Saved: {cm_csv}")

print("\n=== DONE ===")
print(f"Overall accuracy: {overall_acc:.2f}%")
print("Download confusion_matrix.png, roc_curves.png, and the CSV files from /kaggle/working/")
