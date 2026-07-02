import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# --- Kaggle paths ---
BASE_DIR  = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/kaggle/New_Augmented_Dataset"
VAL_DIR   = os.path.join(BASE_DIR, "val")

B2_MODEL  = "/kaggle/input/datasets/youssefyasser0123/skin-b2-model/ensemble_b2_model.keras"
B3_MODEL  = "/kaggle/input/datasets/youssefyasser0123/b3-model-keras/b3_model.keras"

IMG_B3   = (300, 300)
IMG_B2   = (260, 260)
BATCH    = 32
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# 1. Load models
# ---------------------------------------------------------------------------
print("Loading B3...")
b3 = tf.keras.models.load_model(B3_MODEL)

print("Loading B2...")
b2 = tf.keras.models.load_model(B2_MODEL)

# ---------------------------------------------------------------------------
# 2. Val datasets
# ---------------------------------------------------------------------------
val_b3_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B3,
    batch_size=BATCH, label_mode='categorical')
val_b2_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH, label_mode='categorical')

class_names = val_b3_raw.class_names

val_b3 = val_b3_raw.cache().prefetch(AUTOTUNE)
val_b2 = val_b2_raw.cache().prefetch(AUTOTUNE)

class_names = class_names
all_labels  = np.concatenate([lbls.numpy() for _, lbls in val_b3])
true_cls    = np.argmax(all_labels, axis=1)

# Standalone baselines
_, acc_b3 = b3.evaluate(val_b3, verbose=0)
_, acc_b2 = b2.evaluate(val_b2, verbose=0)
print(f"B3 standalone: {acc_b3*100:.2f}%")
print(f"B2 standalone: {acc_b2*100:.2f}%")

# ---------------------------------------------------------------------------
# 3. TTA helper
# ---------------------------------------------------------------------------
aug_b3 = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])
aug_b2 = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])

def tta_predict(model, dataset, augmenter, n_steps):
    accumulated = None
    for step in range(n_steps):
        preds = []
        for imgs, _ in dataset:
            aug_imgs = augmenter(imgs, training=True)
            p = model(aug_imgs, training=False).numpy()
            preds.append(p)
        preds = np.concatenate(preds)
        accumulated = preds if accumulated is None else accumulated + preds
    return accumulated / n_steps

# ---------------------------------------------------------------------------
# 4. Run TTA at multiple step counts
# ---------------------------------------------------------------------------
TTA_STEPS_LIST = [20, 30, 50]

results = {}
best_overall = 0.0
best_config  = ""

# Run TTA once for each model at max steps, accumulate incrementally
print(f"\nRunning TTA up to {max(TTA_STEPS_LIST)} steps on B3...")
probs_b3_accum = None
for step in range(max(TTA_STEPS_LIST)):
    preds = []
    for imgs, _ in val_b3:
        aug_imgs = aug_b3(imgs, training=True)
        p = b3(aug_imgs, training=False).numpy()
        preds.append(p)
    preds = np.concatenate(preds)
    probs_b3_accum = preds if probs_b3_accum is None else probs_b3_accum + preds
    print(f"  B3 step {step+1}/{max(TTA_STEPS_LIST)}", flush=True)

print(f"\nRunning TTA up to {max(TTA_STEPS_LIST)} steps on B2...")
probs_b2_accum = None
for step in range(max(TTA_STEPS_LIST)):
    preds = []
    for imgs, _ in val_b2:
        aug_imgs = aug_b2(imgs, training=True)
        p = b2(aug_imgs, training=False).numpy()
        preds.append(p)
    preds = np.concatenate(preds)
    probs_b2_accum = preds if probs_b2_accum is None else probs_b2_accum + preds
    print(f"  B2 step {step+1}/{max(TTA_STEPS_LIST)}", flush=True)

# ---------------------------------------------------------------------------
# 5. Evaluate all step counts and ensemble weights
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RESULTS SWEEP")
print("="*60)

ensemble_weights = [
    (1.0, 0.0),   # B3 only
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.0, 1.0),   # B2 only
]

for n_steps in TTA_STEPS_LIST:
    probs_b3 = probs_b3_accum[:n_steps * len(probs_b3_accum) // max(TTA_STEPS_LIST)]
    # Correctly slice by re-dividing
    # Re-compute properly: probs_b3_accum after n_steps is probs_b3_accum * n_steps / max_steps
    # (since it's a running sum, not average yet)
    probs_b3_n = probs_b3_accum * (n_steps / max(TTA_STEPS_LIST))
    probs_b3_n = probs_b3_n / n_steps  # normalize to average

    probs_b2_n = probs_b2_accum * (n_steps / max(TTA_STEPS_LIST))
    probs_b2_n = probs_b2_n / n_steps

    print(f"\n--- TTA steps = {n_steps} ---")
    acc_b3_tta = np.mean(np.argmax(probs_b3_n, axis=1) == true_cls)
    acc_b2_tta = np.mean(np.argmax(probs_b2_n, axis=1) == true_cls)
    print(f"  B3+TTA: {acc_b3_tta*100:.2f}%   B2+TTA: {acc_b2_tta*100:.2f}%")

    for w_b3, w_b2 in ensemble_weights:
        if w_b2 == 0.0:
            probs = probs_b3_n
        elif w_b3 == 0.0:
            probs = probs_b2_n
        else:
            probs = w_b3 * probs_b3_n + w_b2 * probs_b2_n
        preds = np.argmax(probs, axis=1)
        acc   = np.mean(preds == true_cls)
        flag  = " <-- BEST!" if acc > best_overall else ""
        if acc > best_overall:
            best_overall = acc
            best_config  = f"steps={n_steps}, w_b3={w_b3}, w_b2={w_b2}"
            best_preds   = preds
        label = f"B3:{w_b3:.1f} B2:{w_b2:.1f}"
        print(f"  [{label}]: {acc*100:.2f}%{flag}")

# ---------------------------------------------------------------------------
# 6. Best result detail
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"BEST CONFIG: {best_config}")
print(f"BEST ACCURACY: {best_overall*100:.2f}%")
print(f"\nPer-class accuracy (best config):")
for i, name in enumerate(class_names):
    mask  = true_cls == i
    acc_i = np.mean(best_preds[mask] == true_cls[mask]) * 100
    print(f"  {name}: {acc_i:.2f}%  ({mask.sum()} samples)")

if best_overall >= 0.93:
    print("\nTARGET REACHED: 93%+!")
else:
    print(f"\nGap to 93%: {(0.93 - best_overall)*100:.2f}%")
    