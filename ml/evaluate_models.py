"""
Model Evaluation Script
Evaluates all three trained models on the test set and reports:
- Overall accuracy and loss
- Per-class accuracy, precision, recall, F1
- Confusion matrix
- TTA accuracy for Phase 3 model
"""

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import os

tf.random.set_seed(42)
np.random.seed(42)

TEST_DIR  = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset/test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES    = ["acne", "eczema", "tinea"]

MODELS = {
    "Phase 1 – Transfer Learning": "mobilenetv2_skin_disease_model.keras",
    "Phase 2 – Fine-Tuning":       "mobilenetv2_finetuned_model.keras",
    "Phase 3 – Anti-Overfitting":  "mobilenetv2_phase1_improved.keras",
}

# ── Load test dataset ────────────────────────────────────────────────────────
print("Loading test dataset …")
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = test_ds_raw.class_names
print(f"Classes detected: {class_names}")

# Count per-class test images
total_per_class = {c: 0 for c in class_names}
y_true_all = []
for _, labels in test_ds_raw:
    idxs = np.argmax(labels.numpy(), axis=1)
    for idx in idxs:
        total_per_class[class_names[idx]] += 1
    y_true_all.extend(idxs)
y_true_all = np.array(y_true_all)

print("\nTest set composition:")
for c in class_names:
    print(f"  {c}: {total_per_class[c]} images")
print(f"  Total: {len(y_true_all)} images")

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds_raw.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).cache().prefetch(AUTOTUNE)


# ── TTA helper ───────────────────────────────────────────────────────────────
def predict_with_tta(model, dataset):
    all_preds = []
    for images, _ in dataset:
        variants = [
            images,
            tf.image.flip_left_right(images),
            tf.image.flip_up_down(images),
            tf.image.flip_left_right(tf.image.flip_up_down(images)),
            tf.image.adjust_brightness(images, delta=0.1),
        ]
        avg = tf.reduce_mean([model(v, training=False) for v in variants], axis=0)
        all_preds.append(avg.numpy())
    return np.concatenate(all_preds, axis=0)


# ── Evaluate each model ──────────────────────────────────────────────────────
results = {}

for phase_name, model_file in MODELS.items():
    print("\n" + "=" * 70)
    print(f"Evaluating: {phase_name}")
    print(f"Model file: {model_file}")
    print("=" * 70)

    if not os.path.exists(model_file):
        print(f"  !! File not found: {model_file}")
        continue

    model = tf.keras.models.load_model(model_file)

    # Standard evaluation
    loss, acc = model.evaluate(test_ds, verbose=0)

    # Predictions for detailed metrics
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Overall Loss     : {loss:.4f}")

    print("\n  Classification Report:")
    report = classification_report(
        y_true_all, y_pred, target_names=class_names, digits=4
    )
    print(report)

    print("  Confusion Matrix (rows=True, cols=Predicted):")
    cm = confusion_matrix(y_true_all, y_pred)
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in class_names)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = f"  {class_names[i]:>10}" + "".join(f"{v:>10}" for v in row)
        print(row_str)

    # Per-class accuracy from confusion matrix
    print("\n  Per-class accuracy:")
    for i, c in enumerate(class_names):
        per_class_acc = cm[i, i] / cm[i].sum() * 100
        print(f"    {c}: {per_class_acc:.2f}%")

    # TTA only for Phase 3
    tta_acc = None
    if "Phase 3" in phase_name:
        print("\n  Running TTA (5 augmentations) …")
        tta_preds = predict_with_tta(model, test_ds)
        tta_y_pred = np.argmax(tta_preds, axis=1)
        tta_acc = accuracy_score(y_true_all, tta_y_pred) * 100
        print(f"  TTA Accuracy: {tta_acc:.2f}%")
        print("\n  TTA Classification Report:")
        print(classification_report(
            y_true_all, tta_y_pred, target_names=class_names, digits=4
        ))
        print("  TTA Confusion Matrix:")
        tta_cm = confusion_matrix(y_true_all, tta_y_pred)
        print(f"  {header}")
        for i, row in enumerate(tta_cm):
            row_str = f"  {class_names[i]:>10}" + "".join(f"{v:>10}" for v in row)
            print(row_str)

    results[phase_name] = {
        "accuracy": acc * 100,
        "loss": loss,
        "tta_accuracy": tta_acc,
        "report": report,
        "cm": cm,
    }

# ── Final summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Phase':<35} {'Accuracy':>10} {'Loss':>10} {'TTA Acc':>10}")
print("-" * 70)
for name, r in results.items():
    tta = f"{r['tta_accuracy']:.2f}%" if r['tta_accuracy'] else "—"
    print(f"{name:<35} {r['accuracy']:>9.2f}% {r['loss']:>10.4f} {tta:>10}")
print("=" * 70)
