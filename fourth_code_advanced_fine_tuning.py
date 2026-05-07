"""
================================================================================
PHASE 4: ADVANCED FINE-TUNING FOR 95% ACCURACY TARGET
================================================================================

STARTING POINT:
  - Phase 2 fine-tuned model (mobilenetv2_finetuned_model.keras)
  - Current accuracy: 84.16% (test set)
  - Phase 2 unfroze layers 100-154 (55 layers) with LR=1e-5

WHY PHASE 3 FAILED:
  - Loaded Phase 1 model (not Phase 2) — lost 8.99% improvement
  - Over-regularized: dropout 0.5 + L2 0.001 + label smoothing 0.2 caused
    underfitting, dropping accuracy to 73.91%
  - Learning rate (1e-6) was too low for meaningful updates

PHASE 4 STRATEGY (what we do differently):
================================================================================

1. DEEPER LAYER UNFREEZING (biggest improvement potential: +3-5%)
   - Unfreeze from layer 70 instead of layer 100
   - Gives 84 trainable base layers vs Phase 2's 55
   - Allows the model to adapt mid-level features (textures, patterns)
     to dermatological imagery more precisely
   - Still keeps early layers (0-69) frozen to preserve low-level ImageNet features

2. ONLINE DATA AUGMENTATION (+2-4%)
   - Phase 2 used no online augmentation (just the pre-augmented dataset)
   - We add: RandomFlip, RandomRotation(+/-10 degrees), RandomZoom(+/-10%)
   - Applied ONLY during training, not validation
   - Increases effective dataset diversity without creating new files

3. TARGETED CLASS WEIGHT BOOST FOR TINEA (+1-3%)
   - Tinea is the worst class: 59.55% -> 71.91% across phases
   - Root cause: class imbalance (Tinea = 22% of data)
   - We boost Tinea's class weight by 1.5× on top of balanced weights
   - Forces the model to pay more attention to Tinea during training

4. MILD LABEL SMOOTHING (0.1 instead of 0.2) (+0.5-1%)
   - Phase 3 used 0.2 (too aggressive, caused underfitting)
   - 0.1 prevents overconfidence without damaging accuracy

5. CALIBRATED LEARNING RATE: 3e-6 (+1-2%)
   - Phase 2 used 1e-5 (may slightly overfit with more unfrozen layers)
   - Phase 3 used 1e-6 (too slow for convergence)
   - 3e-6 is the sweet spot: fast enough to converge, stable enough to not diverge

6. LONGER TRAINING WITH BETTER PATIENCE
   - Epochs: 50 (vs Phase 2's 30)
   - EarlyStopping patience: 8 (vs Phase 2's 5)
   - ReduceLROnPlateau: factor=0.3 (more aggressive reduction), patience=3

7. PER-CLASS EVALUATION
   - Shows accuracy breakdown per class (acne, eczema, tinea)
   - Helps identify remaining bottlenecks

================================================================================
EXPECTED OUTCOME:
  84.16% + ~3-5% (deeper unfreezing) + ~2-4% (augmentation)
          + ~1-3% (Tinea boost) + ~1% (other)
  = Target: 90-95%

================================================================================
"""

import tensorflow as tf
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.utils import class_weight
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 80)
print("PHASE 4: ADVANCED FINE-TUNING — TARGET 95% ACCURACY")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n[CONFIGURATION] Setting up paths and parameters...")

BASE_DIR = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

PHASE2_MODEL_PATH = 'mobilenetv2_finetuned_model.keras'     # Best existing model
PHASE4_MODEL_PATH = 'fourth_code_model.keras'               # Output of this script

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LABEL_MODE = 'categorical'
UNFREEZE_FROM_LAYER = 70        # Deeper than Phase 2's 100
LEARNING_RATE = 3e-6            # Between Phase 2's 1e-5 and Phase 3's 1e-6
LABEL_SMOOTHING = 0.1           # Mild (Phase 3 used 0.2 — too aggressive)
TINEA_WEIGHT_BOOST = 1.5        # Multiply Tinea's class weight by this factor
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3

print(f"  Phase 2 model (input): {PHASE2_MODEL_PATH}")
print(f"  Phase 4 model (output): {PHASE4_MODEL_PATH}")
print(f"  Unfreeze from layer: {UNFREEZE_FROM_LAYER} (Phase 2 used 100)")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Label smoothing: {LABEL_SMOOTHING}")
print(f"  Tinea weight boost: {TINEA_WEIGHT_BOOST}x")
print(f"  Max epochs: {EPOCHS}")
print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================
print("\n[STEP 1] Loading training and validation datasets...")

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)

print(f"  Classes: {class_names}")
print(f"  Training batches: {len(train_ds_raw)}")
print(f"  Validation batches: {len(val_ds_raw)}")

# ============================================================================
# STEP 2: CALCULATE CLASS WEIGHTS WITH TINEA BOOST
# ============================================================================
print("\n[STEP 2] Calculating class weights with Tinea boost...")

y_train = []
for images, labels in train_ds_raw:
    y_train.extend(np.argmax(labels.numpy(), axis=1))
y_train = np.array(y_train)

print(f"  Total training samples: {len(y_train)}")
print(f"  Class distribution:")
for i, name in enumerate(class_names):
    count = np.sum(y_train == i)
    pct = count / len(y_train) * 100
    print(f"    {name}: {count} samples ({pct:.2f}%)")

# Standard balanced class weights
base_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(base_weights))

print(f"\n  Balanced class weights:")
for i, name in enumerate(class_names):
    print(f"    {name}: {class_weight_dict[i]:.4f}")

# Find the Tinea class index (alphabetically: acne=0, eczema=1, tinea=2)
tinea_idx = class_names.index('tinea')
class_weight_dict[tinea_idx] *= TINEA_WEIGHT_BOOST

print(f"\n  After {TINEA_WEIGHT_BOOST}x Tinea boost:")
for i, name in enumerate(class_names):
    print(f"    {name}: {class_weight_dict[i]:.4f}")

# ============================================================================
# STEP 3: BUILD DATA PIPELINE WITH ONLINE AUGMENTATION
# ============================================================================
print("\n[STEP 3] Building data pipeline with online augmentation...")

AUTOTUNE = tf.data.AUTOTUNE

# Online augmentation layer — applied ONLY to training images
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
], name="online_augmentation")

print(f"  Online augmentation layers:")
print(f"    - RandomFlip (horizontal)")
print(f"    - RandomRotation (+/-10 degrees)")
print(f"    - RandomZoom (+/-10%)")
print(f"  (Applied to training only, not validation)")

# Training pipeline: augment -> preprocess -> cache -> prefetch
train_ds = train_ds_raw.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

# Validation pipeline: preprocess -> cache -> prefetch (NO augmentation)
val_ds = val_ds_raw.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print(f"  Training pipeline: augment -> preprocess -> cache -> prefetch")
print(f"  Validation pipeline: preprocess -> cache -> prefetch")

# ============================================================================
# STEP 4: LOAD PHASE 2 MODEL (BEST CHECKPOINT)
# ============================================================================
print(f"\n[STEP 4] Loading Phase 2 model from: {PHASE2_MODEL_PATH}")

model = tf.keras.models.load_model(PHASE2_MODEL_PATH)

print(f"  Model loaded successfully")
print(f"  Total layers in full model: {len(model.layers)}")

# ============================================================================
# STEP 5: DEEPER LAYER UNFREEZING (from layer 70 vs Phase 2's 100)
# ============================================================================
print(f"\n[STEP 5] Unfreezing base model layers from layer {UNFREEZE_FROM_LAYER}...")

# Locate MobileNetV2 base model
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break
if base_model is None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
if base_model is None:
    raise ValueError("Could not locate MobileNetV2 base model in loaded model!")

print(f"  Base model found: {base_model.name}")
print(f"  Total base model layers: {len(base_model.layers)}")

base_model.trainable = True

frozen_count = 0
trainable_count = 0
for i, layer in enumerate(base_model.layers):
    if i < UNFREEZE_FROM_LAYER:
        layer.trainable = False
        frozen_count += 1
    else:
        layer.trainable = True
        trainable_count += 1

print(f"  Frozen layers: {frozen_count} (layers 0-{UNFREEZE_FROM_LAYER - 1})")
print(f"  Trainable layers: {trainable_count} (layers {UNFREEZE_FROM_LAYER}-{len(base_model.layers)-1})")
print(f"  Phase 2 had {len(base_model.layers) - 100} trainable base layers — Phase 4 has {trainable_count}")

# ============================================================================
# STEP 6: RECOMPILE WITH PHASE 4 SETTINGS
# ============================================================================
print("\n[STEP 6] Recompiling model with Phase 4 settings...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

print(f"  Optimizer: Adam (learning_rate={LEARNING_RATE})")
print(f"  Loss: CategoricalCrossentropy (label_smoothing={LABEL_SMOOTHING})")
print(f"  Metrics: accuracy")

# ============================================================================
# STEP 7: SET UP CALLBACKS
# ============================================================================
print("\n[STEP 7] Setting up callbacks...")

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=REDUCE_LR_FACTOR,
    patience=REDUCE_LR_PATIENCE,
    min_lr=1e-8,
    verbose=1,
    mode='min'
)

model_checkpoint = callbacks.ModelCheckpoint(
    PHASE4_MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print(f"  EarlyStopping: patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True")
print(f"  ReduceLROnPlateau: factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}, min_lr=1e-8")
print(f"  ModelCheckpoint: saves to {PHASE4_MODEL_PATH} on val_loss improvement")

# ============================================================================
# STEP 8: EVALUATE PHASE 2 BASELINE (before training)
# ============================================================================
print("\n[STEP 8] Evaluating Phase 2 baseline on validation set...")

phase2_loss, phase2_acc = model.evaluate(val_ds, verbose=0)
print(f"  Phase 2 Baseline Accuracy: {phase2_acc:.4f} ({phase2_acc*100:.2f}%)")
print(f"  Phase 2 Baseline Loss:     {phase2_loss:.4f}")

# ============================================================================
# STEP 9: TRAIN (PHASE 4 FINE-TUNING)
# ============================================================================
print("\n[STEP 9] Starting Phase 4 advanced fine-tuning...")
print("=" * 80)
print(f"  Max epochs: {EPOCHS}")
print(f"  Class weights: {class_weight_dict}")
print("=" * 80)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

print("=" * 80)
print("[STEP 9] Phase 4 training complete!")

# ============================================================================
# STEP 10: LOAD BEST CHECKPOINT AND EVALUATE
# ============================================================================
print(f"\n[STEP 10] Loading best checkpoint from: {PHASE4_MODEL_PATH}")

best_model = tf.keras.models.load_model(PHASE4_MODEL_PATH)

val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
print(f"  Phase 4 Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Phase 4 Loss:     {val_loss:.4f}")

# ============================================================================
# STEP 11: PER-CLASS ACCURACY BREAKDOWN
# ============================================================================
print("\n[STEP 11] Per-class accuracy breakdown...")

all_preds = []
all_labels = []

for images, labels in val_ds:
    preds = best_model(images, training=False)
    all_preds.append(preds.numpy())
    all_labels.append(labels.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

pred_classes = np.argmax(all_preds, axis=1)
true_classes = np.argmax(all_labels, axis=1)

print(f"\n  Per-class accuracy:")
for i, name in enumerate(class_names):
    mask = true_classes == i
    class_acc = np.mean(pred_classes[mask] == true_classes[mask]) * 100
    count = np.sum(mask)
    print(f"    {name:10s}: {class_acc:6.2f}%  ({count} samples)")

# ============================================================================
# STEP 12: TRAINING SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4 COMPLETE — SUMMARY")
print("=" * 80)

epochs_trained = len(history.history['loss'])
best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])
best_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_loss_epoch = np.argmin(history.history['val_loss']) + 1

improvement = val_acc - phase2_acc

print(f"\n  Phase 2 Baseline Accuracy : {phase2_acc:.4f} ({phase2_acc*100:.2f}%)")
print(f"  Phase 4 Final Accuracy    : {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Improvement               : +{improvement:.4f} (+{improvement*100:.2f}%)")
print(f"\n  Epochs trained: {epochs_trained} / {EPOCHS}")
print(f"  Best val_accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch})")
print(f"  Best val_loss:     {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
print(f"\n  Model saved to: {PHASE4_MODEL_PATH}")

print(f"\n  Phase 4 Configuration Applied:")
print(f"    [OK] Loaded from Phase 2 best model (84.16%)")
print(f"    [OK] Unfroze base model from layer {UNFREEZE_FROM_LAYER} ({trainable_count} trainable layers)")
print(f"    [OK] Online augmentation (flip, rotation +/-10 degrees, zoom +/-10%)")
print(f"    [OK] Tinea class weight boosted {TINEA_WEIGHT_BOOST}x")
print(f"    [OK] Label smoothing: {LABEL_SMOOTHING}")
print(f"    [OK] Learning rate: {LEARNING_RATE}")
print(f"    [OK] ReduceLROnPlateau: factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}")
print(f"    [OK] EarlyStopping: patience={EARLY_STOPPING_PATIENCE}")

if val_acc >= 0.95:
    print(f"\n  *** GOAL ACHIEVED: Reached 95%+ accuracy! ***")
elif val_acc >= 0.90:
    print(f"\n  Good progress: {val_acc*100:.2f}% — within 5% of target.")
    print(f"  Suggestion: try unfreezing from layer 50, or increase epochs further.")
elif val_acc >= 0.87:
    print(f"\n  Moderate progress: {val_acc*100:.2f}%.")
    print(f"  Suggestion: consider switching to EfficientNetV2-S architecture.")
else:
    print(f"\n  Accuracy below 87%. Check for overfitting or data issues.")

print("=" * 80)
print("PHASE 4 ADVANCED FINE-TUNING COMPLETE")
print("=" * 80)
