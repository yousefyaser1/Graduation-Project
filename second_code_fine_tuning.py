"""
Phase 2 Fine-Tuning Script for MobileNetV2 Skin Disease Classification Model

This script performs fine-tuning on a previously trained model by:
1. Loading the existing trained model
2. Unfreezing top layers of the MobileNetV2 base (keeping first 100 layers frozen)
3. Recompiling with a drastically reduced learning rate (1e-5)
4. Using advanced callbacks (EarlyStopping, ReduceLROnPlateau)
5. Training for 30 epochs with recalculated class weights
6. Saving the best fine-tuned model
"""

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.utils import class_weight
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 80)
print("PHASE 2 FINE-TUNING - MOBILENETV2 SKIN DISEASE CLASSIFICATION")
print("=" * 80)

# ============================================================================
# STEP 1: Dataset Configuration
# ============================================================================
print("\n[STEP 1] Configuring dataset paths and parameters...")

# Hardcoded base directory as specified
BASE_DIR = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

print(f"  Base Dataset Directory: {BASE_DIR}")
print(f"  Training Directory: {TRAIN_DIR}")
print(f"  Validation Directory: {VAL_DIR}")

# Dataset parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LABEL_MODE = 'categorical'

print(f"  Image Size: {IMAGE_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Label Mode: {LABEL_MODE}")

# ============================================================================
# STEP 2: Load Training Dataset
# ============================================================================
print("\n[STEP 2] Loading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE
)

print(f"  Training dataset loaded successfully")
print(f"  Number of training batches: {len(train_ds)}")

# ============================================================================
# STEP 3: Load Validation Dataset
# ============================================================================
print("\n[STEP 3] Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,  # No shuffle for validation set as specified
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE
)

print(f"  Validation dataset loaded successfully")
print(f"  Number of validation batches: {len(val_ds)}")

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)

print(f"  Number of classes: {num_classes}")
print(f"  Class names: {class_names}")

# ============================================================================
# STEP 4: Extract True Labels for Class Weight Calculation
# ============================================================================
print("\n[STEP 4] Extracting true labels from training dataset for class weight calculation...")

# Extract all labels from the training dataset
y_train = []
for images, labels in train_ds:
    # Convert one-hot encoded labels to class indices
    y_train.extend(np.argmax(labels.numpy(), axis=1))

y_train = np.array(y_train)

print(f"  Total training samples: {len(y_train)}")
print(f"  Label distribution:")
for i, class_name in enumerate(class_names):
    count = np.sum(y_train == i)
    percentage = (count / len(y_train)) * 100
    print(f"    {class_name}: {count} samples ({percentage:.2f}%)")

# ============================================================================
# STEP 5: Calculate Dynamic Class Weights
# ============================================================================
print("\n[STEP 5] Calculating dynamic class weights for imbalanced classes...")

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convert to dictionary format required by Keras
class_weight_dict = dict(enumerate(class_weights))

print(f"  Calculated class weights:")
for i, class_name in enumerate(class_names):
    weight = class_weight_dict[i]
    print(f"    {class_name} (class {i}): {weight:.4f}")

# ============================================================================
# STEP 6: Configure Data Pipeline Performance
# ============================================================================
print("\n[STEP 6] Configuring data pipeline with AUTOTUNE, cache, and prefetch...")

AUTOTUNE = tf.data.AUTOTUNE

# Normalize and optimize training dataset
train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

print(f"  Training dataset optimized with:")
print(f"    - preprocess_input normalization ([-1.0, 1.0] range)")
print(f"    - cache() for faster repeated access")
print(f"    - prefetch() with AUTOTUNE for overlapping CPU/GPU")

# Normalize and optimize validation dataset
val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print(f"  Validation dataset optimized with:")
print(f"    - preprocess_input normalization ([-1.0, 1.0] range)")
print(f"    - cache() for faster repeated access")
print(f"    - prefetch() with AUTOTUNE for overlapping CPU/GPU")

# ============================================================================
# STEP 7: Load Existing Model
# ============================================================================
print("\n[STEP 7] Loading previously trained model...")

model_path = 'mobilenetv2_skin_disease_model.keras'
model = tf.keras.models.load_model(model_path)

print(f"  Model loaded successfully from: {model_path}")
print(f"  Model type: {type(model)}")

# ============================================================================
# STEP 8: Locate and Unfreeze MobileNetV2 Base Model
# ============================================================================
print("\n[STEP 8] Locating MobileNetV2 base model and unfreezing top layers...")

# Find the MobileNetV2 base model within the loaded model
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    # Try alternative approach - check if any layer is a Model instance
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

if base_model is None:
    raise ValueError("Could not locate MobileNetV2 base model in the loaded model!")

print(f"  MobileNetV2 base model found: {base_model.name}")
print(f"  Total layers in base model: {len(base_model.layers)}")

# Set base model to trainable
base_model.trainable = True

# Freeze first 100 layers, unfreeze the rest
print(f"\n  Freezing first 100 layers, unfreezing remaining layers...")
frozen_count = 0
trainable_count = 0

for i, layer in enumerate(base_model.layers):
    if i < 100:
        layer.trainable = False
        frozen_count += 1
    else:
        layer.trainable = True
        trainable_count += 1

print(f"  Frozen layers: {frozen_count} (layers 0-99)")
print(f"  Trainable layers: {trainable_count} (layers 100-{len(base_model.layers)-1})")

# Count total trainable parameters
total_trainable_params = sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers if layer.trainable and len(layer.get_weights()) > 0])
total_params = sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers if len(layer.get_weights()) > 0])

print(f"\n  Total model parameters: {total_params:,}")
print(f"  Trainable parameters: {total_trainable_params:,}")
print(f"  Non-trainable parameters: {total_params - total_trainable_params:,}")

# ============================================================================
# STEP 9: Microscopic Recompilation
# ============================================================================
print("\n[STEP 9] Recompiling model with drastically reduced learning rate...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 0.00001 as specified
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Model recompiled with:")
print(f"    - Optimizer: Adam (learning_rate=1e-5 = 0.00001)")
print(f"    - Loss function: categorical_crossentropy")
print(f"    - Metrics: accuracy")

# ============================================================================
# STEP 10: Set Up Advanced Callbacks
# ============================================================================
print("\n[STEP 10] Setting up advanced training callbacks...")

# EarlyStopping with patience=5 (since fine-tuning is slower)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print(f"  EarlyStopping callback configured:")
print(f"    - Monitor: val_loss")
print(f"    - Patience: 5 epochs (fine-tuning is slower)")
print(f"    - Restore best weights: True")

# ReduceLROnPlateau callback
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

print(f"\n  ReduceLROnPlateau callback configured:")
print(f"    - Monitor: val_loss")
print(f"    - Factor: 0.5 (reduce learning rate by half)")
print(f"    - Patience: 2 epochs")
print(f"    - Minimum learning rate: 1e-7")

# ModelCheckpoint to save the best model
model_checkpoint = callbacks.ModelCheckpoint(
    'mobilenetv2_finetuned_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print(f"\n  ModelCheckpoint callback configured:")
print(f"    - Save path: mobilenetv2_finetuned_model.keras")
print(f"    - Monitor: val_loss")
print(f"    - Save best only: True")

# ============================================================================
# STEP 11: Train the Model (Fine-Tuning)
# ============================================================================
print("\n[STEP 11] Starting Phase 2 fine-tuning...")
print("=" * 80)

# Training parameters
EPOCHS = 30

print(f"  Number of epochs: {EPOCHS}")
print(f"  Class weights will be applied during training")
print(f"  Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
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
print("\n[STEP 11] Phase 2 fine-tuning completed!")

# ============================================================================
# STEP 12: Display Training Results
# ============================================================================
print("\n[STEP 12] Fine-tuning results summary...")

# Get final training and validation metrics
final_train_loss = history.history['loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"  Final Training Loss: {final_train_loss:.4f}")
print(f"  Final Training Accuracy: {final_train_acc:.4f}")
print(f"  Final Validation Loss: {final_val_loss:.4f}")
print(f"  Final Validation Accuracy: {final_val_acc:.4f}")

# ============================================================================
# STEP 13: Display Best Epoch Information
# ============================================================================
print("\n[STEP 13] Best epoch information...")

# Find the epoch with best validation accuracy
best_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_acc = max(history.history['val_accuracy'])

# Find the epoch with best validation loss
best_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = min(history.history['val_loss'])

print(f"  Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch})")
print(f"  Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2 FINE-TUNING COMPLETE - SUMMARY")
print("=" * 80)
print(f"Model Architecture: MobileNetV2 (Fine-Tuned)")
print(f"Dataset: {BASE_DIR}")
print(f"Classes: {class_names}")
print(f"Total Epochs Trained: {len(history.history['loss'])}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Model Saved: mobilenetv2_finetuned_model.keras")
print(f"\nFine-Tuning Configuration:")
print(f"  - First 100 layers frozen")
print(f"  - Layers 100+ unfrozen for fine-tuning")
print(f"  - Learning rate: 1e-5 (0.00001)")
print(f"  - EarlyStopping patience: 5")
print(f"  - ReduceLROnPlateau factor: 0.5, patience: 2, min_lr: 1e-7")
print("=" * 80)
