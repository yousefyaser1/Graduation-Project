"""
MobileNetV2 Transfer Learning Model for Skin Disease Classification
Dataset: New_Augmented_Dataset/train with 3 classes (acne, eczema, tinea)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.utils import class_weight
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 80)
print("STARTING MOBILENETV2 TRANSFER LEARNING TRAINING")
print("=" * 80)

# ============================================================================
# STEP 1: Dataset Configuration
# ============================================================================
print("\n[STEP 1] Configuring dataset paths and parameters...")

# Relative paths for dataset directories
BASE_DIR = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset"
TRAIN_DIR = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset/train"
VAL_DIR = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset/val"

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
    shuffle=False,
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
# STEP 5.5: Dataset Summary
# ============================================================================
print("\n" + "=" * 80)
print("[DATASET SUMMARY]")
print("=" * 80)

# Count total training images
total_train_images = 0
train_class_counts = {class_name: 0 for class_name in class_names}

for images, labels in train_ds:
    batch_size = images.shape[0]
    total_train_images += batch_size
    # Count images per class
    label_indices = np.argmax(labels.numpy(), axis=1)
    for idx in label_indices:
        train_class_counts[class_names[idx]] += 1

print(f"\nTRAINING SET:")
print(f"  Total images: {total_train_images}")
print(f"  Breakdown by class:")
for class_name in class_names:
    count = train_class_counts[class_name]
    percentage = (count / total_train_images) * 100
    print(f"    {class_name}: {count} images ({percentage:.2f}%)")

# Count total validation images
total_val_images = 0
val_class_counts = {class_name: 0 for class_name in class_names}

for images, labels in val_ds:
    batch_size = images.shape[0]
    total_val_images += batch_size
    # Count images per class
    label_indices = np.argmax(labels.numpy(), axis=1)
    for idx in label_indices:
        val_class_counts[class_names[idx]] += 1

print(f"\nVALIDATION SET:")
print(f"  Total images: {total_val_images}")
print(f"  Breakdown by class:")
for class_name in class_names:
    count = val_class_counts[class_name]
    percentage = (count / total_val_images) * 100
    print(f"    {class_name}: {count} images ({percentage:.2f}%)")

print(f"\nTOTAL DATASET:")
print(f"  Total images (train + val): {total_train_images + total_val_images}")

print("\n" + "=" * 80)

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
# STEP 7: Build MobileNetV2 Base Model
# ============================================================================
print("\n[STEP 7] Building MobileNetV2 base model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

print(f"  MobileNetV2 base model created")
print(f"  Input shape: {base_model.input_shape}")
print(f"  Output shape: {base_model.output_shape}")

# ============================================================================
# STEP 8: Freeze the Base Model
# ============================================================================
print("\n[STEP 8] Freezing the base model weights...")

base_model.trainable = False

print(f"  Base model trainable: {base_model.trainable}")
print(f"  Total layers in base model: {len(base_model.layers)}")
print(f"  Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# ============================================================================
# STEP 9: Build Classification Head
# ============================================================================
print("\n[STEP 9] Building classification head...")

inputs = tf.keras.Input(shape=(224, 224, 3))

# Pass inputs through base model
x = base_model(inputs, training=False)

# Add classification head
x = layers.GlobalAveragePooling2D()(x)
print(f"  Added GlobalAveragePooling2D layer")

x = layers.Dropout(0.2)(x)
print(f"  Added Dropout(0.2) layer for regularization")

outputs = layers.Dense(num_classes, activation='softmax')(x)
print(f"  Added Dense output layer with {num_classes} units and softmax activation")

# Create the complete model
model = models.Model(inputs, outputs)

print(f"  Complete model architecture built")

# ============================================================================
# STEP 10: Display Model Summary
# ============================================================================
print("\n[STEP 10] Displaying model summary...")
print("=" * 80)
model.summary()
print("=" * 80)

# ============================================================================
# STEP 11: Compile the Model
# ============================================================================
print("\n[STEP 11] Compiling the model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Model compiled with:")
print(f"    - Optimizer: Adam (learning_rate=0.0001)")
print(f"    - Loss function: categorical_crossentropy")
print(f"    - Metrics: accuracy")

# ============================================================================
# STEP 12: Set Up Callbacks
# ============================================================================
print("\n[STEP 12] Setting up training callbacks...")

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print(f"  EarlyStopping callback configured:")
print(f"    - Monitor: val_loss")
print(f"    - Patience: 3 epochs")
print(f"    - Restore best weights: True")

# ============================================================================
# STEP 13: Train the Model
# ============================================================================
print("\n[STEP 13] Starting model training...")
print("=" * 80)

# Training parameters
EPOCHS = 

print(f"  Number of epochs: {EPOCHS}")
print(f"  Class weights will be applied during training")
print("=" * 80)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

print("=" * 80)
print("\n[STEP 13] Model training completed!")

# ============================================================================
# STEP 14: Display Training Results
# ============================================================================
print("\n[STEP 14] Training results summary...")

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
# STEP 15: Save the Trained Model
# ============================================================================
print("\n[STEP 15] Saving the trained model...")

model_save_path = 'mobilenetv2_skin_disease_model.keras'
model.save(model_save_path)

print(f"  Model saved to: {model_save_path}")

# ============================================================================
# STEP 16: Display Best Epoch Information
# ============================================================================
print("\n[STEP 16] Best epoch information...")

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
print("TRAINING COMPLETE - SUMMARY")
print("=" * 80)
print(f"Model Architecture: MobileNetV2 (Transfer Learning)")
print(f"Dataset: {BASE_DIR}")
print(f"Classes: {class_names}")
print(f"Total Epochs Trained: {len(history.history['loss'])}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Model Saved: {model_save_path}")
print("=" * 80)
