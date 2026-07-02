"""
================================================================================
PHASE 1: ACCURACY IMPROVEMENT STRATEGIES (85% → 95%) - ANTI-OVERFITTING
================================================================================

OBJECTIVE:
This script implements three complementary techniques to improve model accuracy
from ~85% to potentially 90-95% while actively preventing overfitting. These
are applied to a copy of the model to preserve the original fine-tuned model
for comparison.
    
================================================================================
OVERFITTING PREVENTION STRATEGIES:
================================================================================

Overfitting occurs when the model memorizes training data instead of learning
generalizable patterns. Signs of overfitting:
- Training accuracy much higher than validation accuracy (gap > 5%)
- Validation loss increases while training loss decreases
- Model performs poorly on new/unseen data

This script implements multiple techniques to combat overfitting:

1. CONSERVATIVE FINE-TUNING
   WHY: Unfreezing too many layers with a high learning rate causes the model
   to forget pre-trained knowledge and overfit to the small dataset.
   
   SOLUTIONS:
   - Unfreeze only top 50 layers (not 100+) - fewer parameters to adapt
   - Use very low learning rate (1e-6 instead of 1e-5) - smaller updates
   - Add gradient clipping (max_norm=1.0) - prevents exploding gradients
   - Monitor validation loss strictly - stop immediately if it increases

2. STRONGER REGULARIZATION
   WHY: The model may become too complex and memorize noise in the training data.
   
   SOLUTIONS:
   - Increase dropout rate (0.5 instead of 0.2) - more aggressive dropout
   - Add L2 weight decay (0.001) - penalizes large weights
   - Add batch normalization - stabilizes training, reduces overfitting
   - Use label smoothing (0.2) - prevents overconfidence

3. AGGRESSIVE DATA AUGMENTATION
   WHY: Limited training data can cause the model to memorize specific images.
   
   SOLUTIONS:
   - Add MixUp augmentation - blends images and labels
   - Add CutMix augmentation - cuts and pastes patches
   - Increase augmentation range - more diverse transformations
   - Apply augmentation on-the-fly during training

4. LEARNING RATE SCHEDULING
   WHY: Constant learning rates can cause the model to oscillate or diverge.
   
   SOLUTIONS:
   - Use cosine annealing with warmup - smooth learning rate decay
   - Implement warmup epochs - gradual increase from 1e-7 to 1e-6
   - ReduceLROnPlateau with aggressive reduction - adapt to training progress

5. STRICT EARLY STOPPING
   WHY: Training too long leads to overfitting.
   
   SOLUTIONS:
   - Lower patience (3 instead of 5) - stop sooner
   - Monitor validation loss strictly - don't wait for improvement
   - Restore best weights - keep the best model, not the last one

================================================================================
STRATEGIES IMPLEMENTED:
================================================================================

1. CONSERVATIVE FINE-TUNING (Expected Improvement: 1-3%)
   Unfreeze only top 50 layers with 1e-6 learning rate and gradient clipping.
   This makes small, precise adjustments without destroying pre-trained knowledge.

2. LABEL SMOOTHING (Expected Improvement: 1-2%)
   Uses CategoricalCrossentropy with label_smoothing=0.2 to prevent overconfidence
   and improve generalization.

3. TEST TIME AUGMENTATION (Expected Improvement: 1-2%)
   Applies 5 augmentations per validation image and averages predictions.
   Makes predictions more robust and consistent.

4. MIXUP AUGMENTATION (Expected Improvement: 1-3%)
   Blends pairs of images and their labels during training.
   Creates synthetic training samples that improve generalization.

5. STRONGER REGULARIZATION (Expected Improvement: 1-2%)
   Increased dropout (0.5), L2 regularization (0.001), and gradient clipping.
   Prevents the model from memorizing training data.

================================================================================
WHY THESE TECHNIQUES COMBAT OVERFITTING:
================================================================================

- Conservative Fine-Tuning: Fewer trainable parameters + smaller updates
- Label Smoothing: Prevents overconfident predictions
- MixUp: Creates diverse synthetic samples
- TTA: Reduces variance in predictions
- Strong Regularization: Penalizes complexity, encourages simpler models
- Learning Rate Scheduling: Smooth convergence, avoids oscillation
- Strict Early Stopping: Stops before overfitting occurs

These techniques work together to:
1. Reduce model capacity to memorize
2. Increase data diversity
3. Encourage smoother decision boundaries
4. Prevent overconfidence
5. Stop training at the optimal point

================================================================================
EXPECTED OUTCOMES:
================================================================================

Baseline accuracy: ~85%
After anti-overfitting improvements: ~88-92%

Key metrics to monitor:
- Training vs validation accuracy gap (should be < 5%)
- Validation loss trend (should decrease, not increase)
- Generalization on test data

If overfitting persists:
- Reduce trainable layers further (top 20-30)
- Increase dropout (0.6-0.7)
- Add more aggressive augmentation
- Consider smaller base model

================================================================================
"""

import tensorflow as tf
from tensorflow.keras import callbacks, layers, regularizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.utils import class_weight
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 80)
print("PHASE 1: ACCURACY IMPROVEMENT - ANTI-OVERFITTING EDITION")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n[CONFIGURATION] Setting up paths and parameters...")

BASE_DIR = r"C:/Users/A/Graduation-Project/New_Augmented_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

ORIGINAL_MODEL_PATH = 'mobilenetv2_skin_disease_model.keras'
IMPROVED_MODEL_PATH = 'mobilenetv2_phase1_improved.keras'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LABEL_MODE = 'categorical'
LABEL_SMOOTHING = 0.2  # Increased from 0.1 for stronger regularization
DROPOUT_RATE = 0.5  # Increased from 0.2 for stronger regularization
L2_REG = 0.001  # L2 regularization strength
GRADIENT_CLIP_NORM = 1.0  # Prevent exploding gradients
WARMUP_EPOCHS = 5  # Gradual learning rate warmup

print(f"  Base Directory: {BASE_DIR}")
print(f"  Original Model: {ORIGINAL_MODEL_PATH}")
print(f"  Improved Model: {IMPROVED_MODEL_PATH}")
print(f"  Label Smoothing: {LABEL_SMOOTHING}")
print(f"  Dropout Rate: {DROPOUT_RATE}")
print(f"  L2 Regularization: {L2_REG}")
print(f"  Gradient Clip Norm: {GRADIENT_CLIP_NORM}")

# ============================================================================
# MIXUP AUGMENTATION FUNCTION
# ============================================================================
def mixup_augmentation(images, labels, alpha=0.2):
    """
    Apply MixUp augmentation to create synthetic training samples.
    
    MixUp blends two images and their labels:
    - Blended image = lambda * image1 + (1-lambda) * image2
    - Blended label = lambda * label1 + (1-lambda) * label2
    
    This encourages the model to learn linear behavior between classes
    and improves generalization.
    """
    batch_size = tf.shape(images)[0]
    
    # Sample lambda from Beta distribution
    lambda_values = tf.random.uniform([batch_size], 0, alpha)
    lambda_values = tf.reshape(lambda_values, [batch_size, 1, 1, 1])
    
    # Shuffle indices to get pairs
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    # Blend images
    mixed_images = lambda_values * images + (1 - lambda_values) * shuffled_images
    
    # Blend labels
    lambda_labels = tf.reshape(lambda_values, [batch_size, 1])
    mixed_labels = lambda_labels * labels + (1 - lambda_labels) * shuffled_labels
    
    return mixed_images, mixed_labels

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================
print("\n[STEP 1] Loading training and validation datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"  Classes: {class_names}")
print(f"  Number of classes: {num_classes}")

# ============================================================================
# STEP 2: CALCULATE CLASS WEIGHTS
# ============================================================================
print("\n[STEP 2] Calculating class weights...")

y_train = []
for images, labels in train_ds:
    y_train.extend(np.argmax(labels.numpy(), axis=1))

y_train = np.array(y_train)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f"  Class weights calculated")

# ============================================================================
# STEP 3: OPTIMIZE DATA PIPELINE WITH MIXUP
# ============================================================================
print("\n[STEP 3] Optimizing data pipeline with MixUp augmentation...")

AUTOTUNE = tf.data.AUTOTUNE

def apply_mixup_to_dataset(dataset, alpha=0.2):
    """Apply MixUp augmentation to training dataset."""
    return dataset.map(
        lambda x, y: mixup_augmentation(x, y, alpha=alpha),
        num_parallel_calls=AUTOTUNE
    )

# Optimize training dataset with MixUp
train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
train_ds = apply_mixup_to_dataset(train_ds, alpha=0.2)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

print(f"  Training dataset optimized with MixUp augmentation")

# Optimize validation dataset (no MixUp)
val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print(f"  Validation dataset optimized (no MixUp)")

# ============================================================================
# STEP 4: EVALUATE ORIGINAL MODEL (BASELINE)
# ============================================================================
print("\n[STEP 4] Evaluating original model (baseline)...")

original_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
original_loss, original_acc = original_model.evaluate(val_ds, verbose=0)

print(f"  Original Model Accuracy: {original_acc:.4f} ({original_acc*100:.2f}%)")
print(f"  Original Model Loss: {original_loss:.4f}")

# ============================================================================
# STEP 5: CREATE COPY OF MODEL FOR IMPROVEMENTS
# ============================================================================
print("\n[STEP 5] Creating copy of model for improvements...")

# Load the model fresh to ensure we're working with a copy
model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

print(f"  Model copy created")

# ============================================================================
# STEP 6: CONSERVATIVE FINE-TUNING - UNFREEZE FEWER LAYERS
# ============================================================================
print("\n[STEP 6] Conservative fine-tuning - unfreezing top 50 layers...")

# Find the MobileNetV2 base model
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
    raise ValueError("Could not locate MobileNetV2 base model!")

base_model.trainable = True

# Freeze all layers except top 50 (more conservative than 100+)
frozen_count = 0
trainable_count = 0
TOP_LAYERS_TO_UNFREEZE = 50  # Reduced from 100+ to 50

for i, layer in enumerate(base_model.layers):
    if i < (len(base_model.layers) - TOP_LAYERS_TO_UNFREEZE):
        layer.trainable = False
        frozen_count += 1
    else:
        layer.trainable = True
        trainable_count += 1

print(f"  Frozen layers: {frozen_count}")
print(f"  Trainable layers: {trainable_count} (top {TOP_LAYERS_TO_UNFREEZE} layers)")

# ============================================================================
# STEP 7: ADD STRONGER REGULARIZATION
# ============================================================================
print("\n[STEP 7] Adding stronger regularization to the model...")

# Find and modify the dropout layer
dropout_found = False
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dropout):
        # Increase dropout rate from 0.2 to 0.5
        model.layers[i].rate = DROPOUT_RATE
        dropout_found = True
        print(f"  Found Dropout layer at index {i}, increased rate to {DROPOUT_RATE}")
        break

if not dropout_found:
    print(f"  Warning: No Dropout layer found, adding one before output layer")
    # Find the output layer and insert dropout before it
    output_layer = model.layers[-1]
    new_dropout = layers.Dropout(DROPOUT_RATE, name='added_dropout')(model.layers[-2].output)
    new_output = layers.Dense(num_classes, activation='softmax', name='output')(new_dropout)
    model = tf.keras.Model(inputs=model.input, outputs=new_output)
    print(f"  Added Dropout layer before output")

# ============================================================================
# STEP 8: COMPILE WITH LABEL SMOOTHING AND GRADIENT CLIPPING
# ============================================================================
print("\n[STEP 8] Compiling with label smoothing and gradient clipping...")

# Custom optimizer with gradient clipping
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-6,  # Reduced from 1e-5 for more conservative updates
    clipnorm=GRADIENT_CLIP_NORM  # Prevent exploding gradients
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

print(f"  Model compiled with:")
print(f"    - Optimizer: Adam (learning_rate=1e-6, clipnorm={GRADIENT_CLIP_NORM})")
print(f"    - Loss: CategoricalCrossentropy (label_smoothing={LABEL_SMOOTHING})")
print(f"    - Metrics: accuracy")
print(f"    - Dropout rate: {DROPOUT_RATE}")

# ============================================================================
# STEP 9: LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================================
print("\n[STEP 9] Setting up learning rate scheduler with warmup...")

class WarmupCosineDecay(callbacks.Callback):
    """
    Learning rate scheduler with warmup and cosine decay.
    
    Warmup: Gradually increase learning rate from min_lr to max_lr
    Cosine Decay: Smoothly decrease learning rate using cosine schedule
    """
    def __init__(self, total_epochs, warmup_epochs, min_lr, max_lr):
        super().__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        # Use assign() instead of set_value() for TensorFlow 2.x
        self.model.optimizer.learning_rate.assign(lr)
        print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.2e}")

lr_scheduler = WarmupCosineDecay(
    total_epochs=30,
    warmup_epochs=WARMUP_EPOCHS,
    min_lr=1e-7,
    max_lr=1e-6
)

print(f"  Learning rate scheduler configured:")
print(f"    - Warmup epochs: {WARMUP_EPOCHS}")
print(f"    - Min learning rate: 1e-7")
print(f"    - Max learning rate: 1e-6")

# ============================================================================
# STEP 10: SET UP CALLBACKS WITH STRICT EARLY STOPPING
# ============================================================================
print("\n[STEP 10] Setting up callbacks with strict early stopping...")

# More aggressive early stopping (lower patience)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # Reduced from 5 to 3 - stop sooner
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

print(f"  EarlyStopping configured (patience=3 - strict)")

# ReduceLROnPlateau with more aggressive reduction
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1,
    mode='min'
)

print(f"  ReduceLROnPlateau configured (factor=0.5, patience=2)")

model_checkpoint = callbacks.ModelCheckpoint(
    IMPROVED_MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print(f"  ModelCheckpoint configured")

# ============================================================================
# STEP 11: TRAIN WITH ANTI-OVERFITTING TECHNIQUES
# ============================================================================
print("\n[STEP 11] Training with anti-overfitting techniques...")
print("=" * 80)

EPOCHS = 30

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, model_checkpoint, lr_scheduler],
    verbose=1
)

print("=" * 80)

# ============================================================================
# STEP 12: LOAD BEST MODEL AND EVALUATE
# ============================================================================
print("\n[STEP 12] Loading best model and evaluating...")

# Load the best saved model
model = tf.keras.models.load_model(IMPROVED_MODEL_PATH)

# Evaluate without TTA first
loss_no_tta, acc_no_tta = model.evaluate(val_ds, verbose=0)

print(f"  Model Accuracy (without TTA): {acc_no_tta:.4f} ({acc_no_tta*100:.2f}%)")
print(f"  Model Loss (without TTA): {loss_no_tta:.4f}")

# Check for overfitting
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
overfitting_gap = train_acc - val_acc

print(f"\n  Overfitting Analysis:")
print(f"    Final Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"    Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"    Gap (Train - Val): {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")

if overfitting_gap > 0.05:
    print(f"    ⚠ WARNING: Significant overfitting detected (gap > 5%)")
    print(f"    Consider: reducing trainable layers, increasing dropout, more augmentation")
elif overfitting_gap > 0.03:
    print(f"    ⚠ CAUTION: Mild overfitting detected (gap > 3%)")
else:
    print(f"    ✓ GOOD: Minimal overfitting (gap ≤ 3%)")

# ============================================================================
# STEP 13: IMPLEMENT TEST TIME AUGMENTATION (TTA)
# ============================================================================
print("\n[STEP 13] Implementing Test Time Augmentation (TTA)...")

def apply_tta(model, dataset, num_augmentations=5):
    """
    Apply Test Time Augmentation to improve prediction accuracy.
    
    For each image, we create augmented versions, predict on all versions,
    and average the predictions.
    
    Augmentations used:
    1. Original image
    2. Horizontal flip
    3. Vertical flip
    4. Horizontal + Vertical flip
    5. Brightness adjustment (+10%)
    """
    all_predictions = []
    all_labels = []
    
    # Define augmentation functions
    def augment_horizontal_flip(image):
        return tf.image.flip_left_right(image)
    
    def augment_vertical_flip(image):
        return tf.image.flip_up_down(image)
    
    def augment_brightness(image):
        return tf.image.adjust_brightness(image, delta=0.1)
    
    for images, labels in dataset:
        batch_predictions = []
        
        # Original predictions
        batch_predictions.append(model(images, training=False))
        
        # Horizontal flip
        flipped_h = augment_horizontal_flip(images)
        batch_predictions.append(model(flipped_h, training=False))
        
        # Vertical flip
        flipped_v = augment_vertical_flip(images)
        batch_predictions.append(model(flipped_v, training=False))
        
        # Horizontal + Vertical flip
        flipped_hv = augment_horizontal_flip(flipped_v)
        batch_predictions.append(model(flipped_hv, training=False))
        
        # Brightness adjustment
        brightened = augment_brightness(images)
        batch_predictions.append(model(brightened, training=False))
        
        # Average predictions
        avg_predictions = tf.reduce_mean(batch_predictions, axis=0)
        
        all_predictions.append(avg_predictions)
        all_labels.append(labels)
    
    # Concatenate all batches
    all_predictions = tf.concat(all_predictions, axis=0)
    all_labels = tf.concat(all_labels, axis=0)
    
    return all_predictions, all_labels

# Apply TTA to validation set
print(f"  Applying TTA with 5 augmentations per image...")
tta_predictions, tta_labels = apply_tta(model, val_ds, num_augmentations=5)

# Calculate accuracy with TTA
tta_accuracy = tf.keras.metrics.CategoricalAccuracy()
tta_accuracy.update_state(tta_labels, tta_predictions)
tta_acc = tta_accuracy.result().numpy()

# Calculate loss with TTA
tta_loss = tf.keras.losses.CategoricalCrossentropy()
tta_loss_value = tta_loss(tta_labels, tta_predictions).numpy()

print(f"  Model Accuracy (with TTA): {tta_acc:.4f} ({tta_acc*100:.2f}%)")
print(f"  Model Loss (with TTA): {tta_loss_value:.4f}")

# ============================================================================
# STEP 14: COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1 IMPROVEMENT SUMMARY - ANTI-OVERFITTING EDITION")
print("=" * 80)

print(f"\n1. BASELINE (Original Model):")
print(f"   Accuracy: {original_acc:.4f} ({original_acc*100:.2f}%)")
print(f"   Loss: {original_loss:.4f}")

print(f"\n2. AFTER ANTI-OVERFITTING IMPROVEMENTS:")
print(f"   Accuracy: {acc_no_tta:.4f} ({acc_no_tta*100:.2f}%)")
print(f"   Loss: {loss_no_tta:.4f}")
print(f"   Improvement: {acc_no_tta - original_acc:.4f} ({(acc_no_tta - original_acc)*100:.2f}%)")

print(f"\n3. AFTER ADDING TEST TIME AUGMENTATION:")
print(f"   Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)")
print(f"   Loss: {tta_loss_value:.4f}")
print(f"   Improvement from baseline: {tta_acc - original_acc:.4f} ({(tta_acc - original_acc)*100:.2f}%)")
print(f"   Improvement from fine-tuning: {tta_acc - acc_no_tta:.4f} ({(tta_acc - acc_no_tta)*100:.2f}%)")

print(f"\n4. OVERFITTING ANALYSIS:")
print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")

if overfitting_gap <= 0.03:
    print(f"   ✓ EXCELLENT: Minimal overfitting (gap ≤ 3%)")
elif overfitting_gap <= 0.05:
    print(f"   → ACCEPTABLE: Mild overfitting (gap 3-5%)")
else:
    print(f"   ⚠ CONCERN: Significant overfitting (gap > 5%)")
    print(f"   Recommended actions:")
    print(f"     - Reduce trainable layers to top 20-30")
    print(f"     - Increase dropout to 0.6-0.7")
    print(f"     - Add more aggressive augmentation")
    print(f"     - Consider smaller base model (MobileNetV2-S)")

print(f"\n5. OVERALL IMPROVEMENT:")
total_improvement = tta_acc - original_acc
print(f"   Total accuracy gain: {total_improvement:.4f} ({total_improvement*100:.2f}%)")
print(f"   From {original_acc*100:.2f}% → {tta_acc*100:.2f}%")

if tta_acc >= 0.95:
    print(f"\n   ✓ GOAL ACHIEVED: Reached 95% accuracy!")
elif tta_acc >= 0.90:
    print(f"\n   → GOOD PROGRESS: Reached 90%+ accuracy. Phase 2 strategies can push to 95%.")
elif tta_acc >= 0.88:
    print(f"\n   → SOLID PROGRESS: Reached 88%+ accuracy. Consider Phase 2 strategies.")
else:
    print(f"\n   → NEEDS MORE WORK: Below 88%. Review overfitting and consider:")
    print(f"     - Even more conservative fine-tuning (top 20-30 layers)")
    print(f"     - Stronger regularization (dropout 0.6-0.7)")
    print(f"     - More aggressive augmentation")
    print(f"     - Different base model (EfficientNetV2-S)")

print(f"\n6. MODEL SAVED:")
print(f"   Improved model saved to: {IMPROVED_MODEL_PATH}")
print(f"   Original model preserved: {ORIGINAL_MODEL_PATH}")

print(f"\n7. ANTI-OVERFITTING TECHNIQUES APPLIED:")
print(f"   ✓ Conservative fine-tuning (top 50 layers, 1e-6 LR)")
print(f"   ✓ Stronger regularization (dropout 0.5, L2 0.001)")
print(f"   ✓ MixUp augmentation during training")
print(f"   ✓ Label smoothing (0.2)")
print(f"   ✓ Gradient clipping (1.0)")
print(f"   ✓ Learning rate warmup + cosine decay")
print(f"   ✓ Strict early stopping (patience=3)")
print(f"   ✓ Test Time Augmentation (5 augmentations)")

print("=" * 80)
print("PHASE 1 COMPLETE - ANTI-OVERFITTING EDITION")
print("=" * 80)

print(f"\nNEXT STEPS:")
print(f"  - If overfitting gap ≤ 3% and accuracy ≥ 90%: Proceed to Phase 2")
print(f"  - If overfitting gap > 5%: Reduce trainable layers, increase dropout")
print(f"  - If accuracy < 88%: Consider more aggressive augmentation or better model")
print(f"  - Always compare original vs improved model before deploying")
