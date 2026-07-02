"""
Skin Disease Classification using MobileNetV2 Transfer Learning
Training script for classifying skin diseases (acne, eczema, tinea)
"""

# pyre-ignore[import-error]
import tensorflow as tf 
# pyre-ignore[import-error]
from tensorflow.keras.applications import MobileNetV2
# pyre-ignore[import-error]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# pyre-ignore[import-error]
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
# pyre-ignore[import-error]
from tensorflow.keras.models import Model
# pyre-ignore[import-error]
from tensorflow.keras.optimizers import Adam
# pyre-ignore[import-error]
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# Step 1: Data Loading
# =============================================================================
print("=" * 60)
print("Step 1: Loading Dataset")
print("=" * 60)

# Hardcoded dataset directory path
DATASET_DIR = r"C:\Users\A\Graduation-Project\Augmented_Dataset"

# Load training dataset
train_dir = f"{DATASET_DIR}/train"
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)
print(f"Training dataset loaded from: {train_dir}")

# Load validation dataset
val_dir = f"{DATASET_DIR}/val"
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)
print(f"Validation dataset loaded from: {val_dir}")

# Get class names
class_names = train_dataset.class_names
print(f"Classes: {class_names}")
print(f"Number of classes: {len(class_names)}")

# =============================================================================
# Step 2: Normalization with preprocessing and optimization
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Applying Normalization and Optimization")
print("=" * 60)

# Define normalization function using MobileNetV2 preprocessing
def normalize_images(dataset):
    """Apply MobileNetV2 preprocessing to normalize pixels to [-1, 1] range"""
    return dataset.map(lambda x, y: (preprocess_input(x), y))

# Apply normalization to training dataset
train_dataset = normalize_images(train_dataset)

# Apply normalization to validation dataset
val_dataset = normalize_images(val_dataset)

# Optimize loading with AUTOTUNE, cache(), and prefetch()
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Normalization applied using mobilenet_v2.preprocess_input")
print("Dataset optimization applied: cache() and prefetch()")

# =============================================================================
# Step 3: Model Architecture - MobileNetV2 Base
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Building Model Architecture")
print("=" * 60)

# Instantiate MobileNetV2 base model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the base model
base_model.trainable = False

print("MobileNetV2 base model loaded with ImageNet weights")
print("Base model frozen (trainable = False)")

# =============================================================================
# Step 4: Classification Head
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Adding Classification Head")
print("=" * 60)

# Build the complete model
inputs = tf.keras.Input(shape=(224, 224, 3))

# Pass through base model
x = base_model(inputs, training=False)

# Add GlobalAveragePooling2D
x = GlobalAveragePooling2D()(x)

# Add Dropout
x = Dropout(0.2)(x)

# Add Dense output layer with 3 units and softmax activation
outputs = Dense(3, activation='softmax')(x)

# Create the model
model = Model(inputs, outputs)

print("Classification head added:")
print("  - GlobalAveragePooling2D()")
print("  - Dropout(0.2)")
print("  - Dense(3, activation='softmax')")

# =============================================================================
# Step 5: Compilation
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Compiling Model")
print("=" * 60)

# Compile with Adam optimizer, categorical_crossentropy loss, and accuracy
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled with:")
print("  - Optimizer: Adam(learning_rate=0.0001)")
print("  - Loss: categorical_crossentropy")
print("  - Metrics: accuracy")

# =============================================================================
# Step 6: Model Summary Output
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: Model Summary")
print("=" * 60)

# Capture model.summary() into a string list and print
summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))

# Print each line using standard print()
for line in summary_lines:
    print(line)

# =============================================================================
# Step 7: Training Setup and Execution
# =============================================================================
print("\n" + "=" * 60)
print("Step 7: Training Model")
print("=" * 60)

# Set up EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("EarlyStopping callback configured:")
print("  - monitor: val_loss")
print("  - patience: 3")
print("  - restore_best_weights: True")

# Execute model.fit() for 15 epochs
print("\nStarting training for 15 epochs...")
print("-" * 40)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stopping]
)

print("-" * 40)
print("Training completed!")

# Print final results
print("\n" + "=" * 60)
print("Training Results Summary")
print("=" * 60)
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Total epochs trained: {len(history.history['loss'])}")
