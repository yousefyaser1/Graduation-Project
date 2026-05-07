import tensorflow as tf
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.utils import class_weight
import os

tf.random.set_seed(42)
np.random.seed(42)

# --- Paths (update for your environment) ---
TRAIN_DIR    = "New_Augmented_Dataset/train"
VAL_DIR      = "New_Augmented_Dataset/val"
PHASE1_MODEL = "sixth_code_mobilenetv2_phase1.keras"
OUTPUT_MODEL = "sixth_code_mobilenetv2_improved_model.keras"

# --- Hyperparameters ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Phase 1
P1_LR     = 1e-3
P1_EPOCHS = 15

# Phase 2
P2_LR              = 1e-5
P2_EPOCHS          = 40
UNFREEZE_FROM      = 100   # MobileNetV2 has 155 layers; unfreeze top ~55
ES_PATIENCE        = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR   = 0.3

TINEA_OVERSAMPLE   = 2     # repeat tinea samples N times in training pipeline
GAMMA              = 2.0

# ---------------------------------------------------------------------------
# 1. Load raw datasets
# ---------------------------------------------------------------------------
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMAGE_SIZE,
    batch_size=None, label_mode='categorical')   # unbatched for oversampling

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='categorical')

class_names = train_ds_raw.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

tinea_idx = class_names.index('tinea')
print(f"Tinea class index: {tinea_idx}")

# ---------------------------------------------------------------------------
# 2. Class weights + focal-loss alpha
# ---------------------------------------------------------------------------
y_train = []
for _, label in train_ds_raw:
    y_train.append(np.argmax(label.numpy()))
y_train = np.array(y_train)

counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
alpha  = (1.0 / counts) / (1.0 / counts).sum()
alpha  = alpha.tolist()
print("Focal-loss alpha (per class):", [round(a, 4) for a in alpha])

base_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(base_weights))
print("Class weights:", {k: round(v, 3) for k, v in class_weight_dict.items()})

# ---------------------------------------------------------------------------
# 3. Tinea oversampling via tf.data (no file modifications)
# ---------------------------------------------------------------------------
def is_tinea(img, label):
    return tf.equal(tf.argmax(label), tinea_idx)

def is_not_tinea(img, label):
    return tf.not_equal(tf.argmax(label), tinea_idx)

tinea_ds     = train_ds_raw.filter(is_tinea).repeat()
non_tinea_ds = train_ds_raw.filter(is_not_tinea).repeat()

# Sample so that tinea appears ~TINEA_OVERSAMPLE/(TINEA_OVERSAMPLE+1) of minority slots
total = len(y_train)
tinea_count     = int(counts[tinea_idx])
non_tinea_count = total - tinea_count

# target: effective tinea fraction after oversampling
target_tinea = tinea_count * TINEA_OVERSAMPLE
target_total = non_tinea_count + target_tinea
tinea_weight_sample = target_tinea / target_total

oversampled_ds = tf.data.Dataset.sample_from_datasets(
    [non_tinea_ds, tinea_ds],
    weights=[1.0 - tinea_weight_sample, tinea_weight_sample],
    seed=42)

# ---------------------------------------------------------------------------
# 4. Data pipeline
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="augmentation")

train_ds = (oversampled_ds
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (preprocess_input(x), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE))

val_ds = (val_ds_raw
    .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    .cache().prefetch(AUTOTUNE))

# Compute steps per epoch from effective dataset size
steps_per_epoch = (non_tinea_count + target_tinea) // BATCH_SIZE
print(f"Steps per epoch (oversampled): {steps_per_epoch}")

# ---------------------------------------------------------------------------
# 5. Build model
# ---------------------------------------------------------------------------
def build_model(trainable_backbone=False, unfreeze_from=None):
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3))

    if not trainable_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        if unfreeze_from is not None:
            for layer in backbone.layers[:unfreeze_from]:
                layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = backbone(inputs, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# ---------------------------------------------------------------------------
# 6. Phase 1 — frozen backbone, train head only
# ---------------------------------------------------------------------------
print("\n=== Phase 1: Training classification head (backbone frozen) ===")
model = build_model(trainable_backbone=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=P1_LR),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=GAMMA),
    metrics=['accuracy'])

model.summary()

cbs_p1 = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=ES_PATIENCE,
                            restore_best_weights=True, verbose=1, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=REDUCE_LR_FACTOR,
                                patience=REDUCE_LR_PATIENCE, min_lr=1e-8,
                                verbose=1, mode='max'),
    callbacks.ModelCheckpoint(PHASE1_MODEL, monitor='val_accuracy',
                              save_best_only=True, verbose=1, mode='max'),
]

history_p1 = model.fit(train_ds, validation_data=val_ds,
                        epochs=P1_EPOCHS, steps_per_epoch=steps_per_epoch,
                        class_weight=class_weight_dict, callbacks=cbs_p1, verbose=1)

loss1, acc1 = model.evaluate(val_ds, verbose=0)
print(f"Phase 1 result: {acc1*100:.2f}%")

# ---------------------------------------------------------------------------
# 7. Phase 2 — partial backbone unfreeze
#    Load the saved checkpoint instead of rebuilding — preserves BatchNorm stats.
# ---------------------------------------------------------------------------
print(f"\n=== Phase 2: Fine-tuning top layers (unfreeze from layer {UNFREEZE_FROM}) ===")

model_p2 = tf.keras.models.load_model(PHASE1_MODEL)
backbone_p2 = next(l for l in model_p2.layers if 'mobilenetv2' in l.name.lower())
backbone_p2.trainable = True
for layer in backbone_p2.layers[:UNFREEZE_FROM]:
    layer.trainable = False

trainable_count = sum(1 for l in backbone_p2.layers if l.trainable)
print(f"Trainable backbone layers: {trainable_count}")

model_p2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=P2_LR),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=GAMMA),
    metrics=['accuracy'])

cbs_p2 = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=ES_PATIENCE,
                            restore_best_weights=True, verbose=1, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=REDUCE_LR_FACTOR,
                                patience=REDUCE_LR_PATIENCE, min_lr=1e-8,
                                verbose=1, mode='max'),
    callbacks.ModelCheckpoint(OUTPUT_MODEL, monitor='val_accuracy',
                              save_best_only=True, verbose=1, mode='max'),
]

history_p2 = model_p2.fit(train_ds, validation_data=val_ds,
                           epochs=P2_EPOCHS, steps_per_epoch=steps_per_epoch,
                           class_weight=class_weight_dict, callbacks=cbs_p2, verbose=1)

# ---------------------------------------------------------------------------
# 8. Final evaluation
# ---------------------------------------------------------------------------
best = tf.keras.models.load_model(OUTPUT_MODEL)
loss_f, acc_f = best.evaluate(val_ds, verbose=0)
print(f"\nPhase 1 result : {acc1*100:.2f}%")
print(f"Final result   : {acc_f*100:.2f}%")
print(f"Improvement    : {(acc_f - acc1)*100:.2f}%")

# Per-class breakdown
all_preds, all_labels = [], []
for imgs, lbls in val_ds:
    all_preds.append(best(imgs, training=False).numpy())
    all_labels.append(lbls.numpy())
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
pred_cls   = np.argmax(all_preds,  axis=1)
true_cls   = np.argmax(all_labels, axis=1)

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask = true_cls == i
    acc_i = np.mean(pred_cls[mask] == true_cls[mask]) * 100
    print(f"  {name}: {acc_i:.2f}%  ({mask.sum()} samples)")

if acc_f >= 0.95:
    print("\nGOAL ACHIEVED: 95%+ accuracy!")
elif acc_f >= 0.93:
    print("\nTarget reached: 93%+ accuracy!")
