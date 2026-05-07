import tensorflow as tf
from tensorflow.keras import callbacks, layers
import numpy as np
from sklearn.utils import class_weight
import os

tf.random.set_seed(42)
np.random.seed(42)

# --- Kaggle paths ---
BASE_DIR     = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/kaggle/New_Augmented_Dataset"
TRAIN_DIR    = os.path.join(BASE_DIR, "train")
VAL_DIR      = os.path.join(BASE_DIR, "val")
PHASE1_MODEL = "/kaggle/working/seventh_v2_efficientnetb2_phase1.keras"
OUTPUT_MODEL = "/kaggle/working/seventh_v2_efficientnetb2_model.keras"

# --- Hyperparameters ---
IMAGE_SIZE = (260, 260)   # EfficientNetB2 native resolution
BATCH_SIZE = 32

P1_LR     = 1e-3
P1_EPOCHS = 20            # B2 is larger, needs more warmup

P2_LR              = 1e-5
P2_EPOCHS          = 60   # more room to converge
UNFREEZE_FROM      = 220  # B2 has ~339 layers; unfreeze top ~120
ES_PATIENCE        = 15   # was 10 — LR decay killed the run too early last time
REDUCE_LR_PATIENCE = 8    # was 4 — give the model much more time before dropping LR
REDUCE_LR_FACTOR   = 0.3
GAMMA              = 2.0

ECZEMA_WEIGHT_BOOST = 1.2  # reduced from 1.4 — 1.4 over-compressed all classes to 82-84%

# ---------------------------------------------------------------------------
# 1. Load raw datasets
# ---------------------------------------------------------------------------
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='categorical')

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='categorical')

class_names = train_ds_raw.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ---------------------------------------------------------------------------
# 2. Class weights + focal-loss alpha
# ---------------------------------------------------------------------------
y_train = []
for _, labels in train_ds_raw:
    y_train.extend(np.argmax(labels.numpy(), axis=1))
y_train = np.array(y_train)

counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
alpha  = (1.0 / counts) / (1.0 / counts).sum()
alpha  = alpha.tolist()
print("Focal-loss alpha:", [round(a, 4) for a in alpha])

base_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(base_weights))

# Boost eczema (class index 1 in ['acne', 'eczema', 'tinea'])
eczema_idx = class_names.index('eczema')
class_weight_dict[eczema_idx] *= ECZEMA_WEIGHT_BOOST
print("Class weights (after eczema boost):", {k: round(v, 3) for k, v in class_weight_dict.items()})

# ---------------------------------------------------------------------------
# 3. Data pipeline
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.cache().prefetch(AUTOTUNE)
val_ds   = val_ds_raw.cache().prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# 4. Build model
# ---------------------------------------------------------------------------
def build_model(trainable_backbone=False, unfreeze_from=None):
    backbone = tf.keras.applications.EfficientNetB2(
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
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomContrast(0.15)(x)
    x = layers.RandomBrightness(0.15)(x)
    x = backbone(x, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# ---------------------------------------------------------------------------
# 5. Phase 1 — frozen backbone
# ---------------------------------------------------------------------------
print("\n=== Phase 1: frozen backbone ===")
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

history_p1 = model.fit(train_ds, validation_data=val_ds, epochs=P1_EPOCHS,
                        class_weight=class_weight_dict, callbacks=cbs_p1, verbose=1)

loss1, acc1 = model.evaluate(val_ds, verbose=0)
print(f"Phase 1 result: {acc1*100:.2f}%")

# ---------------------------------------------------------------------------
# 6. Phase 2 — unfreeze top layers and fine-tune
# ---------------------------------------------------------------------------
print(f"\n=== Phase 2: unfreeze from layer {UNFREEZE_FROM} ===")
model_p2 = tf.keras.models.load_model(PHASE1_MODEL)

backbone_p2 = next(l for l in model_p2.layers if 'efficientnet' in l.name.lower())
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

history_p2 = model_p2.fit(train_ds, validation_data=val_ds, epochs=P2_EPOCHS,
                           class_weight=class_weight_dict, callbacks=cbs_p2, verbose=1)

# ---------------------------------------------------------------------------
# 7. Final evaluation
# ---------------------------------------------------------------------------
best = tf.keras.models.load_model(OUTPUT_MODEL)
loss_f, acc_f = best.evaluate(val_ds, verbose=0)
print(f"\nPhase 1 result : {acc1*100:.2f}%")
print(f"Final result   : {acc_f*100:.2f}%")
print(f"Improvement    : {(acc_f - acc1)*100:.2f}%")

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
else:
    print(f"\nGap to 93% target: {(0.93 - acc_f)*100:.2f}%")
