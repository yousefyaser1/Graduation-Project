import tensorflow as tf
from tensorflow.keras import callbacks, layers
import numpy as np
from sklearn.utils import class_weight
import os

tf.random.set_seed(42)
np.random.seed(42)

# --- Kaggle paths ---
BASE_DIR  = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/kaggle/New_Augmented_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

B0_PHASE1  = "/kaggle/working/ensemble_b0_phase1.keras"
B0_MODEL   = "/kaggle/working/ensemble_b0_model.keras"
B2_PHASE1  = "/kaggle/working/ensemble_b2_phase1.keras"
B2_MODEL   = "/kaggle/working/ensemble_b2_model.keras"

TTA_STEPS  = 10   # number of augmented passes per image at inference

AUTOTUNE   = tf.data.AUTOTUNE

# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def get_class_info(train_ds_raw, num_classes):
    y_train = []
    for _, labels in train_ds_raw:
        y_train.extend(np.argmax(labels.numpy(), axis=1))
    y_train = np.array(y_train)
    counts  = np.bincount(y_train, minlength=num_classes).astype(float)
    alpha   = ((1.0 / counts) / (1.0 / counts).sum()).tolist()
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train)
    return alpha, dict(enumerate(weights)), y_train


def make_callbacks(phase1_path, model_path, es_patience, reduce_patience):
    return [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=es_patience,
                                restore_best_weights=True, verbose=1, mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                                    patience=reduce_patience, min_lr=1e-8,
                                    verbose=1, mode='max'),
    ], [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=es_patience,
                                restore_best_weights=True, verbose=1, mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                                    patience=reduce_patience, min_lr=1e-8,
                                    verbose=1, mode='max'),
        callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                  save_best_only=True, verbose=1, mode='max'),
    ]


# ===========================================================================
# MODEL A — EfficientNetB0  (224×224)
# ===========================================================================
print("\n" + "="*60)
print("MODEL A: EfficientNetB0")
print("="*60)

IMG_B0    = (224, 224)
BATCH_B0  = 32

train_b0_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMG_B0,
    batch_size=BATCH_B0, label_mode='categorical')
val_b0_raw   = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B0,
    batch_size=BATCH_B0, label_mode='categorical')

class_names = train_b0_raw.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

alpha_b0, cw_b0, _ = get_class_info(train_b0_raw, NUM_CLASSES)
print("B0 focal alpha:", [round(a, 4) for a in alpha_b0])
print("B0 class weights:", {k: round(v, 3) for k, v in cw_b0.items()})

train_b0 = train_b0_raw.cache().prefetch(AUTOTUNE)
val_b0   = val_b0_raw.cache().prefetch(AUTOTUNE)

def build_b0(trainable=False, unfreeze_from=None):
    bb = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(*IMG_B0, 3))
    bb.trainable = trainable
    if trainable and unfreeze_from:
        for l in bb.layers[:unfreeze_from]:
            l.trainable = False
    inp = tf.keras.Input(shape=(*IMG_B0, 3))
    x   = layers.RandomFlip("horizontal_and_vertical")(inp)
    x   = layers.RandomRotation(0.15)(x)
    x   = layers.RandomZoom(0.15)(x)
    x   = layers.RandomContrast(0.1)(x)
    x   = layers.RandomBrightness(0.1)(x)
    x   = bb(x, training=trainable)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inp, out)

# B0 Phase 1
print("\n--- B0 Phase 1: frozen backbone ---")
m_b0 = build_b0(trainable=False)
m_b0.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha_b0, gamma=2.0),
    metrics=['accuracy'])

cbs_b0_p1, _ = make_callbacks(B0_PHASE1, B0_MODEL, es_patience=8, reduce_patience=3)
cbs_b0_p1.append(callbacks.ModelCheckpoint(B0_PHASE1, monitor='val_accuracy',
                                            save_best_only=True, verbose=1, mode='max'))
m_b0.fit(train_b0, validation_data=val_b0, epochs=15,
         class_weight=cw_b0, callbacks=cbs_b0_p1, verbose=1)
_, acc_b0_p1 = m_b0.evaluate(val_b0, verbose=0)
print(f"B0 Phase 1: {acc_b0_p1*100:.2f}%")

# B0 Phase 2
print("\n--- B0 Phase 2: unfreeze top layers ---")
m_b0_p2 = tf.keras.models.load_model(B0_PHASE1)
bb_p2   = next(l for l in m_b0_p2.layers if 'efficientnet' in l.name.lower())
bb_p2.trainable = True
for l in bb_p2.layers[:160]:
    l.trainable = False
print(f"Trainable backbone layers: {sum(1 for l in bb_p2.layers if l.trainable)}")

m_b0_p2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha_b0, gamma=2.0),
    metrics=['accuracy'])

_, cbs_b0_p2 = make_callbacks(B0_PHASE1, B0_MODEL, es_patience=8, reduce_patience=3)
m_b0_p2.fit(train_b0, validation_data=val_b0, epochs=40,
            class_weight=cw_b0, callbacks=cbs_b0_p2, verbose=1)

best_b0    = tf.keras.models.load_model(B0_MODEL)
_, acc_b0  = best_b0.evaluate(val_b0, verbose=0)
print(f"B0 Final: {acc_b0*100:.2f}%")

del m_b0, m_b0_p2   # free memory before B2


# ===========================================================================
# MODEL B — EfficientNetB2  (260×260)
# ===========================================================================
print("\n" + "="*60)
print("MODEL B: EfficientNetB2")
print("="*60)

IMG_B2   = (260, 260)
BATCH_B2 = 32

train_b2_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMG_B2,
    batch_size=BATCH_B2, label_mode='categorical')
val_b2_raw   = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH_B2, label_mode='categorical')

alpha_b2, cw_b2, _ = get_class_info(train_b2_raw, NUM_CLASSES)
eczema_idx          = class_names.index('eczema')
cw_b2[eczema_idx]  *= 1.2
print("B2 focal alpha:", [round(a, 4) for a in alpha_b2])
print("B2 class weights:", {k: round(v, 3) for k, v in cw_b2.items()})

train_b2 = train_b2_raw.cache().prefetch(AUTOTUNE)
val_b2   = val_b2_raw.cache().prefetch(AUTOTUNE)

def build_b2(trainable=False, unfreeze_from=None):
    bb = tf.keras.applications.EfficientNetB2(
        include_top=False, weights='imagenet', input_shape=(*IMG_B2, 3))
    bb.trainable = trainable
    if trainable and unfreeze_from:
        for l in bb.layers[:unfreeze_from]:
            l.trainable = False
    inp = tf.keras.Input(shape=(*IMG_B2, 3))
    x   = layers.RandomFlip("horizontal_and_vertical")(inp)
    x   = layers.RandomRotation(0.15)(x)
    x   = layers.RandomZoom(0.15)(x)
    x   = layers.RandomTranslation(0.1, 0.1)(x)
    x   = layers.RandomContrast(0.15)(x)
    x   = layers.RandomBrightness(0.15)(x)
    x   = bb(x, training=trainable)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.45)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inp, out)

# B2 Phase 1
print("\n--- B2 Phase 1: frozen backbone ---")
m_b2 = build_b2(trainable=False)
m_b2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha_b2, gamma=2.0),
    metrics=['accuracy'])

cbs_b2_p1, _ = make_callbacks(B2_PHASE1, B2_MODEL, es_patience=10, reduce_patience=4)
cbs_b2_p1.append(callbacks.ModelCheckpoint(B2_PHASE1, monitor='val_accuracy',
                                            save_best_only=True, verbose=1, mode='max'))
m_b2.fit(train_b2, validation_data=val_b2, epochs=20,
         class_weight=cw_b2, callbacks=cbs_b2_p1, verbose=1)
_, acc_b2_p1 = m_b2.evaluate(val_b2, verbose=0)
print(f"B2 Phase 1: {acc_b2_p1*100:.2f}%")

# B2 Phase 2
print("\n--- B2 Phase 2: unfreeze top layers ---")
m_b2_p2 = tf.keras.models.load_model(B2_PHASE1)
bb2_p2  = next(l for l in m_b2_p2.layers if 'efficientnet' in l.name.lower())
bb2_p2.trainable = True
for l in bb2_p2.layers[:220]:
    l.trainable = False
print(f"Trainable backbone layers: {sum(1 for l in bb2_p2.layers if l.trainable)}")

m_b2_p2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha_b2, gamma=2.0),
    metrics=['accuracy'])

_, cbs_b2_p2 = make_callbacks(B2_PHASE1, B2_MODEL, es_patience=15, reduce_patience=8)
m_b2_p2.fit(train_b2, validation_data=val_b2, epochs=60,
            class_weight=cw_b2, callbacks=cbs_b2_p2, verbose=1)

best_b2   = tf.keras.models.load_model(B2_MODEL)
_, acc_b2 = best_b2.evaluate(val_b2, verbose=0)
print(f"B2 Final: {acc_b2*100:.2f}%")

del m_b2, m_b2_p2   # free memory


# ===========================================================================
# ENSEMBLE + TTA
# ===========================================================================
print("\n" + "="*60)
print("ENSEMBLE + TTA")
print("="*60)

# Reload both best models
best_b0 = tf.keras.models.load_model(B0_MODEL)
best_b2 = tf.keras.models.load_model(B2_MODEL)

# Val datasets with cache so TTA steps are fast after first pass
val_b0_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B0,
    batch_size=BATCH_B0, label_mode='categorical').cache().prefetch(AUTOTUNE)
val_b2_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH_B2, label_mode='categorical').cache().prefetch(AUTOTUNE)

# Ground-truth labels
all_labels = np.concatenate([lbls.numpy() for _, lbls in val_b0_tta])
true_cls   = np.argmax(all_labels, axis=1)

# Standalone augmenters — applied OUTSIDE the model so BatchNorm stays in
# inference mode (training=False). Using training=True on the full model
# would switch BN to batch-statistics mode and destroy predictions.
aug_b0 = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
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
            aug_imgs = augmenter(imgs, training=True)   # augment only
            p = model(aug_imgs, training=False).numpy() # infer with fixed BN
            preds.append(p)
        preds = np.concatenate(preds)
        accumulated = preds if accumulated is None else accumulated + preds
        print(f"  step {step+1}/{n_steps} done", flush=True)
    return accumulated / n_steps

print(f"\nRunning TTA ({TTA_STEPS} passes) on B0...")
b0_probs = tta_predict(best_b0, val_b0_tta, aug_b0, TTA_STEPS)

print(f"\nRunning TTA ({TTA_STEPS} passes) on B2...")
b2_probs = tta_predict(best_b2, val_b2_tta, aug_b2, TTA_STEPS)

# --- Individual TTA results ---
b0_tta_acc = np.mean(np.argmax(b0_probs, axis=1) == true_cls)
b2_tta_acc = np.mean(np.argmax(b2_probs, axis=1) == true_cls)
print(f"\nB0 + TTA : {b0_tta_acc*100:.2f}%")
print(f"B2 + TTA : {b2_tta_acc*100:.2f}%")

# --- Ensemble: simple average ---
ensemble_probs = (b0_probs + b2_probs) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc   = np.mean(ensemble_preds == true_cls)
print(f"Ensemble (B0+B2) + TTA: {ensemble_acc*100:.2f}%")

# --- Ensemble: weighted average (favour the stronger model) ---
w_b0 = acc_b0 / (acc_b0 + acc_b2)
w_b2 = acc_b2 / (acc_b0 + acc_b2)
weighted_probs = w_b0 * b0_probs + w_b2 * b2_probs
weighted_preds = np.argmax(weighted_probs, axis=1)
weighted_acc   = np.mean(weighted_preds == true_cls)
print(f"Weighted ensemble (B0×{w_b0:.2f} + B2×{w_b2:.2f}) + TTA: {weighted_acc*100:.2f}%")

# --- Per-class breakdown for best ensemble ---
best_preds = weighted_preds if weighted_acc >= ensemble_acc else ensemble_preds
best_label = "Weighted" if weighted_acc >= ensemble_acc else "Simple"
best_acc   = max(weighted_acc, ensemble_acc)

print(f"\nBest ensemble ({best_label}): {best_acc*100:.2f}%")
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask  = true_cls == i
    acc_i = np.mean(best_preds[mask] == true_cls[mask]) * 100
    print(f"  {name}: {acc_i:.2f}%  ({mask.sum()} samples)")

print(f"\n{'='*40}")
print(f"B0 standalone       : {acc_b0*100:.2f}%")
print(f"B2 standalone       : {acc_b2*100:.2f}%")
print(f"B0 + TTA            : {b0_tta_acc*100:.2f}%")
print(f"B2 + TTA            : {b2_tta_acc*100:.2f}%")
print(f"Ensemble + TTA      : {ensemble_acc*100:.2f}%")
print(f"Wtd Ensemble + TTA  : {weighted_acc*100:.2f}%")
print(f"{'='*40}")

if best_acc >= 0.93:
    print("TARGET REACHED: 93%+!")
else:
    print(f"Gap to 93%: {(0.93 - best_acc)*100:.2f}%")
