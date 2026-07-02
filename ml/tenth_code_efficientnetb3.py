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

B2_MODEL_1   = "/kaggle/input/datasets/youssefyasser0123/skin-b2-model/ensemble_b2_model.keras"
B3_PHASE1    = "/kaggle/working/b3_phase1.keras"
B3_MODEL     = "/kaggle/working/b3_model.keras"

IMG_B3    = (300, 300)
IMG_B2    = (260, 260)
BATCH     = 32
TTA_STEPS = 20
AUTOTUNE  = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# 1. Data for B3
# ---------------------------------------------------------------------------
train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMG_B3,
    batch_size=BATCH, label_mode='categorical')
val_raw   = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B3,
    batch_size=BATCH, label_mode='categorical')

class_names = train_raw.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

y_train = []
for _, labels in train_raw:
    y_train.extend(np.argmax(labels.numpy(), axis=1))
y_train = np.array(y_train)

counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
alpha  = ((1.0 / counts) / (1.0 / counts).sum()).tolist()
print("Focal alpha:", [round(a, 4) for a in alpha])

cw = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
cw_dict[class_names.index('eczema')] *= 1.2
print("Class weights:", {k: round(v, 3) for k, v in cw_dict.items()})

train_ds = train_raw.cache().prefetch(AUTOTUNE)
val_ds   = val_raw.cache().prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# 2. Build B3
# ---------------------------------------------------------------------------
def build_b3(trainable=False, unfreeze_from=None):
    bb = tf.keras.applications.EfficientNetB3(
        include_top=False, weights='imagenet', input_shape=(*IMG_B3, 3))
    bb.trainable = trainable
    if trainable and unfreeze_from:
        for l in bb.layers[:unfreeze_from]:
            l.trainable = False
    inp = tf.keras.Input(shape=(*IMG_B3, 3))
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

# ---------------------------------------------------------------------------
# 3. Train B3
# ---------------------------------------------------------------------------
print("\n=== Training EfficientNetB3 ===")

# Phase 1 — frozen backbone
print("\n--- Phase 1: frozen backbone ---")
m = build_b3(trainable=False)
m.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=2.0),
    metrics=['accuracy'])

m.fit(train_ds, validation_data=val_ds, epochs=20,
      class_weight=cw_dict, verbose=1, callbacks=[
          callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                  restore_best_weights=True, verbose=1, mode='max'),
          callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                                      patience=4, min_lr=1e-8, verbose=1, mode='max'),
          callbacks.ModelCheckpoint(B3_PHASE1, monitor='val_accuracy',
                                    save_best_only=True, verbose=1, mode='max'),
      ])
_, acc_p1 = m.evaluate(val_ds, verbose=0)
print(f"Phase 1: {acc_p1*100:.2f}%")

# Phase 2 — unfreeze top layers
# B3 has ~385 layers; unfreeze top ~150 (freeze first 235)
print("\n--- Phase 2: unfreeze top 150 layers ---")
m_p2 = tf.keras.models.load_model(B3_PHASE1)
bb_p2 = next(l for l in m_p2.layers if 'efficientnet' in l.name.lower())
bb_p2.trainable = True
for l in bb_p2.layers[:235]:
    l.trainable = False
print(f"Trainable backbone layers: {sum(1 for l in bb_p2.layers if l.trainable)}")

m_p2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=2.0),
    metrics=['accuracy'])

m_p2.fit(train_ds, validation_data=val_ds, epochs=60,
         class_weight=cw_dict, verbose=1, callbacks=[
             callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                     restore_best_weights=True, verbose=1, mode='max'),
             callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                                         patience=8, min_lr=1e-8, verbose=1, mode='max'),
             callbacks.ModelCheckpoint(B3_MODEL, monitor='val_accuracy',
                                       save_best_only=True, verbose=1, mode='max'),
         ])

b3     = tf.keras.models.load_model(B3_MODEL)
_, acc_b3 = b3.evaluate(val_ds, verbose=0)
print(f"B3 Final: {acc_b3*100:.2f}%")
del m, m_p2

# ---------------------------------------------------------------------------
# 4. Load B2 #1 (260x260) — needs its own val dataset
# ---------------------------------------------------------------------------
print("\nLoading B2 #1...")
b2_v1 = tf.keras.models.load_model(B2_MODEL_1)

val_b2 = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH, label_mode='categorical').cache().prefetch(AUTOTUNE)

_, acc_b2 = b2_v1.evaluate(val_b2, verbose=0)
print(f"B2 #1 accuracy: {acc_b2*100:.2f}%")

# ---------------------------------------------------------------------------
# 5. TTA
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

val_b3_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B3,
    batch_size=BATCH, label_mode='categorical').cache().prefetch(AUTOTUNE)
val_b2_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH, label_mode='categorical').cache().prefetch(AUTOTUNE)

all_labels = np.concatenate([lbls.numpy() for _, lbls in val_b3_tta])
true_cls   = np.argmax(all_labels, axis=1)

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
        print(f"  step {step+1}/{n_steps} done", flush=True)
    return accumulated / n_steps

print(f"\nTTA ({TTA_STEPS} steps) on B3...")
probs_b3 = tta_predict(b3, val_b3_tta, aug_b3, TTA_STEPS)

print(f"\nTTA ({TTA_STEPS} steps) on B2 #1...")
probs_b2 = tta_predict(b2_v1, val_b2_tta, aug_b2, TTA_STEPS)

# ---------------------------------------------------------------------------
# 6. Results
# ---------------------------------------------------------------------------
acc_b3_tta = np.mean(np.argmax(probs_b3, axis=1) == true_cls)
acc_b2_tta = np.mean(np.argmax(probs_b2, axis=1) == true_cls)

# Simple ensemble
ensemble_probs = (probs_b3 + probs_b2) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc   = np.mean(ensemble_preds == true_cls)

# Weighted by standalone accuracy
w_b3 = acc_b3 / (acc_b3 + acc_b2)
w_b2 = acc_b2 / (acc_b3 + acc_b2)
wtd_probs = w_b3 * probs_b3 + w_b2 * probs_b2
wtd_preds = np.argmax(wtd_probs, axis=1)
wtd_acc   = np.mean(wtd_preds == true_cls)

best_preds = wtd_preds if wtd_acc >= ensemble_acc else ensemble_preds
best_acc   = max(wtd_acc, ensemble_acc)
best_label = "Weighted" if wtd_acc >= ensemble_acc else "Simple"

print(f"\n{'='*45}")
print(f"B3 standalone       : {acc_b3*100:.2f}%")
print(f"B2 #1 standalone    : {acc_b2*100:.2f}%")
print(f"B3 + TTA            : {acc_b3_tta*100:.2f}%")
print(f"B2 #1 + TTA         : {acc_b2_tta*100:.2f}%")
print(f"Ensemble + TTA      : {ensemble_acc*100:.2f}%")
print(f"Wtd Ensemble + TTA  : {wtd_acc*100:.2f}%")
print(f"{'='*45}")
print(f"\nBest ({best_label}): {best_acc*100:.2f}%")
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask  = true_cls == i
    acc_i = np.mean(best_preds[mask] == true_cls[mask]) * 100
    print(f"  {name}: {acc_i:.2f}%  ({mask.sum()} samples)")

if best_acc >= 0.93:
    print("\nTARGET REACHED: 93%+!")
else:
    print(f"\nGap to 93%: {(0.93 - best_acc)*100:.2f}%")
