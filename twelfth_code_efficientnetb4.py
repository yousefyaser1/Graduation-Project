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

B2_MODEL  = "/kaggle/input/datasets/youssefyasser0123/skin-b2-model/ensemble_b2_model.keras"
B4_PHASE1 = "/kaggle/working/b4_phase1.keras"
B4_MODEL  = "/kaggle/working/b4_model.keras"

IMG_B4   = (380, 380)
IMG_B2   = (260, 260)
BATCH    = 16          # smaller batch — B4 at 380px is memory-heavy
TTA_STEPS = 20
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMG_B4,
    batch_size=BATCH, label_mode='categorical')
val_raw   = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B4,
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
# 2. Build B4
# ---------------------------------------------------------------------------
def build_b4(trainable=False):
    bb = tf.keras.applications.EfficientNetB4(
        include_top=False, weights='imagenet', input_shape=(*IMG_B4, 3))
    bb.trainable = trainable
    inp = tf.keras.Input(shape=(*IMG_B4, 3))
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
# 3. Train B4
# ---------------------------------------------------------------------------
print("\n=== Training EfficientNetB4 ===")

# Phase 1 — frozen backbone
print("\n--- Phase 1: frozen backbone ---")
m = build_b4(trainable=False)
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
          callbacks.ModelCheckpoint(B4_PHASE1, monitor='val_accuracy',
                                    save_best_only=True, verbose=1, mode='max'),
      ])
_, acc_p1 = m.evaluate(val_ds, verbose=0)
print(f"Phase 1: {acc_p1*100:.2f}%")

# Phase 2 — unfreeze top layers
# B4 has ~475 layers; unfreeze top ~200 (freeze first 275)
print("\n--- Phase 2: unfreeze top 200 layers ---")
m_p2 = tf.keras.models.load_model(B4_PHASE1)
bb_p2 = next(l for l in m_p2.layers if 'efficientnet' in l.name.lower())
bb_p2.trainable = True
for l in bb_p2.layers[:275]:
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
             callbacks.ModelCheckpoint(B4_MODEL, monitor='val_accuracy',
                                       save_best_only=True, verbose=1, mode='max'),
         ])

b4     = tf.keras.models.load_model(B4_MODEL)
_, acc_b4 = b4.evaluate(val_ds, verbose=0)
print(f"B4 Final: {acc_b4*100:.2f}%")
del m, m_p2

# ---------------------------------------------------------------------------
# 4. Load B2 (260x260)
# ---------------------------------------------------------------------------
print("\nLoading B2...")
b2 = tf.keras.models.load_model(B2_MODEL)

val_b2 = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH, label_mode='categorical').cache().prefetch(AUTOTUNE)

_, acc_b2 = b2.evaluate(val_b2, verbose=0)
print(f"B2 accuracy: {acc_b2*100:.2f}%")

# ---------------------------------------------------------------------------
# 5. TTA
# ---------------------------------------------------------------------------
aug_b4 = tf.keras.Sequential([
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

val_b4_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B4,
    batch_size=BATCH, label_mode='categorical').cache().prefetch(AUTOTUNE)
val_b2_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH, label_mode='categorical').cache().prefetch(AUTOTUNE)

all_labels = np.concatenate([lbls.numpy() for _, lbls in val_b4_tta])
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

print(f"\nTTA ({TTA_STEPS} steps) on B4...")
probs_b4 = tta_predict(b4, val_b4_tta, aug_b4, TTA_STEPS)

print(f"\nTTA ({TTA_STEPS} steps) on B2...")
probs_b2 = tta_predict(b2, val_b2_tta, aug_b2, TTA_STEPS)

# ---------------------------------------------------------------------------
# 6. Results — sweep ensemble weights
# ---------------------------------------------------------------------------
acc_b4_tta = np.mean(np.argmax(probs_b4, axis=1) == true_cls)
acc_b2_tta = np.mean(np.argmax(probs_b2, axis=1) == true_cls)

weights = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
           (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.0, 1.0)]

best_acc   = 0.0
best_preds = None
best_label = ""

print(f"\n{'='*45}")
print(f"B4 standalone       : {acc_b4*100:.2f}%")
print(f"B2 standalone       : {acc_b2*100:.2f}%")
print(f"B4 + TTA            : {acc_b4_tta*100:.2f}%")
print(f"B2 + TTA            : {acc_b2_tta*100:.2f}%")
print(f"\nEnsemble sweep:")
for w_b4, w_b2 in weights:
    if w_b2 == 0.0:
        probs = probs_b4
    elif w_b4 == 0.0:
        probs = probs_b2
    else:
        probs = w_b4 * probs_b4 + w_b2 * probs_b2
    preds = np.argmax(probs, axis=1)
    acc   = np.mean(preds == true_cls)
    flag  = " <-- BEST" if acc > best_acc else ""
    if acc > best_acc:
        best_acc   = acc
        best_preds = preds
        best_label = f"B4:{w_b4} B2:{w_b2}"
    print(f"  [B4:{w_b4} B2:{w_b2}]: {acc*100:.2f}%{flag}")

print(f"\nBest config: {best_label} → {best_acc*100:.2f}%")
print(f"\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask  = true_cls == i
    acc_i = np.mean(best_preds[mask] == true_cls[mask]) * 100
    print(f"  {name}: {acc_i:.2f}%  ({mask.sum()} samples)")

if best_acc >= 0.93:
    print("\nTARGET REACHED: 93%+!")
else:
    print(f"\nGap to 93%: {(0.93 - best_acc)*100:.2f}%")
