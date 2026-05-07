import tensorflow as tf
from tensorflow.keras import callbacks, layers
import numpy as np
from sklearn.utils import class_weight
import os

# Seed for model #2 (model #1 was trained with seed=42)
tf.random.set_seed(123)
np.random.seed(123)

# --- Kaggle paths ---
BASE_DIR  = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/kaggle/New_Augmented_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

# Path to the uploaded B2 model #1 (from your dataset)
# Update <your-username> and <dataset-name> to match what you named it on Kaggle
B2_MODEL_1   = "/kaggle/input/datasets/youssefyasser0123/skin-b2-model/ensemble_b2_model.keras"
B2_PHASE1_2  = "/kaggle/working/b2_v2_phase1.keras"
B2_MODEL_2   = "/kaggle/working/b2_v2_model.keras"

IMG_B2    = (260, 260)
BATCH_B2  = 32
TTA_STEPS = 20
AUTOTUNE  = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=123, image_size=IMG_B2,
    batch_size=BATCH_B2, label_mode='categorical')
val_raw   = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH_B2, label_mode='categorical')

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
# 2. Build B2
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Train B2 #2 (seed=123)
# ---------------------------------------------------------------------------
print("\n=== Training B2 #2 (seed=123) ===")

# Phase 1
print("\n--- Phase 1: frozen backbone ---")
m = build_b2(trainable=False)
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
          callbacks.ModelCheckpoint(B2_PHASE1_2, monitor='val_accuracy',
                                    save_best_only=True, verbose=1, mode='max'),
      ])
_, acc_p1 = m.evaluate(val_ds, verbose=0)
print(f"Phase 1: {acc_p1*100:.2f}%")

# Phase 2
print("\n--- Phase 2: unfreeze top 120 layers ---")
m_p2    = tf.keras.models.load_model(B2_PHASE1_2)
bb_p2   = next(l for l in m_p2.layers if 'efficientnet' in l.name.lower())
bb_p2.trainable = True
for l in bb_p2.layers[:220]:
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
             callbacks.ModelCheckpoint(B2_MODEL_2, monitor='val_accuracy',
                                       save_best_only=True, verbose=1, mode='max'),
         ])

b2_v2     = tf.keras.models.load_model(B2_MODEL_2)
_, acc_v2 = b2_v2.evaluate(val_ds, verbose=0)
print(f"B2 #2 Final: {acc_v2*100:.2f}%")
del m, m_p2

# ---------------------------------------------------------------------------
# 4. Load B2 #1
# ---------------------------------------------------------------------------
print("\nLoading B2 #1...")
b2_v1     = tf.keras.models.load_model(B2_MODEL_1)
_, acc_v1 = b2_v1.evaluate(val_ds, verbose=0)
print(f"B2 #1 accuracy: {acc_v1*100:.2f}%")

# ---------------------------------------------------------------------------
# 5. TTA on both models then ensemble
# ---------------------------------------------------------------------------
aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])

val_tta = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_B2,
    batch_size=BATCH_B2, label_mode='categorical').cache().prefetch(AUTOTUNE)

all_labels = np.concatenate([lbls.numpy() for _, lbls in val_tta])
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

print(f"\nTTA ({TTA_STEPS} steps) on B2 #1...")
probs_v1 = tta_predict(b2_v1, val_tta, aug, TTA_STEPS)

print(f"\nTTA ({TTA_STEPS} steps) on B2 #2...")
probs_v2 = tta_predict(b2_v2, val_tta, aug, TTA_STEPS)

# ---------------------------------------------------------------------------
# 6. Results
# ---------------------------------------------------------------------------
acc_v1_tta = np.mean(np.argmax(probs_v1, axis=1) == true_cls)
acc_v2_tta = np.mean(np.argmax(probs_v2, axis=1) == true_cls)

ensemble_probs = (probs_v1 + probs_v2) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc   = np.mean(ensemble_preds == true_cls)

# Weighted by standalone accuracy
w1 = acc_v1 / (acc_v1 + acc_v2)
w2 = acc_v2 / (acc_v1 + acc_v2)
wtd_probs = w1 * probs_v1 + w2 * probs_v2
wtd_preds = np.argmax(wtd_probs, axis=1)
wtd_acc   = np.mean(wtd_preds == true_cls)

best_preds = wtd_preds if wtd_acc >= ensemble_acc else ensemble_preds
best_acc   = max(wtd_acc, ensemble_acc)
best_label = "Weighted" if wtd_acc >= ensemble_acc else "Simple"

print(f"\n{'='*45}")
print(f"B2 #1 standalone    : {acc_v1*100:.2f}%")
print(f"B2 #2 standalone    : {acc_v2*100:.2f}%")
print(f"B2 #1 + TTA         : {acc_v1_tta*100:.2f}%")
print(f"B2 #2 + TTA         : {acc_v2_tta*100:.2f}%")
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
