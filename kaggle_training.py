import tensorflow as tf
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.utils import class_weight
import os

tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/New_Augmented_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
PHASE2_MODEL = "/kaggle/input/datasets/youssefyasser0123/skin-disease-dataset/kaggle/mobilenetv2_finetuned_model.keras"
OUTPUT_MODEL = "/kaggle/working/fourth_code_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
UNFREEZE_FROM_LAYER = 70
LEARNING_RATE = 3e-6
LABEL_SMOOTHING = 0.1
TINEA_WEIGHT_BOOST = 1.5
EPOCHS = 50
ES_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='categorical')

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, label_mode='categorical')

class_names = train_ds_raw.class_names
print("Classes:", class_names)

y_train = []
for _, labels in train_ds_raw:
    y_train.extend(np.argmax(labels.numpy(), axis=1))
y_train = np.array(y_train)

base_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(base_weights))
tinea_idx = class_names.index('tinea')
class_weight_dict[tinea_idx] *= TINEA_WEIGHT_BOOST
print("Class weights:", class_weight_dict)

AUTOTUNE = tf.data.AUTOTUNE
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
])

train_ds = (train_ds_raw
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    .cache().prefetch(AUTOTUNE))

val_ds = (val_ds_raw
    .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    .cache().prefetch(AUTOTUNE))

# Keras 2 -> Keras 3 compatibility fix:
# Model was saved with Keras 2 which stores 'quantization_config' in Dense config.
# Keras 3 does not recognize this key, so we monkey-patch Dense.from_config
# to silently drop it before Keras 3 processes the layer.
_original_dense_from_config = tf.keras.layers.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(cls, config)

tf.keras.layers.Dense.from_config = _patched_dense_from_config

model = tf.keras.models.load_model(PHASE2_MODEL)
print("Phase 2 model loaded")

base_model = next(l for l in model.layers if 'mobilenetv2' in l.name.lower())
base_model.trainable = True
for i, layer in enumerate(base_model.layers):
    layer.trainable = i >= UNFREEZE_FROM_LAYER

trainable = sum(1 for l in base_model.layers if l.trainable)
print("Trainable base layers:", trainable)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy'])

loss0, acc0 = model.evaluate(val_ds, verbose=0)
print("Phase 2 baseline:", round(acc0*100, 2), "%")

cbs = [
    callbacks.EarlyStopping(monitor='val_loss', patience=ES_PATIENCE,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                                patience=REDUCE_LR_PATIENCE, min_lr=1e-8, verbose=1),
    callbacks.ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss',
                              save_best_only=True, verbose=1),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    class_weight=class_weight_dict, callbacks=cbs, verbose=1)

best = tf.keras.models.load_model(OUTPUT_MODEL)
loss, acc = best.evaluate(val_ds, verbose=0)
print("Phase 2 baseline :", round(acc0*100, 2), "%")
print("Phase 4 result   :", round(acc*100, 2), "%")
print("Improvement      :", round((acc-acc0)*100, 2), "%")

all_preds, all_labels = [], []
for imgs, lbls in val_ds:
    all_preds.append(best(imgs, training=False).numpy())
    all_labels.append(lbls.numpy())
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
pred_cls = np.argmax(all_preds, axis=1)
true_cls = np.argmax(all_labels, axis=1)

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask = true_cls == i
    print(name, ":", round(np.mean(pred_cls[mask]==true_cls[mask])*100, 2), "%", "(", mask.sum(), "samples)")

if acc >= 0.95:
    print("GOAL ACHIEVED: 95%+ accuracy!")
