"""
Binary Normal-vs-Disease gate.

Replaces the VAE anomaly gate. The VAE could not separate normal skin from
acne/eczema/tinea (reconstructing skin well also reconstructs the diseases —
the error distributions overlapped and even inverted on-device). A supervised
classifier learns the boundary directly and does not have that failure mode.

Output: P(disease) in [0,1]. The Flutter app runs this first:
    P(disease) >  gate_threshold  -> run the existing 3-class CNN (which disease)
    P(disease) <= gate_threshold  -> "No Disease Detected"

Run on Kaggle GPU (Internet not required — imagenet weights ship with TF).
Retrain v2 (after the on-device false positives on faces / casual photos):
attach gate-kit-zip + lysaapriani + the normal-FACE dataset(s), then roughly:

    !python /kaggle/input/gate-kit-zip/gate_kit/train_normal_gate.py \
        --normal-dirs /kaggle/input/gate-kit-zip/gate_kit/normal_hands \
                      "/kaggle/input/skin-disease-and-normal-skin-dataset/.../normal" \
                      "/kaggle/input/facial-skin-condition-dataset/.../normal-faces" \
        --disease-dirs /kaggle/input/gate-kit-zip/gate_kit/New_Augmented_Dataset \
                       "/kaggle/input/skin-disease-and-normal-skin-dataset/.../disease" \
        --test-dirs /kaggle/input/gate-kit-zip/gate_kit/normal_phone \
        --out /kaggle/working/normal_gate

(normal_phone moved from --normal-dirs to --test-dirs: the user's own photos
are now a pure held-out reliability check, never trained on.)

Disease and normal images are each collected recursively from --disease-dirs /
--normal-dirs (any folder layout), pooled, and split 85/15 train/val. The
disease folders can mix sources (your acne/eczema/tinea + a dermatitis set);
including normal AND disease from the same source stops the model cheating on
"which dataset is this".

v2 changes vs the model that false-flagged on device:
  - skin_bbox_crop applied at decode time to EVERY image (training, val and
    test) — backgrounds (floor tiles, rugs) stop being a class cue. The SAME
    crop runs on-device before the gate (ai_service.dart _cropToSkinRegion).
  - phone_augment: saturation/hue jitter, blur, JPEG artifacts (train only).
  - Brightness/contrast jitter raised 0.2 -> 0.3.
"""
import argparse
import os
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

IMG = 224  # EfficientNetB0 native resolution; small + fast on a phone
AUTOTUNE = tf.data.AUTOTUNE
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(roots):
    files = []
    for root in roots:
        for dp, _, fns in os.walk(root):
            # Skip the reject/prune subfolders left by the patch extractor.
            low = dp.lower()
            if "_rejected" in low or "_pruned" in low:
                continue
            for fn in fns:
                if fn.lower().endswith(EXTS):
                    files.append(os.path.join(dp, fn))
    return files


def skin_bbox_crop(arr):
    """Crop a HxWx3 uint8 image to the bounding box of skin-colored pixels.

    Same per-pixel rule as the Flutter app (_patchIsSkin / _cropToSkinRegion in
    ai_service.dart) so training and on-device inference see the same framing:
        v>30 and v<248 and r>=b-5 and not (sat<0.05 and v>153)
    Falls back to the full frame when the mask is unreliable (low coverage) or
    the box is degenerate. 2nd/98th-percentile bounds ignore stray pixels.
    """
    h, w = arr.shape[:2]
    s = max(1, min(h, w) // 128)  # stride-sample large images for speed
    a = arr[::s, ::s].astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    v = (r + g + b) / 3.0
    mx = a.max(-1)
    mn = a.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    mask = ((v > 30) & (v < 248) & (r >= b - 5)
            & ~((sat < 0.05) & (v > 153))
            & (g > 0.45 * r)    # reject deep-red fabric
            & (r >= g - 5))     # reject green-dominant (walls/plants)
    if mask.mean() < 0.10:
        return arr  # almost no skin found — mask unreliable, keep full frame
    ys, xs = np.nonzero(mask)
    y0, y1 = np.percentile(ys, [2, 98]).astype(int)
    x0, x1 = np.percentile(xs, [2, 98]).astype(int)
    # back to full-res coords with a 5% margin
    my, mx_ = int(0.05 * h), int(0.05 * w)
    y0 = max(0, y0 * s - my); y1 = min(h, (y1 + 1) * s + my)
    x0 = max(0, x0 * s - mx_); x1 = min(w, (x1 + 1) * s + mx_)
    if (y1 - y0) * (x1 - x0) < 0.05 * h * w:
        return arr  # degenerate box
    return arr[y0:y1, x0:x1]


def decode(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.numpy_function(skin_bbox_crop, [img], tf.uint8)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG, IMG])
    img = tf.cast(img, tf.float32)  # [0,255] — EfficientNet rescales internally
    return img, label


def phone_augment(img, label):
    """Train-only degradations mimicking casual phone photos: color cast,
    motion/defocus blur (cheap down-up resize), and JPEG compression. Geometry
    and brightness/contrast jitter live in the model's Keras layers."""
    x = img / 255.0
    x = tf.image.random_saturation(x, 0.6, 1.4)
    x = tf.image.random_hue(x, 0.04)
    if tf.random.uniform([]) < 0.3:  # blur
        f = tf.random.uniform([], 0.35, 0.8)
        s = tf.cast(f * IMG, tf.int32)
        x = tf.image.resize(tf.image.resize(x, [s, s]), [IMG, IMG])
    if tf.random.uniform([]) < 0.4:  # compression artifacts
        x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0.0, 1.0), 25, 80)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0, label


def make_ds(paths, labels, training, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(phone_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def build_model(trainable, unfreeze_from=None):
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
    if not trainable:
        backbone.trainable = False
    else:
        backbone.trainable = True
        if unfreeze_from is not None:
            for layer in backbone.layers[:unfreeze_from]:
                layer.trainable = False

    inp = tf.keras.Input(shape=(IMG, IMG, 3))
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inp)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.2)(x)
    x = tf.keras.layers.RandomTranslation(0.15, 0.15)(x)
    x = tf.keras.layers.RandomContrast(0.3)(x)
    x = tf.keras.layers.RandomBrightness(0.3)(x)
    x = backbone(x, training=trainable)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # P(disease)
    return tf.keras.Model(inp, out)


def evaluate(model, val_ds, val_src):
    """Print a threshold sweep so we can choose a high-disease-recall point."""
    probs, labels = [], []
    for xb, yb in val_ds:
        probs.append(model(xb, training=False).numpy().ravel())
        labels.append(yb.numpy().ravel())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)  # 1 = disease, 0 = normal
    dis = labels == 1
    nor = labels == 0
    print("\nthreshold |  disease-recall (caught)  |  normal-pass (specificity)")
    print("----------+---------------------------+---------------------------")
    for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred_dis = probs > t
        dr = np.mean(pred_dis[dis]) * 100 if dis.any() else 0.0
        npass = np.mean(~pred_dis[nor]) * 100 if nor.any() else 0.0
        print(f"   {t:.2f}   |          {dr:5.1f}%           |          {npass:5.1f}%")
    # Per-source normal pass at t=0.5 — reveals body-part shortcut (e.g. hands
    # pass but phone/face do not).
    if val_src is not None:
        print("\nNormal-pass by source @ t=0.50 (watch for a weak source):")
        for name in sorted(set(val_src[nor])):
            m = nor & (val_src == name)
            if m.any():
                npass = np.mean(probs[m] <= 0.50) * 100
                print(f"  {name:<24} {npass:5.1f}%  ({m.sum()})")
    return probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-dirs", nargs="+", required=True,
                    help="folders of NORMAL skin (collected recursively)")
    ap.add_argument("--disease-dirs", nargs="+", required=True,
                    help="folders of DISEASE skin (collected recursively); any "
                         "layout — nested class folders or flat are both fine")
    ap.add_argument("--test-dirs", nargs="*", default=[],
                    help="held-out NORMAL images (e.g. your phone photos) NEVER used "
                         "in training — reports real-world pass-rate to gauge reliability")
    ap.add_argument("--out", default="/kaggle/working/normal_gate")
    ap.add_argument("--p1-epochs", type=int, default=15)
    ap.add_argument("--p2-epochs", type=int, default=40)
    ap.add_argument("--unfreeze-from", type=int, default=150)  # B0 ~237 layers
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- gather paths ---
    dis_all = list_images(args.disease_dirs)
    nor_all = list_images(args.normal_dirs)

    # tag each normal image with its top-level source dir for the per-source report
    def src_of(p):
        ap_ = os.path.abspath(p)
        for d in args.normal_dirs:
            if ap_.startswith(os.path.abspath(d)):
                return os.path.basename(os.path.normpath(d))
        return "normal"

    rng = np.random.default_rng(42)
    rng.shuffle(nor_all)
    rng.shuffle(dis_all)
    n_val = max(1, int(0.15 * len(nor_all)))
    d_val = max(1, int(0.15 * len(dis_all)))
    nor_val, nor_train = nor_all[:n_val], nor_all[n_val:]
    dis_val, dis_train = dis_all[:d_val], dis_all[d_val:]

    print(f"disease: {len(dis_train)} train / {len(dis_val)} val")
    print(f"normal : {len(nor_train)} train / {len(nor_val)} val")

    train_paths = dis_train + nor_train
    train_lbls = [1.0] * len(dis_train) + [0.0] * len(nor_train)
    val_paths = dis_val + nor_val
    val_lbls = [1.0] * len(dis_val) + [0.0] * len(nor_val)
    val_src = np.array([""] * len(dis_val) + [src_of(p) for p in nor_val])

    train_ds = make_ds(np.array(train_paths), np.array(train_lbls, np.float32), True)
    val_ds = make_ds(np.array(val_paths), np.array(val_lbls, np.float32), False)

    # --- class weights (normal outnumbers disease ~4:1) ---
    n_dis, n_nor = len(dis_train), len(nor_train)
    tot = n_dis + n_nor
    cw = {0: tot / (2.0 * n_nor), 1: tot / (2.0 * n_dis)}
    print("class weights {normal:0, disease:1}:", {k: round(v, 3) for k, v in cw.items()})

    p1_model = os.path.join(args.out, "gate_phase1.keras")
    out_model = os.path.join(args.out, "normal_gate.keras")
    cbs = lambda path: [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=5, min_lr=1e-8, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss",
                                           save_best_only=True, verbose=1),
    ]

    print("\n=== Phase 1: frozen backbone ===")
    model = build_model(trainable=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.p1_epochs,
              class_weight=cw, callbacks=cbs(p1_model), verbose=1)

    print(f"\n=== Phase 2: unfreeze from {args.unfreeze_from} ===")
    model = tf.keras.models.load_model(p1_model)
    bb = next(l for l in model.layers if "efficientnet" in l.name.lower())
    bb.trainable = True
    for layer in bb.layers[:args.unfreeze_from]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.p2_epochs,
              class_weight=cw, callbacks=cbs(out_model), verbose=1)

    best = tf.keras.models.load_model(out_model)
    evaluate(best, val_ds, val_src)

    # Held-out reliability check: normal images from a source the model never saw
    # in training (your phone photos). High pass-rate here = genuine
    # generalization; low pass-rate = the model leaned on a training shortcut.
    test_files = list_images(args.test_dirs) if args.test_dirs else []
    if test_files:
        test_ds = make_ds(np.array(test_files),
                          np.zeros(len(test_files), np.float32), False)
        probs = np.concatenate(
            [best(xb, training=False).numpy().ravel() for xb, _ in test_ds])
        print(f"\n=== HELD-OUT normal test ({len(test_files)} images, never trained) ===")
        for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
            print(f"  threshold {t:.2f}: normal-pass = {np.mean(probs <= t)*100:5.1f}%")

    # --- export tflite (input: [0,255] float32, IMGxIMG, output: P(disease)) ---
    conv = tf.lite.TFLiteConverter.from_keras_model(best)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    tflite_path = os.path.join(args.out, "normal_gate.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tfl)
    print(f"\nSaved {tflite_path} ({len(tfl)//1024} KB)")
    print("Pick a threshold from the sweep (favor disease-recall), then copy "
          "normal_gate.tflite into Flutter/assets/models/.")


if __name__ == "__main__":
    main()
