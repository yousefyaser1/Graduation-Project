"""
INT8 post-training quantization for SkinScan AI TFLite models.

What this does
--------------
Takes the float32 .keras models (B2 + B3 EfficientNet) and produces INT8-
quantized .tflite files.  Typical results:
  - Model size: ~4× smaller (float32 → int8 weights)
  - Inference speed: ~1.5–2× faster on ARM CPUs with NEON/SIMD support
  - Accuracy drop: usually < 1% with a representative calibration dataset

NOTE: EfficientNet was trained with an internal Rescaling(1/255) layer baked
in, so the representative dataset must feed raw [0, 255] uint8 pixels (same
as the Flutter app) — NOT normalized [0, 1] values.

Usage
-----
1.  Place ensemble_b2_model.keras and b3_model.keras in the same directory
    as this script (or update the paths below).
2.  Set DATASET_DIR to a folder containing validation skin images.
3.  Run:  python quantize_tflite_int8.py
4.  Copy the output .tflite files to Flutter/assets/models/ replacing the
    float32 versions.  No Flutter code changes needed.
5.  Verify accuracy with:  python verify_tflite_accuracy.py  (existing script)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
# Adjust if your .keras files live elsewhere
MODEL_B2   = r'C:\Users\A\Downloads\ensemble_b2_model.keras'
MODEL_B3   = r'C:\Users\A\Downloads\b3_model.keras'
DATASET_DIR = 'New_Augmented_Dataset/val'   # validation split for calibration
NUM_CALIB_IMAGES = 200                       # more = more accurate calibration

# Output paths (will overwrite existing float32 .tflite files when ready)
OUT_B2 = 'Flutter/assets/models/cnn_b2_model_int8.tflite'
OUT_B3 = 'Flutter/assets/models/cnn_b3_model_int8.tflite'

# ── Helper ─────────────────────────────────────────────────────────────────────

def collect_images(root, limit):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(dirpath, f))
    np.random.shuffle(paths)
    return paths[:limit]


def make_representative_dataset(image_paths, input_hw):
    """Generator that yields one calibration sample at a time."""
    h, w = input_hw
    def gen():
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB').resize((w, h), Image.BILINEAR)
                data = np.array(img, dtype=np.float32)[np.newaxis, ...]
                # Raw [0, 255] — EfficientNet's internal Rescaling layer handles /255
                yield [data]
            except Exception as exc:
                print(f'  skip {path}: {exc}')
    return gen


def quantize(model_path, output_path, input_hw, calib_images):
    print(f'\n{"─" * 60}')
    print(f'Model  : {model_path}')
    print(f'Input  : {input_hw[0]}×{input_hw[1]}')
    print(f'Calibration images: {len(calib_images)}')

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Full integer ops — weights AND activations are INT8; I/O stays float32
    # for drop-in compatibility with the Flutter app (no tflite code changes).
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback for unsupported ops
    ]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = make_representative_dataset(
        calib_images, input_hw)

    print('Converting…')
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    orig_mb = os.path.getsize(model_path) / 1024 / 1024
    quant_mb = len(tflite_model) / 1024 / 1024
    reduction = (1 - quant_mb / orig_mb) * 100
    print(f'Original : {orig_mb:.1f} MB')
    print(f'Quantized: {quant_mb:.1f} MB  ({reduction:.0f}% smaller)')
    print(f'Saved to : {output_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for path in [MODEL_B2, MODEL_B3]:
        if not os.path.exists(path):
            print(f'ERROR: model not found: {path}')
            sys.exit(1)

    if not os.path.isdir(DATASET_DIR):
        print(f'ERROR: dataset directory not found: {DATASET_DIR}')
        sys.exit(1)

    calib_images = collect_images(DATASET_DIR, NUM_CALIB_IMAGES)
    if len(calib_images) < 50:
        print(f'WARNING: only {len(calib_images)} calibration images found — more is better.')

    # B2 — 260×260 input
    quantize(MODEL_B2, OUT_B2, (260, 260), calib_images)

    # B3 — 300×300 input
    quantize(MODEL_B3, OUT_B3, (300, 300), calib_images)

    print('\n' + '=' * 60)
    print('DONE.  Next steps:')
    print('  1. Verify accuracy:')
    print('       python verify_tflite_accuracy.py')
    print('  2. If accuracy is acceptable (< 1% drop), copy to assets:')
    print(f'       copy {OUT_B2} Flutter/assets/models/cnn_b2_model.tflite')
    print(f'       copy {OUT_B3} Flutter/assets/models/cnn_b3_model.tflite')
    print('  3. Rebuild the Flutter APK.')
    print('  No Flutter code changes required — TFLite runtime handles INT8.')
