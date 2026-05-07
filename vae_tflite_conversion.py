import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
# Extracted PyTorch zip folder  (from vae_skin_model.pth.zip)
PTH_FOLDER  = r"C:\Users\A\Downloads\vae_skin_model"
TFLITE_OUT  = r"C:\Users\A\Graduation-Project\vae_model.tflite"
TEST_IMAGE_DISEASE = r"C:\Users\A\Graduation-Project\.kilo\worktrees\freckle-september\Dermatology_CNN_Dataset\test\acne\07Rhinophyma1.jpg"
TEST_IMAGE_NORMAL  = r"C:\Users\A\Downloads\03ChronicStaisis1.jpg"

PATCH_SIZE       = 64
STRIDE           = 32
ANOMALY_THRESHOLD = 0.008   # 0.006 (PyTorch) + ~0.002 systematic TFLite offset
ANOMALY_RATIO    = 0.20


# -----------------------------------------------------------------------
# 1.  LOAD PYTORCH WEIGHTS  (raw bytes – no PyTorch required)
#
#  The .pth zip format stores each tensor as a raw float32 binary file
#  at  <folder>/data/<index>.  The index order matches parameter order:
#
#   0  encoder.0.weight  (32,  3, 4, 4)
#   1  encoder.0.bias    (32,)
#   2  encoder.2.weight  (64, 32, 4, 4)
#   3  encoder.2.bias    (64,)
#   4  encoder.4.weight  (128,64, 4, 4)
#   5  encoder.4.bias    (128,)
#   6  fc_mu.weight      (32, 8192)
#   7  fc_mu.bias        (32,)
#   8  fc_logvar.weight  (32, 8192)
#   9  fc_logvar.bias    (32,)
#  10  fc_decode.weight  (8192, 32)
#  11  fc_decode.bias    (8192,)
#  12  decoder.0.weight  (128,64, 4, 4)
#  13  decoder.0.bias    (64,)
#  14  decoder.2.weight  (64, 32, 4, 4)
#  15  decoder.2.bias    (32,)
#  16  decoder.4.weight  (32,  3, 4, 4)
#  17  decoder.4.bias    (3,)
# -----------------------------------------------------------------------
def _read_tensor(folder, idx, shape):
    path = os.path.join(folder, 'data', str(idx))
    raw  = np.frombuffer(open(path, 'rb').read(), dtype='<f4')   # little-endian float32
    return raw.reshape(shape).copy()


def load_pytorch_weights(pth_folder):
    d = pth_folder
    return {
        'encoder.0.weight': _read_tensor(d,  0, (32,  3, 4, 4)),
        'encoder.0.bias':   _read_tensor(d,  1, (32,)),
        'encoder.2.weight': _read_tensor(d,  2, (64, 32, 4, 4)),
        'encoder.2.bias':   _read_tensor(d,  3, (64,)),
        'encoder.4.weight': _read_tensor(d,  4, (128,64, 4, 4)),
        'encoder.4.bias':   _read_tensor(d,  5, (128,)),
        'fc_mu.weight':     _read_tensor(d,  6, (32, 8192)),
        'fc_mu.bias':       _read_tensor(d,  7, (32,)),
        'fc_logvar.weight': _read_tensor(d,  8, (32, 8192)),
        'fc_logvar.bias':   _read_tensor(d,  9, (32,)),
        'fc_decode.weight': _read_tensor(d, 10, (8192, 32)),
        'fc_decode.bias':   _read_tensor(d, 11, (8192,)),
        'decoder.0.weight': _read_tensor(d, 12, (128,64, 4, 4)),
        'decoder.0.bias':   _read_tensor(d, 13, (64,)),
        'decoder.2.weight': _read_tensor(d, 14, (64, 32, 4, 4)),
        'decoder.2.bias':   _read_tensor(d, 15, (32,)),
        'decoder.4.weight': _read_tensor(d, 16, (32,  3, 4, 4)),
        'decoder.4.bias':   _read_tensor(d, 17, (3,)),
    }


# -----------------------------------------------------------------------
# 2.  TENSORFLOW VAE  (same architecture, channels-last)
# -----------------------------------------------------------------------
def build_tf_vae():
    """
    Full VAE model used for training / validation.
    Uses the reparameterisation trick so it is NOT suitable for TFLite.
    Use build_tf_inference_model() for TFLite conversion.
    """
    inp = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 3))

    # Encoder
    x = layers.Conv2D(32,  4, strides=2, padding='same', activation='relu')(inp)
    x = layers.Conv2D(64,  4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)          # -> 8 × 8 × 128 = 8192

    mu     = layers.Dense(32, name='mu')(x)
    logvar = layers.Dense(32, name='logvar')(x)

    # Reparameterisation
    eps = tf.random.normal(tf.shape(mu))
    z   = mu + tf.exp(0.5 * logvar) * eps

    # Decoder
    x = layers.Dense(128 * 8 * 8, name='fc_decode')(z)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(3,  4, strides=2, padding='same', activation='sigmoid')(x)

    return keras.Model(inp, [x, mu, logvar], name='vae')


def build_tf_inference_model():
    """
    Deterministic inference model:  uses mu directly (no random sampling).
    Outputs the per-patch MSE reconstruction error as a single scalar.
    Fully compatible with TFLite.
    """
    inp = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 3), name='patch_input')

    # Encoder
    x  = layers.Conv2D(32,  4, strides=2, padding='same', activation='relu', name='enc_conv1')(inp)
    x  = layers.Conv2D(64,  4, strides=2, padding='same', activation='relu', name='enc_conv2')(x)
    x  = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu', name='enc_conv3')(x)
    x  = layers.Flatten(name='flatten')(x)
    mu = layers.Dense(32, name='mu')(x)

    # Decoder  (no sampling — use mu directly)
    x  = layers.Dense(128 * 8 * 8, name='fc_decode')(mu)
    x  = layers.Reshape((8, 8, 128), name='reshape')(x)
    x  = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu', name='dec_conv1')(x)
    x  = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu', name='dec_conv2')(x)
    x  = layers.Conv2DTranspose(3,  4, strides=2, padding='same', activation='sigmoid', name='dec_conv3')(x)

    # Per-patch MSE — computed inside the model so Flutter just reads one number
    diff  = layers.Subtract(name='diff')([x, inp])
    mse   = layers.Lambda(
                lambda d: tf.reduce_mean(tf.square(d), axis=[1, 2, 3], keepdims=False),
                name='mse'
            )(diff)

    return keras.Model(inp, mse, name='vae_inference')


# -----------------------------------------------------------------------
# 3.  WEIGHT TRANSFER  PyTorch -> TensorFlow
# -----------------------------------------------------------------------
def _conv_w(pt_w):
    """PyTorch Conv2d / ConvTranspose2d weight -> TF weight.

    PyTorch Conv2d:       (out_ch, in_ch,  kH, kW)
    PyTorch ConvTranspose: (in_ch, out_ch, kH, kW)  ← note swapped channels
    TF Conv2D:            (kH, kW, in_ch,  out_ch)
    TF Conv2DTranspose:   (kH, kW, out_ch, in_ch)   ← TF also swaps channels

    In both cases the correct numpy transpose is (2, 3, 1, 0).
    """
    return np.transpose(pt_w, (2, 3, 1, 0))


def _dense_w(pt_w):
    """PyTorch Linear weight (out, in) -> TF Dense weight (in, out)."""
    return pt_w.T


# -----------------------------------------------------------------------
# Flatten ordering fix
#
# After the 3rd encoder conv the feature map is (128, 8, 8) in PyTorch
# (NCHW) and (8, 8, 128) in TF (NHWC).  When flattened to 8192 values
# the element ordering differs:
#   PyTorch flat index: i_pt = c*64  + h*8 + w   (C fastest-varying)
#   TF flat index:      i_tf = h*128 + w*128 + c  (wait — H fastest)
#   Actually TF C-order for (H,W,C): i_tf = h*W*C + w*C + c
#
# perm[i_tf] = i_pt maps TF flat index -> PyTorch flat index.
# Use it to permute the Dense weight rows/cols so the same dot product
# is computed regardless of which framework flattened the tensor.
# -----------------------------------------------------------------------
_H, _W, _C = 8, 8, 128
_FLATTEN_PERM = np.array(
    [c * _H * _W + h * _W + w
     for h in range(_H) for w in range(_W) for c in range(_C)],
    dtype=np.int64
)


def transfer_weights(tf_model, pth_folder):
    """Load raw PyTorch weights and copy into the TF inference model."""
    sd = load_pytorch_weights(pth_folder)

    # ---- Encoder convolutions ----
    tf_model.get_layer('enc_conv1').set_weights([
        _conv_w(sd['encoder.0.weight']),
        sd['encoder.0.bias']
    ])
    tf_model.get_layer('enc_conv2').set_weights([
        _conv_w(sd['encoder.2.weight']),
        sd['encoder.2.bias']
    ])
    tf_model.get_layer('enc_conv3').set_weights([
        _conv_w(sd['encoder.4.weight']),
        sd['encoder.4.bias']
    ])

    # ---- Latent mean projection ----
    # Permute rows: W_tf[i, :] = W_pt.T[perm[i], :]
    tf_model.get_layer('mu').set_weights([
        _dense_w(sd['fc_mu.weight'])[_FLATTEN_PERM, :],
        sd['fc_mu.bias']
    ])

    # ---- Decoder FC ----
    # Permute columns: W_tf[:, i] = W_pt.T[:, perm[i]]
    tf_model.get_layer('fc_decode').set_weights([
        _dense_w(sd['fc_decode.weight'])[:, _FLATTEN_PERM],
        sd['fc_decode.bias']
    ])

    # ---- Decoder transposed convolutions ----
    tf_model.get_layer('dec_conv1').set_weights([
        _conv_w(sd['decoder.0.weight']),
        sd['decoder.0.bias']
    ])
    tf_model.get_layer('dec_conv2').set_weights([
        _conv_w(sd['decoder.2.weight']),
        sd['decoder.2.bias']
    ])
    tf_model.get_layer('dec_conv3').set_weights([
        _conv_w(sd['decoder.4.weight']),
        sd['decoder.4.bias']
    ])

    print("Weights transferred successfully.")
    return tf_model


# -----------------------------------------------------------------------
# 4.  VERIFY  weight transfer (sanity check without PyTorch)
# -----------------------------------------------------------------------
def verify_transfer(tf_model):
    """
    Sanity-check the TF model:
    - Run two different patches and confirm errors differ (model is live)
    - Run the same patch twice and confirm identical output (deterministic)
    """
    np.random.seed(0)
    patch_a = np.random.rand(1, PATCH_SIZE, PATCH_SIZE, 3).astype(np.float32)
    patch_b = np.random.rand(1, PATCH_SIZE, PATCH_SIZE, 3).astype(np.float32)

    err_a1 = float(tf_model(patch_a, training=False).numpy()[0])
    err_a2 = float(tf_model(patch_a, training=False).numpy()[0])
    err_b  = float(tf_model(patch_b, training=False).numpy()[0])

    print(f"Patch A  run 1 MSE : {err_a1:.6f}")
    print(f"Patch A  run 2 MSE : {err_a2:.6f}  (should match run 1)")
    print(f"Patch B  MSE       : {err_b:.6f}  (should differ from A)")

    if abs(err_a1 - err_a2) < 1e-8:
        print("PASS — model is deterministic.")
    else:
        print("WARNING — model is not deterministic (unexpected).")

    if abs(err_a1 - err_b) > 1e-6:
        print("PASS — model responds to different inputs.")
    else:
        print("WARNING — both patches gave the same error (model may be dead).")


# -----------------------------------------------------------------------
# 5.  CONVERT TO TFLITE
# -----------------------------------------------------------------------
def convert_to_tflite(tf_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    # No quantisation for the VAE.
    # The anomaly detection relies on precise reconstruction error thresholds —
    # even float16 introduces ~0.002 MSE error which can shift borderline patches.
    # At 1.7 MB the VAE is already small enough for mobile without quantisation.

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite model saved -> {output_path}  ({size_kb:.1f} KB)")
    return tflite_model


# -----------------------------------------------------------------------
# 6.  TFLITE INFERENCE  (sliding window — mirrors vae test 2.py logic)
# -----------------------------------------------------------------------
def run_tflite_inference(tflite_path, image_path,
                         patch_size=PATCH_SIZE,
                         stride=STRIDE,
                         anomaly_threshold=ANOMALY_THRESHOLD,
                         anomaly_ratio_threshold=ANOMALY_RATIO):

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    heatmap      = np.zeros((h, w), dtype=np.float32)
    patch_count  = 0
    anomaly_count = 0

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = img[y:y + patch_size, x:x + patch_size].astype(np.float32) / 255.0
            patch = patch[np.newaxis]                        # (1, 64, 64, 3)

            interpreter.set_tensor(inp_details[0]['index'], patch)
            interpreter.invoke()
            error = float(interpreter.get_tensor(out_details[0]['index'])[0])

            heatmap[y:y + patch_size, x:x + patch_size] += error
            patch_count  += 1
            if error > anomaly_threshold:
                anomaly_count += 1

    ratio = anomaly_count / patch_count if patch_count else 0.0
    label = "ANOMALY" if ratio > anomaly_ratio_threshold else "NORMAL"

    print(f"Total patches : {patch_count}")
    print(f"Anomaly patches: {anomaly_count}")
    print(f"Anomaly ratio  : {ratio:.4f}")
    print(f"FINAL PREDICTION: {label}")

    # Heatmap overlay
    heatmap /= (heatmap.max() + 1e-8)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # Save heatmap to disk (imshow not available in headless environments)
    out_path = image_path.replace('.jpg', '_heatmap.jpg').replace('.png', '_heatmap.png')
    cv2.imwrite(out_path, overlay[:, :, ::-1])
    print(f"Heatmap saved -> {out_path}")

    return label, ratio


# -----------------------------------------------------------------------
# 7.  MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":

    print("=== Step 1: Build TF inference model ===")
    model = build_tf_inference_model()
    model.summary()

    print("\n=== Step 2: Transfer weights from PyTorch zip ===")
    model = transfer_weights(model, PTH_FOLDER)

    print("\n=== Step 3: Verify weight transfer ===")
    verify_transfer(model)

    print("\n=== Step 4: Convert to TFLite ===")
    convert_to_tflite(model, TFLITE_OUT)

    print("\n=== Step 5: Test TFLite inference — disease image ===")
    run_tflite_inference(TFLITE_OUT, TEST_IMAGE_DISEASE)

    print("\n=== Step 5b: Test TFLite inference — normal image ===")
    run_tflite_inference(TFLITE_OUT, TEST_IMAGE_NORMAL)
