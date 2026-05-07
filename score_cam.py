"""
Score-CAM implementation for the skin disease classification pipeline.

Pipeline position:
    VAE (normal/anomaly) -> CNN ensemble (disease class) -> Score-CAM (heatmap)

Score-CAM runs on EfficientNetB3 (the larger model).
Uses top-K channels (default K=30) to keep mobile inference fast:
    ~30 forward passes x ~100ms each = ~3s on a mid-range device.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
B3_KERAS            = r"C:\Users\A\Downloads\b3_model.keras"
B3_TFLITE           = r"C:\Users\A\Graduation-Project\cnn_b3_model.tflite"
FEAT_EXTRACTOR_OUT  = r"C:\Users\A\Graduation-Project\b3_feature_extractor.tflite"
VAE_TFLITE          = r"C:\Users\A\Graduation-Project\vae_model.tflite"
B2_TFLITE           = r"C:\Users\A\Graduation-Project\cnn_b2_model.tflite"

B3_SIZE  = (300, 300)
B2_SIZE  = (260, 260)
CLASSES  = ['Acne', 'Eczema', 'Tinea']   # alphabetical — matches training order
TOP_K    = 30   # number of feature channels to use in Score-CAM

# VAE sliding window params
PATCH_SIZE        = 64
STRIDE            = 32
ANOMALY_THRESHOLD = 0.008   # 0.006 (PyTorch) + ~0.002 systematic TFLite offset
ANOMALY_RATIO     = 0.20


# -----------------------------------------------------------------------
# 1.  BUILD & CONVERT FEATURE EXTRACTOR
#
#  Extracts the last conv feature maps from EfficientNetB3:
#  Input:  (1, 300, 300, 3)
#  Output: (1, 10, 10, 1536)  -- top_activation layer
# -----------------------------------------------------------------------
def build_feature_extractor():
    print("Loading B3 model for feature extractor ...")
    b3 = keras.models.load_model(B3_KERAS)
    backbone = b3.get_layer('efficientnetb3')

    # Use backbone's input (bypasses aug layers) -> top_activation output
    feat_model = keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer('top_activation').output,
        name='b3_feature_extractor'
    )
    print(f"  Feature extractor: {feat_model.input_shape} -> {feat_model.output_shape}")
    return feat_model


def convert_feature_extractor(feat_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(feat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_bytes)
    size_mb = len(tflite_bytes) / (1024 * 1024)
    print(f"  Feature extractor TFLite saved -> {output_path}  ({size_mb:.1f} MB)")


# -----------------------------------------------------------------------
# 2.  SCORE-CAM  (TFLite-based, mobile-ready)
#
#  Algorithm:
#   1. Run feature extractor  -> (10, 10, 1536) feature maps
#   2. Pick top-K channels by mean activation magnitude
#   3. For each channel k:
#       a. Upsample map_k  (10,10) -> (300,300)
#       b. Normalise to [0,1]
#       c. masked_img = original * map_k
#       d. classifier forward pass -> prob[target_class]  (= score_k)
#   4. Normalise scores via softmax so they sum to 1
#   5. heatmap = sum_k( score_k * upsample(map_k) )
#   6. Normalise heatmap to [0,1]
# -----------------------------------------------------------------------
def run_score_cam(image_rgb, target_class_idx,
                  feat_tflite_path, cls_tflite_path,
                  top_k=TOP_K):
    """
    Args:
        image_rgb       : (H, W, 3) float32 numpy array, values in [0,1]
        target_class_idx: integer class index (0=Acne, 1=Eczema, 2=Tinea)
        feat_tflite_path: path to b3_feature_extractor.tflite
        cls_tflite_path : path to cnn_b3_model.tflite
        top_k           : number of feature channels to use

    Returns:
        heatmap_norm    : (H, W) float32, values in [0,1]
        overlay         : (H, W, 3) uint8, heatmap overlaid on original
    """
    H, W = image_rgb.shape[:2]

    # ---- Resize to B3 input size ----
    img_b3 = cv2.resize(image_rgb, B3_SIZE).astype(np.float32)    # (300,300,3)
    img_b3_batch = img_b3[np.newaxis]                               # (1,300,300,3)

    # ---- Load interpreters ----
    feat_interp = tf.lite.Interpreter(model_path=feat_tflite_path)
    cls_interp  = tf.lite.Interpreter(model_path=cls_tflite_path)
    feat_interp.allocate_tensors()
    cls_interp.allocate_tensors()

    feat_in  = feat_interp.get_input_details()[0]
    feat_out = feat_interp.get_output_details()[0]
    cls_in   = cls_interp.get_input_details()[0]
    cls_out  = cls_interp.get_output_details()[0]

    # ---- Step 1: Extract feature maps (1, 10, 10, 1536) ----
    feat_interp.set_tensor(feat_in['index'], img_b3_batch)
    feat_interp.invoke()
    feature_maps = feat_interp.get_tensor(feat_out['index'])[0]    # (10, 10, 1536)

    # ---- Step 2: Select top-K channels ----
    channel_means = np.mean(np.abs(feature_maps), axis=(0, 1))     # (1536,)
    top_k_idx     = np.argsort(channel_means)[-top_k:]             # top-K indices

    # ---- Steps 3-4: Masked forward passes ----
    scores = np.zeros(top_k, dtype=np.float32)

    for i, ch_idx in enumerate(top_k_idx):
        ch_map = feature_maps[:, :, ch_idx]                         # (10, 10)

        # Upsample and normalise
        ch_up  = cv2.resize(ch_map, B3_SIZE)                        # (300, 300)
        ch_min, ch_max = ch_up.min(), ch_up.max()
        if ch_max - ch_min > 1e-8:
            ch_norm = (ch_up - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = np.zeros_like(ch_up)                          # dead channel

        # Masked image
        masked = img_b3 * ch_norm[:, :, np.newaxis]                 # (300,300,3)
        masked_batch = masked[np.newaxis].astype(np.float32)

        # Classifier forward pass
        cls_interp.set_tensor(cls_in['index'], masked_batch)
        cls_interp.invoke()
        probs    = cls_interp.get_tensor(cls_out['index'])[0]       # (3,)
        scores[i] = probs[target_class_idx]

    # ---- Step 4: Normalise scores ----
    scores = np.exp(scores - scores.max())                          # softmax numerically stable
    scores /= scores.sum()

    # ---- Step 5: Weighted heatmap ----
    heatmap = np.zeros(B3_SIZE, dtype=np.float32)
    for i, ch_idx in enumerate(top_k_idx):
        ch_map = feature_maps[:, :, ch_idx]
        ch_up  = cv2.resize(ch_map, B3_SIZE)
        heatmap += scores[i] * ch_up

    # ---- Step 6: Normalise and resize to original image size ----
    heatmap = np.maximum(heatmap, 0)
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    heatmap = cv2.resize(heatmap, (W, H))

    # ---- Colour overlay ----
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig_uint8    = (image_rgb * 255).astype(np.uint8)
    overlay       = cv2.addWeighted(orig_uint8, 0.6, heatmap_color, 0.4, 0)

    return heatmap, overlay


# -----------------------------------------------------------------------
# 3.  FULL PIPELINE  VAE -> CNN ensemble -> Score-CAM
# -----------------------------------------------------------------------
def run_full_pipeline(image_path,
                      vae_tflite   = VAE_TFLITE,
                      b2_tflite    = B2_TFLITE,
                      b3_tflite    = B3_TFLITE,
                      feat_tflite  = FEAT_EXTRACTOR_OUT,
                      save_output  = True):
    """
    Runs the complete 3-stage pipeline on one image and saves results.
    Returns: (stage, predicted_class, confidence, heatmap_path)
        stage = 'normal' | 'anomaly'
    """
    print(f"\n{'='*55}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*55}")

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W     = img_rgb.shape[:2]
    img_f32  = img_rgb.astype(np.float32) / 255.0

    # ----------------------------------------------------------------
    # Stage 1: VAE anomaly detection
    # ----------------------------------------------------------------
    print("\n[Stage 1] VAE anomaly detection ...")
    vae_interp = tf.lite.Interpreter(model_path=vae_tflite)
    vae_interp.allocate_tensors()
    vae_in  = vae_interp.get_input_details()[0]
    vae_out = vae_interp.get_output_details()[0]

    heatmap_vae   = np.zeros((H, W), dtype=np.float32)
    patch_count   = 0
    anomaly_count = 0

    for y in range(0, H - PATCH_SIZE, STRIDE):
        for x in range(0, W - PATCH_SIZE, STRIDE):
            patch = img_f32[y:y+PATCH_SIZE, x:x+PATCH_SIZE][np.newaxis]
            vae_interp.set_tensor(vae_in['index'], patch)
            vae_interp.invoke()
            error = float(vae_interp.get_tensor(vae_out['index'])[0])
            heatmap_vae[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += error
            patch_count += 1
            if error > ANOMALY_THRESHOLD:
                anomaly_count += 1

    anomaly_ratio = anomaly_count / patch_count if patch_count else 0.0
    is_anomaly    = anomaly_ratio > ANOMALY_RATIO
    vae_label     = "ANOMALY" if is_anomaly else "NORMAL"

    print(f"  Patches: {patch_count}  |  Anomalous: {anomaly_count}  |  Ratio: {anomaly_ratio:.3f}")
    print(f"  VAE decision: {vae_label}")

    if not is_anomaly:
        print("\nRESULT: No skin disease detected.")
        return 'normal', None, None, None

    # ----------------------------------------------------------------
    # Stage 2: CNN ensemble classification
    # ----------------------------------------------------------------
    print("\n[Stage 2] CNN ensemble classification ...")
    b2_interp = tf.lite.Interpreter(model_path=b2_tflite)
    b3_interp = tf.lite.Interpreter(model_path=b3_tflite)
    b2_interp.allocate_tensors()
    b3_interp.allocate_tensors()

    img_b2 = cv2.resize(img_f32, B2_SIZE)[np.newaxis].astype(np.float32)
    img_b3 = cv2.resize(img_f32, B3_SIZE)[np.newaxis].astype(np.float32)

    b2_interp.set_tensor(b2_interp.get_input_details()[0]['index'],  img_b2)
    b2_interp.invoke()
    prob_b2 = b2_interp.get_tensor(b2_interp.get_output_details()[0]['index'])[0]

    b3_interp.set_tensor(b3_interp.get_input_details()[0]['index'],  img_b3)
    b3_interp.invoke()
    prob_b3 = b3_interp.get_tensor(b3_interp.get_output_details()[0]['index'])[0]

    ensemble     = (prob_b2 + prob_b3) / 2.0
    pred_idx     = int(np.argmax(ensemble))
    pred_class   = CLASSES[pred_idx]
    confidence   = float(ensemble[pred_idx])

    print(f"  B2: {dict(zip(CLASSES, prob_b2.round(3)))}")
    print(f"  B3: {dict(zip(CLASSES, prob_b3.round(3)))}")
    print(f"  Ensemble: {dict(zip(CLASSES, ensemble.round(3)))}")
    print(f"  Prediction: {pred_class} ({confidence*100:.1f}%)")

    # ----------------------------------------------------------------
    # Stage 3: Score-CAM explainability
    # ----------------------------------------------------------------
    print(f"\n[Stage 3] Score-CAM (top-{TOP_K} channels, target={pred_class}) ...")
    heatmap, overlay = run_score_cam(
        image_rgb        = img_f32,
        target_class_idx = pred_idx,
        feat_tflite_path = feat_tflite,
        cls_tflite_path  = b3_tflite,
        top_k            = TOP_K
    )
    print("  Heatmap generated.")

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    heatmap_path = None
    if save_output:
        base         = os.path.splitext(image_path)[0]
        overlay_path = base + '_scorecam.jpg'
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Score-CAM overlay saved -> {overlay_path}")
        heatmap_path = overlay_path

    print(f"\nFINAL RESULT: {pred_class} | Confidence: {confidence*100:.1f}%")
    return 'anomaly', pred_class, confidence, heatmap_path


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":

    # Step 1: Build and convert feature extractor
    print("=== Step 1: Build B3 feature extractor ===")
    feat_model = build_feature_extractor()

    print("\n=== Step 2: Convert feature extractor to TFLite ===")
    convert_feature_extractor(feat_model, FEAT_EXTRACTOR_OUT)

    # Step 3: Full pipeline test on a disease image
    TEST_IMAGE = r"C:\Users\A\Graduation-Project\.kilo\worktrees\freckle-september\Dermatology_CNN_Dataset\test\acne\07Rhinophyma1.jpg"

    print("\n=== Step 3: Full pipeline test ===")
    run_full_pipeline(TEST_IMAGE)
