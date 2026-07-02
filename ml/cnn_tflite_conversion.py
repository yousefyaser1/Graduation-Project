import numpy as np
import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
B2_KERAS   = r"C:\Users\A\Downloads\ensemble_b2_model.keras"
B3_KERAS   = r"C:\Users\A\Downloads\b3_model.keras"
B2_TFLITE  = r"C:\Users\A\Graduation-Project\Flutter\assets\models\cnn_b2_model.tflite"
B3_TFLITE  = r"C:\Users\A\Graduation-Project\Flutter\assets\models\cnn_b3_model.tflite"

B2_SIZE    = (260, 260)
B3_SIZE    = (300, 300)
CLASSES    = ['Acne', 'Eczema', 'Tinea']   # alphabetical — matches training order


# -----------------------------------------------------------------------
# 1.  STRIP AUGMENTATION LAYERS -> clean inference model
#
#  Training model structure:
#    Input -> [6x random aug] -> EfficientNetBx -> GAP -> Dense -> BN -> Dropout -> Dense
#
#  Inference model (TFLite-friendly):
#    Input -> EfficientNetBx -> GAP -> Dense -> BN -> Dense
#    (Dropout is a no-op at inference; we skip it for a cleaner graph)
# -----------------------------------------------------------------------
def load_trained_model(keras_path, input_size):
    """
    Loads the trained Keras model as-is.
    Aug layers (RandomFlip, RandomRotation, etc.) are no-ops at inference
    (training=False) so they don't affect predictions.
    TFLite converter folds/removes them automatically.
    """
    print(f"Loading {keras_path} ...")
    model = keras.models.load_model(keras_path)
    print(f"  Model loaded: {model.input_shape} -> {model.output_shape}")
    return model


# -----------------------------------------------------------------------
# 2.  VERIFY  — run model twice on same input, confirm deterministic output
# -----------------------------------------------------------------------
def verify_model(model, input_size):
    np.random.seed(42)
    img = np.random.rand(1, *input_size, 3).astype(np.float32)

    prob1 = model(img, training=False).numpy()[0]
    prob2 = model(img, training=False).numpy()[0]

    max_diff = np.max(np.abs(prob1 - prob2))
    pred = CLASSES[int(np.argmax(prob1))]
    print(f"  Probabilities : {dict(zip(CLASSES, prob1.round(4)))}")
    print(f"  Prediction    : {pred}")
    print(f"  Deterministic : diff={max_diff:.2e}  {'PASS' if max_diff < 1e-6 else 'WARNING'}")
    return max_diff < 1e-6


# -----------------------------------------------------------------------
# 3.  CONVERT TO TFLITE
# -----------------------------------------------------------------------
def convert_to_tflite(inference_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    # Do NOT apply DEFAULT (dynamic-range) quantization — it degrades EfficientNet
    # accuracy significantly and pushes predictions toward the dominant class.
    # Float32 models are larger (~2x) but preserve classification accuracy.
    tflite_bytes = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_bytes)

    size_mb = len(tflite_bytes) / (1024 * 1024)
    print(f"  TFLite saved -> {output_path}  ({size_mb:.1f} MB)")
    return tflite_bytes


# -----------------------------------------------------------------------
# 4.  TEST TFLITE — ensemble prediction on a sample image
# -----------------------------------------------------------------------
def run_ensemble_tflite(b2_tflite_path, b3_tflite_path, image_path=None):
    """
    Loads both TFLite models, runs ensemble prediction (50/50 average).
    If no image_path given, uses a random test tensor.
    """
    interp_b2 = tf.lite.Interpreter(model_path=b2_tflite_path)
    interp_b3 = tf.lite.Interpreter(model_path=b3_tflite_path)
    interp_b2.allocate_tensors()
    interp_b3.allocate_tensors()

    b2_in  = interp_b2.get_input_details()[0]
    b2_out = interp_b2.get_output_details()[0]
    b3_in  = interp_b3.get_input_details()[0]
    b3_out = interp_b3.get_output_details()[0]

    if image_path:
        import cv2
        img = cv2.imread(image_path)
        # Keep [0, 255] — EfficientNet B2/B3 include a Rescaling(1/255) layer
        # internally (baked into the TFLite graph).  Do NOT divide here.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_b2 = cv2.resize(img, B2_SIZE)[np.newaxis]
        img_b3 = cv2.resize(img, B3_SIZE)[np.newaxis]
        print(f"  Testing on: {image_path}")
    else:
        # Use a realistic [0,255] range so the test matches actual inference
        np.random.seed(42)
        img_b2 = (np.random.rand(1, *B2_SIZE, 3) * 255).astype(np.float32)
        img_b3 = (np.random.rand(1, *B3_SIZE, 3) * 255).astype(np.float32)
        print("  Testing on: random tensor [0,255]")

    # B2 inference
    interp_b2.set_tensor(b2_in['index'], img_b2)
    interp_b2.invoke()
    prob_b2 = interp_b2.get_tensor(b2_out['index'])[0]

    # B3 inference
    interp_b3.set_tensor(b3_in['index'], img_b3)
    interp_b3.invoke()
    prob_b3 = interp_b3.get_tensor(b3_out['index'])[0]

    # 50/50 ensemble
    ensemble = (prob_b2 + prob_b3) / 2.0
    pred_idx = int(np.argmax(ensemble))

    print(f"\n  B2 probabilities  : {dict(zip(CLASSES, prob_b2.round(4)))}")
    print(f"  B3 probabilities  : {dict(zip(CLASSES, prob_b3.round(4)))}")
    print(f"  Ensemble probs    : {dict(zip(CLASSES, ensemble.round(4)))}")
    print(f"  PREDICTION        : {CLASSES[pred_idx]} ({ensemble[pred_idx]*100:.1f}%)")
    return CLASSES[pred_idx], ensemble


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":

    print("=== Step 1: Load trained models ===")
    b2_model = load_trained_model(B2_KERAS, B2_SIZE)
    b3_model = load_trained_model(B3_KERAS, B3_SIZE)

    print("\n=== Step 2: Verify models are deterministic at inference ===")
    print("B2:")
    verify_model(b2_model, B2_SIZE)
    print("B3:")
    verify_model(b3_model, B3_SIZE)

    print("\n=== Step 3: Convert to TFLite ===")
    print("B2:")
    convert_to_tflite(b2_model, B2_TFLITE)
    print("B3:")
    convert_to_tflite(b3_model, B3_TFLITE)

    print("\n=== Step 4: Test ensemble TFLite inference — random tensor ===")
    run_ensemble_tflite(B2_TFLITE, B3_TFLITE)

    # Test on real images from the local dataset (adjust paths as needed)
    TEST_IMAGES = [
        (r"C:\Users\A\Graduation-Project\New_Augmented_Dataset\test\acne\acne-cystic-117.jpg",   "acne"),
        (r"C:\Users\A\Graduation-Project\New_Augmented_Dataset\test\eczema\eczema-acute-22.jpg", "eczema"),
    ]
    # Add a tinea test image if available
    import os, glob as _glob
    tinea_dir = r"C:\Users\A\Graduation-Project\New_Augmented_Dataset\test\tinea"
    if os.path.isdir(tinea_dir):
        tinea_imgs = _glob.glob(os.path.join(tinea_dir, "*.jpg"))
        if tinea_imgs:
            TEST_IMAGES.append((tinea_imgs[0], "tinea"))

    print("\n=== Step 5: Test on real skin images ===")
    for img_path, true_label in TEST_IMAGES:
        if os.path.exists(img_path):
            print(f"\n  True label: {true_label}")
            run_ensemble_tflite(B2_TFLITE, B3_TFLITE, image_path=img_path)
        else:
            print(f"  Skipping {img_path} (not found)")
