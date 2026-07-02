"""
Validate raising the VAE per-patch MSE threshold (_anomalyThreshold).
Current app: 0.004.  vae_tflite_conversion.py calibrated value: 0.008.

For each candidate threshold we report, over the disease TEST set:
  - DISEASE CAUGHT rate = fraction with anomaly ratio > 0.20 (gate fires -> CNN).
    The complement is the FALSE-NORMAL (missed disease) rate = the safety cost.
And, as the normal-skin false-alarm proxy:
  - normal 64x64 patches (new_patches.zip): per-patch anomalous fraction
  - the failing hairy-arm photo (normal_test.jpg): full-image anomaly ratio + gate

Mirrors verify_pipeline.py / ai_service.dart VAE exactly (no hair removal, same as
the parity tools; clinical disease images aren't hairy so this is negligible for the
safety metric, and slightly conservative for the normal photo).
"""
import os, glob, random, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
DATA = os.path.join(ROOT, "New_Augmented_Dataset", "test")
PATCHES = r"C:\Users\A\Downloads\new_patches_x\new_patches"
NORMAL_IMG = os.path.join(ROOT, "normal_test.jpg")
PATCH, STRIDE, MAX_W, GATE = 64, 64, 1280, 0.20
THRS = [0.004, 0.005, 0.006, 0.007, 0.008]
N_PER_CLASS = 50
random.seed(3)

vae = Interpreter(model_path=os.path.join(MODELS, "vae_model.tflite")); vae.allocate_tensors()
inp, out = vae.get_input_details()[0], vae.get_output_details()[0]

def load_rgb(path):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if w > MAX_W:
        h = int(h * MAX_W / w); w = MAX_W; im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0

def patch_mses(arr):
    h, w, _ = arr.shape; vals = []
    for y in range(0, h - PATCH + 1, STRIDE):
        for x in range(0, w - PATCH + 1, STRIDE):
            p = arr[y:y+PATCH, x:x+PATCH, :]
            vae.set_tensor(inp["index"], p[np.newaxis]); vae.invoke()
            vals.append(float(vae.get_tensor(out["index"]).flatten()[0]))
    return np.array(vals)

# Disease: store per-image MSE arrays once, evaluate every threshold from them.
print("scanning disease test set...", flush=True)
disease = {}
for cls in ["acne", "eczema", "tinea"]:
    fs = glob.glob(os.path.join(DATA, cls, "*")); random.shuffle(fs); fs = fs[:N_PER_CLASS]
    arrs = [patch_mses(load_rgb(f)) for f in fs]
    disease[cls] = arrs
    print(f"  {cls}: {len(arrs)} images", flush=True)

# Normal patches: one MSE each.
pf = glob.glob(os.path.join(PATCHES, "*.png"))
patch_vals = np.array([float(vae.set_tensor(inp["index"],
                       (np.asarray(Image.open(f).convert("RGB"), np.float32)/255.0)[np.newaxis])
                       or vae.invoke() or vae.get_tensor(out["index"]).flatten()[0]) for f in pf])
normal_img_mse = patch_mses(load_rgb(NORMAL_IMG))

print("\n================ THRESHOLD VALIDATION ================")
print(f"{'thr':>6} | {'Acne':>12} {'Eczema':>12} {'Tinea':>12} {'OVERALL':>12} | {'normalPatch':>11} | {'hairyArm':>14}")
print(f"{'':>6} | {'(caught%)':>12}*3{'':>4} | {'anom%':>11} | {'ratio/gate':>14}")
for t in THRS:
    caught = {}
    for cls, arrs in disease.items():
        ratios = np.array([(m > t).mean() for m in arrs])
        caught[cls] = (ratios > GATE).mean() * 100
    overall = np.mean(list(caught.values()))
    patch_anom = (patch_vals > t).mean() * 100
    nr = (normal_img_mse > t).mean()
    gate = "NORMAL" if nr <= GATE else "anom"
    print(f"{t:>6.3f} | {caught['acne']:>11.1f}% {caught['eczema']:>11.1f}% "
          f"{caught['tinea']:>11.1f}% {overall:>11.1f}% | {patch_anom:>10.1f}% | {nr*100:>6.1f}% {gate:>6}")

print("\nReading: DISEASE caught% should stay HIGH (missed = 100-caught = disease wrongly called normal).")
print("normalPatch anom% and hairyArm ratio should DROP (fewer false alarms on normal skin).")
print(f"Gate fires (-> disease path) when ratio > {GATE*100:.0f}%.")
