"""
VAE gate threshold sweep on the diseased test set.

For each image we compute every patch MSE once, then evaluate, for a grid of
(per-patch MSE threshold, anomaly-ratio cutoff), the fraction of *diseased*
images correctly flagged ANOMALY (i.e. detection / sensitivity).

All test images are diseased, so higher detection = fewer false "No Disease".
NOTE: this cannot measure false positives on healthy skin (no negatives here);
it only shows how much the current gate under-detects real disease.
"""
import os, glob, random, numpy as np, cv2
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
TEST = os.path.join(ROOT, "New_Augmented_Dataset", "test")
FOLDERS = {"Acne": "acne", "Eczema": "eczema", "Tinea": "tinea"}
PATCH, STRIDE, MAX_W = 64, 32, 1280
N_PER_CLASS = 25

vae = Interpreter(model_path=os.path.join(MODELS, "vae_model.tflite")); vae.allocate_tensors()
vi, vo = vae.get_input_details()[0], vae.get_output_details()[0]

def patch_mses(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if img.shape[1] > MAX_W:
        img = cv2.resize(img, (MAX_W, int(img.shape[0]*MAX_W/img.shape[1])))
    h, w, _ = img.shape
    out = []
    for y in range(0, h-PATCH, STRIDE):
        for x in range(0, w-PATCH, STRIDE):
            p = img[y:y+PATCH, x:x+PATCH].astype(np.float32)/255.0
            vae.set_tensor(vi["index"], p[np.newaxis]); vae.invoke()
            out.append(float(np.ravel(vae.get_tensor(vo["index"]))[0]))
    return np.array(out)

# collect patch MSEs for a sample of each class
per_img = []
for cls, folder in FOLDERS.items():
    files = sorted(glob.glob(os.path.join(TEST, folder, "*")))
    random.Random(42).shuffle(files)
    for f in files[:N_PER_CLASS]:
        per_img.append(patch_mses(f))
print(f"Collected {len(per_img)} diseased images\n")

MSE_THRESHOLDS = [0.002, 0.004, 0.006, 0.008]
RATIO_CUTOFFS  = [0.05, 0.10, 0.15, 0.20]

print("Detection rate (fraction of diseased images flagged ANOMALY):")
print(f"{'mse_thr \\ ratio':>16}" + "".join(f"{r:>8}" for r in RATIO_CUTOFFS))
for mt in MSE_THRESHOLDS:
    row = []
    for rc in RATIO_CUTOFFS:
        det = np.mean([(np.mean(m > mt) > rc) for m in per_img])
        row.append(det)
    star = "  <- current" if mt == 0.008 else ""
    print(f"{mt:>16}" + "".join(f"{v*100:>7.0f}%" for v in row) + star)
print("\n(current gate = mse_thr 0.008, ratio 0.20 = bottom-right cell)")
