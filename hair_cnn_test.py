"""
A/B test: should the CNN ensemble receive the HAIR-REMOVED image (like the VAE)
instead of the original?

Hypothesis: dense body hair texture resembles Tinea scale to the CNN (trained on
mostly-hairless clinical close-ups), driving the normal-hairy-skin -> Tinea 0.97
failure. DullRazor fills isolated dark lines only, so clinical disease images
should be near-unaffected.

Measures, with the exact device pipeline (resize<=1280, [0,255], 4-way flip TTA,
B2+B3 average — mirrors ai_service.dart / verify_pipeline.py):
  A) normal_test.jpg: ensemble probs original vs hair-removed, and the final app
     verdict after the confidence(0.40)/margin(0.10) gates.
  B) disease test set (N/class): accuracy original vs hair-removed.

Ship rule: hair-removal goes into the CNN path only if (A) meaningfully drops
Tinea on the normal photo AND (B) accuracy doesn't drop.
"""
import os, glob, random
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter
from extract_normal_patches import remove_hair, load_rgb

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
DATA = os.path.join(ROOT, "New_Augmented_Dataset", "test")
CLASSES = ["acne", "eczema", "tinea"]
N_PER_CLASS = 50
MIN_CONF, MIN_MARGIN = 0.40, 0.10
random.seed(7)

b2 = Interpreter(model_path=os.path.join(MODELS, "cnn_b2_model.tflite")); b2.allocate_tensors()
b3 = Interpreter(model_path=os.path.join(MODELS, "cnn_b3_model.tflite")); b3.allocate_tensors()

def _run(interp, x):
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(i["index"], x.astype(np.float32)); interp.invoke()
    return interp.get_tensor(o["index"])[0]

def _cnn(interp, img, size):
    r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    views = [r, r[:, ::-1], r[::-1, :], r[::-1, ::-1]]            # 4-way flip TTA
    return np.mean([_run(interp, v[np.newaxis]) for v in views], axis=0)

def ensemble(img):
    return (_cnn(b2, img, 260) + _cnn(b3, img, 300)) / 2.0

def verdict(p):
    i = int(np.argmax(p)); s = np.sort(p)[::-1]
    if s[0] < MIN_CONF or (s[0] - s[1]) < MIN_MARGIN:
        return "No Disease Detected (uncertain gate)"
    return f"{CLASSES[i].capitalize()} ({s[0]*100:.1f}%)"

def fmt(p):
    return " ".join(f"{c}={p[i]:.3f}" for i, c in enumerate(["Acne", "Eczema", "Tinea"]))

# ── A) the failing normal photo ──────────────────────────────────────────────
img = load_rgb(os.path.join(ROOT, "normal_test.jpg"))
print("A) NORMAL hairy-arm photo")
p_orig = ensemble(img)
print(f"   original    : {fmt(p_orig)}  ->  {verdict(p_orig)}")
p_dh = ensemble(remove_hair(img))
print(f"   hair-removed: {fmt(p_dh)}  ->  {verdict(p_dh)}", flush=True)

# ── B) disease accuracy, both variants from the same images ─────────────────
print(f"\nB) DISEASE test set ({N_PER_CLASS}/class, TTA ensemble)")
acc = {"orig": {}, "dehair": {}}
gate_normal = {"orig": 0, "dehair": 0}   # diseased imgs the uncertainty gate would clear
for cls_idx, cls in enumerate(CLASSES):
    fs = glob.glob(os.path.join(DATA, cls, "*")); random.shuffle(fs); fs = fs[:N_PER_CLASS]
    ok_o = ok_d = 0
    for f in fs:
        im = load_rgb(f)
        if im is None: continue
        po, pd = ensemble(im), ensemble(remove_hair(im))
        ok_o += int(np.argmax(po) == cls_idx)
        ok_d += int(np.argmax(pd) == cls_idx)
        so, sd = np.sort(po)[::-1], np.sort(pd)[::-1]
        gate_normal["orig"]   += int(so[0] < MIN_CONF or so[0]-so[1] < MIN_MARGIN)
        gate_normal["dehair"] += int(sd[0] < MIN_CONF or sd[0]-sd[1] < MIN_MARGIN)
    acc["orig"][cls], acc["dehair"][cls] = 100*ok_o/len(fs), 100*ok_d/len(fs)
    print(f"   {cls:7s}: original {acc['orig'][cls]:5.1f}%   hair-removed {acc['dehair'][cls]:5.1f}%", flush=True)

mo = np.mean(list(acc["orig"].values())); md = np.mean(list(acc["dehair"].values()))
print(f"   OVERALL: original {mo:.1f}%   hair-removed {md:.1f}%   (delta {md-mo:+.1f} pts)")
print(f"   uncertainty-gated to normal (missed disease): original {gate_normal['orig']}, "
      f"hair-removed {gate_normal['dehair']}  (of {3*N_PER_CLASS})")
print("\nSHIP if: normal-photo Tinea drops a lot AND overall delta ~>= -1 pt AND gated count doesn't grow.")
