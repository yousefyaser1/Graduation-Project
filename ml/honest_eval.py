"""
Honest accuracy: evaluate the shipped ensemble ONLY on test images that are not
duplicated (exact or near) in the training set. Removes the leakage that
inflates the headline number, giving a truer estimate of real generalisation.

Note: dHash misses heavy-rotation variants, so some leaked images may survive
into the "clean" set -> the honest number here is still a slight OVER-estimate,
but far closer to reality than the contaminated full-test score.

Run:  python honest_eval.py [hamming_thresh]   (default 6)
"""
import os, sys, glob, hashlib
import numpy as np, cv2
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf; Interpreter = tf.lite.Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(ROOT, "Flutter", "assets", "models")
DATA = os.path.join(ROOT, "New_Augmented_Dataset")
CLASSES = ["acne", "eczema", "tinea"]
B2, B3, MAX_W = (260, 260), (300, 300), 1280
THRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 6

b2 = Interpreter(model_path=os.path.join(M, "cnn_b2_model.tflite")); b2.allocate_tensors()
b3 = Interpreter(model_path=os.path.join(M, "cnn_b3_model.tflite")); b3.allocate_tensors()


def fl(split, cls): return sorted(glob.glob(os.path.join(DATA, split, cls, "*")))
def md5(p):
    with open(p, "rb") as f: return hashlib.md5(f.read()).hexdigest()
def dhash(p, s=8):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (s + 1, s), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((img[:, 1:] > img[:, :-1]).flatten()).tobytes(), "big")
def hmin(q, arr):
    x = np.bitwise_xor(arr, np.uint64(q))
    return int(np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1).min()) if len(arr) else 64
def run(it, x):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x.astype(np.float32)); it.invoke()
    return it.get_tensor(o["index"])[0]
def load(p):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    if img.shape[1] > MAX_W:
        h = int(img.shape[0] * MAX_W / img.shape[1])
        img = cv2.resize(img, (MAX_W, h), interpolation=cv2.INTER_LINEAR)
    return img
def predict(img):
    p2 = run(b2, cv2.resize(img, B2, interpolation=cv2.INTER_LINEAR)[np.newaxis])
    p3 = run(b3, cv2.resize(img, B3, interpolation=cv2.INTER_LINEAR)[np.newaxis])
    return int(np.argmax((p2 + p3) / 2.0))


cm_all = np.zeros((3, 3), int)
cm_clean = np.zeros((3, 3), int)
for ci, cls in enumerate(CLASSES):
    tr_md5 = {md5(f) for f in fl("train", cls)}
    tr_arr = np.array([h for h in (dhash(f) for f in fl("train", cls)) if h is not None], np.uint64)
    for f in fl("test", cls):
        pred = predict(load(f))
        cm_all[ci, pred] += 1
        h = dhash(f)
        leaked = (md5(f) in tr_md5) or (h is not None and hmin(h, tr_arr) <= THRESH)
        if not leaked:
            cm_clean[ci, pred] += 1

def report(cm, title):
    print(f"\n=== {title} ===")
    print(f"{'true/pred':>10}" + "".join(f"{c:>9}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>10}" + "".join(f"{v:>9}" for v in cm[i]))
    tot = cm.sum()
    print(f"  overall: {cm.trace()/tot*100:.1f}%  (n={tot})")
    for i, c in enumerate(CLASSES):
        s = cm[i].sum()
        if s: print(f"    {c}: {cm[i,i]/s*100:.1f}%  ({cm[i,i]}/{s})")

report(cm_all, "FULL test set (leakage-inflated)")
report(cm_clean, "CLEAN test set (leaked images removed) = honest estimate")
