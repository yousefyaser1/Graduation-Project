"""
Test-set ablation: B2-only vs B3-only vs B2+B3 ensemble, on the SAME 362-image
test split, with the SAME leakage filter as honest_eval.py.

Accuracy is device-independent: the shipped .tflite models produce identical
outputs on desktop and phone for identical inputs, so this desktop run is a
faithful measure of the deployed classifier's accuracy.

Run:  python ablation_eval.py [hamming_thresh]   (default 6)

Paste the printed tables back and they go straight into the Results chapter.
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
def probs(img):
    """Return (p_b2, p_b3) softmax vectors for one image."""
    p2 = run(b2, cv2.resize(img, B2, interpolation=cv2.INTER_LINEAR)[np.newaxis])
    p3 = run(b3, cv2.resize(img, B3, interpolation=cv2.INTER_LINEAR)[np.newaxis])
    return p2, p3


# Three confusion matrices per condition (full + clean), one set per model variant
variants = ["B2-only", "B3-only", "B2+B3 ensemble"]
cm_all = {v: np.zeros((3, 3), int) for v in variants}
cm_clean = {v: np.zeros((3, 3), int) for v in variants}

for ci, cls in enumerate(CLASSES):
    tr_md5 = {md5(f) for f in fl("train", cls)}
    tr_arr = np.array([h for h in (dhash(f) for f in fl("train", cls)) if h is not None], np.uint64)
    for f in fl("test", cls):
        p2, p3 = probs(load(f))
        preds = {
            "B2-only": int(np.argmax(p2)),
            "B3-only": int(np.argmax(p3)),
            "B2+B3 ensemble": int(np.argmax((p2 + p3) / 2.0)),
        }
        h = dhash(f)
        leaked = (md5(f) in tr_md5) or (h is not None and hmin(h, tr_arr) <= THRESH)
        for v, pred in preds.items():
            cm_all[v][ci, pred] += 1
            if not leaked:
                cm_clean[v][ci, pred] += 1


def macro_f1(cm):
    f1s = []
    for i in range(3):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))

def report(cms, title):
    print(f"\n================ {title} ================")
    print(f"{'variant':>16}{'accuracy':>12}{'macro-F1':>12}{'n':>8}")
    for v in variants:
        cm = cms[v]; tot = cm.sum()
        acc = cm.trace() / tot * 100 if tot else 0.0
        print(f"{v:>16}{acc:>11.1f}%{macro_f1(cm):>12.3f}{tot:>8}")
    for v in variants:
        cm = cms[v]
        print(f"\n  -- {v} per-class --")
        for i, c in enumerate(CLASSES):
            s = cm[i].sum()
            if s: print(f"     {c}: {cm[i,i]/s*100:.1f}%  ({cm[i,i]}/{s})")

report(cm_all, "FULL test set (leakage-inflated)")
report(cm_clean, "CLEAN test set (leaked images removed) = honest estimate")
print("\nThresholds/paths match honest_eval.py. Paste both tables back.")
