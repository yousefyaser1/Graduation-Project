"""
Generate publication-ready Chapter 5 (Testing & Evaluation) figures + tables
from the REAL shipped models and the held-out test split.

Outputs (in chapter5_figures/):
  confusion_matrix_full.png      full test set (leakage-inflated)
  confusion_matrix_clean.png     near-duplicate-of-train images removed (honest)
  roc_curves.png                 one-vs-rest ROC + AUC for the 3 CNN classes
  classification_report_full.csv per-class precision/recall/F1/support + averages
  classification_report_clean.csv  same, on the leak-removed set
  predictions.csv                per-image: true, pred, p_acne/eczema/tinea, leaked
  metrics_summary.txt            accuracies (full vs clean) + AUCs + leakage counts

Method mirrors honest_eval.py / ai_service.dart CNN path: B2 (260) + B3 (300)
probability averaging, single forward pass (NO TTA) so the headline number is
consistent with the documented honest_eval result. Leakage = exact MD5 dup OR
perceptual dHash within Hamming<=THRESH of any train image of that class.
"""
import os, glob, hashlib, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf; Interpreter = tf.lite.Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(ROOT, "Flutter", "assets", "models")
DATA = os.path.join(ROOT, "New_Augmented_Dataset")
OUT = os.path.join(ROOT, "chapter5_figures")
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Acne", "Eczema", "Tinea"]
FOLDERS = ["acne", "eczema", "tinea"]
B2, B3, MAX_W, THRESH = (260, 260), (300, 300), 1280, 6

b2 = Interpreter(model_path=os.path.join(M, "cnn_b2_model.tflite")); b2.allocate_tensors()
b3 = Interpreter(model_path=os.path.join(M, "cnn_b3_model.tflite")); b3.allocate_tensors()


def fl(split, cls):
    return sorted(glob.glob(os.path.join(DATA, split, cls, "*")))

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def dhash(p, s=8):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (s + 1, s), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((img[:, 1:] > img[:, :-1]).flatten()).tobytes(), "big")

def hmin(q, arr):
    if not len(arr):
        return 64
    x = np.bitwise_xor(arr, np.uint64(q))
    return int(np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1).min())

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
    return (np.asarray(p2) + np.asarray(p3)) / 2.0


# ---- evaluate full test set ----
print("Evaluating test set with B2+B3 ensemble (single pass)...", flush=True)
y_true, y_prob, leaked = [], [], []
for ci, cls in enumerate(FOLDERS):
    tr_md5 = {md5(f) for f in fl("train", cls)}
    tr_arr = np.array([h for h in (dhash(f) for f in fl("train", cls)) if h is not None], np.uint64)
    tef = fl("test", cls)
    for k, f in enumerate(tef):
        y_true.append(ci)
        y_prob.append(predict(load(f)))
        h = dhash(f)
        leaked.append((md5(f) in tr_md5) or (h is not None and hmin(h, tr_arr) <= THRESH))
    print(f"  {cls}: {len(tef)} images done", flush=True)

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = y_prob.argmax(1)
leaked = np.array(leaked, bool)
clean = ~leaked


def confusion(mask):
    cm = np.zeros((3, 3), int)
    for t, p in zip(y_true[mask], y_pred[mask]):
        cm[t, p] += 1
    return cm

def metrics_rows(cm):
    """Return list of (label, precision, recall, f1, support) + accuracy."""
    rows, n = [], cm.sum()
    accs = cm.trace() / n if n else 0.0
    sup = cm.sum(1)
    precs, recs, f1s = [], [], []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
        rows.append((c, prec, rec, f1, int(sup[i])))
    # macro + weighted
    w = sup / sup.sum() if sup.sum() else np.zeros(3)
    rows.append(("Macro Avg", np.mean(precs), np.mean(recs), np.mean(f1s), int(sup.sum())))
    rows.append(("Weighted Avg", np.dot(w, precs), np.dot(w, recs), np.dot(w, f1s), int(sup.sum())))
    return rows, accs

def write_report(path, cm):
    rows, acc = metrics_rows(cm)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "precision", "recall", "f1_score", "support"])
        for c, p, r, fr, s in rows:
            wr.writerow([c, f"{p:.4f}", f"{r:.4f}", f"{fr:.4f}", s])
        wr.writerow(["accuracy", "", "", f"{acc:.4f}", int(cm.sum())])
    return rows, acc

def plot_confusion(cm, title, path):
    acc = cm.trace() / cm.sum() if cm.sum() else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"{title}\noverall accuracy = {acc*100:.1f}%  (n={cm.sum()})")
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(3):
        rowsum = cm[i].sum()
        for j in range(3):
            pct = 100.0 * cm[i, j] / rowsum if rowsum else 0.0
            ax.text(j, i, f"{cm[i,j]}\n{pct:.0f}%", ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def roc_one(y_bin, score):
    order = np.argsort(-score)
    y = y_bin[order]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return None
    tpr = np.concatenate([[0], np.cumsum(y) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / N])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


cm_full, cm_clean = confusion(np.ones_like(leaked)), confusion(clean)
plot_confusion(cm_full, "Confusion matrix — full test set", os.path.join(OUT, "confusion_matrix_full.png"))
plot_confusion(cm_clean, "Confusion matrix — leak-removed (honest)", os.path.join(OUT, "confusion_matrix_clean.png"))
rows_full, acc_full = write_report(os.path.join(OUT, "classification_report_full.csv"), cm_full)
rows_clean, acc_clean = write_report(os.path.join(OUT, "classification_report_clean.csv"), cm_clean)

# ---- ROC ----
fig, ax = plt.subplots(figsize=(5.6, 5.2))
colors = ["#d62728", "#2ca02c", "#1f77b4"]
aucs = {}
for ci, c in enumerate(CLASSES):
    r = roc_one((y_true == ci).astype(int), y_prob[:, ci])
    if r is None:
        continue
    fpr, tpr, auc = r; aucs[c] = auc
    ax.plot(fpr, tpr, color=colors[ci], lw=2, label=f"{c} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("One-vs-rest ROC — B2+B3 ensemble (test set)")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "roc_curves.png"), dpi=200); plt.close(fig)

# ---- per-image predictions ----
with open(os.path.join(OUT, "predictions.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["true", "pred", "p_acne", "p_eczema", "p_tinea", "leaked"])
    for t, p, pr, lk in zip(y_true, y_pred, y_prob, leaked):
        wr.writerow([CLASSES[t], CLASSES[p], f"{pr[0]:.4f}", f"{pr[1]:.4f}", f"{pr[2]:.4f}", int(lk)])

# ---- summary ----
macro_auc = np.mean(list(aucs.values())) if aucs else 0.0
with open(os.path.join(OUT, "metrics_summary.txt"), "w") as f:
    def w(s): f.write(s + "\n"); print(s)
    w("=== CNN ensemble (B2+B3, single-pass) on held-out test set ===")
    w(f"n(full)  = {len(y_true)}   accuracy(full)  = {acc_full*100:.2f}%")
    w(f"n(clean) = {int(clean.sum())}   accuracy(clean) = {acc_clean*100:.2f}%   "
      f"(leaked removed = {int(leaked.sum())}, {leaked.mean()*100:.1f}%)")
    w("")
    w("AUC (one-vs-rest):")
    for c in CLASSES:
        if c in aucs: w(f"  {c:8} {aucs[c]:.4f}")
    w(f"  macro    {macro_auc:.4f}")
    w("")
    w("Per-class (FULL):  class  precision  recall  f1  support")
    for c, p, r, fr, s in rows_full:
        w(f"  {c:14} {p:.4f}  {r:.4f}  {fr:.4f}  {s}")
    w("")
    w("Per-class (CLEAN/honest):  class  precision  recall  f1  support")
    for c, p, r, fr, s in rows_clean:
        w(f"  {c:14} {p:.4f}  {r:.4f}  {fr:.4f}  {s}")

print("\nWrote figures + tables to", OUT)
