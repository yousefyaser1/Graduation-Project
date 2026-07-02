"""
Gate score-distribution figure for the shipped Normal-vs-Disease gate
(normal_gate.tflite, EfficientNetB0). Runs the EXACT shipped model on:
  - NORMAL (held-out phone photos)  : gate_kit/normal_phone  (never trained on)
  - DISEASE (held-out test split)   : New_Augmented_Dataset/test/*

Mirrors ai_service.dart _runGate: resize whole image to 224x224 (antialiased),
feed raw [0,255] RGB; output = P(disease). Decision bands (ai_service.dart):
    P <= 0.60            -> "No Disease Detected"
    0.60 < P < 0.90      -> "Inconclusive" (retake)
    P >= 0.90            -> run CNN (disease)

Outputs (chapter5_figures/):
  gate_score_histogram.png   normalized P(disease) densities, normal vs disease,
                             with the 0.60 / 0.90 bands shaded
  gate_summary.txt           per-band counts, disease-recall + normal-pass rates
"""
import os, glob
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
OUT = os.path.join(ROOT, "chapter5_figures"); os.makedirs(OUT, exist_ok=True)
GATE_SIZE = 224
T_NORMAL, T_DISEASE = 0.60, 0.90
NORMAL_DIR = os.path.join(ROOT, "gate_kit", "normal_phone")
DISEASE_DIR = os.path.join(ROOT, "New_Augmented_Dataset", "test")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

gate = Interpreter(model_path=os.path.join(M, "normal_gate.tflite")); gate.allocate_tensors()
gi, go = gate.get_input_details()[0], gate.get_output_details()[0]


def list_imgs(d):
    out = []
    for dp, _, fns in os.walk(d):
        low = dp.lower()
        if "_rejected" in low or "_pruned" in low:
            continue
        out += [os.path.join(dp, f) for f in fns if f.lower().endswith(EXTS)]
    return sorted(out)


def p_disease(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # antialiased resize straight to 224 (INTER_AREA = the device's antialias path)
    x = cv2.resize(rgb, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gate.set_tensor(gi["index"], x[np.newaxis]); gate.invoke()
    return float(np.ravel(gate.get_tensor(go["index"]))[0])


def scores(files, tag):
    s = []
    for k, f in enumerate(files, 1):
        v = p_disease(f)
        if v is not None:
            s.append(v)
        if k % 50 == 0:
            print(f"  [{tag}] {k}/{len(files)}", flush=True)
    return np.array(s)


normal_files = list_imgs(NORMAL_DIR)
disease_files = list_imgs(DISEASE_DIR)
print(f"normal(held-out phone): {len(normal_files)}   disease(test): {len(disease_files)}", flush=True)
normal = scores(normal_files, "normal")
disease = scores(disease_files, "disease")

# ---- histogram ----
fig, ax = plt.subplots(figsize=(7.2, 4.4))
bins = np.linspace(0, 1, 26)
ax.axvspan(0, T_NORMAL, color="#2ca02c", alpha=0.07)
ax.axvspan(T_NORMAL, T_DISEASE, color="#ff7f0e", alpha=0.07)
ax.axvspan(T_DISEASE, 1.0, color="#d62728", alpha=0.07)
ax.hist(normal, bins=bins, density=True, alpha=0.6, color="#2ca02c",
        label=f"Normal — held-out phone (n={len(normal)})")
ax.hist(disease, bins=bins, density=True, alpha=0.6, color="#d62728",
        label=f"Disease — test split (n={len(disease)})")
ax.axvline(T_NORMAL, color="k", ls="--", lw=1.2)
ax.axvline(T_DISEASE, color="k", ls=":", lw=1.2)
ymax = ax.get_ylim()[1] * 1.34          # headroom so band labels clear the bars
ax.set_ylim(0, ymax)
ax.text(T_NORMAL / 2, ymax * 0.96, "No Disease\nDetected", ha="center", va="top", fontsize=9, color="#1b5e20")
ax.text((T_NORMAL + T_DISEASE) / 2, ymax * 0.96, "Inconclusive\n(retake)", ha="center", va="top", fontsize=9, color="#9c4a00")
ax.text((T_DISEASE + 1) / 2, ymax * 0.96, "Disease\n→ CNN", ha="center", va="top", fontsize=9, color="#8b0000")
ax.set_xlabel("Gate output  P(disease)")
ax.set_ylabel("Normalized density")
ax.set_title("Normal-vs-Disease gate score distribution")
ax.set_xlim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.76), framealpha=0.95)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gate_score_histogram.png"), dpi=200); plt.close(fig)


def band_counts(s):
    n = len(s)
    if n == 0:
        return (0, 0, 0, 0)
    nrm = int((s <= T_NORMAL).sum())
    inc = int(((s > T_NORMAL) & (s < T_DISEASE)).sum())
    dis = int((s >= T_DISEASE).sum())
    return n, nrm, inc, dis


with open(os.path.join(OUT, "gate_summary.txt"), "w") as f:
    def w(s): f.write(s + "\n"); print(s)
    w("=== Normal-vs-Disease gate (normal_gate.tflite) ===")
    w(f"thresholds: normal<= {T_NORMAL}, disease>= {T_DISEASE}")
    for tag, s in [("NORMAL (held-out phone)", normal), ("DISEASE (test split)", disease)]:
        n, nrm, inc, dis = band_counts(s)
        w(f"\n{tag}: n={n}  mean P(disease)={s.mean():.3f}  min={s.min():.3f}  max={s.max():.3f}" if n else f"\n{tag}: n=0")
        if n:
            w(f"   -> Normal band (<= {T_NORMAL}):      {nrm}/{n} ({nrm/n*100:.1f}%)")
            w(f"   -> Inconclusive ({T_NORMAL}-{T_DISEASE}):     {inc}/{n} ({inc/n*100:.1f}%)")
            w(f"   -> Disease band (>= {T_DISEASE}):     {dis}/{n} ({dis/n*100:.1f}%)")
    if len(normal):
        w(f"\nNormal-pass (P<= {T_NORMAL}) on held-out phone: {(normal<=T_NORMAL).mean()*100:.1f}%")
    if len(disease):
        w(f"Disease-fire (P> {T_NORMAL}) on test split:      {(disease>T_NORMAL).mean()*100:.1f}%")
        w(f"Disease->CNN (P>= {T_DISEASE}) on test split:     {(disease>=T_DISEASE).mean()*100:.1f}%")

print("\nWrote gate figure + summary to", OUT)
