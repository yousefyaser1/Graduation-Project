"""
Empirical test: can a feature-space outlier detector flag the failing normal-skin
image WITHOUT any normal training data?

Builds the disease feature distribution from existing train/ images (b3_feature_extractor,
global-average-pooled 1536-vec), measures in-distribution spread on val/ disease images,
then checks where the failing normal image falls. If the normal image is a clear outlier
on any signal, a no-retrain gate is viable; if it sits inside the disease distribution,
it is not.

Mirrors score_cam.py preprocessing exactly (downsample <=1280, resize 300x300 linear,
[0,255], EfficientNet rescales internally).
"""
import os, sys, glob, random
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "Flutter", "assets", "models")
DATA = os.path.join(ROOT, "New_Augmented_Dataset")
CLASSES = ["acne", "eczema", "tinea"]
B3_W = B3_H = 300
MAX_W = 1280
N_TRAIN = 150   # per class, build reference stats
N_VAL = 40      # per class, in-distribution spread
SEED = 0

random.seed(SEED); np.random.seed(SEED)

FEAT = Interpreter(model_path=os.path.join(MODELS, "b3_feature_extractor.tflite")); FEAT.allocate_tensors()
B3 = Interpreter(model_path=os.path.join(MODELS, "cnn_b3_model.tflite")); B3.allocate_tensors()

def _run(interp, x):
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(i["index"], x.astype(np.float32)); interp.invoke()
    return interp.get_tensor(o["index"])[0]

def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None: return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[1] > MAX_W:
        h = int(img.shape[0] * MAX_W / img.shape[1])
        img = cv2.resize(img, (MAX_W, h), interpolation=cv2.INTER_LINEAR)
    return img

def gap_features(path):
    img = load_rgb(path)
    if img is None: return None
    x = cv2.resize(img, (B3_W, B3_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    f = _run(FEAT, x[np.newaxis])           # (10,10,1536)
    return f.reshape(-1, f.shape[-1]).mean(axis=0)   # GAP -> (1536,)

def b3_probs(path):
    img = load_rgb(path)
    x = cv2.resize(img, (B3_W, B3_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return _run(B3, x[np.newaxis])

def sample(cls, split, n):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        files += glob.glob(os.path.join(DATA, split, cls, ext))
    random.shuffle(files)
    return files[:n]

def extract_many(file_lists, tag):
    feats, total = [], sum(len(f) for f in file_lists)
    done = 0
    for cls, files in zip(CLASSES, file_lists):
        cf = []
        for fp in files:
            v = gap_features(fp)
            if v is not None: cf.append(v)
            done += 1
            if done % 50 == 0: print(f"  [{tag}] {done}/{total}", flush=True)
        feats.append(np.array(cf, np.float32))
    return feats

print("== extracting train (reference) ==", flush=True)
train_files = [sample(c, "train", N_TRAIN) for c in CLASSES]
train_feats = extract_many(train_files, "train")     # list of (n,1536)

print("== extracting val (in-distribution) ==", flush=True)
val_files = [sample(c, "val", N_VAL) for c in CLASSES]
val_feats = extract_many(val_files, "val")

all_train = np.concatenate(train_feats, 0)
mu = all_train.mean(0); sd = all_train.std(0) + 1e-6     # standardization
class_means = [cf.mean(0) for cf in train_feats]          # raw per-class means
class_means_std = [(m - mu) / sd for m in class_means]    # standardized
train_std = (all_train - mu) / sd                          # for kNN

def min_maha(v):     # diagonal Mahalanobis (shared var) -> nearest class, normalized by sqrt(D)
    vs = (v - mu) / sd
    d = [np.sqrt(np.mean((vs - cms) ** 2)) for cms in class_means_std]
    return min(d), int(np.argmin(d))

def min_knn_cos(v, k=5):   # 1 - max cosine sim to training features (mean of top-k)
    vs = (v - mu) / sd
    vn = vs / (np.linalg.norm(vs) + 1e-9)
    tn = train_std / (np.linalg.norm(train_std, axis=1, keepdims=True) + 1e-9)
    sims = tn @ vn
    topk = np.sort(sims)[::-1][:k]
    return 1.0 - float(topk.mean())

def feat_norm(v): return float(np.linalg.norm(v))

# In-distribution distributions
val_all = np.concatenate(val_feats, 0)
val_maha = np.array([min_maha(v)[0] for v in val_all])
val_knn  = np.array([min_knn_cos(v) for v in val_all])
val_norm = np.array([feat_norm(v) for v in val_all])

def pct_rank(dist, x): return float((dist < x).mean() * 100)

print("\n================ RESULTS ================")
for name, dist in [("min-Mahalanobis(diag)", val_maha), ("kNN cosine-dist", val_knn), ("feature L2 norm", val_norm)]:
    print(f"\n[{name}] in-distribution (disease val, n={len(dist)}):")
    print(f"   mean={dist.mean():.3f}  p50={np.percentile(dist,50):.3f}  "
          f"p95={np.percentile(dist,95):.3f}  p99={np.percentile(dist,99):.3f}  max={dist.max():.3f}")

# The failing normal image
norm_path = os.path.join(ROOT, "normal_test.jpg")
nv = gap_features(norm_path)
nm, nm_cls = min_maha(nv); nk = min_knn_cos(nv); nn = feat_norm(nv)
p = b3_probs(norm_path)

print("\n--- FAILING NORMAL IMAGE ---")
print(f"   B3 softmax: " + " ".join(f"{c}={p[i]:.3f}" for i,c in enumerate(['Acne','Eczema','Tinea'])))
print(f"   min-Mahalanobis = {nm:.3f}  (nearest class={CLASSES[nm_cls]})  -> percentile vs val = {pct_rank(val_maha, nm):.1f}%")
print(f"   kNN cosine-dist = {nk:.3f}  -> percentile vs val = {pct_rank(val_knn, nk):.1f}%")
print(f"   feature L2 norm = {nn:.3f}  -> percentile vs val = {pct_rank(val_norm, nn):.1f}%")

print("\n--- VERDICT ---")
sep = []
if nm > np.percentile(val_maha, 99): sep.append("Mahalanobis")
if nk > np.percentile(val_knn, 99):  sep.append("kNN-cosine")
if nn > np.percentile(val_norm, 99) or nn < np.percentile(val_norm, 1): sep.append("feature-norm")
if sep:
    print(f"   SEPARABLE on: {', '.join(sep)}  -> a no-retrain OOD gate is VIABLE.")
else:
    print("   NOT separable (normal image sits inside the disease feature distribution)")
    print("   -> a no-retrain feature-OOD gate would NOT reliably catch this. Documenting is the honest call.")
