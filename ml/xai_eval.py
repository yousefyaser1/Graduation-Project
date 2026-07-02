"""
Quantitative Score-CAM faithfulness evaluation on the test set.

This answers "can we trust the heatmap?" with numbers, using ONLY images + the
shipped models (no lesion annotations needed). It reuses score_cam.py's
run_score_cam so the heatmap evaluated here is byte-for-byte the one the app
produces on device.

Metrics (all on the predicted class, [0,255] / 300x300 B3 space):

  Average Drop (%)      mean( max(0, p_full - p_masked) / p_full ) * 100
                        Mask the image to its salient region (img * heatmap).
                        LOWER is better — the highlighted region alone should
                        still support the prediction.

  Increase (%)          % of images where p_masked > p_full.
                        HIGHER is better (Score-CAM "Increase in Confidence").

  Deletion AUC          Progressively zero the most-salient pixels; AUC of
                        target prob vs fraction removed. LOWER is better — prob
                        should collapse quickly when the important pixels go.

  Insertion AUC         Start from a blurred image; progressively reveal the
                        most-salient pixels; AUC of target prob. HIGHER is
                        better — the salient pixels alone should rebuild the
                        prediction.

Sanity baseline: the SAME metrics for a RANDOM heatmap. A real explanation must
beat random (lower Deletion AUC, higher Insertion AUC); if it doesn't, the map
is not faithful. This is the explainability analogue of a sanity check.

Usage:
  python xai_eval.py [--per-class N] [--steps S] [--seed K]
    --per-class  images sampled per class (default 20; use 0 for the full set)
    --steps      deletion/insertion steps (default 20)
"""
import os
import sys
import argparse
import numpy as np
import cv2

# Reuse the parity implementation so we evaluate the EXACT shipped heatmap.
from score_cam import (
    run_score_cam, load_rgb, _run, B3, CLASSES, B3_W, B3_H,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "New_Augmented_Dataset", "test")
# lowercase folder -> canonical class (alphabetical == training order)
FOLDERS = {"acne": "Acne", "eczema": "Eczema", "tinea": "Tinea"}


def _b3_probs(img_b3):
    """img_b3: (300,300,3) float32 [0,255] -> (3,) probs."""
    return np.asarray(_run(B3, img_b3[np.newaxis]), dtype=np.float64)


def avg_drop_increase(img_b3, heatmap, pred, p_full):
    """Score-CAM masking metrics on the predicted class."""
    masked = img_b3 * heatmap[:, :, np.newaxis]          # keep salient region
    p_masked = _b3_probs(masked)[pred]
    drop = max(0.0, p_full - p_masked) / (p_full + 1e-12)
    inc = 1.0 if p_masked > p_full else 0.0
    return drop, inc


def deletion_insertion_auc(img_b3, heatmap, pred, steps):
    """
    Deletion: zero the most-salient pixels first.
    Insertion: reveal the most-salient pixels first, over a blurred baseline.
    Returns (deletion_auc, insertion_auc), both prob-vs-fraction trapezoid AUCs.
    """
    order = np.argsort(heatmap.ravel())[::-1]            # most salient first
    n = order.size
    blur = cv2.GaussianBlur(img_b3, (0, 0), sigmaX=11)   # insertion baseline

    flat_img = img_b3.reshape(-1, 3)
    flat_blur = blur.reshape(-1, 3)

    del_canvas = flat_img.copy()
    ins_canvas = flat_blur.copy()
    del_probs = [p_full_cache[0]]                         # fraction 0 == full image
    ins_probs = [_b3_probs(blur.reshape(B3_H, B3_W, 3))[pred]]  # fraction 0 == blurred

    chunk = max(1, n // steps)
    for s in range(steps):
        idx = order[s * chunk:(s + 1) * chunk]
        del_canvas[idx] = 0.0                            # remove salient pixels
        ins_canvas[idx] = flat_img[idx]                  # insert salient pixels
        del_probs.append(_b3_probs(del_canvas.reshape(B3_H, B3_W, 3))[pred])
        ins_probs.append(_b3_probs(ins_canvas.reshape(B3_H, B3_W, 3))[pred])

    xs = np.linspace(0, 1, len(del_probs))
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x renamed trapz
    return float(_trap(del_probs, xs)), float(_trap(ins_probs, xs))


# small cache so deletion fraction-0 point reuses the unmasked prob
p_full_cache = [0.0]


def sample_paths(per_class, rng):
    paths = []
    for folder, cls in FOLDERS.items():
        d = os.path.join(TEST_DIR, folder)
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if per_class > 0 and len(files) > per_class:
            files = [files[i] for i in rng.choice(len(files), per_class, replace=False)]
        paths += [(p, cls) for p in files]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(TEST_DIR):
        sys.exit(f"test dir not found: {TEST_DIR}")

    rng = np.random.default_rng(args.seed)
    paths = sample_paths(args.per_class, rng)
    print(f"Evaluating Score-CAM faithfulness on {len(paths)} images "
          f"({args.per_class or 'all'}/class, {args.steps} steps)\n")

    agg = {c: [] for c in CLASSES}        # per-true-class metric rows
    rand_del, rand_ins = [], []           # random-heatmap baseline
    real_del, real_ins = [], []
    drops, incs = [], []

    for k, (path, true_cls) in enumerate(paths, 1):
        img = load_rgb(path)
        img_b3 = cv2.resize(img, (B3_W, B3_H),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32)
        probs = _b3_probs(img_b3)
        pred = int(np.argmax(probs))
        p_full = probs[pred]
        p_full_cache[0] = p_full

        heatmaps, _ = run_score_cam(img)
        hm = heatmaps[CLASSES[pred]]

        drop, inc = avg_drop_increase(img_b3, hm, pred, p_full)
        d_auc, i_auc = deletion_insertion_auc(img_b3, hm, pred, args.steps)

        rng_hm = rng.random((B3_H, B3_W)).astype(np.float32)
        rd_auc, ri_auc = deletion_insertion_auc(img_b3, rng_hm, pred, args.steps)

        drops.append(drop); incs.append(inc)
        real_del.append(d_auc); real_ins.append(i_auc)
        rand_del.append(rd_auc); rand_ins.append(ri_auc)
        agg[true_cls].append((drop, inc, d_auc, i_auc))

        print(f"  [{k}/{len(paths)}] {os.path.basename(path):<28} "
              f"true={true_cls:<7} pred={CLASSES[pred]:<7} "
              f"drop={drop*100:5.1f}% del={d_auc:.3f} ins={i_auc:.3f}")

    def m(x):
        return float(np.mean(x)) if x else 0.0

    print("\n" + "=" * 64)
    print("PER-CLASS (true label)")
    print(f"{'class':<8} {'n':>3}  {'AvgDrop':>8} {'Increase':>9} "
          f"{'DelAUC':>7} {'InsAUC':>7}")
    for c in CLASSES:
        rows = agg[c]
        if not rows:
            continue
        dr, ic, da, ia = (np.mean([r[i] for r in rows]) for i in range(4))
        print(f"{c:<8} {len(rows):>3}  {dr*100:7.1f}% {ic*100:8.1f}% "
              f"{da:7.3f} {ia:7.3f}")

    print("\nOVERALL")
    print(f"  Average Drop     : {m(drops)*100:.1f}%   (lower better)")
    print(f"  Increase in Conf : {m(incs)*100:.1f}%   (higher better)")
    print(f"  Deletion AUC     : {m(real_del):.3f}   (lower better)")
    print(f"  Insertion AUC    : {m(real_ins):.3f}   (higher better)")

    print("\nRANDOM-HEATMAP BASELINE (sanity)")
    print(f"  Deletion AUC     : {m(rand_del):.3f}")
    print(f"  Insertion AUC    : {m(rand_ins):.3f}")
    del_ok = m(real_del) < m(rand_del)
    ins_ok = m(real_ins) > m(rand_ins)
    verdict = "PASS" if (del_ok and ins_ok) else "FAIL"
    print(f"\n  Score-CAM beats random?  Deletion {'Y' if del_ok else 'N'} | "
          f"Insertion {'Y' if ins_ok else 'N'}  ->  {verdict}")
    if verdict == "PASS":
        print("  The heatmap is faithful: salient pixels genuinely drive the "
              "prediction, far more than random pixels do.")
    else:
        print("  WARNING: explanation not clearly better than random — "
              "investigate before trusting the heatmaps.")


if __name__ == "__main__":
    main()
