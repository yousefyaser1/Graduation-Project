"""
Train/test leakage audit for New_Augmented_Dataset.

If augmentation was applied BEFORE the train/val/test split, near-duplicate
copies of the same source image land in different splits — which inflates the
reported test accuracy and means the model generalises worse than the numbers
suggest. This script detects that.

Two detectors:
  1. EXACT duplicates  — identical file bytes (MD5) across splits.
  2. NEAR duplicates   — perceptual dHash (64-bit) within Hamming distance
                         <= THRESH. Catches augmented variants (flip, rotate,
                         brightness, crop) of the same source image.

Reports, per class, how many TEST images have a near-duplicate in TRAIN.
A high percentage = leakage = the 90%+ accuracy is optimistic.

Run:  python leakage_audit.py [hamming_thresh]   (default 6)
"""
import os, sys, glob, hashlib
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "New_Augmented_Dataset")
CLEAN = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "Finalized_Clean_Data")
CLASSES = ["acne", "eczema", "tinea"]
THRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def files(base, split, cls):
    return sorted(glob.glob(os.path.join(base, split, cls, "*")))


def dhash(path, size=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]            # 64 booleans
    bits = np.packbits(diff.flatten())
    return int.from_bytes(bits.tobytes(), "big")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def hamming_min(query, ref_arr):
    """Min Hamming distance of uint64 `query` to any uint64 in ref_arr."""
    x = np.bitwise_xor(ref_arr, np.uint64(query))
    # popcount on uint64 via view as uint8
    pc = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(axis=1)
    return int(pc.min()) if len(pc) else 64


def audit(base, label):
    print(f"\n===== {label}  ({base}) =====")
    if not os.path.isdir(base):
        print("  MISSING"); return
    # exact dupes across splits (per class)
    print("-- exact (MD5) duplicates train<->test --")
    for cls in CLASSES:
        tr = {md5(f): f for f in files(base, "train", cls)}
        dups = [f for f in files(base, "test", cls) if md5(f) in tr]
        print(f"  {cls}: {len(dups)} exact test images also in train")

    # near dupes
    print(f"-- near-duplicate (dHash, Hamming<= {THRESH}) test-in-train --")
    for cls in CLASSES:
        tr_hashes = [h for h in (dhash(f) for f in files(base, "train", cls)) if h is not None]
        tr_arr = np.array(tr_hashes, dtype=np.uint64)
        te = files(base, "test", cls)
        leak = 0
        for f in te:
            h = dhash(f)
            if h is not None and hamming_min(h, tr_arr) <= THRESH:
                leak += 1
        pct = 100.0 * leak / len(te) if te else 0.0
        print(f"  {cls}: {leak}/{len(te)} test images ({pct:.1f}%) have a near-dup in train")


if __name__ == "__main__":
    audit(DATA, "New_Augmented_Dataset")
    audit(CLEAN, "Finalized_Clean_Data")
