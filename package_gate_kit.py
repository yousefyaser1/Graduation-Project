"""
Stage + resize the Normal-vs-Disease gate training data into one folder, then
zip it for Kaggle. Re-runnable: pass --user-photos to fold in freshly taken
phone normals when they are ready.

Layout produced (all images downscaled to <=256px to keep the zip small):
  gate_kit/
    train_normal_gate.py
    New_Augmented_Dataset/{train,val}/{acne,eczema,tinea}/   (disease)
    normal_hands/                                            (sampled 11k-hands)
    normal_phone/                                            (your own phone shots)
    normal_user/                                             (new phone photos, optional)
"""
import argparse
import os
import shutil
import zipfile
from PIL import Image

REPO = r"C:\Users\A\Graduation-Project-AMA"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAXPX = 256


def resize_into(src_files, dst_dir, limit=None):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for i, src in enumerate(src_files):
        if limit and n >= limit:
            break
        try:
            im = Image.open(src).convert("RGB")
            im.thumbnail((MAXPX, MAXPX), Image.LANCZOS)
            im.save(os.path.join(dst_dir, f"{n:06d}.jpg"), "JPEG", quality=88)
            n += 1
        except Exception:
            continue
    return n


def walk_imgs(root, skip_reject=True):
    out = []
    for dp, _, fns in os.walk(root):
        low = dp.lower()
        if skip_reject and ("_rejected" in low or "_pruned" in low):
            continue
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-photos", default=None,
                    help="folder of freshly taken normal phone photos")
    ap.add_argument("--hands-cap", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(REPO, "gate_kit"))
    ap.add_argument("--zip", default=os.path.join(REPO, "gate_kit.zip"))
    args = ap.parse_args()

    stage = args.out
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    # 1. disease — copy/resize the New_Augmented_Dataset train+val, preserving the
    #    acne/eczema/tinea subfolders the trainer expects.
    src_ds = os.path.join(REPO, "New_Augmented_Dataset")
    for split in ("train", "val"):
        for cls in ("acne", "eczema", "tinea"):
            s = os.path.join(src_ds, split, cls)
            if os.path.isdir(s):
                d = os.path.join(stage, "New_Augmented_Dataset", split, cls)
                n = resize_into(walk_imgs(s), d)
                print(f"disease {split}/{cls}: {n}")

    # 2. normal — hands (sampled), phone, and optional new user photos.
    import random
    random.seed(42)
    hands = walk_imgs(os.path.join(REPO, "normal_datasets", "11k_hands"))
    random.shuffle(hands)
    print("normal_hands:", resize_into(hands, os.path.join(stage, "normal_hands"),
                                       limit=args.hands_cap))

    # The user's own phone shots are now a HELD-OUT test set (never trained on)
    # so the per-source / held-out pass-rate is an honest reliability gauge.
    # Trainer command: pass gate_kit/normal_phone to --test-dirs, not --normal-dirs.
    phone = walk_imgs(r"C:\Users\A\Downloads\manually_taken_normal_skin")
    print("normal_phone (HELD-OUT test):",
          resize_into(phone, os.path.join(stage, "normal_phone")))

    if args.user_photos and os.path.isdir(args.user_photos):
        u = walk_imgs(args.user_photos)
        print("normal_user:", resize_into(u, os.path.join(stage, "normal_user")))

    # 3. trainer
    shutil.copy(os.path.join(REPO, "train_normal_gate.py"),
                os.path.join(stage, "train_normal_gate.py"))

    # 4. zip
    if os.path.exists(args.zip):
        os.remove(args.zip)
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fns in os.walk(stage):
            for fn in fns:
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, os.path.dirname(stage)))
    mb = os.path.getsize(args.zip) / 1e6
    print(f"\nWrote {args.zip} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
