"""
Extract 64x64 normal-skin patches from real phone photos, in the SAME
distribution the on-device VAE sees at inference. Output feeds finetune_vae.py.

Why this exists: the VAE flags real normal skin (esp. hairy limbs) as anomalous
because it was never trained on that domain. new_patches.zip is clean FACIAL skin
and doesn't cover it. Point this script at phone photos of YOUR normal skin
(hairy arms/legs, varied lighting/body parts) to build the missing domain patches.

Mirrors ai_service.dart exactly:
  * downsample to <=1280 px wide (linear)  -> matches `original`
  * DullRazor hair removal (default ON)     -> matches `_removeHair` before the VAE
  * 64x64 tiling at stride 64               -> matches the VAE sliding window grid
  * output PNGs are plain [0,255] RGB; finetune_vae.py normalises to [0,1]

Hair-removal note: the device hair-removes BEFORE the VAE, so we do too by default
(train == inference). If you instead plan to REMOVE the hair-removal step in the app
and let the VAE learn hair directly, pass --no-hair-removal here AND drop the
_removeHair call in ai_service.dart so the two stay consistent.

Skin filter: phone photos contain background (walls, furniture). Training the VAE on
those teaches it to call background "normal". A lenient warm-foreground filter drops
obvious non-skin patches; rejects are saved to <out>/_rejected for you to eyeball.
It is intentionally lenient to avoid dropping dark skin tones — review the output and
pass --no-skin-filter if it's mis-dropping.

Usage:
  python extract_normal_patches.py --input "C:/path/to/normal_photos" --out normal_patches_domain
  python extract_normal_patches.py --input normal_test.jpg --out normal_patches_domain
"""
import os, sys, glob, argparse
import numpy as np
import cv2

PATCH, MAX_W = 64, 1280
HAIR_RADIUS, HAIR_DARKNESS = 3, 0.16   # == _hairRadius / _hairDarkness in ai_service.dart


def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[1] > MAX_W:
        h = int(img.shape[0] * MAX_W / img.shape[1])
        img = cv2.resize(img, (MAX_W, h), interpolation=cv2.INTER_LINEAR)
    return img


def remove_hair(rgb_u8, radius=HAIR_RADIUS, darkness=HAIR_DARKNESS):
    """Vectorised DullRazor matching ai_service.dart `_removeHair`.

    A pixel is hair when the local-max luminance in a (2r+1)^2 window exceeds it by
    >= `darkness` (relative, [0,1]). Hair pixels are replaced by the mean of their
    non-hair neighbours in the same window.
    """
    rgb = rgb_u8.astype(np.float32) / 255.0
    lum = rgb.mean(axis=2)
    k = 2 * radius + 1
    kernel = np.ones((k, k), np.uint8)
    maxlum = cv2.dilate(lum, kernel)                       # local max luminance
    hair = (maxlum - lum) >= darkness                      # bool HxW
    nonhair = (~hair).astype(np.float32)
    den = cv2.boxFilter(nonhair, -1, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)
    out = rgb.copy()
    for c in range(3):
        num = cv2.boxFilter(rgb[:, :, c] * nonhair, -1, (k, k),
                            normalize=False, borderType=cv2.BORDER_REFLECT)
        fill = np.where(den > 0, num / np.maximum(den, 1e-6), rgb[:, :, c])
        out[:, :, c] = np.where(hair, fill, rgb[:, :, c])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def is_skin_patch(patch_u8, min_frac=0.80):
    """Per-PIXEL warm-foreground coverage test (lenient on tone, strict on coverage).

    A pixel is skin-like when it is warm (R>=B), not near-black, not blown out, and
    not flat bright grey. We keep the patch only if >= `min_frac` of its pixels pass
    — this drops white-background-edge slivers and mostly-background patches that a
    mean-colour test would wrongly accept. Still tone-agnostic (warm holds across
    skin tones), so it does not reject dark skin."""
    p = patch_u8.astype(np.float32) / 255.0
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    v = (r + g + b) / 3.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    skin = (v > 0.12) & (v < 0.97) & (r >= b - 0.02) & ~((sat < 0.05) & (v > 0.6))
    return float(skin.mean()) >= min_frac


def tile(img, stride):
    h, w, _ = img.shape
    for y in range(0, h - PATCH + 1, stride):
        for x in range(0, w - PATCH + 1, stride):
            yield img[y:y + PATCH, x:x + PATCH]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file, folder, or glob")
    ap.add_argument("--out", default="normal_patches_domain")
    ap.add_argument("--stride", type=int, default=PATCH)
    ap.add_argument("--hair-removal", dest="hair", action="store_true", default=True)
    ap.add_argument("--no-hair-removal", dest="hair", action="store_false")
    ap.add_argument("--skin-filter", dest="skin", action="store_true", default=True)
    ap.add_argument("--no-skin-filter", dest="skin", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="random sample when used with --limit")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = []
        for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp"):
            files += glob.glob(os.path.join(args.input, e))
    elif any(ch in args.input for ch in "*?["):
        files = glob.glob(args.input)
    else:
        files = [args.input]
    if not files:
        sys.exit(f"No images found at: {args.input}")
    if args.shuffle:
        import random as _r; _r.seed(0); _r.shuffle(files)
    if args.limit > 0:
        files = files[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    rej_dir = os.path.join(args.out, "_rejected")
    if args.skin:
        os.makedirs(rej_dir, exist_ok=True)

    kept = dropped = 0
    for f in sorted(files):
        img = load_rgb(f)
        if img is None:
            print(f"  skip (unreadable): {f}"); continue
        if args.hair:
            img = remove_hair(img)
        stem = os.path.splitext(os.path.basename(f))[0]
        k = d = 0
        for i, patch in enumerate(tile(img, args.stride)):
            keep = (not args.skin) or is_skin_patch(patch)
            dst = os.path.join(args.out if keep else rej_dir, f"{stem}_p{i:04d}.png")
            cv2.imwrite(dst, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            if keep: k += 1
            else:    d += 1
        kept += k; dropped += d
        print(f"  {os.path.basename(f)}: kept {k}, dropped {d}")

    print(f"\nDONE. {len(files)} image(s) -> {kept} patches in '{args.out}'"
          + (f"  ({dropped} background patches in _rejected/ — review them)" if args.skin else ""))
    print("Next: combine these with the new_patches.zip patches and run finetune_vae.py.")


if __name__ == "__main__":
    main()
