"""
Fine-tune the skin VAE so it reconstructs (i.e. ACCEPTS) real normal skin, fixing the
"normal skin -> ANOMALY -> CNN -> Tinea" failure WITHOUT touching the CNN.

Runs in PyTorch (Kaggle/Colab — there's no torch on the dev machine). It:
  1. rebuilds the exact architecture from vae_tflite_conversion.py,
  2. loads the existing weights from the extracted .pth folder (vae_skin_model/data/*),
  3. fine-tunes on your normal patches (new_patches.zip + extract_normal_patches.py output),
  4. EACH EPOCH validates that disease is still flagged (so we don't trade the
     false-positive problem for missed diseases), and
  5. saves the fine-tuned weights back in the SAME raw layout, so the existing
     vae_tflite_conversion.py converts to TFLite unchanged.

ARCHITECTURE (must match vae_tflite_conversion.py):
  enc: Conv(3->32->64->128, k4 s2 p1, ReLU) -> flatten 8192 -> mu/logvar (32)
  dec: Linear(32->8192) -> reshape(128,8,8) -> ConvT(128->64->32->3, k4 s2 p1),
       ReLU,ReLU,Sigmoid. Inference uses mu (no sampling); anomaly = mean MSE per patch.

WEIGHT FILE ORDER (index -> name -> shape) — identical to vae_tflite_conversion.py:
  0 encoder.0.weight (32,3,4,4)   1 encoder.0.bias (32,)
  2 encoder.2.weight (64,32,4,4)  3 encoder.2.bias (64,)
  4 encoder.4.weight (128,64,4,4) 5 encoder.4.bias (128,)
  6 fc_mu.weight (32,8192)        7 fc_mu.bias (32,)
  8 fc_logvar.weight (32,8192)    9 fc_logvar.bias (32,)
 10 fc_decode.weight (8192,32)   11 fc_decode.bias (8192,)
 12 decoder.0.weight (128,64,4,4)13 decoder.0.bias (64,)
 14 decoder.2.weight (64,32,4,4) 15 decoder.2.bias (32,)
 16 decoder.4.weight (32,3,4,4)  17 decoder.4.bias (3,)

USAGE (Colab/Kaggle):
  python finetune_vae.py \
      --normal-dirs new_patches_x/new_patches normal_patches_domain \
      --init-raw vae_skin_model \
      --disease-dir New_Augmented_Dataset/test \
      --out vae_finetuned --epochs 12 --lr 1e-4 --beta 1e-3
Then point vae_tflite_conversion.py's PTH_FOLDER at  vae_finetuned/  and run it.

GOAL when reading the per-epoch table: normal pass% UP toward ~100, while disease
caught% stays near its baseline (don't let it fall much — that's missed disease).
"""
import os, glob, argparse
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit("PyTorch not found. Run this in Kaggle/Colab (pip install torch).")

try:
    import cv2
    def imread_rgb(p):
        b = cv2.imread(p); return None if b is None else cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    def resize_w(img, W):
        h = int(img.shape[0] * W / img.shape[1]); return cv2.resize(img, (W, h), interpolation=cv2.INTER_LINEAR)
except ImportError:
    from PIL import Image
    def imread_rgb(p):
        try: return np.asarray(Image.open(p).convert("RGB"))
        except Exception: return None
    def resize_w(img, W):
        from PIL import Image as I
        im = I.fromarray(img); h = int(img.shape[0] * W / img.shape[1])
        return np.asarray(im.resize((W, h), I.BILINEAR))

PATCH, MAX_W, GATE = 64, 1280, 0.20
WEIGHT_SPEC = [  # (idx, name, shape)
    (0,  "encoder.0.weight", (32, 3, 4, 4)),   (1,  "encoder.0.bias", (32,)),
    (2,  "encoder.2.weight", (64, 32, 4, 4)),  (3,  "encoder.2.bias", (64,)),
    (4,  "encoder.4.weight", (128, 64, 4, 4)), (5,  "encoder.4.bias", (128,)),
    (6,  "fc_mu.weight", (32, 8192)),          (7,  "fc_mu.bias", (32,)),
    (8,  "fc_logvar.weight", (32, 8192)),      (9,  "fc_logvar.bias", (32,)),
    (10, "fc_decode.weight", (8192, 32)),      (11, "fc_decode.bias", (8192,)),
    (12, "decoder.0.weight", (128, 64, 4, 4)), (13, "decoder.0.bias", (64,)),
    (14, "decoder.2.weight", (64, 32, 4, 4)),  (15, "decoder.2.bias", (32,)),
    (16, "decoder.4.weight", (32, 3, 4, 4)),   (17, "decoder.4.bias", (3,)),
]


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(8192, 32)
        self.fc_logvar = nn.Linear(8192, 32)
        self.fc_decode = nn.Linear(32, 8192)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 128, 8, 8))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def patch_mse(self, x):                      # deterministic (mu) — matches inference
        recon, _, _ = self.forward(x, sample=False)
        return ((recon - x) ** 2).mean(dim=[1, 2, 3])


def load_raw(model, folder):
    sd = model.state_dict()
    for idx, name, shape in WEIGHT_SPEC:
        raw = np.frombuffer(open(os.path.join(folder, "data", str(idx)), "rb").read(), "<f4")
        sd[name] = torch.tensor(raw.reshape(shape).copy())
    model.load_state_dict(sd)
    print(f"loaded existing weights from {folder}/data/*")


def save_raw(model, folder):
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)
    sd = model.state_dict()
    for idx, name, _ in WEIGHT_SPEC:
        sd[name].detach().cpu().numpy().astype("<f4").tofile(os.path.join(folder, "data", str(idx)))
    torch.save(model.state_dict(), os.path.join(folder, "vae_finetuned_state.pth"))
    print(f"saved fine-tuned weights -> {folder}/data/*  (+ vae_finetuned_state.pth)")
    print(f"  Now set PTH_FOLDER = r'{os.path.abspath(folder)}' in vae_tflite_conversion.py and run it.")


class PatchDS(Dataset):
    def __init__(self, dirs):
        self.files = []
        for d in dirs:
            for e in ("*.png", "*.jpg", "*.jpeg"):
                self.files += glob.glob(os.path.join(d, e))
        if not self.files:
            raise SystemExit(f"No patches found in {dirs}")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        a = imread_rgb(self.files[i])
        if a.shape[0] != PATCH or a.shape[1] != PATCH:
            a = a[:PATCH, :PATCH]  # safety; patches should already be 64x64
        return torch.tensor(a.astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def disease_caught(model, disease_dir, thr, n_per_class, device):
    """Fraction of disease images still flagged ANOMALY (ratio>GATE). Higher = safer."""
    model.eval(); out = {}
    for cls in sorted(os.listdir(disease_dir)):
        cdir = os.path.join(disease_dir, cls)
        if not os.path.isdir(cdir): continue
        fs = []
        for e in ("*.jpg", "*.jpeg", "*.png"): fs += glob.glob(os.path.join(cdir, e))
        fs = fs[:n_per_class]; caught = 0
        for f in fs:
            img = imread_rgb(f)
            if img is None: continue
            if img.shape[1] > MAX_W: img = resize_w(img, MAX_W)
            h, w, _ = img.shape; patches = []
            for y in range(0, h - PATCH + 1, PATCH):
                for x in range(0, w - PATCH + 1, PATCH):
                    patches.append(img[y:y+PATCH, x:x+PATCH])
            if not patches: continue
            t = torch.tensor(np.stack(patches).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
            mse = model.patch_mse(t).cpu().numpy()
            if (mse > thr).mean() > GATE: caught += 1
        out[cls] = 100.0 * caught / max(1, len(fs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-dirs", nargs="+", required=True, help="folders of 64x64 normal patches")
    ap.add_argument("--init-raw", default="vae_skin_model", help="folder with data/<idx> existing weights")
    ap.add_argument("--init-state", default=None, help="alt: a .pth state_dict to start from")
    ap.add_argument("--disease-dir", default=None, help="e.g. New_Augmented_Dataset/test (for validation)")
    ap.add_argument("--out", default="vae_finetuned")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1e-3, help="KL weight (small: prioritise reconstruction)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.004, help="per-patch MSE anomaly threshold (== app)")
    ap.add_argument("--val-disease-n", type=int, default=40)
    ap.add_argument("--disease-drop", type=float, default=3.0,
                    help="max allowed drop (pts) in mean disease-caught vs baseline when picking best epoch")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE().to(device)
    if args.init_state:
        model.load_state_dict(torch.load(args.init_state, map_location=device)); print(f"loaded {args.init_state}")
    else:
        load_raw(model, args.init_raw)

    ds = PatchDS(args.normal_dirs)
    n_val = max(1, int(0.1 * len(ds)))
    val_idx = set(range(len(ds))[::max(1, len(ds)//n_val)][:n_val])
    train = torch.utils.data.Subset(ds, [i for i in range(len(ds)) if i not in val_idx])
    valn  = torch.utils.data.Subset(ds, sorted(val_idx))
    dl = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    print(f"normal patches: {len(ds)} ({len(train)} train / {len(valn)} val)  device={device}")

    @torch.no_grad()
    def normal_pass(m):
        m.eval(); below = tot = 0
        for i in range(0, len(valn), args.batch):
            xb = torch.stack([valn[j] for j in range(i, min(i+args.batch, len(valn)))]).to(device)
            mse = m.patch_mse(xb).cpu().numpy(); below += int((mse <= args.thr).sum()); tot += len(mse)
        return 100.0 * below / max(1, tot)

    def report(tag, m):
        npass = normal_pass(m)
        dc = disease_caught(m, args.disease_dir, args.thr, args.val_disease_n, device) if args.disease_dir else {}
        ds_str = "  ".join(f"{k}={v:.0f}%" for k, v in dc.items()) if dc else "(no --disease-dir)"
        dc_mean = (sum(dc.values()) / len(dc)) if dc else float("nan")
        print(f"  [{tag}] normal-pass={npass:5.1f}%  | disease-caught: {ds_str}"
              + (f"  (mean {dc_mean:.0f}%)" if dc else ""))
        return npass, dc_mean

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("\nbaseline (before fine-tuning):")
    _, base_dc = report("ep0", model)
    floor = (base_dc - args.disease_drop) if base_dc == base_dc else -1.0  # NaN-safe
    print(f"\ntraining — goal: normal-pass UP while disease-caught stays >= {floor:.0f}% "
          f"(baseline {base_dc:.0f}% - {args.disease_drop}):")

    # Save EACH epoch to its own folder so no checkpoint is lost, and auto-track
    # the best: highest normal-pass among epochs whose disease-caught held the floor.
    best = {"ep": 0, "npass": -1.0}
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for xb in dl:
            xb = xb.to(device)
            recon, mu, logvar = model(xb, sample=True)
            rec = F.mse_loss(recon, xb, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + args.beta * kl
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep:2d}  loss={tot/len(dl):.5f}", end="")
        npass, dc_mean = report(f"ep{ep}", model)
        ep_dir = os.path.join(args.out, f"ep{ep}")
        save_raw(model, ep_dir)
        if (dc_mean != dc_mean or dc_mean >= floor) and npass > best["npass"]:
            best = {"ep": ep, "npass": npass, "dc": dc_mean, "dir": ep_dir}

    print("\n" + "=" * 64)
    if best["ep"]:
        print(f"RECOMMENDED: epoch {best['ep']}  (normal-pass {best['npass']:.0f}%, "
              f"disease-caught {best.get('dc', float('nan')):.0f}%)  ->  {best['dir']}")
        print(f"Set PTH_FOLDER = r'{os.path.abspath(best['dir'])}' in vae_tflite_conversion.py,")
    else:
        print("No epoch kept disease-caught above the floor — fine-tuning can't fix it without")
        print("hurting disease detection. Try more/cleaner normal data, or drop _removeHair and")
        print("retrain on raw hairy patches. Pick an epoch folder manually if you still want one.")
    print("run it to export vae_model.tflite, and copy that into Flutter/assets/models/.")
    print("(Every epoch is saved under", args.out + "/ep<N>/ — you can choose a different one.)")


if __name__ == "__main__":
    main()
