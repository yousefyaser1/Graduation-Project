"""
Go/no-go: are 11k-Hands skin patches good VAE fine-tuning data for OUR failure?

We need patches that are (a) genuinely normal skin and (b) CURRENTLY mis-flagged by
the shipped VAE (high MSE) — those are exactly the domain the VAE must learn to accept.
If the hand patches are already low-MSE, the VAE accepts them and they won't help.

Reference points (from earlier runs):
  facial new_patches.zip : 51% of patches anomalous (MSE>0.004)
  failing hairy-arm photo : 89% of patches anomalous
A good training set should sit in that high-MSE range.
"""
import os, glob, random, sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from extract_normal_patches import load_rgb, remove_hair, is_skin_patch, tile, PATCH

HANDS = r"C:\Users\A\Downloads\11k_hands_x\Hands"
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Flutter", "assets", "models")
THR = 0.004
N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
MAX_PATCHES = 6000
random.seed(11)

vae = Interpreter(model_path=os.path.join(MODELS, "vae_model.tflite"))
inp_i = vae.get_input_details()[0]["index"]
out_i = vae.get_output_details()[0]["index"]

def vae_mse_batch(patches):
    """patches: (n,64,64,3) float32 [0,1] -> (n,) MSE via batched invoke."""
    vae.resize_tensor_input(inp_i, list(patches.shape)); vae.allocate_tensors()
    vae.set_tensor(inp_i, patches.astype(np.float32)); vae.invoke()
    return vae.get_tensor(out_i).flatten()

files = glob.glob(os.path.join(HANDS, "*.jpg"))
random.shuffle(files); files = files[:N_IMAGES]
print(f"sampling {len(files)} hand images; extracting skin patches (hair-removed, bg-filtered)...", flush=True)

kept, total_tiles = [], 0
for f in files:
    img = load_rgb(f)
    if img is None: continue
    img = remove_hair(img)
    for p in tile(img, PATCH):
        total_tiles += 1
        if is_skin_patch(p):
            kept.append(p.astype(np.float32) / 255.0)
    if len(kept) >= MAX_PATCHES:
        break

kept = np.stack(kept[:MAX_PATCHES])
print(f"skin patches kept: {len(kept)}  ({100*len(kept)/max(1,total_tiles):.0f}% of tiles; rest = white bg)")

mses = np.concatenate([vae_mse_batch(kept[i:i+256]) for i in range(0, len(kept), 256)])
anom = (mses > THR).mean() * 100
print("\n=== VAE on 11k-Hands skin patches ===")
print(f"  mean MSE = {mses.mean():.5f}  median = {np.median(mses):.5f}  "
      f"p90 = {np.percentile(mses,90):.5f}  max = {mses.max():.5f}")
print(f"  anomalous (MSE>{THR}) = {anom:.1f}%")
print("\n  reference: facial patches 51% | hairy-arm photo 89%")
if anom >= 35:
    print("  VERDICT: GOOD — these patches sit in the high-MSE failure region the VAE")
    print("           currently rejects, so fine-tuning on them should teach it to accept")
    print("           this skin domain. Proceed to full extraction + fine-tune.")
else:
    print("  VERDICT: WEAK — the VAE mostly already accepts these (low MSE), so they won't")
    print("           move the needle on the hairy-arm case. Prefer YOUR phone photos.")
