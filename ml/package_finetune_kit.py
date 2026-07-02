"""Bundle the COMPLETE VAE fine-tune kit into one zip (single-pass, STORED = fast).
Pulls every normal-patch source on disk, including the user's own phone photos.

Inside the archive:
  vae_finetune_kit/
    finetune_vae.py, vae_tflite_conversion.py, RUN.txt
    normal/facial/   (Downloads/new_patches_x/new_patches)
    normal/hands/    (normal_patches_domain/hands)
    normal/mendeley/ (normal_patches_domain/mendeley)
    normal/phone/    (normal_patches_domain/phone  <- user's own skin/camera)
    vae_skin_model/data/0..17  (original weights for --init-raw)
    disease_val/{acne,eczema,tinea}/  (N/class for the disease-caught safety metric)
"""
import os, glob, random, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
NPD = os.path.join(ROOT, "normal_patches_domain")
NORMAL_SOURCES = [   # (arc_name, source_dir)
    ("facial",   r"C:\Users\A\Downloads\new_patches_x\new_patches"),
    ("hands",    os.path.join(NPD, "hands")),
    ("mendeley", os.path.join(NPD, "mendeley")),
    ("phone",    os.path.join(NPD, "phone")),
]
WEIGHTS = r"C:\Users\A\Downloads\vae_skin_model"
DISEASE = os.path.join(ROOT, "New_Augmented_Dataset", "test")
SCRIPT = os.path.join(ROOT, "finetune_vae.py")
CONV = os.path.join(ROOT, "vae_tflite_conversion.py")
OUT = os.path.join(ROOT, "vae_finetune_kit_full.zip")
DIS_PER_CLASS = 50
random.seed(0)

RUN = r"""COLAB / KAGGLE — complete VAE fine-tune (GPU runtime).

# Cell 1 (Colab): upload this kit
from google.colab import files; files.upload()            # vae_finetune_kit_full.zip

# Cell 2: setup + fine-tune (prints baseline + per-epoch table, then RECOMMENDED epoch)
!pip -q install torch opencv-python-headless numpy
!unzip -o -q vae_finetune_kit_full.zip
%cd vae_finetune_kit
!python finetune_vae.py \
    --normal-dirs normal/facial normal/hands normal/mendeley normal/phone \
    --init-raw vae_skin_model \
    --disease-dir disease_val \
    --out vae_finetuned --epochs 12 --lr 1e-4 --beta 1e-3
# normal-pass% should rise while disease-caught% holds near the ep0 baseline.

# Cell 3: export the chosen epoch to TFLite + download
BEST = 8                                                   # <- RECOMMENDED epoch from Cell 2
import vae_tflite_conversion as V
m = V.build_tf_inference_model()
V.transfer_weights(m, f"vae_finetuned/ep{BEST}")
V.convert_to_tflite(m, "vae_model.tflite")
from google.colab import files; files.download("vae_model.tflite")
# Then: copy vae_model.tflite into Flutter/assets/models/ and `flutter run`.

# KAGGLE note: upload the unzipped folders as a Dataset, set --out /kaggle/working/...
"""

def add_dir(z, src, arc, pattern="*.png"):
    if not os.path.isdir(src):
        print(f"  WARN: missing {arc} dir {src} — skipped", flush=True); return 0
    n = 0
    for f in glob.glob(os.path.join(src, pattern)):
        z.write(f, f"vae_finetune_kit/normal/{arc}/{os.path.basename(f)}"); n += 1
    return n

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    z.write(SCRIPT, "vae_finetune_kit/finetune_vae.py")
    z.write(CONV, "vae_finetune_kit/vae_tflite_conversion.py")
    z.writestr("vae_finetune_kit/RUN.txt", RUN)
    total = 0
    for arc, src in NORMAL_SOURCES:
        c = add_dir(z, src, arc); print(f"{arc}: {c}", flush=True); total += c
    for i in range(18):
        z.write(os.path.join(WEIGHTS, "data", str(i)), f"vae_finetune_kit/vae_skin_model/data/{i}")
    nd = 0
    for cls in ("acne", "eczema", "tinea"):
        fs = []
        for e in ("*.jpg", "*.jpeg", "*.png"):
            fs += glob.glob(os.path.join(DISEASE, cls, e))
        random.shuffle(fs)
        for f in fs[:DIS_PER_CLASS]:
            z.write(f, f"vae_finetune_kit/disease_val/{cls}/{os.path.basename(f)}"); nd += 1
    print(f"disease_val: {nd}  | normal total: {total}", flush=True)

print(f"\nDONE -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
