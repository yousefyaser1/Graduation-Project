# Explainable AI (XAI) Powered Mobile App for Skin Disease Detection

**Project:** Graduation Project — MSA University, Computer Science Engineering  
**Student:** Yousef Yaser

**Repository URL:** https://github.com/yousefyaser1/Graduation-Project

---

## Overview

An offline, on-device mobile application for automated skin disease screening using a three-stage AI pipeline:

1. **VAE (Variational Autoencoder)** — detects whether the image contains anomalous skin
2. **EfficientNet CNN Ensemble (B2 + B3)** — classifies the disease: Acne, Eczema, or Tinea
3. **Score-CAM (XAI Heatmap)** — highlights the skin regions that influenced the diagnosis

All inference runs locally on-device using TensorFlow Lite. No internet connection or server is required.

---

## Step 1 — Install Java Development Kit (JDK 17)

Download and install **JDK 17** from:  
https://www.oracle.com/java/technologies/downloads/#java17

Accept all defaults during installation.

Verify by opening a terminal (Command Prompt or PowerShell) and running:
```
java -version
```
Expected output: `java version "17.x.x"`

---

## Step 2 — Install Android Studio

Download **Android Studio** (latest stable) from:  
https://developer.android.com/studio

During installation:
- Select **Standard** installation type
- Accept all SDK licence agreements

After installation, open Android Studio and wait for the initial SDK download to finish.

### Install Required SDK Components

Go to **File → Settings → Appearance & Behavior → System Settings → Android SDK**

- **SDK Platforms tab** — check: **Android 14.0 (API 34)**
- **SDK Tools tab** — check: **Android SDK Build-Tools**, **Android Emulator**, **Android SDK Platform-Tools**

Click **Apply** and wait for the downloads to complete.

### Create a Virtual Android Device (Emulator)

1. In Android Studio, open **Device Manager** (right-side panel, or **View → Tool Windows → Device Manager**)
2. Click **Create Device**
3. Select **Phone → Pixel 6** → click **Next**
4. Under **System Image**, select **API Level 34 (Android 14)** — download it if prompted
5. Click **Next → Finish**

The virtual device will appear in the Device Manager list.

---

## Step 3 — Install Flutter SDK

Download the Flutter SDK (stable channel) from:  
https://docs.flutter.dev/get-started/install/windows

1. Extract the downloaded `.zip` to `C:\flutter` (avoid paths with spaces)
2. Add Flutter to your system PATH:
   - Open **Start → Search → "Edit the system environment variables"**
   - Click **Environment Variables**
   - Under **User variables**, select **Path** → click **Edit → New**
   - Enter: `C:\flutter\bin`
   - Click **OK** on all dialogs

3. Open a **new** terminal and verify:
```
flutter --version
```
Expected output: `Flutter 3.x.x channel stable`

4. Run the Flutter doctor to validate your setup:
```
flutter doctor
```
You should see green checkmarks next to Flutter, Android toolchain, and Android Studio.

If licences are not yet accepted, run:
```
flutter doctor --android-licenses
```
Type `y` and press Enter for each prompt.

---

## Step 4 — Install Git and Download the Project

### Install Git

Download Git from: https://git-scm.com/download/win

Accept all defaults during installation. Verify by opening a new terminal and running:
```
git --version
```

### Clone the repository

Open a terminal and run:
```
git clone https://github.com/yousefyaser1/Graduation-Project.git
```

This downloads the full project — source code, all four TFLite AI model files, and assets — into a folder called `Graduation-Project` in your current directory.

---

## Step 5 — Run the Application

Navigate into the Flutter app folder:
```
cd Graduation-Project\Flutter
```

Install all Dart/Flutter packages:
```
flutter pub get
```

Start the Android emulator: open Android Studio → **Device Manager** → click the **Play (▶)** button next to the Pixel 6 device. Wait for it to fully boot to the Android home screen (1–2 minutes on first launch).

Confirm Flutter detects the running emulator:
```
flutter devices
```
Expected output includes a line such as:
```
sdk gphone64 x86 64 (mobile) • emulator-5554 • android-x64 • Android 14 (API 34)
```

Build and launch the app:
```
flutter run
```

The first build takes 3–5 minutes. When the terminal shows a `✓ Built` message, the **DermAI** splash screen will appear on the emulator.

---

## Using the Application

| Step | Action |
|------|--------|
| 1 | **Sign Up** — create a patient account (name, email, password). No real email verification is needed. |
| 2 | **Home Screen** — tap **Capture** to use the camera, or tap the **Gallery** icon to upload an image. |
| 3 | **Body Part Selection** — choose the affected area and confirm. |
| 4 | **Analysis** — the app runs the three-stage AI pipeline on-device and shows a live progress screen. |
| 5 | **Results** — view the diagnosis (Acne / Eczema / Tinea / No Disease Detected), confidence score, Score-CAM heatmap, and per-stage inference timing. |
| 6 | **History** — all past scans are saved locally and accessible from the bottom navigation bar. |

> **Uploading images to the emulator:** Drag and drop a `.jpg` file onto the running emulator window, or use **Android Studio → Device Manager → ⋮ → Send files**. The image will then appear in the emulator's gallery.

---

## Project Structure

```
Graduation-Project/
├── Flutter/                              # Mobile application (Flutter/Dart)
│   ├── lib/
│   │   ├── features/                     # UI screens
│   │   │   ├── onboarding/               # Login, signup, role selection
│   │   │   ├── core_workflow/            # Home, capture, history, profile
│   │   │   └── results/                  # Diagnosis result, Score-CAM heatmap
│   │   ├── models/scan_result.dart       # Core data model
│   │   ├── services/
│   │   │   ├── ai/ai_service.dart        # Three-stage AI pipeline
│   │   │   └── database/                 # SQLite scan history
│   │   └── core/                         # Theme, routing, shared widgets
│   ├── assets/models/                    # TFLite model files (on-device inference)
│   │   ├── vae_model.tflite              # Anomaly detector (3.4 MB)
│   │   ├── cnn_b2_model.tflite           # EfficientNetB2 classifier (8.6 MB)
│   │   ├── cnn_b3_model.tflite           # EfficientNetB3 classifier
│   │   └── b3_feature_extractor.tflite   # Score-CAM feature extraction
│   └── pubspec.yaml
│
├── model1/                               # Python model training scripts (Kaggle)
├── thirteenth_code_evaluation_metrics.py # CNN evaluation (confusion matrix, ROC, AUC)
├── fourteenth_code_vae_evaluation.py     # VAE anomaly detection evaluation
├── Chapter4_Implementation.tex           # Thesis Chapter 4 — Implementation
├── Chapter5_Testing.tex                  # Thesis Chapter 5 — Testing & Results
└── README.md                             # This file
```

---

## AI Pipeline Details

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| 1 — Anomaly Detection | VAE (`vae_model.tflite`) | 64×64 image patches | Anomaly ratio; pipeline halts if skin is normal |
| 2 — Classification | EfficientNetB2 + B3 ensemble | 260×260 / 300×300 | Acne / Eczema / Tinea probabilities (50/50 average, 20-step TTA) |
| 3 — Explainability | Score-CAM (`b3_feature_extractor.tflite`) | 300×300 | Heatmap overlay highlighting lesion regions |

**Validation accuracy: 92.62%** on a 325-image held-out split (test set preserved for future fine-tuning).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `flutter: command not found` | Close and reopen the terminal after adding Flutter to PATH |
| `Android licence not accepted` | Run `flutter doctor --android-licenses` and type `y` for each prompt |
| Emulator won't start | Enable virtualisation in BIOS (Intel VT-x / AMD-V), or install **Intel HAXM** via Android Studio SDK Tools |
| `Gradle build failed` | Confirm JDK 17 is installed; set `JAVA_HOME` to the JDK 17 directory |
| App crashes during analysis | Verify all four `.tflite` files are present under `Flutter/assets/models/` |
| Camera unavailable in emulator | Use Gallery upload instead; drag a `.jpg` onto the emulator window |

---

## Software Versions Used

| Tool | Version |
|------|---------|
| Flutter SDK | 3.x stable |
| Dart | Bundled with Flutter |
| Android SDK | API 34 (Android 14) |
| Java JDK | 17 |
| TensorFlow Lite | 0.12.0 (via `tflite_flutter`) |
| Minimum Android version | API 21 (Android 5.0) |
