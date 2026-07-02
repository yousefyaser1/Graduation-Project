Explainable AI (XAI) Powered Mobile App for Skin Disease Detection
===================================================================

Project: Graduation Project - MSA University, Computer Science Engineering
Student: Yousef Yaser

Repository URL: https://github.com/yousefyaser1/Graduation-Project

-------------------------------------------------------------------
Overview
-------------------------------------------------------------------

An offline, on-device mobile application for automated skin disease
screening using a three-stage AI pipeline:

  1. VAE (Variational Autoencoder)        - detects whether the image contains anomalous skin
  2. EfficientNet CNN Ensemble (B2 + B3)  - classifies the disease: Acne, Eczema, or Tinea
  3. Score-CAM (XAI Heatmap)             - highlights the skin regions that influenced the diagnosis

All inference runs locally on-device using TensorFlow Lite.
No internet connection or server is required.

-------------------------------------------------------------------
Choose Your Scenario
-------------------------------------------------------------------

Before following any steps, identify which scenario applies to you:

  SCENARIO A - You have an Android phone and a Windows PC
    Connect it to your PC via USB cable and follow Section A below.

  SCENARIO B - You have an iPhone and a Mac
    You can run the app on your iPhone from your Mac.
    Follow Section B below.

  SCENARIO C - You have an iPhone and a Windows PC
    Apple does not allow building iOS apps on Windows.
    You will need to use an Android emulator on your PC instead.
    Follow Section C below.

  SCENARIO D - You have an Android phone and a Mac
    Connect it to your Mac via USB cable and follow Section D below.

===================================================================
SCENARIO A - Android Phone via USB (Windows PC)
===================================================================

This is the simplest setup. No emulator needed.

-------------------------------------------------------------------
A1 - Install Java Development Kit (JDK 17)
-------------------------------------------------------------------

Download and install JDK 17 from:
https://www.oracle.com/java/technologies/downloads/#java17

Accept all defaults during installation.

Verify by opening a terminal (Command Prompt or PowerShell):

    java -version

Expected output: java version "17.x.x"

-------------------------------------------------------------------
A2 - Install Android SDK Command-Line Tools
-------------------------------------------------------------------

Android Studio is NOT required. You only need the Android SDK
command-line tools to build the app.

1. Go to: https://developer.android.com/studio
2. Scroll down to the section "Command line tools only"
3. Download the Windows ZIP (commandlinetools-win-XXXXXXX_latest.zip)
4. Create this folder on your PC:

       C:\android-sdk\cmdline-tools\latest\

5. Extract the ZIP contents into that folder so the structure is:

       C:\android-sdk\cmdline-tools\latest\bin\
       C:\android-sdk\cmdline-tools\latest\lib\

6. Add the following to your system PATH
   (Start > Search > "Edit the system environment variables" >
    Environment Variables > User variables > Path > Edit > New):

       C:\android-sdk\cmdline-tools\latest\bin
       C:\android-sdk\platform-tools

7. Also add a new User variable:
       Variable name : ANDROID_HOME
       Variable value: C:\android-sdk

   Click OK on all dialogs.

8. Open a NEW terminal and install the required SDK packages:

       sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"

   Type y and press Enter when prompted.

9. Accept all Android licences:

       sdkmanager --licenses

   Type y and press Enter for each prompt.

-------------------------------------------------------------------
A3 - Install Flutter SDK
-------------------------------------------------------------------

Download the Flutter SDK (stable channel) from:
https://docs.flutter.dev/get-started/install/windows

1. Extract the downloaded .zip to C:\flutter (avoid paths with spaces)
2. Add Flutter to your system PATH:
   - Open Start > Search > "Edit the system environment variables"
   - Click Environment Variables > User variables > Path > Edit > New
   - Enter: C:\flutter\bin
   - Click OK on all dialogs

3. Open a NEW terminal and verify:

       flutter --version

4. Validate your setup:

       flutter doctor

   Green checkmarks should appear next to Flutter and Android toolchain.
   If licences are flagged, run:

       flutter doctor --android-licenses

   Type y and press Enter for each prompt.

-------------------------------------------------------------------
A4 - Prepare Your Android Phone
-------------------------------------------------------------------

Enable Developer Options on the phone:
  1. Open Settings > About phone
  2. Tap "Build number" seven times in a row
  3. You will see "You are now a developer"

Enable USB Debugging:
  1. Go to Settings > Developer options
  2. Turn on USB Debugging and tap OK

Connect the phone:
  1. Plug the phone into your PC with a USB cable
  2. On the phone, tap Allow when asked "Allow USB debugging?"
  3. If asked for a connection type, select File Transfer (MTP)

-------------------------------------------------------------------
A5 - Install Git and Download the Project
-------------------------------------------------------------------

Download Git from: https://git-scm.com/download/win
Accept all defaults. Verify:

    git --version

Clone the repository:

    git clone https://github.com/yousefyaser1/Graduation-Project.git

-------------------------------------------------------------------
A6 - Run the Application
-------------------------------------------------------------------

    cd Graduation-Project\Flutter
    flutter pub get

Verify your phone is detected:

    flutter devices

You should see your phone listed by model name.
If it does not appear, unplug, replug, and tap Allow on the phone.

Build and run:

    flutter run

The first build takes 3-5 minutes. The app will install and open
automatically on your phone.

===================================================================
SCENARIO B - iPhone via USB (Mac)
===================================================================

Apple only allows building iOS apps on macOS with Xcode.
If you have a Mac and an iPhone, follow these steps.

-------------------------------------------------------------------
B1 - Install Xcode
-------------------------------------------------------------------

1. Open the App Store on your Mac
2. Search for "Xcode" and install it (this is a large download, ~7 GB)
3. After installation, open Xcode once to accept the licence agreement
4. Install the Xcode command-line tools by opening Terminal and running:

       sudo xcode-select --install

-------------------------------------------------------------------
B2 - Install CocoaPods
-------------------------------------------------------------------

Flutter uses CocoaPods to manage iOS dependencies. In Terminal:

    sudo gem install cocoapods

-------------------------------------------------------------------
B3 - Install Flutter SDK (Mac)
-------------------------------------------------------------------

Download the Flutter SDK (stable, macOS build) from:
https://docs.flutter.dev/get-started/install/macos

1. Extract the downloaded .zip to your home folder, e.g. ~/flutter
2. Add Flutter to your PATH by editing ~/.zshrc (or ~/.bash_profile):

       export PATH="$HOME/flutter/bin:$PATH"

3. Save the file, then reload it:

       source ~/.zshrc

4. Verify:

       flutter --version

5. Validate your setup:

       flutter doctor

   Green checkmarks should appear next to Flutter and Xcode.

-------------------------------------------------------------------
B4 - Prepare Your iPhone
-------------------------------------------------------------------

1. Connect your iPhone to the Mac with a Lightning or USB-C cable
2. On the iPhone, tap Trust when asked "Trust This Computer?"
3. Enter your iPhone passcode if prompted
4. Open Xcode > Window > Devices and Simulators and confirm
   your iPhone appears in the list

-------------------------------------------------------------------
B5 - Install Git and Download the Project
-------------------------------------------------------------------

Git is already included with Xcode command-line tools. Verify:

    git --version

Clone the repository:

    git clone https://github.com/yousefyaser1/Graduation-Project.git

-------------------------------------------------------------------
B6 - Run the Application
-------------------------------------------------------------------

    cd Graduation-Project/Flutter
    flutter pub get

Verify your iPhone is detected:

    flutter devices

You should see your iPhone listed by model name.

Build and run:

    flutter run

The first build takes 5-10 minutes (iOS builds are slower than Android).
The app will install and open on your iPhone automatically.

NOTE: If you see a "Trust" or "Untrusted Developer" error on the iPhone:
  Go to Settings > General > VPN & Device Management
  Find the developer entry and tap Trust.

===================================================================
SCENARIO C - iPhone + Windows PC (Android Emulator Fallback)
===================================================================

Building iOS apps on Windows is not possible due to Apple restrictions.
The only option in this case is to run an Android emulator on your PC.
This requires Android Studio.

-------------------------------------------------------------------
C1 - Install Java Development Kit (JDK 17)
-------------------------------------------------------------------

Download and install JDK 17 from:
https://www.oracle.com/java/technologies/downloads/#java17

Accept all defaults. Verify:

    java -version

Expected output: java version "17.x.x"

-------------------------------------------------------------------
C2 - Install Android Studio
-------------------------------------------------------------------

Download Android Studio (latest stable) from:
https://developer.android.com/studio

During installation:
  - Select "Standard" installation type
  - Accept all SDK licence agreements

After installation, open Android Studio and wait for the initial
SDK download to finish.

Install Required SDK Components:
  Go to File > Settings > Appearance & Behavior > System Settings > Android SDK

  SDK Platforms tab: check Android 14.0 (API 34)
  SDK Tools tab    : check Android SDK Build-Tools
                          Android Emulator
                          Android SDK Platform-Tools

  Click Apply and wait for the downloads to complete.

Create a Virtual Android Device (Emulator):
  1. Open Device Manager (right-side panel or View > Tool Windows > Device Manager)
  2. Click Create Device
  3. Select Phone > Pixel 6 > click Next
  4. Under System Image, select API Level 34 (Android 14) - download if prompted
  5. Click Next > Finish

-------------------------------------------------------------------
C3 - Install Flutter SDK
-------------------------------------------------------------------

Download the Flutter SDK (stable channel) from:
https://docs.flutter.dev/get-started/install/windows

1. Extract the downloaded .zip to C:\flutter (avoid paths with spaces)
2. Add Flutter to your system PATH:
   - Open Start > Search > "Edit the system environment variables"
   - Click Environment Variables > User variables > Path > Edit > New
   - Enter: C:\flutter\bin
   - Click OK on all dialogs

3. Open a NEW terminal and verify:

       flutter --version

4. Validate your setup:

       flutter doctor

   If licences are flagged, run:

       flutter doctor --android-licenses

   Type y and press Enter for each prompt.

-------------------------------------------------------------------
C4 - Install Git and Download the Project
-------------------------------------------------------------------

Download Git from: https://git-scm.com/download/win
Accept all defaults. Verify:

    git --version

Clone the repository:

    git clone https://github.com/yousefyaser1/Graduation-Project.git

-------------------------------------------------------------------
C5 - Run the Application
-------------------------------------------------------------------

Start the Android emulator:
  Open Android Studio > Device Manager > click the Play button
  next to the Pixel 6 device. Wait for it to fully boot
  to the Android home screen (1-2 minutes on first launch).

Then in the terminal:

    cd Graduation-Project\Flutter
    flutter pub get

Verify Flutter detects the emulator:

    flutter devices

Expected output includes a line such as:
  sdk gphone64 x86 64 (mobile) . emulator-5554 . android-x64 . Android 14

Build and run:

    flutter run

The first build takes 3-5 minutes. The DermAI splash screen
will appear on the emulator window.

NOTE - Uploading images to the emulator:
  Drag and drop a .jpg file onto the running emulator window, or use
  Android Studio > Device Manager > three-dot menu > Send files.
  The image will appear in the emulator's gallery.

===================================================================
SCENARIO D - Android Phone via USB (Mac)
===================================================================

If you have an Android phone and a Mac, follow these steps.
No emulator needed.

-------------------------------------------------------------------
D1 - Install Java Development Kit (JDK 17)
-------------------------------------------------------------------

Download and install JDK 17 for macOS from:
https://www.oracle.com/java/technologies/downloads/#java17

After installation, open Terminal and verify:

    java -version

Expected output: java version "17.x.x"

-------------------------------------------------------------------
D2 - Install Android SDK Command-Line Tools
-------------------------------------------------------------------

Android Studio is NOT required. You only need the command-line tools.

1. Go to: https://developer.android.com/studio
2. Scroll down to "Command line tools only"
3. Download the macOS ZIP (commandlinetools-mac-XXXXXXX_latest.zip)
4. Create this folder:

       mkdir -p ~/android-sdk/cmdline-tools/latest

5. Extract the ZIP contents into that folder so the structure is:

       ~/android-sdk/cmdline-tools/latest/bin/
       ~/android-sdk/cmdline-tools/latest/lib/

6. Add the following to your ~/.zshrc (or ~/.bash_profile):

       export ANDROID_HOME=$HOME/android-sdk
       export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH
       export PATH=$ANDROID_HOME/platform-tools:$PATH

7. Reload the file:

       source ~/.zshrc

8. Install the required SDK packages:

       sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"

   Type y and press Enter when prompted.

9. Accept all Android licences:

       sdkmanager --licenses

   Type y and press Enter for each prompt.

-------------------------------------------------------------------
D3 - Install Flutter SDK (Mac)
-------------------------------------------------------------------

Download the Flutter SDK (stable, macOS build) from:
https://docs.flutter.dev/get-started/install/macos

1. Extract the downloaded .zip to your home folder, e.g. ~/flutter
2. Add Flutter to your PATH by editing ~/.zshrc:

       export PATH="$HOME/flutter/bin:$PATH"

3. Save and reload:

       source ~/.zshrc

4. Verify:

       flutter --version

5. Validate your setup:

       flutter doctor

   Green checkmarks should appear next to Flutter and Android toolchain.
   If licences are flagged, run:

       flutter doctor --android-licenses

   Type y and press Enter for each prompt.

-------------------------------------------------------------------
D4 - Prepare Your Android Phone
-------------------------------------------------------------------

Enable Developer Options on the phone:
  1. Open Settings > About phone
  2. Tap "Build number" seven times in a row
  3. You will see "You are now a developer"

Enable USB Debugging:
  1. Go to Settings > Developer options
  2. Turn on USB Debugging and tap OK

Connect the phone:
  1. Plug the phone into your Mac with a USB cable
  2. On the phone, tap Allow when asked "Allow USB debugging?"
  3. If asked for a connection type, select File Transfer (MTP)

-------------------------------------------------------------------
D5 - Install Git and Download the Project
-------------------------------------------------------------------

Git is already included with macOS. Verify in Terminal:

    git --version

If prompted to install developer tools, click Install and wait.

Clone the repository:

    git clone https://github.com/yousefyaser1/Graduation-Project.git

-------------------------------------------------------------------
D6 - Run the Application
-------------------------------------------------------------------

    cd Graduation-Project/Flutter
    flutter pub get

Verify your phone is detected:

    flutter devices

You should see your phone listed by model name.
If it does not appear, unplug, replug, and tap Allow on the phone.

Build and run:

    flutter run

The first build takes 3-5 minutes. The app will install and open
automatically on your phone.

-------------------------------------------------------------------
Troubleshooting (All Scenarios)
-------------------------------------------------------------------

Problem: flutter: command not found
Fix    : Close and reopen the terminal after editing PATH

Problem: Android licence not accepted
Fix    : Run "flutter doctor --android-licenses" and type y for each prompt

Problem: Android phone not detected by "flutter devices"
Fix    : Unplug and replug the USB cable; tap Allow on the phone;
         confirm USB Debugging is enabled in Developer Options;
         try a different USB cable or port

Problem: iPhone not detected by "flutter devices"
Fix    : Tap Trust on the iPhone; make sure Xcode shows the device
         under Window > Devices and Simulators

Problem: "Untrusted Developer" error on iPhone
Fix    : Settings > General > VPN & Device Management > tap Trust

Problem: Gradle build failed (Android)
Fix    : Confirm JDK 17 is installed; set JAVA_HOME to the JDK 17 directory

Problem: CocoaPods not found (Mac/iOS)
Fix    : Run "sudo gem install cocoapods" then try again

Problem: Android emulator won't start (Scenario C)
Fix    : Enable virtualisation in BIOS (Intel VT-x / AMD-V),
         or install Intel HAXM via Android Studio SDK Tools

Problem: App crashes during analysis
Fix    : Verify all four .tflite files are present under Flutter/assets/models/

-------------------------------------------------------------------
Project Structure
-------------------------------------------------------------------

Graduation-Project/
|-- Flutter/                              Mobile application (Flutter/Dart)
|   |-- lib/
|   |   |-- features/                     UI screens
|   |   |   |-- onboarding/               Login, signup, role selection
|   |   |   |-- core_workflow/            Home, capture, history, profile
|   |   |   +-- results/                  Diagnosis result, Score-CAM heatmap
|   |   |-- models/scan_result.dart       Core data model
|   |   |-- services/
|   |   |   |-- ai/ai_service.dart        Three-stage AI pipeline
|   |   |   +-- database/                 SQLite scan history
|   |   +-- core/                         Theme, routing, shared widgets
|   |-- assets/models/                    TFLite model files (on-device inference)
|   |   |-- vae_model.tflite              Anomaly detector (3.4 MB)
|   |   |-- cnn_b2_model.tflite           EfficientNetB2 classifier (8.6 MB)
|   |   |-- cnn_b3_model.tflite           EfficientNetB3 classifier
|   |   +-- b3_feature_extractor.tflite   Score-CAM feature extraction
|   +-- pubspec.yaml
|
|-- model1/                               Python model training scripts (Kaggle)
|-- thirteenth_code_evaluation_metrics.py CNN evaluation (confusion matrix, ROC, AUC)
|-- fourteenth_code_vae_evaluation.py     VAE anomaly detection evaluation
|-- Chapter4_Implementation.tex           Thesis Chapter 4 - Implementation
|-- Chapter5_Testing.tex                  Thesis Chapter 5 - Testing & Results
+-- README.txt                            This file

-------------------------------------------------------------------
AI Pipeline Details
-------------------------------------------------------------------

Stage 1 - Anomaly Detection
  Model : vae_model.tflite
  Input : 64x64 image patches
  Output: Anomaly ratio; pipeline halts if skin appears normal

Stage 2 - Classification
  Model : EfficientNetB2 + B3 ensemble (cnn_b2_model.tflite + cnn_b3_model.tflite)
  Input : 260x260 / 300x300
  Output: Acne / Eczema / Tinea probabilities (50/50 average, 20-step TTA)

Stage 3 - Explainability (Score-CAM)
  Model : b3_feature_extractor.tflite
  Input : 300x300
  Output: Heatmap overlay highlighting lesion regions

Validation accuracy: 92.62% on a 325-image held-out split
(test set preserved for future fine-tuning).

-------------------------------------------------------------------
Software Versions Used
-------------------------------------------------------------------

  Flutter SDK           : 3.x stable
  Dart                  : Bundled with Flutter
  Android SDK           : API 34 (Android 14)
  Xcode (iOS only)      : Latest stable from App Store
  Java JDK              : 17 (Android/Windows only)
  TensorFlow Lite       : 0.12.0 (via tflite_flutter)
  Minimum Android       : API 21 (Android 5.0)
  Minimum iOS           : iOS 12.0
