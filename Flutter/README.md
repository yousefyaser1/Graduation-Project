# Dermatology AI App

An offline-capable, cross-platform mobile application for dermatological screening using Explainable AI (XAI).

## Project Overview

This Flutter application uses TensorFlow Lite models (MobileNet V2 and VAE) for skin disease detection and provides explainable AI visualizations through heatmaps. The app features local SQLite database storage for scan history tracking.

## Features

- **Onboarding Flow**: Welcome, Login, Role Selection, and Personal Information screens
- **Core Workflow**: Dashboard, Body Part Selection, and Camera/Gallery capture
- **Results**: Diagnosis output with confidence scores and XAI heatmap visualization
- **Offline Capability**: All AI inference runs locally using TensorFlow Lite
- **Local Storage**: SQLite database for scan history and user data
- **Dark/Light Theme**: Automatic theme switching based on system preferences

## Project Structure

```
Flutter/
├── lib/
│   ├── main.dart                      # App entry point
│   ├── core/                          # Core application logic
│   │   ├── routing/
│   │   │   └── app_router.dart        # Navigation configuration
│   │   └── theme/
│   │       └── app_theme.dart         # Theme definitions
│   ├── features/                      # Feature-based organization
│   │   ├── onboarding/                # Onboarding flow screens
│   │   │   └── screens/
│   │   │       ├── welcome_screen.dart
│   │   │       ├── login_screen.dart
│   │   │       ├── role_selection_screen.dart
│   │   │       └── personal_info_screen.dart
│   │   ├── core_workflow/             # Main app workflow screens
│   │   │   └── screens/
│   │   │       ├── body_part_selection_screen.dart
│   │   │       └── capture_screen.dart
│   │   └── results/                   # Results and visualization screens
│   │       └── screens/
│   │           ├── diagnosis_result_screen.dart
│   │           └── xai_heatmap_screen.dart
│   ├── models/                        # Data models
│   │   ├── scan_result.dart
│   │   └── user.dart
│   ├── providers/                     # Riverpod state management
│   │   ├── scan_provider.dart
│   │   └── user_provider.dart
│   └── services/                      # Business logic services
│       ├── ai/
│       │   └── ai_service.dart        # TensorFlow Lite inference
│       └── database/
│           └── database_service.dart  # SQLite operations
├── assets/                            # Static assets
│   ├── images/                        # App images and icons
│   └── models/                        # TFLite models (Phase 2)
├── pubspec.yaml                       # Dependencies and configuration
├── analysis_options.yaml              # Linting rules
└── .gitignore                         # Git ignore rules
```

## Technology Stack

- **Framework**: Flutter 3.x
- **Language**: Dart 3.x
- **State Management**: Riverpod
- **Routing**: go_router
- **Database**: SQLite (sqflite)
- **AI/ML**: TensorFlow Lite (to be added in Phase 2)
- **Camera**: camera package
- **Image Handling**: image_picker, image_cropper

## Setup Instructions

### Prerequisites

1. Install Flutter SDK (3.0.0 or higher)
2. Install Android Studio / Xcode for mobile development
3. Ensure you have a connected device or emulator

### Installation

1. Navigate to the Flutter directory:
   ```bash
   cd Flutter
   ```

2. Install dependencies:
   ```bash
   flutter pub get
   ```

3. Run the app:
   ```bash
   flutter run
   ```

## Development Phases

### Phase 1: Project Setup & Base Architecture (Current)
- ✅ Project initialization
- ✅ Folder structure
- ✅ Dependencies configuration
- ✅ Routing setup
- ✅ Placeholder screens
- ✅ Theme configuration

### Phase 2: AI/ML Integration
- TensorFlow Lite model integration
- Camera and image picker implementation
- Image preprocessing
- Model inference
- XAI heatmap generation

### Phase 3: Database & Persistence
- SQLite database implementation
- Scan history CRUD operations
- User data management
- Local image storage

### Phase 4: UI/UX Enhancement
- Improved screen designs
- Animations and transitions
- Loading states
- Error handling
- Accessibility improvements

## Routes

| Route | Description |
|-------|-------------|
| `/welcome` | Welcome screen with app introduction |
| `/login` | User login screen |
| `/role-selection` | Role selection (Patient/Healthcare Provider) |
| `/personal-info` | Personal information collection |
| `/dashboard` | Main dashboard with quick actions |
| `/body-part-selection` | Body part selection for scanning |
| `/capture` | Camera/Gallery capture screen |
| `/diagnosis-result` | Diagnosis results display |
| `/xai-heatmap` | XAI heatmap visualization |

## License

This project is part of a graduation project for dermatological screening.

## Disclaimer

This application uses AI-powered diagnosis and should not replace professional medical advice. Always consult a dermatologist for accurate diagnosis and treatment.
