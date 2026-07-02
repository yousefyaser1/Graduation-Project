# Explainable AI (XAI) Powered Mobile App for Skin Disease Classification

## A Comprehensive Academic Report

---

## Table of Contents

1. [Introduction & Problem Statement](#1-introduction--problem-statement)
2. [Dataset Collection & Description](#2-dataset-collection--description)
3. [Data Cleaning & Preparation](#3-data-cleaning--preparation)
4. [Data Augmentation (Offline)](#4-data-augmentation-offline)
5. [Dataset Analysis & Verification](#5-dataset-analysis--verification)
6. [Phase 1 — Transfer Learning (Base Model Training)](#6-phase-1--transfer-learning-base-model-training)
7. [Phase 2 — Fine-Tuning (Layer Unfreezing)](#7-phase-2--fine-tuning-layer-unfreezing)
8. [Phase 3 — Anti-Overfitting & Accuracy Improvement](#8-phase-3--anti-overfitting--accuracy-improvement)
9. [Mobile Application Development (Flutter)](#9-mobile-application-development-flutter)
10. [Discussion](#10-discussion)
11. [Conclusion & Future Work](#11-conclusion--future-work)

---

## 1. Introduction & Problem Statement

### 1.1 Clinical Motivation

Skin diseases represent one of the most prevalent categories of medical conditions worldwide, affecting populations across all demographics and geographic regions. Dermatological diagnosis traditionally relies on expert visual inspection by trained dermatologists, a process that is inherently subjective, time-consuming, and limited by the availability of specialized clinicians. In many developing regions, access to dermatologists remains severely constrained, leading to delayed diagnoses and suboptimal patient outcomes.

The challenge of skin disease classification is compounded by the visual similarity between conditions. Diseases such as acne, eczema, and tinea can present with overlapping morphological features, making accurate differentiation difficult even for experienced practitioners. This diagnostic ambiguity underscores the need for automated classification systems that can provide consistent, objective, and rapid preliminary assessments.

### 1.2 Deep Learning for Dermatology

Convolutional Neural Networks (CNNs) have demonstrated remarkable performance in image classification tasks, particularly in the medical imaging domain. The hierarchical feature extraction capability of CNNs makes them well-suited for dermatological image analysis, as they can learn to identify subtle visual patterns — such as texture, color distribution, lesion morphology, and spatial arrangement — that are indicative of specific skin conditions.

Transfer learning, wherein a model pretrained on a large general-purpose dataset (e.g., ImageNet) is adapted to a domain-specific task, has proven especially effective when labeled medical data is limited. By leveraging features learned from millions of natural images, transfer learning reduces the data requirements and training time while improving generalization performance.

### 1.3 Project Objectives

This graduation project pursues the following objectives:

1. **Three-class skin disease classification**: Develop a CNN-based classifier capable of distinguishing between Acne, Eczema, and Tinea from clinical dermatological images.
2. **Explainable AI (XAI) integration**: Incorporate interpretability mechanisms (e.g., Grad-CAM heatmaps) to provide visual explanations of model predictions, enhancing clinical trust and transparency.
3. **Mobile deployment**: Build a cross-platform Flutter mobile application that enables end-users to capture skin images and receive real-time classification results with XAI visualizations.

### 1.4 Scope

The project encompasses the full machine learning pipeline — from dataset curation and augmentation through multi-phase model training — as well as the development of a Flutter-based mobile application. The classification target is restricted to three dermatological conditions: Acne, Eczema, and Tinea. The model architecture is MobileNetV2, selected for its balance between classification performance and computational efficiency on mobile devices.

---

## 2. Dataset Collection & Description

### 2.1 Original Dataset Structure

The initial dataset was organized under the directory `Dermatology_CNN_Dataset/`, structured into three class folders corresponding to the target skin conditions:

```
Dermatology_CNN_Dataset/
├── train/
│   ├── acne/
│   ├── eczema/
│   └── tinea/
└── test/
    ├── acne/
    ├── eczema/
    └── tinea/
```

### 2.2 Image Sources

The dataset comprises clinical dermatological photographs sourced from publicly available dermatology atlases and medical image repositories. Images vary in resolution, lighting conditions, and anatomical location, reflecting the diversity encountered in real-world clinical settings. File formats are predominantly JPEG (`.jpg`), with images depicting various subtypes within each class — for example, cystic acne, pustular acne, and comedonal acne within the Acne category; nummular eczema, asteatotic eczema, and hand eczema within the Eczema category; and tinea corporis, tinea pedis, and tinea versicolor within the Tinea category.

### 2.3 Initial Class Distribution

The original dataset exhibited a train/test split with no dedicated validation partition. An inherent class imbalance was observed, with the Eczema class containing a larger proportion of images relative to Acne and Tinea. This imbalance necessitated the application of class weighting strategies during model training, as discussed in subsequent sections.

---

## 3. Data Cleaning & Preparation

### 3.1 Transition to Finalized Clean Data

The raw dataset underwent a systematic cleaning process, resulting in the curated directory `Finalized_Clean_Data/`. This cleaning phase addressed several data quality concerns inherent in web-sourced medical imagery.

### 3.2 Quality Filtering

The following quality control measures were applied:

- **Duplicate removal**: Visually and computationally identical images were identified and removed to prevent data leakage and inflated performance metrics.
- **Irrelevant image removal**: Non-dermatological images (e.g., histology slides, diagrams, or unrelated photographs) were manually reviewed and excluded.
- **Mislabeled image correction**: Images that were clearly misclassified based on visual inspection were either relabeled or removed.
- **Resolution filtering**: Extremely low-resolution or heavily watermarked images that would impede feature extraction were excluded.

### 3.3 Train/Validation/Test Splitting Strategy

The cleaned dataset was partitioned into three subsets:

- **Training set (~90%)**: Used for model parameter optimization.
- **Validation set (~5%)**: Used for hyperparameter tuning and early stopping decisions during training.
- **Test set (~5%)**: Held out entirely and used only for final model evaluation.

The split was performed at the image level with stratification to preserve class proportions across all three subsets. This three-way split — as opposed to the original two-way train/test arrangement — enabled proper model selection through validation-based early stopping without contaminating the test set.

---

## 4. Data Augmentation (Offline)

### 4.1 Augmentation Script

Offline data augmentation was performed on the training set only, using the script `New_Augmented_Dataset/augmentation_script_on_Clean_Data.py`. The augmentation pipeline was implemented using Keras preprocessing layers encapsulated in a `tf.keras.Sequential` model, ensuring deterministic and reproducible transformations via a fixed random seed.

### 4.2 Augmentation Pipeline

The augmentation pipeline was constructed as follows:

```python
def create_augmentation_layer():
    augmentation_pipeline = tf.keras.Sequential([
        # Random Horizontal Flip (50% probability)
        tf.keras.layers.RandomFlip(
            mode="horizontal",
            seed=42
        ),
        # Random Vertical Flip (50% probability)
        tf.keras.layers.RandomFlip(
            mode="vertical",
            seed=42
        ),
        # Random Rotation (-15 degrees to +15 degrees)
        tf.keras.layers.RandomRotation(
            factor=0.0417,  # 15/360 ~ 0.0417
            fill_mode='reflect',
            seed=42
        ),
        # Random Zoom (up to 10%)
        tf.keras.layers.RandomZoom(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode='reflect',
            seed=42
        ),
        # Random Brightness (+/-10%)
        tf.keras.layers.RandomBrightness(
            factor=0.1,
            seed=42
        ),
    ])
    return augmentation_pipeline
```

Each original training image was processed three times through this pipeline, producing three distinct augmented copies. Augmented images were saved with the naming convention `aug_{original_name}_{index}.jpg`, where index ranges from 1 to 3.

### 4.3 Augmentation Techniques Summary

| Technique | Keras Layer | Parameters | Rationale |
|-----------|-------------|------------|-----------|
| Horizontal Flip | `RandomFlip("horizontal")` | 50% probability | Skin lesions have no canonical left-right orientation |
| Vertical Flip | `RandomFlip("vertical")` | 50% probability | Augments viewpoint diversity |
| Random Rotation | `RandomRotation(0.0417)` | +/-15 degrees, reflect fill | Simulates camera angle variation |
| Random Zoom | `RandomZoom(0.1, 0.1)` | +/-10%, reflect fill | Mimics varying capture distances |
| Random Brightness | `RandomBrightness(0.1)` | +/-10% | Accounts for lighting variation |

All transformations used `fill_mode='reflect'` (where applicable) to avoid introducing artificial black borders, and `seed=42` was set for reproducibility.

### 4.4 Augmentation Configuration

| Parameter | Value |
|-----------|-------|
| Source directory | `Finalized_Clean_Data/train/` |
| Target classes | acne, eczema, tinea |
| Augmentations per image | 3 |
| Random seed | 42 |
| Output format | JPEG (`.jpg`) |
| Filename prefix | `aug_` |
| Pixel value clipping | [0, 255] |

### 4.5 Resulting Dataset Counts

The augmented dataset, stored in `New_Augmented_Dataset/`, contains the following image counts:

| Split | Acne | Eczema | Tinea | Total |
|-------|------|--------|-------|-------|
| Train | 2,880 | 3,460 | 1,816 | 8,156 |
| Val   | 93   | 142    | 90    | 325   |
| Test  | 96   | 137    | 89    | 322   |
| **Total** | **3,069** | **3,739** | **1,995** | **8,803** |

Note that augmentation was applied exclusively to the training split. The validation and test sets retain only original (non-augmented) images to ensure unbiased evaluation.

---

## 5. Dataset Analysis & Verification

### 5.1 Analysis Script

Dataset verification was performed using the script `model1/analyze_Augmented_dataset.py`. This script iterates over all split directories and class folders, counting images by file extension (`.jpg`, `.jpeg`, `.png`) and reporting per-class and per-split statistics.

```python
DATASET_ROOT = Path("Augmented_Dataset")
SPLITS = ["train", "test", "val"]
CATEGORIES = ["acne", "eczema", "tinea"]

def count_images_in_category(split: str, category: str) -> int:
    category_dir = DATASET_ROOT / split / category
    if not category_dir.exists():
        return 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = sum(1 for file in category_dir.iterdir()
                if file.suffix in image_extensions)
    return count
```

### 5.2 Class Distribution Analysis

The analysis of the augmented training set reveals the following class distribution:

| Class | Training Images | Percentage |
|-------|----------------|------------|
| Eczema | 3,460 | ~42.4% |
| Acne | 2,880 | ~35.3% |
| Tinea | 1,816 | ~22.3% |
| **Total** | **8,156** | **100%** |

### 5.3 Class Imbalance Identification

The distribution analysis identifies a notable class imbalance: Eczema comprises approximately 42% of the training data, while Tinea accounts for only approximately 22%. This imbalance, if unaddressed, would bias the model toward predicting the majority class. To mitigate this, dynamic class weighting was employed during training (see Section 6), computed using scikit-learn's `compute_class_weight('balanced')` function, which assigns inversely proportional weights to each class based on its frequency.

---

## 6. Phase 1 — Transfer Learning (Base Model Training)

### 6.1 Architecture Selection: MobileNetV2

MobileNetV2 was selected as the base architecture for the following reasons:

1. **Lightweight design**: MobileNetV2 uses depthwise separable convolutions and inverted residual blocks, resulting in a model with significantly fewer parameters than architectures such as ResNet or VGG, making it suitable for deployment on mobile devices with constrained computational resources.
2. **ImageNet pretraining**: The model is pretrained on ImageNet (1.4 million images, 1,000 classes), providing rich low-level and mid-level feature representations that transfer effectively to medical image classification.
3. **Mobile deployment compatibility**: MobileNetV2 is natively supported by TensorFlow Lite, facilitating future conversion for on-device inference in the Flutter application.

### 6.2 Training Script

The base model training was implemented in `train_mobilenetv2.py`. The script follows a structured pipeline: dataset loading, class weight computation, data pipeline optimization, model construction, compilation, training, and model serialization.

### 6.3 Model Construction

The model architecture consists of a frozen MobileNetV2 backbone with a custom classification head:

```python
# Load pretrained MobileNetV2 (excluding top classification layers)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all base model layers
base_model.trainable = False

# Build classification head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
```

The classification head comprises:
- **Global Average Pooling (GAP)**: Reduces the spatial dimensions of the feature map to a single vector, serving as a regularizer and eliminating the need for fully connected layers.
- **Dropout (rate=0.2)**: Randomly deactivates 20% of neurons during training to reduce overfitting.
- **Dense layer (3 units, softmax)**: Produces a probability distribution over the three target classes.

### 6.4 Preprocessing and Data Pipeline

Input images were preprocessed using MobileNetV2's native `preprocess_input` function, which scales pixel values from [0, 255] to [-1, 1]:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
```

The `.cache()` operation stores the preprocessed dataset in memory after the first epoch, eliminating redundant I/O and preprocessing in subsequent epochs. The `.prefetch(AUTOTUNE)` operation overlaps data loading with model training, allowing the CPU to prepare the next batch while the GPU processes the current one.

### 6.5 Class Weight Calculation

To address class imbalance, dynamic class weights were computed using scikit-learn:

```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
```

The `'balanced'` mode assigns weights inversely proportional to class frequencies: $w_c = \frac{N}{k \cdot n_c}$, where $N$ is the total number of samples, $k$ is the number of classes, and $n_c$ is the number of samples in class $c$. This ensures that underrepresented classes (e.g., Tinea) contribute proportionally more to the loss function during training.

### 6.6 Training Configuration

| Parameter | Value |
|-----------|-------|
| Input size | 224 x 224 x 3 |
| Preprocessing | `preprocess_input` (scales to [-1, 1]) |
| Batch size | 32 |
| Optimizer | Adam (learning rate = 0.0001) |
| Loss function | Categorical Crossentropy |
| Metrics | Accuracy |
| Classification head | GAP -> Dropout(0.2) -> Dense(3, softmax) |
| Class weights | `compute_class_weight('balanced')` |
| Callbacks | EarlyStopping (monitor=val_loss, patience=3, restore_best_weights=True) |
| Data pipeline | `.cache()` + `.prefetch(AUTOTUNE)` |
| Random seed | 42 |

### 6.7 Output

The trained model was serialized in the Keras native format:

- **File**: `mobilenetv2_skin_disease_model.keras`
- **Size**: 9.22 MB

The compact model size reflects the frozen base model: only the classification head parameters (GAP, Dropout, and Dense layers) were optimized during this phase, keeping the total trainable parameter count minimal.

### 6.8 Test Set Evaluation Results

The Phase 1 model was evaluated on the held-out test set (322 images: 96 Acne, 137 Eczema, 89 Tinea).

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **75.16%** |
| Overall Loss | 0.6093 |

**Per-class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Acne | 0.7857 | 0.8021 | 0.7938 | 96 |
| Eczema | 0.7887 | 0.8175 | 0.8029 | 137 |
| Tinea | 0.6463 | 0.5955 | 0.6199 | 89 |
| **Macro avg** | **0.7403** | **0.7384** | **0.7389** | 322 |
| **Weighted avg** | **0.7485** | **0.7516** | **0.7496** | 322 |

**Confusion Matrix:**

|  | Predicted Acne | Predicted Eczema | Predicted Tinea |
|---|---|---|---|
| **True Acne** | 77 | 6 | 13 |
| **True Eczema** | 9 | 112 | 16 |
| **True Tinea** | 12 | 24 | 53 |

**Per-class Accuracy:** Acne 80.21% | Eczema 81.75% | Tinea 59.55%

**Analysis**: The Phase 1 model establishes a solid baseline at 75.16% overall accuracy. Acne and Eczema are classified well (F1 ~0.79–0.80), while Tinea shows the weakest performance (F1 = 0.62, recall 59.55%). The primary confusion is between Tinea and Eczema (24 Tinea samples misclassified as Eczema), reflecting the visual overlap between fungal infections and inflammatory skin conditions. This result is expected given that the MobileNetV2 backbone was entirely frozen — no domain-specific adaptation was applied.

---
$$$$
## 7. Phase 2 — Fine-Tuning (Layer Unfreezing)

### 7.1 Rationale

While transfer learning with a frozen base model provides a strong baseline, the pretrained features from ImageNet may not perfectly align with the visual characteristics of dermatological images. Fine-tuning — selectively unfreezing and retraining upper layers of the base model — allows the network to adapt its learned representations to the domain-specific features of skin disease images, such as lesion texture, erythema patterns, and scaling morphology.

### 7.2 Fine-Tuning Script

Fine-tuning was implemented in `finetune_mobilenetv2_phase2.py`. The script loads the Phase 1 model, identifies the MobileNetV2 backbone within the model hierarchy, and selectively unfreezes layers for further training.

### 7.3 Layer Unfreezing Strategy

The MobileNetV2 base model contains 155 layers. The first 100 layers — which capture low-level and mid-level features (edges, textures, shapes) — were kept frozen to preserve their pretrained representations. Layers 100 onward, which encode higher-level semantic features, were unfrozen for fine-tuning:

```python
# Locate MobileNetV2 within the loaded model
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break

base_model.trainable = True

# Freeze first 100 layers, unfreeze the rest
for i, layer in enumerate(base_model.layers):
    if i < 100:
        layer.trainable = False
    else:
        layer.trainable = True
```

### 7.4 Learning Rate Reduction

A critical consideration in fine-tuning is the prevention of catastrophic forgetting — the phenomenon where aggressive weight updates destroy the useful representations learned during pretraining. To mitigate this, the learning rate was reduced by an order of magnitude from Phase 1 (1e-4) to Phase 2 (1e-5):

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 7.5 Advanced Callbacks

Three callbacks were employed to manage the fine-tuning process:

```python
# EarlyStopping with increased patience for fine-tuning
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau for adaptive learning rate
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ModelCheckpoint to persist the best model
model_checkpoint = callbacks.ModelCheckpoint(
    'mobilenetv2_finetuned_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
```

- **EarlyStopping** (patience=5): Halts training if validation loss does not improve for 5 consecutive epochs, restoring the weights from the best epoch. The patience was increased from 3 (Phase 1) to 5 to accommodate the slower convergence characteristic of fine-tuning with a reduced learning rate.
- **ReduceLROnPlateau** (factor=0.5, patience=2): Halves the learning rate when validation loss plateaus for 2 epochs, with a minimum learning rate floor of 1e-7. This enables the optimizer to make progressively finer adjustments as training converges.
- **ModelCheckpoint**: Saves the model weights whenever a new best validation loss is achieved, ensuring that the final saved model corresponds to the optimal training state rather than the last epoch.

### 7.6 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Loaded from Phase 1 (`mobilenetv2_skin_disease_model.keras`) |
| Frozen layers | First 100 (of 155 total in MobileNetV2) |
| Unfrozen layers | Layers 100–154 (55 layers) |
| Optimizer | Adam (learning rate = 1e-5) |
| Loss function | Categorical Crossentropy |
| Epochs | 30 (maximum) |
| Batch size | 32 |
| Class weights | `compute_class_weight('balanced')` |
| Callbacks | EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-7), ModelCheckpoint |
| Data pipeline | `.cache()` + `.prefetch(AUTOTUNE)` |

### 7.7 Output

- **File**: `mobilenetv2_finetuned_model.keras`
- **Size**: 23.45 MB

The increased file size relative to Phase 1 (9.22 MB vs. 23.45 MB) reflects the additional trainable parameters: the unfrozen upper layers of MobileNetV2 now store optimized weights specific to the dermatological classification task, rather than retaining their original ImageNet-pretrained values.

### 7.8 Test Set Evaluation Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **84.16%** |
| Overall Loss | 0.4889 |

**Per-class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Acne | 0.9213 | 0.8542 | 0.8865 | 96 |
| Eczema | 0.8065 | 0.9124 | 0.8562 | 137 |
| Tinea | 0.8205 | 0.7191 | 0.7665 | 89 |
| **Macro avg** | **0.8494** | **0.8286** | **0.8364** | 322 |
| **Weighted avg** | **0.8446** | **0.8416** | **0.8404** | 322 |

**Confusion Matrix:**

|  | Predicted Acne | Predicted Eczema | Predicted Tinea |
|---|---|---|---|
| **True Acne** | 82 | 9 | 5 |
| **True Eczema** | 3 | 125 | 9 |
| **True Tinea** | 4 | 21 | 64 |

**Per-class Accuracy:** Acne 85.42% | Eczema 91.24% | Tinea 71.91%
$$$$
**Analysis**: Fine-tuning produced a substantial improvement of **+8.99 percentage points** over Phase 1 (75.16% → 84.16%). All three classes improved markedly, with Eczema achieving the highest recall (91.24%) and Acne achieving the highest precision (92.13%). Tinea showed the largest gain — from 59.55% to 71.91% recall — demonstrating that adapting the upper convolutional layers to dermatological features significantly improved discrimination of the most visually challenging class. The loss also improved from 0.6093 to 0.4889, indicating a more confident and better-calibrated model.

---

## 8. Phase 3 — Anti-Overfitting & Accuracy Improvement

### 8.1 Motivation

Following Phase 2 fine-tuning, an analysis of training dynamics revealed potential overfitting: the gap between training accuracy and validation accuracy indicated that the model was beginning to memorize training-specific patterns rather than learning generalizable features. Phase 3 was designed as a comprehensive anti-overfitting intervention, implementing multiple complementary regularization and augmentation techniques.

### 8.2 Improvement Script

The anti-overfitting strategies were implemented in `phase1_improve_accuracy.py`. This script loads the Phase 1 base model (not the Phase 2 fine-tuned model) to apply a more conservative fine-tuning regimen with enhanced regularization.

### 8.3 Technique 1: Conservative Fine-Tuning

Unlike Phase 2, which unfroze 55 layers (layers 100–154), Phase 3 adopted a more conservative approach by unfreezing only the top 50 layers:

```python
TOP_LAYERS_TO_UNFREEZE = 50

for i, layer in enumerate(base_model.layers):
    if i < (len(base_model.layers) - TOP_LAYERS_TO_UNFREEZE):
        layer.trainable = False
    else:
        layer.trainable = True
```

By reducing the number of trainable parameters, the model has fewer degrees of freedom to memorize training data, thereby encouraging the learning of more generalizable representations.

### 8.4 Technique 2: Enhanced Regularization

Multiple regularization mechanisms were strengthened:

- **Dropout rate**: Increased from 0.2 (Phase 1/2) to 0.5, providing more aggressive stochastic regularization.
- **L2 weight decay**: Set to 0.001, penalizing large weight magnitudes in the loss function.
- **Gradient clipping**: Applied with `clipnorm=1.0` to prevent exploding gradients during fine-tuning.

```python
DROPOUT_RATE = 0.5
L2_REG = 0.001
GRADIENT_CLIP_NORM = 1.0

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-6,
    clipnorm=GRADIENT_CLIP_NORM
)
```

### 8.5 Technique 3: Label Smoothing

Label smoothing with a factor of 0.2 was applied to the categorical crossentropy loss:

```python
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
    metrics=['accuracy']
)
```

Label smoothing replaces the hard one-hot target vector $[0, 0, 1]$ with a softened version $[0.067, 0.067, 0.867]$ (for 3 classes with smoothing factor 0.2). This prevents the model from becoming overconfident in its predictions, encourages better calibration of output probabilities, and acts as a form of regularization that improves generalization.

### 8.6 Technique 4: MixUp Augmentation (Online)

MixUp augmentation was applied as an online data transformation during training. MixUp generates synthetic training samples by linearly interpolating between pairs of images and their corresponding labels:

```python
def mixup_augmentation(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]

    # Sample lambda from Beta distribution
    lambda_values = tf.random.uniform([batch_size], 0, alpha)
    lambda_values = tf.reshape(lambda_values, [batch_size, 1, 1, 1])

    # Shuffle indices to get pairs
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Blend images and labels
    mixed_images = lambda_values * images + (1 - lambda_values) * shuffled_images
    lambda_labels = tf.reshape(lambda_values, [batch_size, 1])
    mixed_labels = lambda_labels * labels + (1 - lambda_labels) * shuffled_labels

    return mixed_images, mixed_labels
```

Given two training samples $(x_i, y_i)$ and $(x_j, y_j)$, MixUp produces:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where $\lambda \sim \text{Uniform}(0, \alpha)$ with $\alpha = 0.2$. By training on these interpolated samples, the model learns smoother decision boundaries between classes, reducing the tendency to memorize discrete training examples.

### 8.7 Technique 5: Warmup Cosine Decay Learning Rate Schedule

A custom learning rate scheduler was implemented with two phases:

1. **Warmup phase** (epochs 1–5): The learning rate increases linearly from $1 \times 10^{-7}$ to $1 \times 10^{-6}$, allowing the model to stabilize before making larger updates.
2. **Cosine decay phase** (epochs 6–30): The learning rate decreases following a cosine schedule, enabling smooth convergence.

```python
class WarmupCosineDecay(callbacks.Callback):
    def __init__(self, total_epochs, warmup_epochs, min_lr, max_lr):
        super().__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (epoch / self.warmup_epochs)
        else:
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))

        self.model.optimizer.learning_rate.assign(lr)

lr_scheduler = WarmupCosineDecay(
    total_epochs=30,
    warmup_epochs=5,
    min_lr=1e-7,
    max_lr=1e-6
)
```

### 8.8 Technique 6: Test Time Augmentation (TTA)

Test Time Augmentation improves inference robustness by generating multiple augmented versions of each input image, obtaining predictions for each variant, and averaging the resulting probability distributions:

```python
def apply_tta(model, dataset, num_augmentations=5):
    all_predictions = []
    all_labels = []

    for images, labels in dataset:
        batch_predictions = []

        # 1. Original image
        batch_predictions.append(model(images, training=False))

        # 2. Horizontal flip
        flipped_h = tf.image.flip_left_right(images)
        batch_predictions.append(model(flipped_h, training=False))

        # 3. Vertical flip
        flipped_v = tf.image.flip_up_down(images)
        batch_predictions.append(model(flipped_v, training=False))

        # 4. Horizontal + Vertical flip
        flipped_hv = tf.image.flip_left_right(flipped_v)
        batch_predictions.append(model(flipped_hv, training=False))

        # 5. Brightness adjustment (+10%)
        brightened = tf.image.adjust_brightness(images, delta=0.1)
        batch_predictions.append(model(brightened, training=False))

        # Average predictions
        avg_predictions = tf.reduce_mean(batch_predictions, axis=0)
        all_predictions.append(avg_predictions)
        all_labels.append(labels)

    return tf.concat(all_predictions, axis=0), \
           tf.concat(all_labels, axis=0)
```

The five TTA variants are:

| Variant | Transformation |
|---------|---------------|
| 1 | Original (no transformation) |
| 2 | Horizontal flip |
| 3 | Vertical flip |
| 4 | Horizontal + Vertical flip |
| 5 | Brightness increase (+10%) |

By ensembling predictions across these variants, TTA reduces the variance of individual predictions and improves classification consistency, particularly for ambiguous or borderline cases.

### 8.9 Phase 3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Loaded from Phase 1 (`mobilenetv2_skin_disease_model.keras`) |
| Unfrozen layers | Top 50 (conservative fine-tuning) |
| Optimizer | Adam (learning rate = 1e-6, clipnorm = 1.0) |
| Loss function | CategoricalCrossentropy (label_smoothing = 0.2) |
| Dropout rate | 0.5 |
| L2 regularization | 0.001 |
| MixUp alpha | 0.2 |
| Warmup epochs | 5 (1e-7 -> 1e-6) |
| LR schedule | Cosine decay after warmup |
| Max epochs | 30 |
| TTA variants | 5 |
| Early stopping | patience = 3, restore_best_weights = True |
| ReduceLROnPlateau | factor = 0.5, patience = 2, min_lr = 1e-7 |

### 8.10 Output

- **File**: `mobilenetv2_phase1_improved.keras`
- **Size**: 23.40 MB

### 8.11 Test Set Evaluation Results

**Without TTA:**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **73.91%** |
| Overall Loss | 0.9272 |

**Per-class Performance (without TTA):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Acne | 0.8068 | 0.7396 | 0.7717 | 96 |
| Eczema | 0.7842 | 0.7956 | 0.7899 | 137 |
| Tinea | 0.6105 | 0.6517 | 0.6304 | 89 |
| **Macro avg** | **0.7338** | **0.7290** | **0.7307** | 322 |
| **Weighted avg** | **0.7429** | **0.7391** | **0.7404** | 322 |

**Confusion Matrix (without TTA):**

|  | Predicted Acne | Predicted Eczema | Predicted Tinea |
|---|---|---|---|
| **True Acne** | 71 | 9 | 16 |
| **True Eczema** | 7 | 109 | 21 |
| **True Tinea** | 10 | 21 | 58 |

**With Test Time Augmentation (5 variants):**

| Metric | Value |
|--------|-------|
| **TTA Overall Accuracy** | **76.09%** |

**Per-class Performance (with TTA):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Acne | 0.8523 | 0.7812 | 0.8152 | 96 |
| Eczema | 0.7887 | 0.8175 | 0.8029 | 137 |
| Tinea | 0.6304 | 0.6517 | 0.6409 | 89 |
| **Macro avg** | **0.7571** | **0.7502** | **0.7530** | 322 |
| **Weighted avg** | **0.7639** | **0.7609** | **0.7618** | 322 |

**TTA Confusion Matrix:**

|  | Predicted Acne | Predicted Eczema | Predicted Tinea |
|---|---|---|---|
| **True Acne** | 75 | 7 | 14 |
| **True Eczema** | 5 | 112 | 20 |
| **True Tinea** | 8 | 23 | 58 |

**Analysis**: Phase 3 records 73.91% accuracy without TTA, and 76.09% with TTA — a modest +2.18 percentage point improvement from ensembling. This phase underperforms Phase 2 (84.16%) on the test set, despite applying more aggressive regularization. The elevated loss (0.9272 vs. 0.4889) indicates that the label smoothing and stronger dropout, combined with a much reduced learning rate (1e-6), caused under-fitting relative to the Phase 2 configuration. Specifically, restarting fine-tuning from the Phase 1 base model (rather than continuing from Phase 2) and restricting trainable layers to only 50 (vs. 55 in Phase 2) limited the model's ability to re-acquire the domain-adapted representations that Phase 2 had already learned. TTA provides a consistent improvement (+2.18%), confirming its value as an inference-time ensemble technique.

---

## 9. Mobile Application Development (Flutter)

### 9.1 Framework Selection

The mobile application was developed using Flutter, Google's open-source UI toolkit for building natively compiled applications from a single codebase. Flutter was selected for its cross-platform capability (Android and iOS from a single Dart codebase), rich widget library, high rendering performance, and native support for TensorFlow Lite integration through community packages.

### 9.2 Application Architecture

The application follows a feature-based directory structure with clear separation of concerns:

```
Flutter/lib/
├── main.dart                          # Application entry point
├── core/
│   ├── routing/
│   │   └── app_router.dart            # GoRouter navigation configuration
│   └── theme/
│       └── app_theme.dart             # Light/Dark theme definitions
├── features/
│   ├── onboarding/screens/            # Onboarding flow
│   │   ├── welcome_screen.dart
│   │   ├── login_screen.dart
│   │   ├── role_selection_screen.dart
│   │   └── personal_info_screen.dart
│   ├── core_workflow/screens/         # Main application workflow
│   │   ├── dashboard_screen.dart
│   │   ├── body_part_selection_screen.dart
│   │   └── capture_screen.dart
│   └── results/screens/              # Diagnosis results and XAI
│       ├── diagnosis_result_screen.dart
│       └── xai_heatmap_screen.dart
├── models/                            # Data models
│   ├── scan_result.dart
│   └── user.dart
├── providers/                         # Riverpod state providers
│   ├── scan_provider.dart
│   └── user_provider.dart
└── services/                          # Business logic services
    ├── ai/
    │   └── ai_service.dart            # TensorFlow Lite inference
    └── database/
        └── database_service.dart      # SQLite operations
```

### 9.3 State Management: Riverpod

The application employs Riverpod for state management, wrapping the root widget in a `ProviderScope`:

```dart
void main() {
  WidgetsFlutterBinding.ensureInitialized();

  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  runApp(
    const ProviderScope(
      child: DermatologyAIApp(),
    ),
  );
}

class DermatologyAIApp extends ConsumerWidget {
  const DermatologyAIApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);
    final themeMode = ref.watch(themeModeProvider);

    return MaterialApp.router(
      title: 'Dermatology AI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: themeMode,
      routerConfig: router,
    );
  }
}
```

Riverpod was chosen over alternatives (Provider, BLoC) for its compile-time safety, testability, independence from the widget tree, and support for asynchronous state management — all critical qualities for managing AI inference states and scan history.

### 9.4 Routing: GoRouter

Declarative routing is managed via `go_router`, with routes defined as constants in the `AppRoutes` class:

```dart
class AppRoutes {
  static const String welcome = '/welcome';
  static const String login = '/login';
  static const String roleSelection = '/role-selection';
  static const String personalInfo = '/personal-info';
  static const String dashboard = '/dashboard';
  static const String bodyPartSelection = '/body-part-selection';
  static const String capture = '/capture';
  static const String diagnosisResult = '/diagnosis-result';
  static const String xaiHeatmap = '/xai-heatmap';
}
```

### 9.5 Application Workflow

The application follows a sequential user workflow:

1. **Onboarding**: Welcome screen -> Login -> Role Selection (Patient / Healthcare Provider) -> Personal Information
2. **Core Workflow**: Dashboard (quick actions) -> Body Part Selection -> Image Capture (camera or gallery)
3. **Results**: Diagnosis Result (classification with confidence scores) -> XAI Heatmap Visualization

### 9.6 Screen Routes

| Route | Screen | Description |
|-------|--------|-------------|
| `/welcome` | WelcomeScreen | App introduction and entry point |
| `/login` | LoginScreen | User authentication |
| `/role-selection` | RoleSelectionScreen | Patient or Healthcare Provider role |
| `/personal-info` | PersonalInfoScreen | User demographic data collection |
| `/dashboard` | DashboardScreen | Main hub with quick actions |
| `/body-part-selection` | BodyPartSelectionScreen | Anatomical region selection for scanning |
| `/capture` | CaptureScreen | Camera/gallery image capture |
| `/diagnosis-result` | DiagnosisResultScreen | Classification results with confidence |
| `/xai-heatmap` | XAIHeatmapScreen | Grad-CAM heatmap visualization |

### 9.7 Technology Stack

| Component | Technology | Package | Purpose |
|-----------|-----------|---------|---------|
| Framework | Flutter 3.x | `flutter` | Cross-platform UI |
| Language | Dart 3.x | — | Application logic |
| State management | Riverpod | `flutter_riverpod: ^2.4.9` | Reactive state |
| Navigation | GoRouter | `go_router: ^13.0.0` | Declarative routing |
| Camera | Camera API | `camera: ^0.10.5+5` | Live camera preview |
| Image picker | Gallery/Camera | `image_picker: ^1.0.5` | Image selection |
| Local database | SQLite | `sqflite: ^2.3.0` | Scan history storage |
| File paths | Path provider | `path_provider: ^2.1.1` | Platform-specific paths |
| Permissions | Handler | `permission_handler: ^11.1.0` | Runtime permissions |
| Fonts | Google Fonts | `google_fonts: ^6.1.0` | Typography |
| SVG rendering | Flutter SVG | `flutter_svg: ^2.0.9` | Vector graphics |
| Date formatting | Intl | `intl: ^0.18.1` | Localized formatting |
| Unique IDs | UUID | `uuid: ^4.2.1` | Scan record identification |
| Logging | Logger | `logger: ^2.0.2+1` | Debug logging |
| AI inference | TFLite (planned) | `tflite_flutter` | On-device inference |

### 9.8 Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Project setup, architecture, routing, placeholder screens, theming | Completed |
| Phase 2 | TensorFlow Lite model integration, camera implementation, image preprocessing, model inference, XAI heatmap generation | Planned |
| Phase 3 | SQLite database implementation, scan history CRUD, user data management, local image storage | Planned |
| Phase 4 | UI/UX enhancement, animations, transitions, loading states, error handling, accessibility | Planned |

---

## 10. Discussion

### 10.1 Model Progression Summary

| Phase | Key Technique | Test Accuracy | Loss | TTA Accuracy | Model File | Size |
|-------|---------------|--------------|------|-------------|------------|------|
| Phase 1 | Transfer learning, frozen backbone | 75.16% | 0.6093 | — | `mobilenetv2_skin_disease_model.keras` | 9.22 MB |
| Phase 2 | Fine-tuning, top 55 layers unfrozen | **84.16%** | **0.4889** | — | `mobilenetv2_finetuned_model.keras` | 23.45 MB |
| Phase 3 | Anti-overfitting, top 50 layers, MixUp, label smoothing | 73.91% | 0.9272 | 76.09% | `mobilenetv2_phase1_improved.keras` | 23.40 MB |

**Best model: Phase 2** (84.16% test accuracy, loss 0.4889).

**Per-class accuracy across phases:**

| Class | Phase 1 | Phase 2 | Phase 3 (no TTA) | Phase 3 (TTA) |
|-------|---------|---------|-----------------|---------------|
| Acne | 80.21% | 85.42% | 73.96% | 78.12% |
| Eczema | 81.75% | **91.24%** | 79.56% | 81.75% |
| Tinea | 59.55% | 71.91% | 65.17% | 65.17% |
| **Overall** | **75.16%** | **84.16%** | **73.91%** | **76.09%** |

### 10.2 Contribution of Individual Techniques

**Class Weighting** (all phases): By assigning balanced class weights inversely proportional to class frequency, the model was prevented from developing a bias toward the majority class (Eczema, ~42%). This is particularly critical for medical applications where false negatives for minority classes can have significant clinical consequences.

**Transfer Learning** (Phase 1): Leveraging ImageNet-pretrained features provided a strong initialization, yielding 75.16% test accuracy with only the classification head trained. The frozen backbone approach minimized the risk of overfitting during initial training.

**Fine-Tuning** (Phase 2): Unfreezing the top 55 layers with a learning rate of 1e-5 produced the most significant single improvement: +8.99 percentage points (75.16% → 84.16%). Tinea recall improved from 59.55% to 71.91%, and Eczema recall reached 91.24%, the highest of any class across all phases. The ReduceLROnPlateau callback (halving LR on plateau, min 1e-7) ensured smooth convergence without catastrophic forgetting of pretrained representations.

**Enhanced Regularization, Label Smoothing, MixUp, and Conservative Fine-Tuning** (Phase 3): Phase 3 did not improve upon Phase 2 in absolute test accuracy (73.91% without TTA vs. 84.16%), primarily because it restarted fine-tuning from the Phase 1 base model rather than continuing from Phase 2, and restricted trainable layers to 50 rather than 55. The very low learning rate (1e-6) combined with label smoothing (0.2) also increased the test loss to 0.9272, suggesting that these combined constraints prevented the model from fully re-learning the domain-specific representations that Phase 2 had acquired.

**Test Time Augmentation** (Phase 3): TTA consistently provided a +2.18 percentage point improvement over the non-TTA baseline for Phase 3 (73.91% → 76.09%), confirming its value as an inference-time ensemble technique that reduces prediction variance without retraining.

### 10.3 Overfitting Mitigation Strategies and Observations

The evaluation results reveal an important finding regarding the overfitting mitigation strategies:

| Phase | Test Accuracy | Loss | Observation |
|-------|--------------|------|-------------|
| Phase 1 | 75.16% | 0.6093 | Underfitting — frozen backbone cannot adapt to dermatological features |
| Phase 2 | **84.16%** | **0.4889** | Optimal balance — fine-tuning improved generalization significantly |
| Phase 3 | 73.91% | 0.9272 | Under-fitting — excessive regularization and restarting from Phase 1 reduced performance |

The overfitting gap (difference between training and validation accuracy) serves as the primary diagnostic metric. A gap exceeding 5% indicates significant overfitting; a gap under 3% indicates effective regularization. In this project, the main generalization challenge was not overfitting per se, but rather finding the right fine-tuning depth and regularization balance. Phase 2's configuration (55 unfrozen layers, LR=1e-5, early stopping patience=5) proved to be the optimal trade-off. Phase 3's over-regularization demonstrates that for a dataset of ~8,000 augmented training images, a dropout rate of 0.5 combined with label smoothing of 0.2 and a learning rate as low as 1e-6 imposes too strong a constraint, preventing the model from converging to a good minimum.

### 10.4 Mobile Deployment Considerations

MobileNetV2 was selected specifically for its mobile deployment characteristics:

- **Parameter efficiency**: ~3.4 million parameters (vs. ~25.6 million for ResNet50), enabling fast inference on mobile hardware.
- **Depthwise separable convolutions**: Reduce computational cost by factoring standard convolutions into depthwise and pointwise operations.
- **TFLite compatibility**: Native support for conversion to TensorFlow Lite format, with additional optimization options including quantization (INT8, FP16) for further size and latency reduction.
- **Model size**: At 9.22–23.45 MB (depending on the phase), the model is practical for bundling within a mobile application package.

---

## 11. Conclusion & Future Work

### 11.1 Summary of Achievements

This project developed a complete pipeline for skin disease classification encompassing:

1. **Dataset curation**: A cleaned and augmented dataset of 8,803 dermatological images across three classes (Acne, Eczema, Tinea), with a stratified train/validation/test split and 4× augmentation of the training set.
2. **Multi-phase model training**: A three-phase training methodology with the following quantified outcomes on the 322-image test set:
   - Phase 1 (Transfer Learning): **75.16% accuracy**, loss 0.6093
   - Phase 2 (Fine-Tuning): **84.16% accuracy**, loss 0.4889 — best performing model (+8.99 pp over Phase 1)
   - Phase 3 (Anti-Overfitting): **73.91% accuracy** (76.09% with TTA), loss 0.9272
3. **Mobile application foundation**: A Flutter-based cross-platform application with a complete user workflow (onboarding, body part selection, image capture, diagnosis display, and XAI visualization), Riverpod state management, GoRouter navigation, and a modular feature-based architecture prepared for AI model integration.

### 11.2 Future Work

The following directions are planned for continued development:

1. **TFLite Conversion**: Convert the best-performing Keras model to TensorFlow Lite format with post-training quantization (INT8 or FP16) for on-device inference, reducing model size and inference latency.
2. **Grad-CAM XAI Integration**: Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to generate class-discriminative heatmaps overlaid on input images, providing visual explanations of which image regions most influenced the classification decision.
3. **Full Application Deployment**: Complete Phases 2–4 of the Flutter application, integrating the TFLite model, implementing SQLite persistence for scan history, and refining the user interface with animations, loading states, and accessibility features.
4. **Dataset Expansion**: Increase the dataset size and diversity by incorporating additional image sources, adding more skin disease classes beyond the current three, and improving demographic representation to reduce potential biases.
5. **Clinical Validation**: Conduct formal evaluation of the deployed system in a clinical setting, comparing model predictions against dermatologist diagnoses to assess real-world diagnostic accuracy and clinical utility.

---

## References

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *International Conference on Learning Representations (ICLR)*.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [Label smoothing]
- TensorFlow Documentation. *tf.keras.applications.MobileNetV2*. https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
- Flutter Documentation. https://docs.flutter.dev/
