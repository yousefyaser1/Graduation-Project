"""
Generates a professional academic PDF for Chapter 4: Implementation
Uses Windows TrueType fonts (Times New Roman, Georgia, Courier New)
for full Unicode support — eliminates all black-box rendering artifacts.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

# ─────────────────────────────────────────────────────────────────────────────
# REGISTER TRUETYPE FONTS  (full Unicode support, no black boxes)
# ─────────────────────────────────────────────────────────────────────────────
FONTS = r'C:\Windows\Fonts'

pdfmetrics.registerFont(TTFont('TNR',       os.path.join(FONTS, 'times.ttf')))
pdfmetrics.registerFont(TTFont('TNR-Bold',  os.path.join(FONTS, 'timesbd.ttf')))
pdfmetrics.registerFont(TTFont('TNR-Italic',os.path.join(FONTS, 'timesi.ttf')))
pdfmetrics.registerFont(TTFont('TNR-BI',    os.path.join(FONTS, 'timesbi.ttf')))
pdfmetrics.registerFont(TTFont('Cour',      os.path.join(FONTS, 'cour.ttf')))
pdfmetrics.registerFont(TTFont('Cour-Bold', os.path.join(FONTS, 'courbd.ttf')))

from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('TNR',
    normal='TNR', bold='TNR-Bold',
    italic='TNR-Italic', boldItalic='TNR-BI')

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
CLR_CODE_BG    = HexColor('#F5F5F5')
CLR_TABLE_HDR  = HexColor('#2C3E50')
CLR_TABLE_ALT  = HexColor('#EBF5FB')
CLR_TABLE_BRD  = HexColor('#BDC3C7')
CLR_HEADING    = HexColor('#1A252F')
CLR_SUBHEADING = HexColor('#1F3A4A')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def make_footer(canvas, doc):
    canvas.saveState()
    w, _ = A4
    canvas.setStrokeColor(HexColor('#AAAAAA'))
    canvas.setLineWidth(0.5)
    canvas.line(2.5*cm, 1.55*cm, w - 2.5*cm, 1.55*cm)
    canvas.setFont('TNR', 9)
    canvas.setFillColor(HexColor('#555555'))
    canvas.drawString(2.5*cm, 1.15*cm, 'Chapter 4: Implementation')
    canvas.drawCentredString(w / 2.0, 1.15*cm, f'- {doc.page} -')
    canvas.restoreState()

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
def S(**kw):
    """Shorthand ParagraphStyle factory."""
    name = kw.pop('name', '_anon')
    return ParagraphStyle(name, **kw)

def build_styles():
    st = {}

    st['chapter_label'] = S(name='chapter_label',
        fontName='TNR-Italic', fontSize=12, leading=18,
        textColor=HexColor('#555555'), alignment=TA_CENTER, spaceAfter=4)

    st['chapter'] = S(name='chapter',
        fontName='TNR-Bold', fontSize=22, leading=28,
        textColor=CLR_HEADING, alignment=TA_CENTER, spaceAfter=4)

    st['h2'] = S(name='h2',
        fontName='TNR-Bold', fontSize=14, leading=20,
        textColor=CLR_HEADING, spaceBefore=20, spaceAfter=6, keepWithNext=1)

    st['h3'] = S(name='h3',
        fontName='TNR-Bold', fontSize=12, leading=17,
        textColor=CLR_SUBHEADING, spaceBefore=12, spaceAfter=4, keepWithNext=1)

    st['body'] = S(name='body',
        fontName='TNR', fontSize=11, leading=17.5,
        alignment=TA_JUSTIFY, spaceAfter=8)

    st['bullet'] = S(name='bullet',
        fontName='TNR', fontSize=11, leading=16,
        alignment=TA_LEFT, spaceAfter=4,
        leftIndent=20, firstLineIndent=0)

    st['code'] = S(name='code',
        fontName='Cour', fontSize=8.5, leading=12.5,
        alignment=TA_LEFT)

    st['caption'] = S(name='caption',
        fontName='TNR-BI', fontSize=10, leading=14,
        alignment=TA_CENTER, spaceAfter=4, spaceBefore=4,
        textColor=HexColor('#444444'))

    st['endnote'] = S(name='endnote',
        fontName='TNR-Italic', fontSize=10, leading=14,
        alignment=TA_CENTER, textColor=HexColor('#666666'))

    # Inline table cell styles
    st['th'] = S(name='th',
        fontName='TNR-Bold', fontSize=10, leading=14,
        textColor=colors.white, spaceAfter=0)
    st['td'] = S(name='td',
        fontName='TNR', fontSize=10, leading=14,
        textColor=colors.black, spaceAfter=0)

    return st


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sp(n=8):
    return Spacer(1, n)

def rule():
    return HRFlowable(width='100%', thickness=0.6,
                      color=HexColor('#AAAAAA'), spaceAfter=4, spaceBefore=4)

def H2(text, st):
    return Paragraph(text, st['h2'])

def H3(text, st):
    return Paragraph(text, st['h3'])

def P(text, st):
    return Paragraph(text, st['body'])

def bullets(items, st):
    out = []
    for item in items:
        out.append(Paragraph(f'&#x2022;&#160;&#160;{item}', st['bullet']))
    return out

def code_block(lines, st):
    """Renders a code snippet in a light-grey bordered box."""
    escaped = []
    for line in lines:
        line = (line
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))
        escaped.append(line)
    text = '<br/>'.join(escaped)
    p = Paragraph(
        f'<font name="Cour" size="8.5">{text}</font>',
        S(name='ci', fontName='Cour', fontSize=8.5, leading=12.5,
          alignment=TA_LEFT, leftIndent=6, rightIndent=6))
    t = Table([[p]], colWidths=[14.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), CLR_CODE_BG),
        ('BOX',           (0,0), (-1,-1), 0.6, HexColor('#CCCCCC')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
    ]))
    return t

def data_table(header, rows, col_widths, caption, st):
    """Styled data table with header shading and alternating rows."""
    all_rows = []
    for ci, cell in enumerate(header):
        _ = ci  # unused
    formatted_header = [Paragraph(str(c), st['th']) for c in header]
    all_rows.append(formatted_header)
    for ri, row in enumerate(rows):
        all_rows.append([Paragraph(str(c), st['td']) for c in row])

    t = Table(all_rows, colWidths=col_widths)
    bg_commands = []
    for ri in range(1, len(rows) + 1):
        bg = colors.white if ri % 2 == 1 else CLR_TABLE_ALT
        bg_commands.append(('BACKGROUND', (0, ri), (-1, ri), bg))

    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), CLR_TABLE_HDR),
        ('GRID',          (0, 0), (-1, -1), 0.5, CLR_TABLE_BRD),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 7),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 7),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ] + bg_commands))

    return [Paragraph(caption, st['caption']), t, sp(12)]


# ─────────────────────────────────────────────────────────────────────────────
# STORY CONTENT
# ─────────────────────────────────────────────────────────────────────────────
def build_story(st):
    story = []

    # ── Title block ───────────────────────────────────────────────────────────
    story += [sp(10),
              Paragraph('Chapter 4', st['chapter_label']),
              Paragraph('Implementation', st['chapter']),
              rule(), sp(14)]

    # =========================================================================
    # 4.1  System Overview
    # =========================================================================
    story += [H2('4.1  System Overview', st),
    P('The implemented system is a mobile-first, AI-powered dermatology assistant capable '
      'of classifying three common skin conditions -- Acne, Eczema, and Tinea -- from '
      'clinical photographs, while providing visual explanations that indicate which regions '
      'of the image drove the prediction. The system integrates three interconnected '
      'components: a multi-phase deep learning pipeline trained on a curated dermatological '
      'dataset, a three-stage on-device inference engine, and a cross-platform Flutter '
      'mobile application.', st),
    P('The end-to-end workflow follows this sequence: the user captures or uploads a skin '
      'image through the application; the image is passed through a Variational Autoencoder '
      '(VAE) that acts as an anomaly detection gate to confirm the presence of a skin lesion; '
      'if a lesion is detected, the image is fed into an ensemble of two EfficientNet '
      'classifiers (EfficientNetB2 and EfficientNetB3) that produce a three-class probability '
      'distribution; finally, a Score-CAM algorithm generates a heatmap overlay highlighting '
      'the lesion regions most relevant to the classification decision. All inference runs '
      'entirely on-device using TensorFlow Lite, requiring no network connectivity after '
      'installation.', st)]

    # =========================================================================
    # 4.2  Development Environment
    # =========================================================================
    story += [H2('4.2  Development Environment and Tools', st),
    P('The project utilized two distinct environments: a local Windows 10 workstation for '
      'application development and initial prototyping, and Kaggle cloud GPU notebooks for '
      'computationally intensive model training.', st)]

    story.append(H3('4.2.1  Machine Learning Stack', st))
    story += data_table(
        ['Component', 'Tool / Version'],
        [
            ['Deep learning framework',   'TensorFlow / Keras (Python)'],
            ['GPU training platform',     'Kaggle (NVIDIA P100 / T4)'],
            ['Model architectures',       'MobileNetV2, EfficientNetB0 - B3'],
            ['Image processing (Python)', 'OpenCV (cv2), NumPy'],
            ['Class balancing',           'scikit-learn (compute_class_weight)'],
            ['TFLite conversion',         'tf.lite.TFLiteConverter'],
            ['Random seed',               '42 (all experiments)'],
        ],
        [6.5*cm, 8.0*cm],
        'Table 4.1 -- Machine Learning Tools and Frameworks', st)

    story.append(H3('4.2.2  Mobile Application Stack', st))
    story += data_table(
        ['Component', 'Tool / Version'],
        [
            ['Framework',               'Flutter 3.x'],
            ['Language',                'Dart 3.x'],
            ['State management',        'Riverpod (flutter_riverpod ^2.4.9)'],
            ['Navigation',              'GoRouter (go_router ^13.0.0)'],
            ['On-device AI inference',  'tflite_flutter'],
            ['Image processing (Dart)', 'image package'],
            ['Local persistence',       'SQLite (sqflite ^2.3.0)'],
            ['Camera integration',      'camera ^0.10.5+5'],
            ['Image picker',            'image_picker ^1.0.5'],
            ['File paths',              'path_provider ^2.1.1'],
            ['Runtime permissions',     'permission_handler ^11.1.0'],
        ],
        [5.5*cm, 9.0*cm],
        'Table 4.2 -- Flutter Application Dependencies', st)

    # =========================================================================
    # 4.3  Dataset Preparation
    # =========================================================================
    story.append(H2('4.3  Dataset Preparation', st))

    story.append(H3('4.3.1  Raw Dataset and Cleaning', st))
    story += [
    P('The initial dataset was organized into three class folders -- Acne, Eczema, and '
      'Tinea -- under a train/test directory structure sourced from publicly available '
      'dermatological image repositories. Before any model training, the dataset underwent '
      'systematic quality control. Duplicate images were identified and removed to prevent '
      'data leakage and artificially inflated performance metrics. Images that were clearly '
      'non-dermatological (histology slides, diagrams, or unrelated photographs), severely '
      'low-resolution, or heavily watermarked were excluded. Misclassified images identified '
      'through visual inspection were either relabeled or discarded. The cleaned output was '
      'stored in a dedicated directory, <i>Finalized_Clean_Data/</i>.', st),
    P('The original two-way (train/test) split was restructured into a three-way '
      '(train/validation/test) split with stratification to preserve class proportions across '
      'all three partitions. Approximately 90% of images were allocated to training, with 5% '
      'each for validation and test. The validation set was used exclusively for '
      'hyperparameter tuning and early stopping decisions; the test set was held out entirely '
      'and used only for final evaluation.', st)]

    story.append(H3('4.3.2  Offline Data Augmentation', st))
    story += [
    P('Data augmentation was applied offline to the training split only, using a '
      'deterministic Keras preprocessing pipeline. Each original training image was '
      'transformed three times, producing three distinct augmented copies. '
      'The augmentation pipeline was implemented as follows:', st),
    code_block([
        'augmentation_pipeline = tf.keras.Sequential([',
        '    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),',
        '    tf.keras.layers.RandomFlip(mode="vertical",   seed=42),',
        '    tf.keras.layers.RandomRotation(factor=0.0417,   # +/- 15 degrees',
        '                                   fill_mode="reflect", seed=42),',
        '    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1,',
        '                              fill_mode="reflect", seed=42),',
        '    tf.keras.layers.RandomBrightness(factor=0.1, seed=42),',
        '])',
    ], st), sp(8)]

    story += data_table(
        ['Technique', 'Parameters', 'Rationale'],
        [
            ['Horizontal Flip',   '50% probability',        'Lesions have no canonical left-right orientation'],
            ['Vertical Flip',     '50% probability',        'Augments viewpoint diversity'],
            ['Random Rotation',   '+/- 15 deg, reflect',    'Simulates camera angle variation'],
            ['Random Zoom',       '+/- 10%, reflect fill',  'Mimics varying capture distances'],
            ['Random Brightness', '+/- 10%',                'Accounts for lighting variation'],
        ],
        [3.8*cm, 3.8*cm, 6.9*cm],
        'Table 4.3 -- Data Augmentation Techniques Applied to Training Split', st)

    story.append(P(
        'The choice of <i>fill_mode="reflect"</i> avoided black border artifacts that could '
        'be misinterpreted as disease features, and a fixed seed of 42 ensured full '
        'reproducibility. The validation and test sets were left unmodified to ensure '
        'unbiased evaluation.', st))

    story.append(H3('4.3.3  Final Dataset Statistics', st))
    story.append(P(
        'The augmented dataset, stored in <i>New_Augmented_Dataset/</i>, has the following '
        'composition after augmentation was applied exclusively to the training split:', st))
    story += data_table(
        ['Split', 'Acne', 'Eczema', 'Tinea', 'Total'],
        [
            ['Training',   '2,880', '3,460', '1,816', '8,156'],
            ['Validation', '93',    '142',   '90',    '325'],
            ['Test',       '96',    '137',   '89',    '322'],
            ['<b>Total</b>','<b>3,069</b>','<b>3,739</b>','<b>1,995</b>','<b>8,803</b>'],
        ],
        [3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm],
        'Table 4.4 -- Final Dataset Distribution Across Splits and Classes', st)

    story.append(P(
        'A notable class imbalance exists in the training set: Eczema comprises approximately '
        '42.4% of training samples, while Tinea accounts for only 22.3%. This imbalance was '
        'addressed throughout training using dynamically computed class weights.', st))

    # =========================================================================
    # 4.4  Model Development
    # =========================================================================
    story.append(H2('4.4  Model Development', st))

    story.append(H3('4.4.1  Training Configuration and Data Pipeline', st))
    story += [
    P('All training pipelines followed a consistent data loading and optimization pattern. '
      'Images were loaded using <font name="Cour" size="10">tf.keras.utils.'
      'image_dataset_from_directory</font> with categorical labels, then processed through '
      'a pipeline combining in-memory caching and prefetching:', st),
    code_block([
        'AUTOTUNE = tf.data.AUTOTUNE',
        'train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y),',
        '                        num_parallel_calls=AUTOTUNE)',
        'train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)',
    ], st), sp(8),
    P('The <font name="Cour" size="10">.cache()</font> call stored the preprocessed '
      'dataset in memory after the first epoch, eliminating redundant I/O in subsequent '
      'epochs. The <font name="Cour" size="10">.prefetch(AUTOTUNE)</font> operation '
      'overlapped CPU data preparation with GPU computation, maximizing GPU utilization '
      'during training on Kaggle.', st),
    P('To address class imbalance, dynamic class weights were computed using scikit-learn '
      'before each training run, assigning weights inversely proportional to class '
      'frequency: w_c = N / (k * n_c), where N is total samples, k is number of classes, '
      'and n_c is the count for class c. Tinea, as the minority class, received the '
      'highest weight throughout all training phases.', st)]

    story.append(H3('4.4.2  Phase 1: MobileNetV2 Transfer Learning', st))
    story += [
    P('The first training phase established a baseline using MobileNetV2 pretrained on '
      'ImageNet as a frozen feature extractor. MobileNetV2 was selected for its lightweight '
      'depthwise separable convolution architecture, which offered a strong balance between '
      'classification accuracy and mobile inference efficiency.', st),
    P('The entire MobileNetV2 backbone was frozen '
      '(<font name="Cour" size="10">base_model.trainable = False</font>), and a '
      'lightweight classification head was appended:', st),
    code_block([
        'base_model = MobileNetV2(weights="imagenet", include_top=False,',
        '                          input_shape=(224, 224, 3))',
        'base_model.trainable = False',
        '',
        'inputs  = tf.keras.Input(shape=(224, 224, 3))',
        'x       = base_model(inputs, training=False)',
        'x       = layers.GlobalAveragePooling2D()(x)',
        'x       = layers.Dropout(0.2)(x)',
        'outputs = layers.Dense(3, activation="softmax")(x)',
        'model   = models.Model(inputs, outputs)',
    ], st), sp(8),
    P('The model was compiled with Adam (learning rate = 1e-4) and categorical '
      'cross-entropy loss, trained for up to 20 epochs with EarlyStopping '
      '(patience = 3, monitoring validation loss). Input images were scaled to '
      'the range [-1, 1] using MobileNetV2\'s native preprocess_input function.', st),
    P('This phase achieved <b>75.16% test accuracy</b>, with strong performance on Acne '
      '(80.21%) and Eczema (81.75%), but weaker on Tinea (59.55%) due to the frozen '
      'backbone\'s inability to adapt to dermatology-specific features.', st)]

    story += data_table(
        ['Parameter', 'Value'],
        [
            ['Input size',       '224 x 224 x 3'],
            ['Preprocessing',    'preprocess_input (scales to [-1, 1])'],
            ['Batch size',       '32'],
            ['Optimizer',        'Adam (learning rate = 1e-4)'],
            ['Loss function',    'Categorical Cross-Entropy'],
            ['Classification head', 'GAP -> Dropout(0.2) -> Dense(3, softmax)'],
            ['Callback',         'EarlyStopping (patience=3, restore best weights)'],
            ['Test accuracy',    '75.16%'],
        ],
        [5.5*cm, 9.0*cm],
        'Table 4.5 -- Phase 1 (MobileNetV2 Transfer Learning) Configuration', st)

    story.append(H3('4.4.3  Phase 2: Fine-Tuning (Layer Unfreezing)', st))
    story += [
    P('Phase 2 loaded the Phase 1 model and selectively unfroze the upper layers of the '
      'MobileNetV2 backbone to allow domain adaptation. The first 100 of MobileNetV2\'s '
      '155 layers -- which encode low-level features such as edges and textures -- remained '
      'frozen, while layers 100 onward were made trainable:', st),
    code_block([
        'base_model.trainable = True',
        'for i, layer in enumerate(base_model.layers):',
        '    layer.trainable = (i >= 100)',
    ], st), sp(8),
    P('A drastically reduced learning rate of 1e-5 -- one tenth of Phase 1\'s rate -- '
      'was critical to prevent catastrophic forgetting of the pretrained representations. '
      'Three callbacks were employed: <b>EarlyStopping</b> (patience=5, increased from 3 '
      'to accommodate the slower convergence of fine-tuning), <b>ReduceLROnPlateau</b> '
      '(factor=0.5, patience=2, min_lr=1e-7), and <b>ModelCheckpoint</b> (saving only '
      'the epoch with the best validation loss).', st),
    P('This phase produced the most significant single improvement across all MobileNetV2 '
      'experiments: <b>84.16% test accuracy</b>, an improvement of 8.99 percentage points '
      'over Phase 1. Tinea recall improved from 59.55% to 71.91%, and Eczema recall '
      'reached 91.24%. The model was saved as '
      '<font name="Cour" size="10">mobilenetv2_finetuned_model.keras</font> (23.45 MB).', st)]

    story += data_table(
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        [
            ['Acne',         '0.9213', '0.8542', '0.8865', '96'],
            ['Eczema',       '0.8065', '0.9124', '0.8562', '137'],
            ['Tinea',        '0.8205', '0.7191', '0.7665', '89'],
            ['Macro avg',    '0.8494', '0.8286', '0.8364', '322'],
            ['Weighted avg', '0.8446', '0.8416', '0.8404', '322'],
        ],
        [3.5*cm, 2.7*cm, 2.7*cm, 2.7*cm, 2.9*cm],
        'Table 4.6 -- Phase 2 Per-Class Classification Report (Test Set, 322 images)', st)

    story.append(H3('4.4.4  Phase 3: Anti-Overfitting Strategies', st))
    story += [
    P('Phase 3 explored a comprehensive suite of regularization techniques applied on top '
      'of the Phase 1 base model, with the aim of further improving generalization. Five '
      'complementary strategies were implemented simultaneously:', st)]
    story += bullets([
        '<b>Conservative Fine-Tuning:</b> Only the top 50 layers of the MobileNetV2 '
        'backbone were made trainable (vs. 55 in Phase 2), reducing the number of free '
        'parameters.',
        '<b>Enhanced Regularization:</b> Dropout was increased from 0.2 to 0.5; L2 weight '
        'decay of 0.001 was added; gradient clipping (clipnorm=1.0) prevented exploding '
        'gradients.',
        '<b>Label Smoothing:</b> A smoothing factor of 0.2 was applied to the categorical '
        'cross-entropy loss, replacing hard one-hot targets with soft distributions to '
        'discourage overconfidence and improve calibration.',
        '<b>Online MixUp Augmentation:</b> Synthetic samples were generated by linearly '
        'interpolating between image pairs and their labels '
        '(lambda ~ Uniform(0, 0.2)), encouraging smoother decision boundaries.',
        '<b>Warmup Cosine Decay Schedule:</b> A custom learning rate callback implemented a '
        '5-epoch linear warmup from 1e-7 to 1e-6, followed by cosine annealing for the '
        'remaining epochs.',
    ], st)
    story += [sp(6),
    P('Despite the sophistication of these techniques, Phase 3 recorded <b>73.91% test '
      'accuracy</b> without Test Time Augmentation (TTA) and 76.09% with TTA -- both '
      'below Phase 2\'s 84.16%. The under-performance stemmed from restarting fine-tuning '
      'from the Phase 1 base model rather than continuing from Phase 2\'s already-adapted '
      'weights, combined with an excessively low learning rate (1e-6) and strong '
      'regularization that together prevented the model from converging to an equally good '
      'minimum. This result confirmed that Phase 2\'s configuration -- 55 unfrozen layers, '
      'lr=1e-5, moderate dropout -- represented the optimal trade-off for this dataset '
      'size (~8,000 training images).', st)]

    story.append(H3('4.4.5  Transition to EfficientNet', st))
    story += [
    P('Following the MobileNetV2 experiments, the architecture was upgraded to the '
      'EfficientNet family to push accuracy further. EfficientNet models are designed '
      'through compound scaling -- simultaneously scaling network depth, width, and input '
      'resolution using a principled coefficient -- resulting in significantly higher '
      'accuracy at equivalent parameter budgets compared to MobileNetV2.', st),
    P('Two key design decisions distinguished the EfficientNet training:', st)]
    story += bullets([
        '<b>Focal Loss:</b> Standard categorical cross-entropy was replaced with '
        'CategoricalFocalCrossentropy(alpha, gamma=2.0), where alpha is the per-class '
        'inverse-frequency weighting. Focal loss down-weights well-classified examples '
        'and focuses training on hard, misclassified samples -- particularly beneficial '
        'for the Tinea class.',
        '<b>Online Data Augmentation Within the Model Graph:</b> Augmentation layers '
        '(RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast, '
        'RandomBrightness) were embedded directly as the first layers of the model, '
        'applied stochastically during each training pass and automatically bypassed at '
        'inference time (training=False).',
    ], st)
    story.append(sp(6))

    story.append(H3('4.4.6  EfficientNetB2 Training', st))
    story += [
    P('EfficientNetB2 uses a native input resolution of 260 x 260 pixels and has '
      'approximately 339 layers. Training followed a two-phase approach executed entirely '
      'on Kaggle GPU:', st),
    P('<b>Phase 1 -- Frozen Backbone:</b> The backbone was entirely frozen and only the '
      'classification head was trained using Adam (lr = 1e-3) with focal loss. The head '
      'included a 256-unit Dense layer with BatchNormalization and Dropout(0.45), followed '
      'by the 3-class softmax output.', st),
    P('<b>Phase 2 -- Selective Unfreezing:</b> The best Phase 1 checkpoint was loaded and '
      'the top ~120 layers (from layer 220 onward) of the EfficientNetB2 backbone were '
      'unfrozen. The model was recompiled with lr = 1e-5 and trained for up to 60 epochs. '
      'An Eczema weight boost factor of 1.2x was applied on top of the balanced class '
      'weights, calibrated through experimentation to prevent the majority class from '
      'compressing per-class performance.', st)]

    story += data_table(
        ['Parameter', 'Phase 1', 'Phase 2'],
        [
            ['Input resolution',      '260 x 260',    '260 x 260'],
            ['Backbone',              'Frozen',        'Top ~120 layers unfrozen'],
            ['Learning rate',         '1e-3',          '1e-5'],
            ['Loss',                  'Focal (g=2.0)', 'Focal (g=2.0)'],
            ['Max epochs',            '20',            '60'],
            ['EarlyStopping patience','15',             '15'],
            ['ReduceLROnPlateau',     'factor=0.3, patience=8', 'factor=0.3, patience=8'],
        ],
        [5.0*cm, 4.25*cm, 5.25*cm],
        'Table 4.7 -- EfficientNetB2 Training Configuration', st)

    story.append(H3('4.4.7  EfficientNetB3 Training', st))
    story += [
    P('EfficientNetB3 operates at 300 x 300 pixels and has approximately 385 layers, '
      'offering greater representational capacity than B2. The same two-phase training '
      'procedure was applied: frozen backbone training at lr = 1e-3, followed by '
      'fine-tuning with the top 150 layers unfrozen (backbone layers 235 onward) at '
      'lr = 1e-5, with up to 60 epochs and EarlyStopping (patience=15).', st),
    P('EfficientNetB3 serves a dual role in the final pipeline: as a classifier in the '
      'ensemble and as the backbone for Score-CAM feature extraction. Its '
      '<font name="Cour" size="10">top_activation</font> layer produces feature maps '
      'of shape (1, 10, 10, 1536), which form the basis for the Score-CAM heatmap '
      'computation described in Section 4.5.3.', st)]

    # =========================================================================
    # 4.5  Three-Stage Inference Pipeline
    # =========================================================================
    story.append(H2('4.5  Three-Stage Inference Pipeline', st))
    story.append(P(
        'The final deployed system implements a sequential three-stage inference pipeline. '
        'Each stage acts as a gate or enrichment step, adding progressively more specific '
        'information about the input image. The pipeline design ensures that the most '
        'computationally expensive operations (CNN classification and Score-CAM) are only '
        'invoked when a genuine skin lesion has been confirmed by the preceding stage.', st))

    story.append(H3('4.5.1  Stage 1: VAE Anomaly Detection Gate', st))
    story += [
    P('The first stage uses a Variational Autoencoder (VAE) trained on normal skin images '
      'as an anomaly gate. Images that do not exhibit skin disease characteristics are '
      'flagged as normal and rejected before reaching the classifier, preventing spurious '
      'disease classifications on non-clinical photographs.', st),
    P('The VAE is applied using a sliding window approach over the input image. The image '
      'is first downsampled to a maximum width of 1,280 pixels (if larger) using linear '
      'interpolation. A 64 x 64-pixel patch is then extracted at each position with a '
      'stride of 32 pixels, producing overlapping coverage of the entire image. Each '
      'patch is fed into the VAE, which reconstructs it; the mean squared error (MSE) '
      'between the original and reconstructed patch is the anomaly signal.', st),
    code_block([
        'for (int y = 0; y <= h - patchSize; y += stride) {',
        '  for (int x = 0; x <= w - patchSize; x += stride) {',
        '    final patch = img.copyCrop(original,',
        '        x: x, y: y, width: patchSize, height: patchSize);',
        '    _fillFloat32(patch, inputBuf, patchSize, patchSize); // normalize to [0,1]',
        '    _vae!.invoke();',
        '    final mse = _vae!.getOutputTensor(0).data',
        '                    .buffer.asFloat32List()[0];',
        '    patches++;',
        '    if (mse > anomalyThreshold) anomalous++;',
        '  }',
        '}',
    ], st), sp(8),
    P('An MSE exceeding the calibrated threshold of <b>0.008</b> (adjusted from the '
      'original PyTorch threshold of 0.006 to account for a systematic TFLite '
      'quantization offset of ~0.002) marks a patch as anomalous. If the ratio of '
      'anomalous patches to total patches exceeds <b>20%</b>, the image proceeds to '
      'Stage 2; otherwise, the pipeline returns a "No skin disease detected" result '
      'immediately.', st)]

    story.append(H3('4.5.2  Stage 2: CNN Ensemble Classification', st))
    story += [
    P('Images flagged as anomalous proceed to the classification stage, where an ensemble '
      'of two independently trained EfficientNet models produces the final diagnosis. '
      'Ensemble methods reduce prediction variance and improve generalization compared to '
      'any single model. The two models\' different input resolutions (260 x 260 for B2 '
      'vs. 300 x 300 for B3) expose them to slightly different feature scales, making '
      'their predictions complementary.', st),
    P('Because both EfficientNet models include an internal Rescaling(1/255) layer baked '
      'into the backbone, the application passes raw pixel values in the range [0, 255]. '
      'Sending pre-normalized [0, 1] values would result in double-normalization and '
      'near-black inputs, causing the classifier to produce near-uniform probability '
      'distributions. This contrast with the VAE -- which requires [0, 1] normalized '
      'input -- was a critical distinction implemented through separate preprocessing '
      'helper functions.', st),
    code_block([
        '// Ensemble: 50/50 average of B2 and B3 softmax outputs',
        'List<double> ensemble = List.generate(',
        '    3, (i) => (probsB2[i] + probsB3[i]) / 2.0);',
        '',
        'final predIdx    = _argmax(ensemble);',
        'final diagnosis  = _classes[predIdx];   // "Acne" | "Eczema" | "Tinea"',
        'final confidence = ensemble[predIdx];',
    ], st), sp(8)]

    story.append(H3('4.5.3  Stage 3: Score-CAM Explainability', st))
    story += [
    P('The third stage generates a Score-CAM (Score-weighted Class Activation Map) '
      'heatmap that visually highlights which regions of the skin image most strongly '
      'influenced the classification decision. Unlike Grad-CAM, Score-CAM is '
      'gradient-free: it uses the classifier\'s forward-pass response to channel-masked '
      'images to assign importance scores. This makes it directly compatible with TFLite '
      'models, which do not support gradient computation on-device.', st),
    P('The algorithm proceeds through six steps:', st)]
    story += bullets([
        '<b>Feature Extraction:</b> The image (resized to 300 x 300) is fed into the B3 '
        'feature extractor TFLite model, producing a tensor of shape (1, 10, 10, 1536) '
        '-- 1,536 channels, each a 10 x 10 spatial activation map.',
        '<b>Top-K Channel Selection:</b> The 1,536 channels are ranked by mean absolute '
        'activation magnitude. The top K=30 channels are selected, reducing masked '
        'forward passes from 1,536 to 30 (approx. 3 seconds on a mid-range device).',
        '<b>Masked Forward Passes:</b> For each selected channel, the 10 x 10 activation '
        'map is upsampled to 300 x 300 and normalized to [0, 1], then applied as a '
        'spatial mask: masked_image = original_image x attention_mask. The masked image '
        'is passed through the B3 classifier and the target class probability is '
        'recorded as the channel\'s importance score.',
        '<b>Score Normalization:</b> The 30 raw scores are passed through a numerically '
        'stable softmax to produce weights summing to 1.',
        '<b>Weighted Heatmap:</b> The final heatmap is the weighted sum of the 30 '
        'upsampled feature maps. A ReLU removes negative contributions, and the result '
        'is min-max normalized to [0, 1].',
        '<b>Visualization:</b> The heatmap is colorized using a jet colormap (blue = low '
        'importance, red = high importance), resized to the original image dimensions, '
        'and blended with the original at a 60% / 40% ratio (original / heatmap). '
        'The overlay JPEG is saved to the device\'s temporary directory.',
    ], st)
    story.append(sp(6))

    # =========================================================================
    # 4.6  TFLite Conversion
    # =========================================================================
    story.append(H2('4.6  TFLite Model Conversion and Deployment', st))
    story += [
    P('Four TFLite models are bundled in the application\'s assets directory '
      '(<font name="Cour" size="10">assets/models/</font>). Conversion was performed '
      'using <font name="Cour" size="10">tf.lite.TFLiteConverter.from_keras_model()</font> '
      'with dynamic range quantization '
      '(<font name="Cour" size="10">tf.lite.Optimize.DEFAULT</font>) to reduce model '
      'size and improve inference latency on mobile hardware.', st)]

    story += data_table(
        ['File', 'Purpose', 'Input Shape', 'Input Range'],
        [
            ['vae_model.tflite',            'VAE anomaly gate',          '(1, 64, 64, 3)',   '[0, 1]'],
            ['cnn_b2_model.tflite',         'EfficientNetB2 classifier', '(1, 260, 260, 3)', '[0, 255]'],
            ['cnn_b3_model.tflite',         'EfficientNetB3 classifier', '(1, 300, 300, 3)', '[0, 255]'],
            ['b3_feature_extractor.tflite', 'Score-CAM feature maps',   '(1, 300, 300, 3)', '[0, 255]'],
        ],
        [4.5*cm, 3.8*cm, 3.5*cm, 2.7*cm],
        'Table 4.8 -- TFLite Models Bundled in the Application (all float32)', st)

    story.append(P(
        'The training-time augmentation layers embedded in the EfficientNet model graph '
        'are automatically removed by the converter, as they are no-ops at inference time '
        '(training=False). Prior to deployment, each TFLite model was verified for '
        'determinism: two forward passes on the same input produced identical outputs '
        '(maximum absolute difference < 1e-6), ensuring consistent results on-device.', st))

    # =========================================================================
    # 4.7  Flutter Application
    # =========================================================================
    story.append(H2('4.7  Flutter Mobile Application', st))

    story.append(H3('4.7.1  Application Architecture', st))
    story += [
    P('The Flutter application is structured following a feature-based directory '
      'organization, cleanly separating concerns across onboarding, core workflow, '
      'and results domains:', st),
    code_block([
        'Flutter/lib/',
        '|-- main.dart                         # App entry, ProviderScope',
        '|-- core/',
        '|   |-- routing/app_router.dart       # GoRouter configuration',
        '|   |-- theme/app_theme.dart          # Light and dark theme definitions',
        '|   +-- widgets/                      # Shared UI components',
        '|-- features/',
        '|   |-- onboarding/screens/           # Welcome, Login, RoleSelection,',
        '|   |                                   PersonalInfo, AllSet, HowItWorks',
        '|   |-- core_workflow/screens/        # Home, BodyPartSelection, Capture,',
        '|   |                                   Analyzing, History, Profile,',
        '|   |                                   EditProfile, Notifications, HelpSupport',
        '|   +-- results/screens/              # DiagnosisResult',
        '|-- models/',
        '|   |-- scan_result.dart              # Scan data model',
        '|   +-- user.dart                     # User data model',
        '|-- providers/',
        '|   |-- scan_provider.dart            # Scan state management',
        '|   +-- user_provider.dart            # User / auth state management',
        '+-- services/',
        '    |-- ai/ai_service.dart            # Full inference pipeline (singleton)',
        '    +-- database/database_service.dart# SQLite CRUD operations',
    ], st), sp(8)]

    story.append(H3('4.7.2  State Management with Riverpod', st))
    story += [
    P('The application uses Riverpod for all reactive state management. The root widget '
      'is wrapped in a <font name="Cour" size="10">ProviderScope</font>, enabling '
      'compile-time safe access to providers throughout the widget tree. '
      '<font name="Cour" size="10">DermatologyAIApp</font> is a '
      '<font name="Cour" size="10">ConsumerWidget</font> that watches both a router '
      'provider and a theme mode provider, ensuring the entire application rebuilds '
      'correctly in response to either state change.', st),
    P('Riverpod was preferred over BLoC or the basic Provider package for its compile-time '
      'type safety, its independence from the widget tree (providers are accessible without '
      'a BuildContext), and its clean handling of asynchronous inference states through '
      '<font name="Cour" size="10">AsyncValue</font>.', st)]

    story.append(H3('4.7.3  Navigation with GoRouter', st))
    story += [
    P('All application navigation is handled declaratively through '
      '<font name="Cour" size="10">go_router</font>. Route paths are defined as typed '
      'constants in an <font name="Cour" size="10">AppRoutes</font> class, preventing '
      'hard-coded string errors. The router handles the complete user journey in three '
      'phases:', st)]
    story += bullets([
        '<b>Onboarding Flow:</b> Welcome -- Login -- Role Selection -- Personal Info -- All Set',
        '<b>Core Scanning Workflow:</b> Home Dashboard -- Body Part Selection -- Capture '
        '-- Analyzing -- Diagnosis Result',
        '<b>Auxiliary Screens:</b> Scan History, Profile, Edit Profile, Notifications, '
        'Help and Support, Specialist Dashboard (Healthcare Provider role)',
    ], st)
    story.append(sp(6))

    story += data_table(
        ['Route', 'Screen', 'Description'],
        [
            ['/welcome',             'WelcomeScreen',          'App introduction and entry point'],
            ['/login',               'LoginScreen',            'User authentication'],
            ['/role-selection',      'RoleSelectionScreen',    'Patient or Healthcare Provider'],
            ['/personal-info',       'PersonalInfoScreen',     'Demographic data collection'],
            ['/dashboard',           'HomeScreen',             'Main hub with quick actions'],
            ['/body-part-selection', 'BodyPartSelectionScreen','Anatomical region selection'],
            ['/capture',             'CaptureScreen',          'Camera or gallery image capture'],
            ['/analyzing',           'AnalyzingScreen',        'Real-time inference progress'],
            ['/diagnosis-result',    'DiagnosisResultScreen',  'Classification result with heatmap'],
            ['/history',             'HistoryScreen',          'Chronological scan history'],
        ],
        [3.8*cm, 4.5*cm, 6.2*cm],
        'Table 4.9 -- Application Screen Routes', st)

    story.append(H3('4.7.4  AI Service Integration', st))
    story += [
    P('The <font name="Cour" size="10">AIService</font> class is implemented as a '
      'singleton and provides the full three-stage inference pipeline as a single '
      'asynchronous entry point, <font name="Cour" size="10">analyzeImage()</font>. '
      'It accepts an image file path and an optional progress callback that fires with '
      'a step index (0-3) as each stage begins, enabling the UI to display a real-time '
      'progress indicator during analysis:', st),
    code_block([
        'Future<AnalysisResult> analyzeImage({',
        '  required String imagePath,',
        '  void Function(int step)? onStepChange,',
        '  // 0=preprocess  1=VAE  2=CNN  3=Score-CAM',
        '}) async { ... }',
    ], st), sp(8),
    P('The four TFLite interpreters (VAE, B2, B3, and B3 feature extractor) are loaded '
      'lazily on first call via '
      '<font name="Cour" size="10">Interpreter.fromAsset()</font> and held in memory '
      'for the application\'s lifetime. To keep the UI responsive during the 30-iteration '
      'Score-CAM computation, the loop periodically yields to the Flutter event loop:', st),
    code_block([
        '// Yield every 5 iterations to keep animations running',
        'if (ki % 5 == 0) await Future.delayed(Duration.zero);',
    ], st), sp(8)]

    story.append(H3('4.7.5  Local Database Layer', st))
    story += [
    P('Scan history and user account data are persisted locally using SQLite, accessed '
      'through the <font name="Cour" size="10">sqflite</font> package. The '
      '<font name="Cour" size="10">DatabaseService</font> singleton ensures a single '
      'database connection. The schema contains two tables:', st),
    code_block([
        'CREATE TABLE scans (',
        '  id                  TEXT  PRIMARY KEY,',
        '  image_path          TEXT  NOT NULL,',
        '  body_part           TEXT  NOT NULL,',
        '  diagnosis           TEXT  NOT NULL,',
        '  confidence          REAL  NOT NULL,',
        '  class_probabilities TEXT  NOT NULL,  -- JSON-encoded map',
        '  timestamp           INTEGER NOT NULL,',
        '  notes               TEXT,',
        '  heatmap_path        TEXT',
        ');',
        '',
        'CREATE TABLE users (',
        '  id               TEXT     PRIMARY KEY,',
        '  name             TEXT     NOT NULL,',
        '  email            TEXT     NOT NULL UNIQUE,',
        '  password         TEXT     NOT NULL,',
        '  role             TEXT     NOT NULL DEFAULT "patient",',
        '  age              INTEGER,',
        '  gender           TEXT,',
        '  medical_history  TEXT,',
        '  created_at       INTEGER  NOT NULL',
        ');',
    ], st), sp(8),
    P('The schema supports non-destructive versioned migrations through an '
      '<font name="Cour" size="10">onUpgrade</font> callback. The application supports '
      'two user roles -- Patient and Healthcare Provider -- captured during onboarding; '
      'Healthcare Providers have access to a specialist dashboard providing an aggregated '
      'view over scan history and patient data.', st)]

    story.append(H3('4.7.6  User Interface and Screen Flow', st))
    story += [
    P('The application implements a complete, production-ready user interface with both '
      'light and dark themes defined centrally in the '
      '<font name="Cour" size="10">AppTheme</font> class. Key screens include:', st)]
    story += bullets([
        '<b>Welcome Screen:</b> Application introduction with animated entry, leading '
        'to login or registration.',
        '<b>Login and Registration Flow:</b> Email/password authentication with role '
        'selection (Patient or Healthcare Provider) and demographic data collection '
        '(age, gender, medical history).',
        '<b>Home Dashboard:</b> Quick-action hub presenting recent scan history, a scan '
        'initiation button, and navigation to all secondary screens.',
        '<b>Body Part Selection:</b> An anatomical map enabling the user to specify the '
        'affected region before capture, providing clinical context stored with each '
        'scan record.',
        '<b>Capture Screen:</b> Integrates the device camera or gallery image picker '
        'with a preview before submitting to the analysis pipeline.',
        '<b>Analyzing Screen:</b> A step-by-step progress display that updates in real '
        'time as each pipeline stage completes (preprocessing -- VAE -- CNN -- Score-CAM).',
        '<b>Diagnosis Result Screen:</b> Presents the primary diagnosis, confidence '
        'percentage, per-class probability breakdown, and the Score-CAM heatmap overlay. '
        'A notes field allows annotation before saving.',
        '<b>History Screen:</b> A chronological list of all saved scans, each showing '
        'the diagnosis, body part, confidence score, and timestamp.',
    ], st)
    story.append(sp(6))

    # =========================================================================
    # 4.8  Challenges and Solutions
    # =========================================================================
    story.append(H2('4.8  Challenges and Solutions', st))
    story.append(P(
        'Several technical challenges arose during implementation. Each was diagnosed '
        'systematically and resolved before the final system was deployed.', st))

    challenges = [
        (
            'Challenge 1: Preprocessing Mismatch Between Training and Inference',
            'EfficientNetB2 and B3 were trained with an internal Rescaling(1/255) layer '
            'embedded in the model graph. During initial TFLite integration, the application '
            'was normalizing pixel values to [0, 1] before passing them to the interpreter, '
            'resulting in double normalization and near-zero classifier activations. This '
            'caused the models to output near-uniform probability distributions regardless '
            'of the input image.',
            'Passing raw [0, 255] pixel values to both EfficientNet TFLite models, while '
            'retaining [0, 1] normalization only for the VAE. Two separate helper '
            'functions -- fillFloat32Raw() for EfficientNet and fillFloat32() for '
            'the VAE -- were implemented to handle the two normalization modes cleanly.'
        ),
        (
            'Challenge 2: VAE Threshold Calibration Across Frameworks',
            'The VAE was originally calibrated in PyTorch with an MSE anomaly threshold '
            'of 0.006. After conversion to TFLite, a systematic offset of approximately '
            '+0.002 was observed in reconstruction errors, attributable to differences in '
            'floating-point precision and dynamic range quantization behavior between the '
            'two runtimes.',
            'The threshold was recalibrated empirically to 0.008 for the TFLite runtime, '
            'producing equivalent anomaly detection behavior to the original PyTorch model.'
        ),
        (
            'Challenge 3: UI Responsiveness During Score-CAM Computation',
            'The 30-iteration Score-CAM loop ran synchronously on the main Dart isolate, '
            'causing dropped frames and a frozen analyzing screen animation during inference.',
            'Inserting "await Future.delayed(Duration.zero)" every 5 loop iterations '
            'yielded control back to the Flutter event loop, allowing animations to '
            'continue without perceptible interruption while inference proceeded.'
        ),
        (
            'Challenge 4: Overfitting Reversal in MobileNetV2 Phase 3',
            'The aggressive combination of high dropout (0.5), label smoothing (0.2), '
            'and a very low learning rate (1e-6) in Phase 3 caused under-fitting rather '
            'than improved generalization. The model could not converge from the Phase 1 '
            'starting point given the combined constraints, producing lower accuracy than '
            'Phase 2 despite more sophisticated regularization.',
            'Root cause identified as restarting from a less-adapted Phase 1 checkpoint '
            'rather than continuing from Phase 2\'s domain-adapted weights. This insight '
            'informed the transition to EfficientNet with a cleaner two-phase protocol '
            'and a less aggressive regularization regime.'
        ),
    ]

    for title, prob, sol in challenges:
        rows = [
            [Paragraph(f'<b>{title}</b>', st['th'])],
            [Table(
                [[Paragraph('<b>Problem</b>', st['td']),
                  Paragraph(prob, st['td'])],
                 [Paragraph('<b>Resolution</b>', st['td']),
                  Paragraph(sol, st['td'])]],
                colWidths=[2.5*cm, 12.0*cm],
                style=TableStyle([
                    ('VALIGN',        (0,0),(-1,-1),'TOP'),
                    ('TOPPADDING',    (0,0),(-1,-1), 5),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                    ('LEFTPADDING',   (0,0),(-1,-1), 7),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 7),
                    ('FONTNAME',      (0,0),(0, -1),'TNR-Bold'),
                    ('BACKGROUND',    (0,0),(0, -1), CLR_TABLE_ALT),
                    ('GRID',          (0,0),(-1,-1), 0.4, CLR_TABLE_BRD),
                ])
            )]
        ]
        outer = Table(rows, colWidths=[14.5*cm])
        outer.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1, 0), CLR_TABLE_HDR),
            ('TOPPADDING',    (0,0),(-1, 0), 6),
            ('BOTTOMPADDING', (0,0),(-1, 0), 6),
            ('LEFTPADDING',   (0,0),(-1, 0), 8),
            ('BOX',           (0,0),(-1,-1), 0.6, CLR_TABLE_BRD),
            ('TOPPADDING',    (0,1),(-1,-1), 0),
            ('BOTTOMPADDING', (0,1),(-1,-1), 0),
            ('LEFTPADDING',   (0,1),(-1,-1), 0),
            ('RIGHTPADDING',  (0,1),(-1,-1), 0),
        ]))
        story.append(KeepTogether([outer, sp(12)]))

    # =========================================================================
    # 4.9  Model Performance Summary
    # =========================================================================
    story.append(H2('4.9  Model Performance Summary', st))
    story.append(P(
        'The following table summarizes the classification performance achieved across '
        'all training phases on the held-out test set of 322 images, and identifies the '
        'final models deployed in the mobile application:', st))

    story += data_table(
        ['Phase', 'Architecture', 'Key Technique', 'Test Accuracy'],
        [
            ['Phase 1', 'MobileNetV2',     'Frozen backbone, GAP head',            '75.16%'],
            ['Phase 2', 'MobileNetV2',     'Top 55 layers unfrozen, lr=1e-5',       '84.16% (best)'],
            ['Phase 3', 'MobileNetV2',     'MixUp, label smoothing, cosine LR',    '73.91% / 76.09% (TTA)'],
            ['Phase 5', 'EfficientNetB2',  'Focal loss, online aug, two-phase',     'Deployed model'],
            ['Phase 10','EfficientNetB3',  'Focal loss, 300x300, two-phase',        'Deployed model'],
            ['Final',   'B2 + B3 Ensemble','Score-CAM, TFLite on-device inference', 'Production system'],
        ],
        [1.8*cm, 3.5*cm, 5.5*cm, 3.7*cm],
        'Table 4.10 -- Training Phase Summary and Model Progression', st)

    # Footer note
    story += [sp(20), rule(), sp(6),
              Paragraph('<i>End of Chapter 4 -- Implementation</i>', st['endnote'])]

    return story


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    output = r'C:\Users\A\Downloads\Chapter4_Implementation.pdf'

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.8*cm,
        title='Chapter 4: Implementation',
        author='Graduation Project -- MSA University',
        subject='Explainable AI Skin Disease Classification',
    )

    styles = build_styles()
    story  = build_story(styles)
    doc.build(story, onFirstPage=make_footer, onLaterPages=make_footer)

    size_kb = os.path.getsize(output) / 1024
    print(f'PDF saved to: {output}')
    print(f'Size: {size_kb:.1f} KB')


if __name__ == '__main__':
    main()
