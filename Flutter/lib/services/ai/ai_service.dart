import 'dart:developer' as dev;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

InterpreterOptions _makeOptions() {
  final numThreads = math.min(4, Platform.numberOfProcessors);
  // We deliberately do NOT add an XNNPackDelegate.
  //
  // The prebuilt XNNPACK delegate shipped in tflite_flutter's
  // libtensorflowlite_jni.so segfaults inside
  // TfLiteXNNPackDelegateCreateWithThreadpool on some Android SoCs (confirmed
  // on a MediaTek-based Samsung A35 / Android 16, arm64 — hard SIGSEGV in the
  // background isolate the instant the first interpreter is created). The
  // delegate is opt-in here (the plain interpreter never auto-applies it), so
  // omitting it falls back to the multi-threaded reference CPU kernels:
  // slightly slower, but stable across all devices. This mirrors the path that
  // already ran correctly on Windows.
  return InterpreterOptions()..threads = numThreads;
}

/// Pipeline diagnostic log. Mirrors to both DevTools (`dart:developer`) and
/// stdout, because `dart:developer`'s `log()` does NOT reach `adb logcat`,
/// which makes on-device debugging via logcat impossible. `print` shows up in
/// logcat under the `flutter` tag, so the full pipeline trace can be captured
/// from a connected device.
void _dlog(String message, {String? name}) {
  final tag = name ?? 'AIService';
  dev.log(message, name: tag);
  // ignore: avoid_print
  print('$tag: $message');
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const List<String> _classes = ['Acne', 'Eczema', 'Tinea'];

// VAE sliding-window params
const int    _patchSize        = 64;
// Stride MUST be ≤ _patchSize. A stride of 96 (the previous value) is larger
// than the 64px patch, so it leaves 32px-wide bands of the image that no patch
// ever covers — small/localised lesions falling in those gaps are never seen by
// the VAE, producing false "No Disease Detected" results. 64 = exact tiling
// (every pixel covered exactly once, no gaps, no redundant compute).
const int    _stride           = 64;
// VAE gate thresholds — validated 2026-05-29 on raw archive test set:
//   disease images:  min ratio = 0.388 (Tinea), mean = 0.64–0.82
//   solid skin-tone: ratio = 0.000 (correctly NORMAL)
//
// _anomalyThreshold: per-patch MSE cutoff. 0.004 keeps high disease sensitivity.
// _anomalyRatio: fraction of patches that must exceed the MSE threshold before
//   the image is treated as diseased. Raised from 0.15 → 0.20 to reduce false
//   positives on real-world photos of normal skin whose natural texture (pores,
//   fine hairs) triggers a small fraction of patches. All disease test images
//   had ratio ≥ 0.388, so 0.20 still catches every confirmed disease case.
const double _anomalyThreshold = 0.004;
const double _anomalyRatio     = 0.20;

// Normal-vs-Disease gate (EfficientNetB0) — REPLACES the VAE anomaly gate.
// The VAE could not separate normal skin from disease on-device (its
// reconstruction-error distributions overlapped and even inverted: a tinea
// photo scored 0.121 while normal skin scored 0.307). This supervised gate
// learns the boundary directly. Input: full image resized to 224, raw [0,255]
// (EfficientNet rescales internally). Output: P(disease) in [0,1].
// _gateThreshold: P(disease) above this -> run the CNN; at/below -> "No Disease
// Detected". Measured locally on the shipped model (ai_edge_litert, identical
// [0,255] preprocessing): DISEASE val scores 0.95-0.99 (min 0.638), the user's
// own normal phone photos score mean 0.19 / 95th-pct 0.43 (one hard outlier at
// 0.79) — a clean bimodal split. t=0.60 -> 100% disease-recall / 96% normal-
// pass. Lower toward 0.40 to catch more disease; raise toward 0.70 to pass more
// borderline normal skin (costs <1% disease-recall).
const int    _gateSize      = 224;
const double _gateThreshold = 0.60;

// Skin-crop before the gate. The SHIPPED model was trained on UNCROPPED images,
// so cropping at inference is a train/serve mismatch — measured to HURT
// (normal-pass 96%->92%, created a new false positive). Leave false unless the
// gate is retrained with skin_bbox_crop (train_normal_gate.py) on the training
// images too, so both sides match.
const bool   _cropBeforeGate = false;

// Gate "gray zone" upper bound. P(disease) in (_gateThreshold,
// _gateConfidentDiseaseMin) is too uncertain to commit: past the normal ceiling
// but not confidently disease (e.g. a hard normal-skin photo, or a borderline
// lesion). We ABSTAIN there — "Inconclusive, retake" — instead of forcing the
// disease-only CNN, which emits a confident (usually Tinea) label on skin it
// can't actually read. With the antialiased gate resize the user's normals top
// out ~0.87 and confident disease is ~0.90+ (median 0.99), so 0.90 keeps the
// hardest normal out of a disease verdict; the ~6% of borderline disease in
// [0.60,0.90] reads "Inconclusive" (safe — it never says "normal"). Lower toward
// 0.80 if a real disease shows "Inconclusive" in a demo.
const double _gateConfidentDiseaseMin = 0.90;

// Gate explainability — occlusion-sensitivity ("disease-evidence") map.
// Built for Normal / Inconclusive results, which have NO Score-CAM (that runs
// only once the CNN commits to a disease). A neutral square is slid over the
// gate's 224² input on a grid; wherever covering the skin makes P(disease) fall
// the most, that region was the evidence FOR disease, so it is coloured warm.
// This needs only the gate classifier itself — it is model-agnostic, which is
// why it works where Score-CAM cannot: TFLite hands us no gradients for
// Grad-CAM, and we ship no feature extractor for the gate model.
const int    _occSize   = _gateSize ~/ 4; // 56px occluding square
const int    _occStride = _gateSize ~/ 8; // 28px stride → 7×7 = 49 passes
const int    _occBatch  = 8;              // gate invokes batched per chunk
// Below this peak P(disease) drop the evidence is too diffuse to localise — we
// skip the map so the UI can say so honestly rather than amplifying noise.
const double _occMinPeakDrop = 0.02;

// Background/edge masking for the VAE ratio.
// The anomaly ratio is anomalous / patches. Without masking, patches that are
// background, letterbox bars, or a limb's silhouette edge (skin meeting a dark
// background) all count — and on a narrow limb against a dark backdrop those
// edge patches alone can push the ratio past _anomalyRatio even when the skin
// itself is normal (confirmed via on-device VAE heatmaps: the anomaly fired on
// the limb outline, not the skin). We exclude patches that are not mostly skin
// from BOTH the numerator and denominator, using the same warm-foreground rule
// as the capture-time skin-coverage check. A patch counts only if at least
// _minPatchSkinFraction of its sampled pixels look like skin.
const bool   _maskNonSkinPatches   = true;
const double _minPatchSkinFraction = 0.50;
// If almost no patch is skin the frame is out-of-distribution (mostly
// background); fall back to the unmasked ratio rather than dividing by ~0.
const double _minSkinPatchCoverage = 0.10;

// Minimum CNN confidence required to report a disease.
// If the highest class probability (after calibration) is below this value
// the model is uncertain — the image likely does not match any of the three
// trained disease classes (e.g. normal skin that slipped past the VAE gate).
// Validated: disease images scored ≥ 0.39 confidence; uncertain/normal images
// are expected to distribute probability more evenly (max < 0.50).
//
// 0.50 was too aggressive for a 3-class softmax: a legitimate but borderline
// call such as 0.46 / 0.30 / 0.24 was being forced to "No Disease Detected"
// even though the top class clearly dominated. We lower the absolute floor to
// 0.40 and instead rely primarily on the top1−top2 *margin* below, which is a
// better "is the model actually committing?" signal than the raw peak.
const double _minCnnConfidence = 0.40;

// Minimum gap between the highest and second-highest class probability. A small
// margin means the model is split between two diseases, so we report
// INCONCLUSIVE rather than guessing. Lowered 0.10 -> 0.05: the gate now rejects
// normal skin upstream, so an image reaching the CNN is almost certainly a real
// disease — a 9.6%-margin Eczema-over-Acne call (real on-device case) is a
// genuine lean and should show as Eczema (with the low-separation caution note),
// not be thrown away. Only a near-coin-flip (<5%) is treated as undecidable.
const double _minCnnMargin = 0.05;

// CNN input sizes
const int _b2W = 260, _b2H = 260;
const int _b3W = 300, _b3H = 300;

// Test-time augmentation: average each backbone's softmax over the 4 flip
// variants (identity, horizontal, vertical, both). Costs 4× CNN forward passes
// but the models are flip-invariant by training, so it only stabilises/improves
// the result. Toggle off to compare against the single-pass baseline.
const bool _useTta = true;

// Hair-removal (DullRazor) params — applied to VAE input only.
// A pixel is classified as hair when its luminance is ≥ _hairDarkness below
// the local maximum in a (2×_hairRadius+1)² window.  Hair pixels are replaced
// by the mean of their non-hair neighbours in the same window.
//
// Why hair fools the VAE: the VAE was trained on smooth clinical patches.
// A dark 1–3 px hair line through a 64×64 patch produces a reconstruction
// error the VAE cannot minimise, so the patch is flagged ANOMALY even on
// perfectly normal skin.
//
// Why disease is safe: lesion pixels are surrounded by other dark/abnormal
// pixels, so the local maximum within the window is also dark — the relative
// darkness is small and the lesion is NOT removed.  Only isolated dark lines
// (hair) exceed the _hairDarkness threshold relative to their bright
// surroundings.
const int    _hairRadius   = 3;    // 7×7 window — covers hair widths of 1–5 px
const double _hairDarkness = 0.16; // relative darkness threshold (16 % of max)

// Score-CAM params (shapes read dynamically from model at runtime)
const int _topK  = 10;  // reduced from 20 → 50% fewer masked passes

// Post-hoc class-probability calibration.
//
// Verified on D:\GRAD-XAI\archive\test (50 images/class, 2026-05-29):
//   Raw model:  Acne 92 %  Eczema 88 %  Tinea 72 %  overall 84 %
//   Main error: Tinea→Eczema confusion (13/50 = 26 % miss rate).
//   The model does NOT strongly over-predict Tinea for Acne/Eczema images
//   (only 2 and 3 cases respectively out of 50).
//
// The user's "I often get Tinea" complaint is likely caused by real-world /
// out-of-distribution camera photos, not by systematic Acne/Eczema mis-
// labelling.  A mild Tinea reduction (0.85) softens borderline Tinea calls
// without significantly hurting true-Tinea recall.
//
// Tuning guide (re-run verify_pipeline.py after each retrain):
//   • Tinea still predicted for Acne/Eczema images → lower toward 0.85.
//   • True Tinea images now mis-classified as Eczema → raise toward 1.05.
//   • Reset to 1.0 after retraining with ×1.4 Tinea weight; re-measure first.
const double _tineaPriorScale = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

class AnalysisResult {
  final bool isNormal;
  final bool isInconclusive;       // gate gray zone — abstained, retake advised
  final String diagnosis;          // empty when isNormal == true
  final double confidence;
  final Map<String, double> classProbabilities;
  final String? heatmapPath;        // path to Score-CAM overlay JPEG (predicted class)
  final Map<String, String> classHeatmapPaths; // per-class Score-CAM overlays
  final String? xaiRationale;       // region-aware text describing where the model looked
  final String? vaeHeatmapPath;     // path to VAE anomaly heatmap JPEG on device
  final double anomalyRatio;        // VAE sliding-window anomaly ratio [0,1]
  final int preprocessMs;           // image decode + resize duration (ms)
  final int vaeMs;                  // VAE stage wall-clock duration (ms)
  final int cnnMs;                  // CNN ensemble duration (ms); 0 when isNormal
  final int scoreCamMs;             // Score-CAM duration (ms); 0 when isNormal

  int get totalMs => preprocessMs + vaeMs + cnnMs + scoreCamMs;

  const AnalysisResult({
    required this.isNormal,
    this.isInconclusive = false,
    required this.diagnosis,
    required this.confidence,
    required this.classProbabilities,
    required this.anomalyRatio,
    this.heatmapPath,
    this.classHeatmapPaths = const {},
    this.xaiRationale,
    this.vaeHeatmapPath,
    required this.preprocessMs,
    required this.vaeMs,
    required this.cnnMs,
    required this.scoreCamMs,
  });

  Map<String, dynamic> toIsolateMap() => {
    'isNormal': isNormal,
    'isInconclusive': isInconclusive,
    'diagnosis': diagnosis,
    'confidence': confidence,
    'classProbabilities': Map<String, double>.from(classProbabilities),
    'anomalyRatio': anomalyRatio,
    'heatmapPath': heatmapPath,
    'classHeatmapPaths': Map<String, String>.from(classHeatmapPaths),
    'xaiRationale': xaiRationale,
    'vaeHeatmapPath': vaeHeatmapPath,
    'preprocessMs': preprocessMs,
    'vaeMs': vaeMs,
    'cnnMs': cnnMs,
    'scoreCamMs': scoreCamMs,
  };

  static AnalysisResult fromIsolateMap(Map<dynamic, dynamic> m) => AnalysisResult(
    isNormal: m['isNormal'] as bool,
    isInconclusive: m['isInconclusive'] as bool? ?? false,
    diagnosis: m['diagnosis'] as String,
    confidence: (m['confidence'] as num).toDouble(),
    classProbabilities: Map<String, double>.from(
        (m['classProbabilities'] as Map).map((k, v) => MapEntry(k as String, (v as num).toDouble()))),
    anomalyRatio: (m['anomalyRatio'] as num).toDouble(),
    heatmapPath: m['heatmapPath'] as String?,
    classHeatmapPaths: Map<String, String>.from(
        (m['classHeatmapPaths'] as Map).map((k, v) => MapEntry(k as String, v as String))),
    xaiRationale: m['xaiRationale'] as String?,
    vaeHeatmapPath: m['vaeHeatmapPath'] as String?,
    preprocessMs: m['preprocessMs'] as int,
    vaeMs: m['vaeMs'] as int,
    cnnMs: m['cnnMs'] as int,
    scoreCamMs: m['scoreCamMs'] as int,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AIService
// ─────────────────────────────────────────────────────────────────────────────

class AIService {
  static final AIService _instance = AIService._internal();
  factory AIService() => _instance;
  AIService._internal();

  Interpreter? _gate;
  Interpreter? _vae;
  Interpreter? _b2;
  Interpreter? _b3;
  Interpreter? _feat;
  bool _ready = false;

  // ── Initialise ─────────────────────────────────────────────────────────────

  Future<void> initialize() async {
    if (_ready) return;
    _gate = await Interpreter.fromAsset('assets/models/normal_gate.tflite', options: _makeOptions());
    _vae  = await Interpreter.fromAsset('assets/models/vae_model.tflite', options: _makeOptions());
    _b2   = await Interpreter.fromAsset('assets/models/cnn_b2_model.tflite', options: _makeOptions());
    _b3   = await Interpreter.fromAsset('assets/models/cnn_b3_model.tflite', options: _makeOptions());
    _feat = await Interpreter.fromAsset('assets/models/b3_feature_extractor.tflite', options: _makeOptions());
    _gate!.allocateTensors();
    _vae!.allocateTensors();
    _b2!.allocateTensors();
    _b3!.allocateTensors();
    _feat!.allocateTensors();
    _ready = true;
  }

  /// Initialise from pre-loaded bytes (used from background isolates where
  /// Flutter's rootBundle is unavailable).
  void initializeFromBuffers({
    required Uint8List gateBytes,
    required Uint8List vaeBytes,
    required Uint8List b2Bytes,
    required Uint8List b3Bytes,
    required Uint8List featBytes,
  }) {
    if (_ready) return;
    _gate = Interpreter.fromBuffer(gateBytes, options: _makeOptions());
    _vae  = Interpreter.fromBuffer(vaeBytes, options: _makeOptions());
    _b2   = Interpreter.fromBuffer(b2Bytes, options: _makeOptions());
    _b3   = Interpreter.fromBuffer(b3Bytes, options: _makeOptions());
    _feat = Interpreter.fromBuffer(featBytes, options: _makeOptions());
    _gate!.allocateTensors();
    _vae!.allocateTensors();
    _b2!.allocateTensors();
    _b3!.allocateTensors();
    _feat!.allocateTensors();
    _ready = true;
  }

  // ── Public entry point ─────────────────────────────────────────────────────

  /// Run the full pipeline on [imagePath].
  ///
  /// [onStepChange] is called with step index 0-3 as each stage starts:
  ///   0 = preprocessing
  ///   1 = VAE anomaly detection
  ///   2 = CNN classification
  ///   3 = Score-CAM heatmap
  Future<AnalysisResult> analyzeImage({
    required String imagePath,
    void Function(int step)? onStepChange,
  }) async {
    if (!_ready) await initialize();

    // Step 0 – decode & (if huge) downsample image
    onStepChange?.call(0);
    final preprocessStart = DateTime.now().millisecondsSinceEpoch;
    final rawBytes = await File(imagePath).readAsBytes();
    final decoded0 = img.decodeImage(rawBytes);
    if (decoded0 == null) throw Exception('Failed to decode image: $imagePath');

    // Bake EXIF orientation into the pixels. Android camera photos store
    // sideways sensor pixels plus an EXIF "rotate" flag: Flutter's Image widget
    // honours it (so the user SEES the photo upright) but img.decodeImage does
    // NOT, leaving every model stage to process a 90°-rotated image. The gate,
    // CNN and Score-CAM are all trained on upright images, so a rotated input
    // inflates P(disease) — measured: normal-skin false positives jump 1/25 ->
    // 4/25 under 90° rotation, which is exactly the on-device "normal -> Tinea"
    // failure — and it also misaligns the heatmap overlay. Baking fixes all
    // three. No-op for already-upright images (gallery / WhatsApp / no EXIF).
    final decoded = img.bakeOrientation(decoded0);

    // Normalise to uint8 RGB once so every downstream stage receives the
    // correct [0,255] integer range.  The image pkg 4.x can return float32
    // [0,1] for HDR / HEIC sources; feeding that to EfficientNet's internal
    // Rescaling(1/255) double-normalises to near-black and collapses the
    // classifier to a single class (empirically: Tinea).  We rebuild the
    // image explicitly via the *Normalized accessors (always [0,1] for any
    // source format) rather than `convert()`, which on some float sources
    // clamps to 0/1 instead of rescaling.
    final decodedU8 = _toUint8Rgb(decoded);

    // Only downsample very large photos so the VAE sliding window doesn't
    // explode on 12MP camera shots. Keep well above the VAE's training scale
    // (patches must still look like the training data) and ALWAYS use linear
    // interpolation — the default is nearest-neighbor which introduces
    // aliasing artefacts that the VAE flags as anomalies.
    final original = decodedU8.width > 1280
        ? img.copyResize(decodedU8,
            width: 1280, interpolation: img.Interpolation.linear)
        : decodedU8;
    final preprocessMs = DateTime.now().millisecondsSinceEpoch - preprocessStart;

    // Step 1 – VAE anomaly gate
    // De-hair before the VAE: arm/leg hair produces dark thin lines in 64×64
    // patches that the VAE cannot reconstruct → spurious high MSE → normal skin
    // is incorrectly flagged as ANOMALY and forwarded to the CNN.
    // The CNN always receives `original` (unmodified); only the VAE sees the
    // hair-removed copy so disease texture is preserved for classification.
    var stageStart = DateTime.now().millisecondsSinceEpoch;
    onStepChange?.call(1);
    final pDisease = _runGate(original);
    final vaeMs = DateTime.now().millisecondsSinceEpoch - stageStart;
    final gateDecision = pDisease <= _gateThreshold
        ? 'NORMAL'
        : (pDisease < _gateConfidentDiseaseMin ? 'INCONCLUSIVE' : 'DISEASE');
    _dlog(
      '[GATE] P(disease)=${pDisease.toStringAsFixed(3)} '
      'normal<=$_gateThreshold disease>=$_gateConfidentDiseaseMin '
      'decision=$gateDecision',
      name: 'AIService',
    );

    if (pDisease <= _gateThreshold) {
      // Normal — but still produce a visual explanation. Score-CAM only runs
      // once the CNN commits to a disease, so without this a "normal" verdict
      // shows no heatmap at all. Occlusion sensitivity on the gate answers
      // "where was the most disease-like skin, and was it enough?" using only
      // the gate classifier (no feature maps / gradients required).
      onStepChange?.call(3);
      final camStart = DateTime.now().millisecondsSinceEpoch;
      final gateXai = await _runGateSaliency(original, pDisease);
      final scoreCamMs = DateTime.now().millisecondsSinceEpoch - camStart;
      _dlog(
        '[TIMING] preprocess=${preprocessMs}ms gate=${vaeMs}ms cnn=0ms xai=${scoreCamMs}ms total=${preprocessMs + vaeMs + scoreCamMs}ms | NORMAL',
        name: 'AIService',
      );
      return AnalysisResult(
        isNormal: true,
        diagnosis: '',
        confidence: 1.0 - pDisease,
        classProbabilities: const {},
        anomalyRatio: pDisease,
        vaeHeatmapPath: gateXai.heatmapPath,
        xaiRationale: gateXai.rationale,
        preprocessMs: preprocessMs,
        vaeMs: vaeMs,
        cnnMs: 0,
        scoreCamMs: scoreCamMs,
      );
    }

    // Gate gray zone → abstain. See _gateConfidentDiseaseMin. We deliberately do
    // NOT run the CNN: on skin the gate can't confidently call, the disease-only
    // CNN returns a confident (usually Tinea) label — the exact false positive
    // we're preventing. "Inconclusive — retake" is honest in both directions.
    if (pDisease < _gateConfidentDiseaseMin) {
      // Gray-zone abstain — the most informative place for the gate evidence
      // map: it shows the borderline region the gate could not commit on.
      onStepChange?.call(3);
      final camStart = DateTime.now().millisecondsSinceEpoch;
      final gateXai = await _runGateSaliency(original, pDisease);
      final scoreCamMs = DateTime.now().millisecondsSinceEpoch - camStart;
      _dlog(
        '[TIMING] preprocess=${preprocessMs}ms gate=${vaeMs}ms cnn=0ms xai=${scoreCamMs}ms total=${preprocessMs + vaeMs + scoreCamMs}ms | INCONCLUSIVE',
        name: 'AIService',
      );
      return AnalysisResult(
        isNormal: false,
        isInconclusive: true,
        diagnosis: '',
        confidence: pDisease,
        classProbabilities: const {},
        anomalyRatio: pDisease,
        vaeHeatmapPath: gateXai.heatmapPath,
        xaiRationale: gateXai.rationale,
        preprocessMs: preprocessMs,
        vaeMs: vaeMs,
        cnnMs: 0,
        scoreCamMs: scoreCamMs,
      );
    }

    // Step 2 – CNN ensemble classification
    stageStart = DateTime.now().millisecondsSinceEpoch;
    onStepChange?.call(2);
    final probs = await _runCnnEnsemble(original);
    final cnnMs = DateTime.now().millisecondsSinceEpoch - stageStart;
    final predIdx = _argmax(probs);
    final diagnosis = _classes[predIdx];
    final confidence = probs[predIdx];
    final probMap = {
      for (int i = 0; i < _classes.length; i++) _classes[i]: probs[i]
    };
    // top1 − top2 margin: how decisively the model favours its top class.
    final sortedProbs = [...probs]..sort();
    final margin =
        sortedProbs[sortedProbs.length - 1] - sortedProbs[sortedProbs.length - 2];
    _dlog(
      '[CNN] Acne=${probs[0].toStringAsFixed(3)} '
      'Eczema=${probs[1].toStringAsFixed(3)} '
      'Tinea=${probs[2].toStringAsFixed(3)} '
      '→ $diagnosis (${(confidence * 100).toStringAsFixed(1)}%)',
      name: 'AIService',
    );

    // Low-confidence gate: if the CNN cannot commit to any class with at least
    // _minCnnConfidence, OR is split between two diseases (margin below
    // _minCnnMargin), it can't commit to a type. The GATE already decided this
    // image is a disease (P >= _gateConfidentDiseaseMin), so the honest result
    // is INCONCLUSIVE ("a condition is likely, type unclear — retake/consult"),
    // NOT "No Disease Detected" — returning normal here would be false
    // reassurance on skin the gate was ~99% sure is diseased. The class
    // probabilities are kept so the UI can still show the leaning (e.g. Eczema
    // 42% vs Acne 32%) with its low-separation caution note.
    if (confidence < _minCnnConfidence || margin < _minCnnMargin) {
      _dlog(
        '[CNN] confidence ${(confidence * 100).toStringAsFixed(1)}% '
        '(floor ${(_minCnnConfidence * 100).toStringAsFixed(0)}%) '
        'margin ${(margin * 100).toStringAsFixed(1)}% '
        '(floor ${(_minCnnMargin * 100).toStringAsFixed(0)}%) → INCONCLUSIVE (disease likely, type unclear)',
        name: 'AIService',
      );
      return AnalysisResult(
        isNormal: false,
        isInconclusive: true,
        diagnosis: '',
        confidence: confidence,
        classProbabilities: probMap,
        anomalyRatio: pDisease,
        vaeHeatmapPath: null,
        preprocessMs: preprocessMs,
        vaeMs: vaeMs,
        cnnMs: cnnMs,
        scoreCamMs: 0,
      );
    }

    // Step 3 – Score-CAM heatmap
    stageStart = DateTime.now().millisecondsSinceEpoch;
    onStepChange?.call(3);
    final scoreCam = await _runScoreCam(original, imagePath, predIdx);
    final heatmapPath = scoreCam.classPaths[diagnosis];
    final scoreCamMs = DateTime.now().millisecondsSinceEpoch - stageStart;

    final total = preprocessMs + vaeMs + cnnMs + scoreCamMs;
    _dlog(
      '[TIMING] preprocess=${preprocessMs}ms gate=${vaeMs}ms cnn=${cnnMs}ms scorecam=${scoreCamMs}ms total=${total}ms | $diagnosis',
      name: 'AIService',
    );
    return AnalysisResult(
      isNormal: false,
      diagnosis: diagnosis,
      confidence: confidence,
      classProbabilities: probMap,
      anomalyRatio: pDisease,
      vaeHeatmapPath: null,
      heatmapPath: heatmapPath,
      classHeatmapPaths: scoreCam.classPaths,
      xaiRationale: scoreCam.rationale,
      preprocessMs: preprocessMs,
      vaeMs: vaeMs,
      cnnMs: cnnMs,
      scoreCamMs: scoreCamMs,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Stage 1 — Normal-vs-Disease gate (replaces the VAE gate)
  // ─────────────────────────────────────────────────────────────────────────

  /// Returns P(disease) in [0,1] for the whole image. EfficientNetB0 expects a
  /// raw [0,255] RGB tensor of _gateSize×_gateSize (it rescales internally).
  double _runGate(img.Image original) {
    final input = _cropBeforeGate ? _cropToSkinRegion(original) : original;
    // ANTIALIASED downscale (area filter), NOT linear. A phone photo is ~7×
    // larger than 224, and a plain bilinear/linear resize aliases that big a
    // reduction: on textured skin (faces, fine detail) the high-frequency noise
    // survives as artefacts that the gate reads as disease. Measured on-device,
    // `linear` inflated a normal photo from its true ~0.79 up to 0.95 (confident
    // Tinea); `average` samples every source pixel and removes the aliasing, so
    // the score tracks the validated off-device value. Disease scores are
    // unchanged by this (clinical images are already well-sampled).
    final resized = img.copyResize(input,
        width: _gateSize, height: _gateSize,
        interpolation: img.Interpolation.average);
    final buf = Float32List(_gateSize * _gateSize * 3);
    _fillFloat32Raw(resized, buf, _gateSize, _gateSize); // [0,255]
    _gate!.resizeInputTensor(0, [1, _gateSize, _gateSize, 3]);
    _gate!.allocateTensors();
    _gate!.getInputTensor(0).data = buf.buffer.asUint8List();
    _gate!.invoke();
    final out = _gate!.getOutputTensor(0).data.buffer.asFloat32List();
    return out[0].clamp(0.0, 1.0).toDouble();
  }

  /// Crop to the bounding box of skin-colored pixels so background (floor
  /// tiles, rugs, clothing) stops influencing the gate. MUST stay in sync with
  /// skin_bbox_crop in train_normal_gate.py — the gate is trained on images
  /// cropped by the exact same rule. Falls back to the full frame when skin
  /// coverage is too low (mask unreliable) or the box is degenerate.
  img.Image _cropToSkinRegion(img.Image src) {
    final w = src.width, h = src.height;
    final step = math.max(1, math.min(w, h) ~/ 128);
    final gw = (w + step - 1) ~/ step, gh = (h + step - 1) ~/ step;
    final colCnt = List<int>.filled(gw, 0);
    final rowCnt = List<int>.filled(gh, 0);
    int skin = 0, total = 0;
    for (int gy = 0; gy < gh; gy++) {
      for (int gx = 0; gx < gw; gx++) {
        final p = src.getPixel(gx * step, gy * step);
        final r = p.r.toDouble(), g = p.g.toDouble(), b = p.b.toDouble();
        final v = (r + g + b) / 3.0;
        final mx = math.max(r, math.max(g, b));
        final mn = math.min(r, math.min(g, b));
        final sat = mx > 0 ? (mx - mn) / mx : 0.0;
        // Keep this rule identical to skin_bbox_crop in train_normal_gate.py.
        final isSkin = v > 30 &&
            v < 248 &&
            r >= b - 5 &&
            !(sat < 0.05 && v > 153) &&
            g > 0.45 * r && // reject deep-red fabric
            r >= g - 5; //     reject green-dominant (walls/plants)
        total++;
        if (isSkin) {
          skin++;
          colCnt[gx]++;
          rowCnt[gy]++;
        }
      }
    }
    if (total == 0 || skin == 0 || skin / total < 0.10) return src;
    // 2nd/98th-percentile bounds (as in skin_bbox_crop) so a few stray
    // skin-toned background pixels don't blow the box out to the full frame.
    int trimLo(List<int> cnt) {
      final cut = 0.02 * skin;
      double cum = 0;
      for (int i = 0; i < cnt.length; i++) {
        cum += cnt[i];
        if (cum >= cut) return i;
      }
      return 0;
    }

    int trimHi(List<int> cnt) {
      final cut = 0.02 * skin;
      double cum = 0;
      for (int i = cnt.length - 1; i >= 0; i--) {
        cum += cnt[i];
        if (cum >= cut) return i;
      }
      return cnt.length - 1;
    }

    // 5% margin, clamped to the frame.
    final my = (0.05 * h).round(), mxr = (0.05 * w).round();
    final x0 = math.max(0, trimLo(colCnt) * step - mxr);
    final y0 = math.max(0, trimLo(rowCnt) * step - my);
    final x1 = math.min(w, (trimHi(colCnt) + 1) * step + mxr);
    final y1 = math.min(h, (trimHi(rowCnt) + 1) * step + my);
    if ((x1 - x0) * (y1 - y0) < 0.05 * w * h) return src;
    _dlog(
      '[GATE] skin crop ($x0,$y0)-($x1,$y1) of ${w}x$h '
      'coverage=${(skin / total).toStringAsFixed(2)}',
      name: 'AIService',
    );
    return img.copyCrop(src, x: x0, y: y0, width: x1 - x0, height: y1 - y0);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Gate explainability — occlusion-sensitivity ("disease-evidence") map
  // ─────────────────────────────────────────────────────────────────────────

  /// Build a disease-evidence heatmap for the gate decision using occlusion
  /// sensitivity (Zeiler & Fergus, 2014): slide a neutral square over a grid of
  /// positions on the gate's input, re-run the gate on each occluded image, and
  /// measure how much P(disease) DROPS. A large drop means that region was
  /// driving the disease signal, so it is coloured warm. Model-agnostic — it
  /// needs only the gate classifier, which is why it works where Score-CAM
  /// cannot (no gradients from TFLite, no feature extractor for the gate).
  ///
  /// Used for NORMAL and INCONCLUSIVE results, which otherwise have no visual
  /// explanation. Returns the overlay JPEG path + a short rationale, or
  /// (null, null) when the evidence is too diffuse to localise, so the UI can
  /// say so instead of drawing an amplified-noise map. Non-fatal on any error.
  Future<({String? heatmapPath, String? rationale})> _runGateSaliency(
      img.Image original, double pDisease) async {
    try {
      final input = _cropBeforeGate ? _cropToSkinRegion(original) : original;
      // Match exactly what the gate scored (same antialiased resize as _runGate)
      // so the occlusion drops are measured against the real decision.
      final base = img.copyResize(input,
          width: _gateSize, height: _gateSize,
          interpolation: img.Interpolation.average);
      const n = _gateSize * _gateSize;
      final baseBuf = Float32List(n * 3); // [0,255]
      _fillFloat32Raw(base, baseBuf, _gateSize, _gateSize);

      // Neutral occluder = image mean. Replacing a region with the mean removes
      // its information without injecting a hard edge — a black box would itself
      // read as anomalous to the gate and corrupt the attribution.
      double sr = 0, sg = 0, sb = 0;
      for (int i = 0; i < n; i++) {
        sr += baseBuf[i * 3];
        sg += baseBuf[i * 3 + 1];
        sb += baseBuf[i * 3 + 2];
      }
      final occR = sr / n, occG = sg / n, occB = sb / n;

      const gw = (_gateSize - _occSize) ~/ _occStride + 1;
      const gh = (_gateSize - _occSize) ~/ _occStride + 1;
      const total = gw * gh;
      final drops = Float32List(total);

      const imgFloats = _gateSize * _gateSize * 3;
      _gate!.resizeInputTensor(0, [_occBatch, _gateSize, _gateSize, 3]);
      _gate!.allocateTensors();
      final buf = Float32List(_occBatch * imgFloats);

      for (int baseIdx = 0; baseIdx < total; baseIdx += _occBatch) {
        final count = math.min(_occBatch, total - baseIdx);
        for (int bi = 0; bi < count; bi++) {
          final cell = baseIdx + bi;
          final gx = cell % gw, gy = cell ~/ gw;
          final x0 = gx * _occStride, y0 = gy * _occStride;
          final off = bi * imgFloats;
          buf.setRange(off, off + imgFloats, baseBuf); // copy clean image
          for (int yy = y0; yy < y0 + _occSize; yy++) {
            final rowBase = off + yy * _gateSize * 3;
            for (int xx = x0; xx < x0 + _occSize; xx++) {
              final idx = rowBase + xx * 3;
              buf[idx]     = occR;
              buf[idx + 1] = occG;
              buf[idx + 2] = occB;
            }
          }
        }
        _gate!.getInputTensor(0).data = buf.buffer.asUint8List();
        _gate!.invoke();
        final out = _gate!.getOutputTensor(0).data.buffer.asFloat32List();
        for (int bi = 0; bi < count; bi++) {
          final occluded = out[bi].clamp(0.0, 1.0).toDouble();
          // Positive drop = occluding this region lowered P(disease) = it was
          // evidence FOR disease. ReLU: only disease-supporting regions are mapped.
          final drop = pDisease - occluded;
          drops[baseIdx + bi] = drop > 0 ? drop : 0.0;
        }
        await Future.delayed(Duration.zero); // keep UI/animations responsive
      }

      // Normalise. If even the strongest drop is negligible the evidence is
      // diffuse — bail so the UI shows honest text instead of amplified noise.
      double dMax = 0;
      for (final v in drops) {
        if (v > dMax) dMax = v;
      }
      if (dMax < _occMinPeakDrop) {
        _dlog(
          '[GATE-XAI] evidence too diffuse (peakDrop=${dMax.toStringAsFixed(3)} '
          '< $_occMinPeakDrop) — no localised map',
          name: 'AIService',
        );
        return (heatmapPath: null, rationale: _gateRationale(drops, gw, gh, pDisease));
      }
      for (int i = 0; i < total; i++) {
        drops[i] /= dMax;
      }

      // Small grid → jet colour → bilinear upsample → 60/40 blend (Score-CAM style).
      final gridImg = img.Image(width: gw, height: gh);
      for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
          final jet = _jetColor(drops[gy * gw + gx]);
          gridImg.setPixelRgb(gx, gy, jet[0], jet[1], jet[2]);
        }
      }
      final heatFull = img.copyResize(gridImg,
          width: original.width, height: original.height,
          interpolation: img.Interpolation.linear);
      final overlay = img.Image(width: original.width, height: original.height);
      for (int y = 0; y < original.height; y++) {
        for (int x = 0; x < original.width; x++) {
          final op = original.getPixel(x, y);
          final hp = heatFull.getPixel(x, y);
          overlay.setPixelRgb(
            x, y,
            (op.r * 0.6 + hp.r * 0.4).round().clamp(0, 255),
            (op.g * 0.6 + hp.g * 0.4).round().clamp(0, 255),
            (op.b * 0.6 + hp.b * 0.4).round().clamp(0, 255),
          );
        }
      }

      final tmpDir = await getTemporaryDirectory();
      final outPath =
          '${tmpDir.path}/gate_evidence_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(outPath).writeAsBytes(img.encodeJpg(overlay, quality: 88));
      _dlog(
        '[GATE-XAI] occlusion map saved=$outPath grid=${gw}x$gh '
        'passes=$total peakDrop=${dMax.toStringAsFixed(3)}',
        name: 'AIService',
      );
      return (heatmapPath: outPath, rationale: _gateRationale(drops, gw, gh, pDisease));
    } catch (e, st) {
      // Non-fatal — a missing gate map just means the UI shows text only.
      _dlog('[GATE-XAI] failed: $e\n$st', name: 'AIService');
      return (heatmapPath: null, rationale: null);
    }
  }

  /// One-sentence description of WHERE the gate's disease evidence concentrated,
  /// from the normalised occlusion-drop grid. [drops] may be unnormalised when
  /// the peak was negligible (the centroid is still meaningful).
  String _gateRationale(Float32List drops, int gw, int gh, double pDisease) {
    double sum = 0, sx = 0, sy = 0;
    for (int y = 0; y < gh; y++) {
      for (int x = 0; x < gw; x++) {
        final v = drops[y * gw + x];
        sum += v;
        sx += v * x;
        sy += v * y;
      }
    }
    final pct = (pDisease * 100).round();
    final outcome = pDisease <= _gateThreshold ? 'normal' : 'inconclusive';
    if (sum < 1e-6) {
      return 'No single region looked disease-like — the $pct% disease score was '
          'spread evenly across the skin, so the result is reported as $outcome.';
    }
    final cx = (sx / sum) / gw, cy = (sy / sum) / gh;
    final vert = cy < 0.4 ? 'upper' : (cy > 0.6 ? 'lower' : 'central');
    final horiz = cx < 0.4 ? 'left' : (cx > 0.6 ? 'right' : 'centre');
    final loc = (vert == 'central' && horiz == 'centre')
        ? 'the centre of the image'
        : 'the $vert-$horiz region';
    return 'The most disease-like skin was in $loc. At a $pct% disease score '
        'that stayed below the action threshold, so the result is reported as '
        '$outcome rather than a specific condition.';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // (legacy) VAE sliding-window anomaly detection — no longer gates the
  // pipeline; kept for reference / optional heatmap use.
  // ─────────────────────────────────────────────────────────────────────────

  // Parked: the VAE gate was superseded by the EfficientNet normal-gate but is
  // retained for the planned VAE re-integration.
  // ignore: unused_element
  Future<({bool isAnomaly, double ratio, String? vaeHeatmapPath})> _runVae(
      img.Image original) async {
    final h = original.height;
    final w = original.width;

    // Pre-compute grid dimensions matching the sliding-window loop
    final gridH = h >= _patchSize ? (h - _patchSize) ~/ _stride + 1 : 0;
    final gridW = w >= _patchSize ? (w - _patchSize) ~/ _stride + 1 : 0;

    if (gridH == 0 || gridW == 0) {
      return (isAnomaly: false, ratio: 0.0, vaeHeatmapPath: null);
    }

    final mseGrid = Float32List(gridH * gridW);
    // Per-patch skin flag: true = mostly skin (counts toward the ratio),
    // false = background/letterbox/silhouette edge (excluded).
    final skinMask = List<bool>.filled(gridH * gridW, true);
    int anomalousAll = 0;  // patches over MSE threshold, whole frame
    int anomalousSkin = 0; // patches over MSE threshold, skin patches only
    int skinPatches = 0;

    // Batch one grid row per invoke. The VAE graph has a dynamic batch dim and
    // batched output is bit-identical to per-patch invokes (verified off-device),
    // but one invoke per row lets TFLite parallelise across the batch instead of
    // paying per-invoke overhead 192 times — a large wall-clock win on CPU.
    const patchFloats = _patchSize * _patchSize * 3;
    _vae!.resizeInputTensor(0, [gridW, _patchSize, _patchSize, 3]);
    _vae!.allocateTensors();
    final rowBuf = Float32List(gridW * patchFloats);

    for (int gy = 0; gy < gridH; gy++) {
      final y = gy * _stride;
      for (int gx = 0; gx < gridW; gx++) {
        final x = gx * _stride;
        final patch = img.copyCrop(original,
            x: x, y: y, width: _patchSize, height: _patchSize);
        if (_maskNonSkinPatches) {
          skinMask[gy * gridW + gx] = _patchIsSkin(patch);
        }
        _fillFloat32(patch, rowBuf, _patchSize, _patchSize,
            offset: gx * patchFloats);
      }
      _vae!.getInputTensor(0).data = rowBuf.buffer.asUint8List();
      _vae!.invoke();
      final mses = _vae!.getOutputTensor(0).data.buffer.asFloat32List();
      for (int gx = 0; gx < gridW; gx++) {
        final idx = gy * gridW + gx;
        final mse = mses[gx];
        mseGrid[idx] = mse;
        final over = mse > _anomalyThreshold;
        if (over) anomalousAll++;
        if (skinMask[idx]) {
          skinPatches++;
          if (over) anomalousSkin++;
        }
      }
      // Yield to event loop every row so animations keep running
      await Future.delayed(Duration.zero);
    }

    final patches = gridH * gridW;
    // Default to skin-only patches. If almost the whole frame is non-skin the
    // mask is unreliable (out-of-distribution photo) — fall back to all patches
    // so we don't divide by a near-zero count.
    final skinCoverage = patches > 0 ? skinPatches / patches : 0.0;
    final useMask = _maskNonSkinPatches && skinCoverage >= _minSkinPatchCoverage;
    final denom = useMask ? skinPatches : patches;
    final anomalous = useMask ? anomalousSkin : anomalousAll;
    final ratio = denom > 0 ? anomalous / denom : 0.0;
    _dlog(
      '[VAE] patches=$patches skinPatches=$skinPatches '
      'skinCoverage=${skinCoverage.toStringAsFixed(2)} '
      'anomalous=$anomalous ratio=${ratio.toStringAsFixed(3)} '
      '${useMask ? "(skin-masked)" : "(unmasked fallback)"} '
      'threshold=$_anomalyThreshold decision=${ratio > _anomalyRatio ? "ANOMALY" : "NORMAL"}',
      name: 'AIService',
    );

    final vaeHeatmapPath =
        await _buildVaeHeatmap(original, mseGrid, gridH, gridW);

    return (
      isAnomaly: ratio > _anomalyRatio,
      ratio: ratio,
      vaeHeatmapPath: vaeHeatmapPath,
    );
  }

  /// True when a 64×64 patch is mostly skin (and so should count toward the VAE
  /// anomaly ratio). Uses the same warm-foreground rule as the capture-time
  /// skin-coverage check so the two agree. Samples every 4th pixel for speed —
  /// 256 samples per patch is plenty to judge background vs skin.
  bool _patchIsSkin(img.Image patch) {
    int skin = 0, total = 0;
    for (int y = 0; y < patch.height; y += 4) {
      for (int x = 0; x < patch.width; x += 4) {
        final p = patch.getPixel(x, y);
        final r = p.r.toDouble(), g = p.g.toDouble(), b = p.b.toDouble();
        final v = (r + g + b) / 3.0;
        final mx = math.max(r, math.max(g, b));
        final mn = math.min(r, math.min(g, b));
        final sat = mx > 0 ? (mx - mn) / mx : 0.0;
        final isSkin =
            v > 30 && v < 248 && r >= b - 5 && !(sat < 0.05 && v > 153);
        if (isSkin) skin++;
        total++;
      }
    }
    return total > 0 && skin / total >= _minPatchSkinFraction;
  }

  Future<String?> _buildVaeHeatmap(
      img.Image original,
      Float32List mseGrid,
      int gridH,
      int gridW) async {
    try {
      // Normalize MSE values to [0, 1]
      double mseMin = mseGrid[0], mseMax = mseGrid[0];
      for (final v in mseGrid) {
        if (v < mseMin) mseMin = v;
        if (v > mseMax) mseMax = v;
      }
      final mseRange = (mseMax - mseMin) > 1e-8 ? mseMax - mseMin : 1.0;

      // Build small grid image with jet colormap
      final gridImg = img.Image(width: gridW, height: gridH);
      for (int gy = 0; gy < gridH; gy++) {
        for (int gx = 0; gx < gridW; gx++) {
          final norm = (mseGrid[gy * gridW + gx] - mseMin) / mseRange;
          final jet = _jetColor(norm);
          gridImg.setPixelRgb(gx, gy, jet[0], jet[1], jet[2]);
        }
      }

      // Bilinear upsample to original image dimensions
      final heatmapFull = img.copyResize(
        gridImg,
        width: original.width,
        height: original.height,
        interpolation: img.Interpolation.linear,
      );

      // Blend: 60% original + 40% heatmap (matches Score-CAM style)
      final overlay =
          img.Image(width: original.width, height: original.height);
      for (int row = 0; row < original.height; row++) {
        for (int col = 0; col < original.width; col++) {
          final op = original.getPixel(col, row);
          final hp = heatmapFull.getPixel(col, row);
          overlay.setPixelRgb(
            col, row,
            (op.r * 0.6 + hp.r * 0.4).round().clamp(0, 255),
            (op.g * 0.6 + hp.g * 0.4).round().clamp(0, 255),
            (op.b * 0.6 + hp.b * 0.4).round().clamp(0, 255),
          );
        }
      }

      final tmpDir = await getTemporaryDirectory();
      final outPath =
          '${tmpDir.path}/vae_heatmap_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(outPath).writeAsBytes(img.encodeJpg(overlay, quality: 88));
      _dlog('[VAE] heatmap saved=$outPath', name: 'AIService');
      return outPath;
    } catch (e, st) {
      _dlog('[VAE] heatmap failed: $e\n$st', name: 'AIService');
      return null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Stage 2 — CNN B2 + B3 ensemble
  // ─────────────────────────────────────────────────────────────────────────

  Future<List<double>> _runCnnEnsemble(img.Image original) async {
    final probsB2 = _runCnn(_b2!, original, _b2W, _b2H, 'B2');
    final probsB3 = _runCnn(_b3!, original, _b3W, _b3H, 'B3');
    _dlog(
      '[CNN-B2] Acne=${probsB2[0].toStringAsFixed(3)} '
      'Eczema=${probsB2[1].toStringAsFixed(3)} '
      'Tinea=${probsB2[2].toStringAsFixed(3)}',
      name: 'AIService',
    );
    _dlog(
      '[CNN-B3] Acne=${probsB3[0].toStringAsFixed(3)} '
      'Eczema=${probsB3[1].toStringAsFixed(3)} '
      'Tinea=${probsB3[2].toStringAsFixed(3)}',
      name: 'AIService',
    );
    final raw = List.generate(
        _classes.length, (i) => (probsB2[i] + probsB3[i]) / 2.0);
    return _calibrateProbs(raw);
  }

  /// Reduce the Tinea class prior to counteract the model's training bias.
  /// Multiplies Tinea's raw probability by [_tineaPriorScale] then renormalises
  /// so all three probabilities still sum to 1.0.
  List<double> _calibrateProbs(List<double> probs) {
    final w = [probs[0], probs[1], probs[2] * _tineaPriorScale];
    final sum = w[0] + w[1] + w[2];
    final calibrated = w.map((v) => v / sum).toList();
    _dlog(
      '[Calibration] raw  Acne=${probs[0].toStringAsFixed(3)} '
      'Eczema=${probs[1].toStringAsFixed(3)} '
      'Tinea=${probs[2].toStringAsFixed(3)} | '
      'cal  Acne=${calibrated[0].toStringAsFixed(3)} '
      'Eczema=${calibrated[1].toStringAsFixed(3)} '
      'Tinea=${calibrated[2].toStringAsFixed(3)}',
      name: 'AIService',
    );
    return calibrated;
  }

  List<double> _runCnn(
      Interpreter interp, img.Image original, int w, int h, String tag) {
    final resized = img.copyResize(original,
        width: w, height: h, interpolation: img.Interpolation.linear);

    // Test-time augmentation. Both backbones were trained with
    // RandomFlip("horizontal_and_vertical"), so they are flip-invariant by
    // design — averaging the softmax over the four flip variants reduces
    // prediction variance (the "flips between classes on near-identical photos"
    // symptom) and usually improves accuracy slightly on real-world images.
    // Flips are applied to CLONES so the identity view (`resized`) is never
    // mutated by the in-place transforms. Set _useTta=false to A/B against the
    // single-pass path — keep verify_pipeline.py / parity_check.py in sync.
    final views = _useTta
        ? <img.Image>[
            resized,
            img.flipHorizontal(img.Image.from(resized)),
            img.flipVertical(img.Image.from(resized)),
            img.flip(img.Image.from(resized), direction: img.FlipDirection.both),
          ]
        : <img.Image>[resized];

    // All views in ONE batched invoke (batch dim is dynamic; batched output is
    // bit-identical to sequential, verified off-device). 4 TTA views per model
    // thus cost 1 invoke instead of 4.
    final n = views.length;
    final viewFloats = w * h * 3;
    interp.resizeInputTensor(0, [n, h, w, 3]);
    interp.allocateTensors();
    final buf = Float32List(n * viewFloats);
    for (int k = 0; k < n; k++) {
      _fillFloat32Raw(views[k], buf, w, h, offset: k * viewFloats); // [0,255]
    }
    _guardCnnRange(buf, tag); // hard safeguard against [0,1] slipping through
    interp.getInputTensor(0).data = buf.buffer.asUint8List();
    interp.invoke();
    final out = interp.getOutputTensor(0).data.buffer.asFloat32List();

    final sum = List<double>.filled(_classes.length, 0.0);
    for (int k = 0; k < n; k++) {
      for (int i = 0; i < _classes.length; i++) {
        sum[i] += out[k * _classes.length + i];
      }
    }
    final inv = 1.0 / n;
    return [for (int i = 0; i < _classes.length; i++) sum[i] * inv];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Stage 3 — Score-CAM
  // ─────────────────────────────────────────────────────────────────────────

  Future<({Map<String, String> classPaths, String? rationale})> _runScoreCam(
      img.Image original, String imagePath, int targetClass) async {
    try {
      final resized = img.copyResize(original,
          width: _b3W, height: _b3H, interpolation: img.Interpolation.linear);
      final imgBuf = Float32List(_b3W * _b3H * 3);
      _fillFloat32Raw(resized, imgBuf, _b3W, _b3H); // [0,255] — EfficientNet rescales internally
      _guardCnnRange(imgBuf, 'feat');               // hard safeguard against [0,1] slipping through

      // ── Extract feature maps ─────────────────────────────────────────────
      _feat!.getInputTensor(0).data = imgBuf.buffer.asUint8List();
      _feat!.invoke();
      // Read actual tensor shape dynamically — avoids hardcoded shape bugs
      final featShape = _feat!.getOutputTensor(0).shape; // [1, H, W, C]
      final featH = featShape[1];
      final featW = featShape[2];
      final featC = featShape[3];
      final features = Float32List.fromList(
          _feat!.getOutputTensor(0).data.buffer.asFloat32List()); // owned copy — native TFLite buffer is invalidated on next invoke

      // ── Select top-K channels by mean activation ─────────────────────────
      final means = List<double>.generate(featC, (c) {
        double sum = 0;
        for (int h = 0; h < featH; h++) {
          for (int w = 0; w < featW; w++) {
            sum += (features[(h * featW + w) * featC + c]).abs();
          }
        }
        return sum / (featH * featW);
      });

      final topKIdx = List<int>.generate(featC, (i) => i)
        ..sort((a, b) => means[b].compareTo(means[a]));
      final selectedChannels = topKIdx.sublist(0, _topK);

      // ── Per-channel masked forward passes ────────────────────────────────
      // rawUpsampled is class-independent (it's the channel activation upsampled).
      // scoresPerClass[cls][ki] captures the masked probability for EVERY class
      // so a separate heatmap can be assembled per class from the same passes.
      final scoresPerClass = List<List<double>>.generate(
          _classes.length, (_) => List<double>.filled(_topK, 0.0));
      final rawUpsampled =
          List<Float32List>.generate(_topK, (_) => Float32List(_b3W * _b3H));

      // Masked passes run batched in chunks (batch dim is dynamic; batched
      // output is bit-identical to sequential, verified off-device): topK=10
      // costs 3 invokes instead of 10. NOTE: this resize is also required for
      // correctness — _runCnn leaves _b3's input at the TTA batch shape, so a
      // single-image buffer would no longer match the tensor size.
      const camBatch = 4;
      const imgFloats = _b3W * _b3H * 3;
      _b3!.resizeInputTensor(0, [camBatch, _b3H, _b3W, 3]);
      _b3!.allocateTensors();
      final chunkBuf = Float32List(camBatch * imgFloats);

      for (int base = 0; base < _topK; base += camBatch) {
        final count = math.min(camBatch, _topK - base);
        for (int bi = 0; bi < count; bi++) {
          final ki = base + bi;
          final c = selectedChannels[ki];

          // Extract raw channel map (featH x featW)
          final rawChan = Float32List(featH * featW);
          double cMin = double.infinity, cMax = double.negativeInfinity;
          for (int h = 0; h < featH; h++) {
            for (int w = 0; w < featW; w++) {
              final v = features[(h * featW + w) * featC + c];
              rawChan[h * featW + w] = v;
              if (v < cMin) cMin = v;
              if (v > cMax) cMax = v;
            }
          }

          // Build per-channel-normalized image for bilinear upsample (masking only)
          final chanImg = img.Image(width: featW, height: featH);
          final range = (cMax - cMin).abs() > 1e-8 ? cMax - cMin : 1.0;
          for (int h = 0; h < featH; h++) {
            for (int w = 0; w < featW; w++) {
              final norm = ((rawChan[h * featW + w] - cMin) / range * 255)
                  .round()
                  .clamp(0, 255);
              chanImg.setPixelRgb(w, h, norm, norm, norm);
            }
          }
          final upNorm = img.copyResize(chanImg,
              width: _b3W, height: _b3H, interpolation: img.Interpolation.linear);

          // Write masked input into this view's slice + store raw upsampled
          // values in one pass. upNorm encodes (rawValue - cMin) / range as a
          // [0,1] pixel. Inverting: rawValue = maskVal * range + cMin.
          final off = bi * imgFloats;
          for (int h = 0; h < _b3H; h++) {
            for (int w = 0; w < _b3W; w++) {
              final maskVal = upNorm.getPixel(w, h).r.toDouble() / 255.0;
              rawUpsampled[ki][h * _b3W + w] = (maskVal * range + cMin).toDouble();
              final idx = (h * _b3W + w) * 3;
              chunkBuf[off + idx]     = imgBuf[idx]     * maskVal;
              chunkBuf[off + idx + 1] = imgBuf[idx + 1] * maskVal;
              chunkBuf[off + idx + 2] = imgBuf[idx + 2] * maskVal;
            }
          }
        }

        // Unused slots in a short final chunk hold stale data; their outputs
        // are simply never read below.
        _b3!.getInputTensor(0).data = chunkBuf.buffer.asUint8List();
        _b3!.invoke();
        final probsOut = _b3!.getOutputTensor(0).data.buffer.asFloat32List();
        for (int bi = 0; bi < count; bi++) {
          for (int cls = 0; cls < _classes.length; cls++) {
            scoresPerClass[cls][base + bi] =
                probsOut[bi * _classes.length + cls].toDouble();
          }
        }

        // Yield between chunks to keep UI responsive
        await Future.delayed(Duration.zero);
      }

      // ── Assemble one heatmap per class ────────────────────────────────────
      final tmpDir = await getTemporaryDirectory();
      final classPaths = <String, String>{};
      String? rationale;

      for (int cls = 0; cls < _classes.length; cls++) {
        final heatmap = _assembleHeatmap(scoresPerClass[cls], rawUpsampled);
        final path = await _saveHeatmapOverlay(
            heatmap, original, tmpDir.path, 'scorecam_${_classes[cls]}');
        if (path != null) classPaths[_classes[cls]] = path;
        if (cls == targetClass) {
          rationale = _heatmapRationale(heatmap, _classes[cls]);
        }
        await Future.delayed(Duration.zero);
      }

      _dlog(
        '[ScoreCAM] done — featShape=${featH}x${featW}x$featC '
        'classes=${classPaths.keys.join(",")}',
        name: 'AIService',
      );
      return (classPaths: classPaths, rationale: rationale);
    } catch (e, st) {
      // Score-CAM failure is non-fatal — return empty, show no heatmap
      _dlog('[ScoreCAM] failed: $e\n$st', name: 'AIService');
      return (classPaths: <String, String>{}, rationale: null);
    }
  }

  /// Softmax the per-channel [scores] into weights, sum the [rawUpsampled]
  /// channel maps by those weights, ReLU, and normalise to [0,1].
  Float32List _assembleHeatmap(
      List<double> scores, List<Float32List> rawUpsampled) {
    final maxScore = scores.reduce(math.max);
    final expScores = scores.map((s) => math.exp(s - maxScore)).toList();
    final sumExp = expScores.reduce((a, b) => a + b);
    final normScores = expScores.map((e) => e / sumExp).toList();

    final heatmap = Float32List(_b3W * _b3H);
    for (int ki = 0; ki < scores.length; ki++) {
      final wgt = normScores[ki];
      final ru = rawUpsampled[ki];
      for (int p = 0; p < heatmap.length; p++) {
        heatmap[p] += wgt * ru[p];
      }
    }
    for (int p = 0; p < heatmap.length; p++) {
      if (heatmap[p] < 0) heatmap[p] = 0;
    }
    double hMin = heatmap[0], hMax = heatmap[0];
    for (final v in heatmap) {
      if (v < hMin) hMin = v;
      if (v > hMax) hMax = v;
    }
    final hRange = (hMax - hMin) > 1e-8 ? hMax - hMin : 1.0;
    for (int p = 0; p < heatmap.length; p++) {
      heatmap[p] = (heatmap[p] - hMin) / hRange;
    }
    return heatmap;
  }

  /// Jet-colour a normalised [_b3W]×[_b3H] heatmap, upsample to [original]'s
  /// size, blend 60/40 over the photo, and write a JPEG. Returns its path.
  Future<String?> _saveHeatmapOverlay(Float32List heatmap, img.Image original,
      String dirPath, String tag) async {
    final heatmapImg = img.Image(width: _b3W, height: _b3H);
    for (int h = 0; h < _b3H; h++) {
      for (int w = 0; w < _b3W; w++) {
        final jet = _jetColor(heatmap[h * _b3W + w]);
        heatmapImg.setPixelRgb(w, h, jet[0], jet[1], jet[2]);
      }
    }
    final heatmapResized = img.copyResize(heatmapImg,
        width: original.width,
        height: original.height,
        interpolation: img.Interpolation.linear);
    final overlay = img.Image(width: original.width, height: original.height);
    for (int h = 0; h < original.height; h++) {
      for (int w = 0; w < original.width; w++) {
        final op = original.getPixel(w, h);
        final hp = heatmapResized.getPixel(w, h);
        overlay.setPixelRgb(
          w, h,
          (op.r * 0.6 + hp.r * 0.4).round().clamp(0, 255),
          (op.g * 0.6 + hp.g * 0.4).round().clamp(0, 255),
          (op.b * 0.6 + hp.b * 0.4).round().clamp(0, 255),
        );
      }
    }
    final outPath =
        '$dirPath/${tag}_${DateTime.now().millisecondsSinceEpoch}.jpg';
    await File(outPath).writeAsBytes(img.encodeJpg(overlay, quality: 90));
    return outPath;
  }

  /// Describe WHERE the model looked from a normalised heatmap: weighted
  /// centroid (location), spatial spread (focal vs diffuse), and the fraction
  /// of strongly-attended pixels (coverage).
  String _heatmapRationale(Float32List heatmap, String className) {
    const n = _b3W * _b3H;
    double sum = 0, sx = 0, sy = 0;
    int highCount = 0;
    const thr = 0.5;
    for (int y = 0; y < _b3H; y++) {
      for (int x = 0; x < _b3W; x++) {
        final v = heatmap[y * _b3W + x];
        sum += v;
        sx += v * x;
        sy += v * y;
        if (v > thr) highCount++;
      }
    }
    if (sum < 1e-6) {
      return 'The attention map for $className was diffuse with no dominant region.';
    }
    final cx = (sx / sum) / _b3W;
    final cy = (sy / sum) / _b3H;
    final ccx = cx * _b3W, ccy = cy * _b3H;
    double varSum = 0;
    for (int y = 0; y < _b3H; y++) {
      for (int x = 0; x < _b3W; x++) {
        final v = heatmap[y * _b3W + x];
        final dx = x - ccx, dy = y - ccy;
        varSum += v * (dx * dx + dy * dy);
      }
    }
    final spread = math.sqrt(varSum / sum) /
        math.sqrt((_b3W * _b3W + _b3H * _b3H).toDouble());
    final coverage = highCount / n;

    final vert = cy < 0.4 ? 'upper' : (cy > 0.6 ? 'lower' : 'central');
    final horiz = cx < 0.4 ? 'left' : (cx > 0.6 ? 'right' : 'centre');
    final loc = (vert == 'central' && horiz == 'centre')
        ? 'the centre of the image'
        : 'the $vert-$horiz region';
    final focal = spread < 0.22
        ? 'a single focal area'
        : (spread < 0.32 ? 'a moderately spread area' : 'a diffuse area');

    return 'For $className, the model concentrated on $focal in $loc, with '
        'high-attention pixels covering about ${(coverage * 100).round()}% of the image.';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Helpers
  // ─────────────────────────────────────────────────────────────────────────

  /// Simplified DullRazor hair removal — VAE preprocessing only.
  ///
  /// Detects dark thin lines (hair) via a local bottom-hat comparison and
  /// fills them with the mean colour of surrounding non-hair pixels.
  /// The CNN always receives the original image; only the VAE sees this copy.
  // ignore: unused_element  (parked alongside _runVae for VAE re-integration)
  img.Image _removeHair(img.Image src) {
    final w = src.width, h = src.height;
    const r = _hairRadius;

    // Pre-extract luminance into a flat Float32 array.
    // Using array indexing for the inner loops is ~10× faster than repeated
    // getPixel dispatch inside the nested neighbourhood scan.
    final lum = Float32List(w * h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = src.getPixel(x, y);
        lum[y * w + x] =
            (p.rNormalized + p.gNormalized + p.bNormalized) / 3.0;
      }
    }

    final out = img.Image.from(src);
    int hairPixels = 0;

    for (int y = r; y < h - r; y++) {
      for (int x = r; x < w - r; x++) {
        final myLum = lum[y * w + x];

        // Local maximum luminance in (2r+1)² window
        double maxLum = myLum;
        for (int dy = -r; dy <= r; dy++) {
          final row = (y + dy) * w;
          for (int dx = -r; dx <= r; dx++) {
            final v = lum[row + x + dx];
            if (v > maxLum) maxLum = v;
          }
        }

        if (maxLum - myLum < _hairDarkness) continue; // not hair

        // Hair pixel — replace with mean of non-hair neighbours
        double rs = 0, gs = 0, bs = 0;
        int cnt = 0;
        for (int dy = -r; dy <= r; dy++) {
          final row = (y + dy) * w;
          for (int dx = -r; dx <= r; dx++) {
            if (maxLum - lum[row + x + dx] <= _hairDarkness) {
              final np = src.getPixel(x + dx, y + dy);
              rs += np.rNormalized;
              gs += np.gNormalized;
              bs += np.bNormalized;
              cnt++;
            }
          }
        }
        if (cnt > 0) {
          out.setPixelRgb(
            x, y,
            (rs / cnt * 255).round().clamp(0, 255),
            (gs / cnt * 255).round().clamp(0, 255),
            (bs / cnt * 255).round().clamp(0, 255),
          );
          hairPixels++;
        }
      }
    }
    _dlog(
      '[Hair] removed $hairPixels hair pixels '
      '(${(hairPixels / (w * h) * 100).toStringAsFixed(1)}% of image)',
      name: 'AIService',
    );
    return out;
  }

  /// Last-line safeguard before a buffer is handed to an EfficientNet model.
  ///
  /// The B2/B3/feature-extractor graphs contain a baked-in Rescaling(1/255),
  /// so they REQUIRE [0,255] input. If pixels somehow arrive in [0,1] (an
  /// unforeseen decode format slipping past `_toUint8Rgb` + `_fillFloat32Raw`),
  /// the model sees near-black and collapses to a single class (Tinea). This
  /// guard detects that case at the model boundary and rescales in-place, so
  /// the bias can never silently reappear. A correct [0,255] image is a no-op.
  ///
  /// Threshold 1.5: any natural photo spans well above 1.5 after correct
  /// scaling; a buffer whose maximum is ≤1.5 is almost certainly [0,1] (or a
  /// genuinely black frame, where rescaling is harmless).
  void _guardCnnRange(Float32List buf, String tag) {
    double maxVal = 0;
    for (final v in buf) {
      if (v > maxVal) maxVal = v;
    }
    if (maxVal > 0 && maxVal <= 1.5) {
      _dlog(
        '[$tag] SELF-CORRECT pixel max=$maxVal looked like [0,1]; '
        'rescaling ×255 to [0,255] before inference.',
        name: 'AIService',
      );
      for (int i = 0; i < buf.length; i++) {
        buf[i] *= 255.0;
      }
    }
  }

  /// Return a guaranteed uint8 RGB [0,255] copy of [src].
  ///
  /// If [src] is already uint8 it is returned unchanged. Otherwise we build a
  /// fresh uint8 image by reading each pixel's *Normalized value ([0,1] for any
  /// format) and scaling to [0,255] — correct for float32 / uint16 sources
  /// where `Image.convert` may clamp rather than rescale. Downstream code
  /// (model fills via `_fillFloat32Raw`, heatmap overlay blends) relies on
  /// `original` being a true [0,255] image.
  img.Image _toUint8Rgb(img.Image src) {
    if (src.format == img.Format.uint8 && src.numChannels >= 3) return src;
    final out = img.Image(width: src.width, height: src.height);
    for (int y = 0; y < src.height; y++) {
      for (int x = 0; x < src.width; x++) {
        final p = src.getPixel(x, y);
        out.setPixelRgb(
          x, y,
          (p.rNormalized * 255).round().clamp(0, 255),
          (p.gNormalized * 255).round().clamp(0, 255),
          (p.bNormalized * 255).round().clamp(0, 255),
        );
      }
    }
    return out;
  }

  /// Fill [buf] with normalised [0,1] RGB — used by the VAE, which was
  /// built from scratch with no internal preprocessing layer.
  ///
  /// Uses the *Normalized channel accessors so the result is [0,1] regardless
  /// of the decoded image's pixel format (uint8 0-255, uint16 0-65535, or
  /// float32 0-1). Reading raw `p.r` would yield a format-dependent range.
  void _fillFloat32(img.Image image, Float32List buf, int w, int h,
      {int offset = 0}) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final p = image.getPixel(col, row);
        final i = offset + (row * w + col) * 3;
        buf[i]     = p.rNormalized.toDouble();
        buf[i + 1] = p.gNormalized.toDouble();
        buf[i + 2] = p.bNormalized.toDouble();
      }
    }
  }

  /// Fill [buf] with raw [0,255] RGB — used by EfficientNetB2/B3 and the
  /// B3 feature extractor, which include an internal Rescaling(1/255) layer
  /// baked into the TFLite graph.  Sending [0,1] would double-normalise to
  /// [0, ~0.004] and the backbone would see near-black images — which makes
  /// the classifier collapse to a single class (empirically: Tinea).
  ///
  /// We read the *Normalized accessors ([0,1] for any pixel format) and scale
  /// to [0,255]. This is bulletproof against HEIC/HDR float decodes and 16-bit
  /// PNGs, where raw `p.r` is NOT in [0,255] — the root cause of the bias.
  void _fillFloat32Raw(img.Image image, Float32List buf, int w, int h,
      {int offset = 0}) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final p = image.getPixel(col, row);
        final i = offset + (row * w + col) * 3;
        buf[i]     = p.rNormalized.toDouble() * 255.0;
        buf[i + 1] = p.gNormalized.toDouble() * 255.0;
        buf[i + 2] = p.bNormalized.toDouble() * 255.0;
      }
    }
  }

  int _argmax(List<double> list) {
    int best = 0;
    for (int i = 1; i < list.length; i++) {
      if (list[i] > list[best]) best = i;
    }
    return best;
  }

  /// Jet colormap: t in [0,1] → [r, g, b] each 0-255.
  List<int> _jetColor(double t) {
    t = t.clamp(0.0, 1.0);
    double r, g, b;
    if (t < 0.125) {
      r = 0; g = 0; b = 0.5 + t * 4;
    } else if (t < 0.375) {
      r = 0; g = (t - 0.125) * 4; b = 1.0;
    } else if (t < 0.625) {
      r = (t - 0.375) * 4; g = 1.0; b = 1.0 - (t - 0.375) * 4;
    } else if (t < 0.875) {
      r = 1.0; g = 1.0 - (t - 0.625) * 4; b = 0.0;
    } else {
      r = 1.0 - (t - 0.875) * 4; g = 0.0; b = 0.0;
    }
    return [
      (r * 255).round().clamp(0, 255),
      (g * 255).round().clamp(0, 255),
      (b * 255).round().clamp(0, 255),
    ];
  }

  List<String> getSupportedClasses() => _classes;
}

/// Riverpod provider
final aiServiceProvider = Provider<AIService>((ref) => AIService());
