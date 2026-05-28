import 'dart:developer' as developer;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const List<String> _classes = ['Acne', 'Eczema', 'Tinea'];

// VAE sliding-window params
const int    _patchSize        = 64;
const int    _stride           = 32;
// Recalibrated on the labelled test set (verify_pipeline.py / vae_threshold_sweep.py):
// the previous 0.008 / 0.20 gate flagged only ~57% of genuinely diseased images
// as anomalous — i.e. it silently dropped ~43% of real cases as "No Disease
// Detected" before the CNN ever ran. The assumed "+0.002 TFLite offset" had
// over-corrected the per-patch threshold. 0.004 / 0.15 recovers ~96% disease
// detection. (NOTE: validate against healthy-skin samples to bound false
// positives — the test set contains only diseased images.)
const double _anomalyThreshold = 0.004;
const double _anomalyRatio     = 0.15;

// CNN input sizes
const int _b2W = 260, _b2H = 260;
const int _b3W = 300, _b3H = 300;

// Score-CAM params (shapes read dynamically from model at runtime)
const int _topK  = 30;

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

class AnalysisResult {
  final bool isNormal;
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
}

// ─────────────────────────────────────────────────────────────────────────────
// AIService
// ─────────────────────────────────────────────────────────────────────────────

class AIService {
  static final AIService _instance = AIService._internal();
  factory AIService() => _instance;
  AIService._internal();

  Interpreter? _vae;
  Interpreter? _b2;
  Interpreter? _b3;
  Interpreter? _feat;
  bool _ready = false;

  // ── Initialise ─────────────────────────────────────────────────────────────

  Future<void> initialize() async {
    if (_ready) return;
    _vae  = await Interpreter.fromAsset('assets/models/vae_model.tflite');
    _b2   = await Interpreter.fromAsset('assets/models/cnn_b2_model.tflite');
    _b3   = await Interpreter.fromAsset('assets/models/cnn_b3_model.tflite');
    _feat = await Interpreter.fromAsset('assets/models/b3_feature_extractor.tflite');
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
    final decoded = img.decodeImage(rawBytes);
    if (decoded == null) throw Exception('Failed to decode image: $imagePath');

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
    var stageStart = DateTime.now().millisecondsSinceEpoch;
    onStepChange?.call(1);
    final (:isAnomaly, :ratio, :vaeHeatmapPath) = await _runVae(original);
    final vaeMs = DateTime.now().millisecondsSinceEpoch - stageStart;

    if (!isAnomaly) {
      developer.log(
        '[TIMING] preprocess=${preprocessMs}ms vae=${vaeMs}ms cnn=0ms scorecam=0ms total=${preprocessMs + vaeMs}ms | NORMAL',
        name: 'AIService',
      );
      return AnalysisResult(
        isNormal: true,
        diagnosis: '',
        confidence: 1.0,
        classProbabilities: const {},
        anomalyRatio: ratio,
        vaeHeatmapPath: vaeHeatmapPath,
        preprocessMs: preprocessMs,
        vaeMs: vaeMs,
        cnnMs: 0,
        scoreCamMs: 0,
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
    developer.log(
      '[CNN] Acne=${probs[0].toStringAsFixed(3)} '
      'Eczema=${probs[1].toStringAsFixed(3)} '
      'Tinea=${probs[2].toStringAsFixed(3)} '
      '→ $diagnosis (${(confidence * 100).toStringAsFixed(1)}%)',
      name: 'AIService',
    );

    // Step 3 – Score-CAM heatmap
    stageStart = DateTime.now().millisecondsSinceEpoch;
    onStepChange?.call(3);
    final scoreCam = await _runScoreCam(original, imagePath, predIdx);
    final heatmapPath = scoreCam.classPaths[diagnosis];
    final scoreCamMs = DateTime.now().millisecondsSinceEpoch - stageStart;

    final total = preprocessMs + vaeMs + cnnMs + scoreCamMs;
    developer.log(
      '[TIMING] preprocess=${preprocessMs}ms vae=${vaeMs}ms cnn=${cnnMs}ms scorecam=${scoreCamMs}ms total=${total}ms | $diagnosis',
      name: 'AIService',
    );
    return AnalysisResult(
      isNormal: false,
      diagnosis: diagnosis,
      confidence: confidence,
      classProbabilities: probMap,
      anomalyRatio: ratio,
      vaeHeatmapPath: vaeHeatmapPath,
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
  // Stage 1 — VAE sliding-window anomaly detection
  // ─────────────────────────────────────────────────────────────────────────

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
    int anomalous = 0;
    final inputBuf = Float32List(_patchSize * _patchSize * 3);

    for (int gy = 0; gy < gridH; gy++) {
      final y = gy * _stride;
      for (int gx = 0; gx < gridW; gx++) {
        final x = gx * _stride;
        final patch = img.copyCrop(original,
            x: x, y: y, width: _patchSize, height: _patchSize);
        _fillFloat32(patch, inputBuf, _patchSize, _patchSize);

        _vae!.getInputTensor(0).data = inputBuf.buffer.asUint8List();
        _vae!.invoke();
        final mse =
            _vae!.getOutputTensor(0).data.buffer.asFloat32List()[0];

        mseGrid[gy * gridW + gx] = mse;
        if (mse > _anomalyThreshold) anomalous++;
      }
      // Yield to event loop every row so animations keep running
      await Future.delayed(Duration.zero);
    }

    final patches = gridH * gridW;
    final ratio = anomalous / patches;
    developer.log(
      '[VAE] patches=$patches anomalous=$anomalous ratio=${ratio.toStringAsFixed(3)} '
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
      developer.log('[VAE] heatmap saved=$outPath', name: 'AIService');
      return outPath;
    } catch (e, st) {
      developer.log('[VAE] heatmap failed: $e\n$st', name: 'AIService');
      return null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Stage 2 — CNN B2 + B3 ensemble
  // ─────────────────────────────────────────────────────────────────────────

  Future<List<double>> _runCnnEnsemble(img.Image original) async {
    final probsB2 = _runCnn(_b2!, original, _b2W, _b2H, 'B2');
    final probsB3 = _runCnn(_b3!, original, _b3W, _b3H, 'B3');
    developer.log(
      '[CNN-B2] Acne=${probsB2[0].toStringAsFixed(3)} '
      'Eczema=${probsB2[1].toStringAsFixed(3)} '
      'Tinea=${probsB2[2].toStringAsFixed(3)}',
      name: 'AIService',
    );
    developer.log(
      '[CNN-B3] Acne=${probsB3[0].toStringAsFixed(3)} '
      'Eczema=${probsB3[1].toStringAsFixed(3)} '
      'Tinea=${probsB3[2].toStringAsFixed(3)}',
      name: 'AIService',
    );
    return List.generate(
        _classes.length, (i) => (probsB2[i] + probsB3[i]) / 2.0);
  }

  List<double> _runCnn(
      Interpreter interp, img.Image original, int w, int h, String tag) {
    final resized = img.copyResize(original,
        width: w, height: h, interpolation: img.Interpolation.linear);

    final buf = Float32List(w * h * 3);
    _fillFloat32Raw(resized, buf, w, h);   // [0,255] — EfficientNet rescales internally
    _guardCnnRange(buf, tag);              // hard safeguard against [0,1] slipping through

    interp.getInputTensor(0).data = buf.buffer.asUint8List();
    interp.invoke();
    final out = interp.getOutputTensor(0).data.buffer.asFloat32List();
    return List<double>.from(out);
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

      for (int ki = 0; ki < _topK; ki++) {
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

        // Create masked input + store raw upsampled values in one pass.
        // upNorm encodes (rawValue - cMin) / range as a [0,1] pixel.
        // Inverting: rawValue = maskVal * range + cMin  — used for heatmap sum.
        final maskedBuf = Float32List(_b3W * _b3H * 3);
        for (int h = 0; h < _b3H; h++) {
          for (int w = 0; w < _b3W; w++) {
            final maskVal = upNorm.getPixel(w, h).r.toDouble() / 255.0;
            // Recover approx raw feature value (for weighted heatmap, matches Python ch_up)
            rawUpsampled[ki][h * _b3W + w] = (maskVal * range + cMin).toDouble();
            final idx = (h * _b3W + w) * 3;
            maskedBuf[idx]     = imgBuf[idx]     * maskVal;
            maskedBuf[idx + 1] = imgBuf[idx + 1] * maskVal;
            maskedBuf[idx + 2] = imgBuf[idx + 2] * maskVal;
          }
        }

        _b3!.getInputTensor(0).data = maskedBuf.buffer.asUint8List();
        _b3!.invoke();
        final probs = _b3!.getOutputTensor(0).data.buffer.asFloat32List();
        for (int cls = 0; cls < _classes.length; cls++) {
          scoresPerClass[cls][ki] = probs[cls].toDouble();
        }

        // Yield every 5 iterations to keep UI responsive
        if (ki % 5 == 0) await Future.delayed(Duration.zero);
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

      developer.log(
        '[ScoreCAM] done — featShape=${featH}x${featW}x$featC '
        'classes=${classPaths.keys.join(",")}',
        name: 'AIService',
      );
      return (classPaths: classPaths, rationale: rationale);
    } catch (e, st) {
      // Score-CAM failure is non-fatal — return empty, show no heatmap
      developer.log('[ScoreCAM] failed: $e\n$st', name: 'AIService');
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
      developer.log(
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
  void _fillFloat32(img.Image image, Float32List buf, int w, int h) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final p = image.getPixel(col, row);
        final i = (row * w + col) * 3;
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
  void _fillFloat32Raw(img.Image image, Float32List buf, int w, int h) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final p = image.getPixel(col, row);
        final i = (row * w + col) * 3;
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
