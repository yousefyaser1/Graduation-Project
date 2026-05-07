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
const double _anomalyThreshold = 0.008;   // calibrated for TFLite (PyTorch 0.006 + ~0.002 offset)
const double _anomalyRatio     = 0.20;

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
  final String? heatmapPath;        // path to Score-CAM overlay JPEG on device
  final String? vaeHeatmapPath;     // path to VAE anomaly heatmap JPEG on device
  final double anomalyRatio;        // VAE sliding-window anomaly ratio [0,1]
  final int vaeMs;                  // VAE stage wall-clock duration (ms)
  final int cnnMs;                  // CNN ensemble duration (ms); 0 when isNormal
  final int scoreCamMs;             // Score-CAM duration (ms); 0 when isNormal

  const AnalysisResult({
    required this.isNormal,
    required this.diagnosis,
    required this.confidence,
    required this.classProbabilities,
    required this.anomalyRatio,
    this.heatmapPath,
    this.vaeHeatmapPath,
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
    final rawBytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(rawBytes);
    if (decoded == null) throw Exception('Failed to decode image: $imagePath');

    // Normalise to uint8 RGB once so every downstream stage receives the
    // correct [0,255] integer range.  The image pkg 4.x can return float32
    // [0,1] for HDR / HEIC sources, which breaks EfficientNet's internal
    // Rescaling(1/255) and causes the model to always predict Acne.
    final decodedU8 = (decoded.format == img.Format.uint8)
        ? decoded
        : decoded.convert(format: img.Format.uint8, numChannels: 3);

    // Only downsample very large photos so the VAE sliding window doesn't
    // explode on 12MP camera shots. Keep well above the VAE's training scale
    // (patches must still look like the training data) and ALWAYS use linear
    // interpolation — the default is nearest-neighbor which introduces
    // aliasing artefacts that the VAE flags as anomalies.
    final original = decodedU8.width > 1280
        ? img.copyResize(decodedU8,
            width: 1280, interpolation: img.Interpolation.linear)
        : decodedU8;

    // Step 1 – VAE anomaly gate
    var stageStart = DateTime.now().millisecondsSinceEpoch;
    onStepChange?.call(1);
    final (:isAnomaly, :ratio, :vaeHeatmapPath) = await _runVae(original);
    final vaeMs = DateTime.now().millisecondsSinceEpoch - stageStart;

    if (!isAnomaly) {
      return AnalysisResult(
        isNormal: true,
        diagnosis: '',
        confidence: 1.0,
        classProbabilities: const {},
        anomalyRatio: ratio,
        vaeHeatmapPath: vaeHeatmapPath,
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
    final heatmapPath = await _runScoreCam(original, imagePath, predIdx);
    final scoreCamMs = DateTime.now().millisecondsSinceEpoch - stageStart;

    return AnalysisResult(
      isNormal: false,
      diagnosis: diagnosis,
      confidence: confidence,
      classProbabilities: probMap,
      anomalyRatio: ratio,
      vaeHeatmapPath: vaeHeatmapPath,
      heatmapPath: heatmapPath,
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

    // Diagnostic: warn if pixels look like [0,1] instead of [0,255]
    final maxVal = buf.reduce(math.max);
    if (maxVal < 1.1) {
      developer.log(
        '[$tag] WARNING pixel max=$maxVal — values are in [0,1] not [0,255]; '
        'image format conversion may have failed.',
        name: 'AIService',
      );
    }

    interp.getInputTensor(0).data = buf.buffer.asUint8List();
    interp.invoke();
    final out = interp.getOutputTensor(0).data.buffer.asFloat32List();
    return List<double>.from(out);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Stage 3 — Score-CAM
  // ─────────────────────────────────────────────────────────────────────────

  Future<String?> _runScoreCam(
      img.Image original, String imagePath, int targetClass) async {
    try {
      final resized = img.copyResize(original,
          width: _b3W, height: _b3H, interpolation: img.Interpolation.linear);
      final imgBuf = Float32List(_b3W * _b3H * 3);
      _fillFloat32Raw(resized, imgBuf, _b3W, _b3H); // [0,255] — EfficientNet rescales internally

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
      // rawUpsampled: raw feature values after bilinear upsampling (for heatmap sum)
      // normMask: per-channel-normalized [0,1] mask (for image masking)
      final scores = List<double>.filled(_topK, 0.0);
      final rawUpsampled  = List<Float32List>.generate(_topK, (_) => Float32List(_b3W * _b3H));

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
        scores[ki] = probs[targetClass].toDouble();

        // Yield every 5 iterations to keep UI responsive
        if (ki % 5 == 0) await Future.delayed(Duration.zero);
      }

      // ── Softmax over scores ───────────────────────────────────────────────
      final maxScore = scores.reduce(math.max);
      final expScores = scores.map((s) => math.exp(s - maxScore)).toList();
      final sumExp = expScores.reduce((a, b) => a + b);
      final normScores = expScores.map((e) => e / sumExp).toList();

      // ── Weighted heatmap sum (raw maps, matching Python score_cam.py) ─────
      final heatmap = Float32List(_b3W * _b3H);
      for (int ki = 0; ki < _topK; ki++) {
        for (int p = 0; p < _b3W * _b3H; p++) {
          heatmap[p] += normScores[ki] * rawUpsampled[ki][p];
        }
      }

      // ReLU — remove negative contributions (matches Python np.maximum(heatmap, 0))
      for (int p = 0; p < heatmap.length; p++) {
        if (heatmap[p] < 0) heatmap[p] = 0;
      }

      // Normalise to [0, 1]
      double hMin = heatmap[0], hMax = heatmap[0];
      for (final v in heatmap) {
        if (v < hMin) hMin = v;
        if (v > hMax) hMax = v;
      }
      final hRange = (hMax - hMin) > 1e-8 ? hMax - hMin : 1.0;
      for (int p = 0; p < heatmap.length; p++) {
        heatmap[p] = (heatmap[p] - hMin) / hRange;
      }

      // Resize heatmap back to original image size
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

      // Blend with original (60% original + 40% heatmap)
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

      // Save to temp directory
      final tmpDir = await getTemporaryDirectory();
      final outPath =
          '${tmpDir.path}/scorecam_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(outPath).writeAsBytes(img.encodeJpg(overlay, quality: 90));
      developer.log(
        '[ScoreCAM] done — featShape=${featH}x${featW}x${featC} '
        'heatRange=[${hMin.toStringAsFixed(4)}, ${hMax.toStringAsFixed(4)}] '
        'saved=$outPath',
        name: 'AIService',
      );
      return outPath;
    } catch (e, st) {
      // Score-CAM failure is non-fatal — return null, show no heatmap
      developer.log('[ScoreCAM] failed: $e\n$st', name: 'AIService');
      return null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Helpers
  // ─────────────────────────────────────────────────────────────────────────

  /// Fill [buf] with normalised [0,1] RGB — used by the VAE, which was
  /// built from scratch with no internal preprocessing layer.
  void _fillFloat32(img.Image image, Float32List buf, int w, int h) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final p = image.getPixel(col, row);
        final i = (row * w + col) * 3;
        buf[i]     = p.r.toDouble() / 255.0;
        buf[i + 1] = p.g.toDouble() / 255.0;
        buf[i + 2] = p.b.toDouble() / 255.0;
      }
    }
  }

  /// Fill [buf] with raw [0,255] RGB — used by EfficientNetB2/B3 and the
  /// B3 feature extractor, which include an internal Rescaling(1/255) layer
  /// baked into the TFLite graph.  Sending [0,1] would double-normalise to
  /// [0, ~0.004] and the backbone would see near-black images.
  void _fillFloat32Raw(img.Image image, Float32List buf, int w, int h) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final p = image.getPixel(col, row);
        final i = (row * w + col) * 3;
        buf[i]     = p.r.toDouble();   // [0,255]
        buf[i + 1] = p.g.toDouble();
        buf[i + 2] = p.b.toDouble();
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
