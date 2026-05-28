import 'dart:convert';
import 'package:equatable/equatable.dart';

/// Model representing a scan result
class ScanResult extends Equatable {
  final String id;
  final String imagePath;
  final String bodyPart;
  final String diagnosis;
  final double confidence;
  final Map<String, double> classProbabilities;
  final DateTime timestamp;
  final String? notes;
  final String? heatmapPath;       // Score-CAM overlay image (predicted class)
  final Map<String, String> classHeatmapPaths; // per-class Score-CAM overlays
  final String? xaiRationale;      // region-aware text describing where the model looked
  final double? anomalyRatio;      // VAE anomaly ratio
  final String? vaeHeatmapPath;    // VAE anomaly heatmap image
  final int? preprocessMs;         // Per-stage inference timing (ms)
  final int? vaeMs;
  final int? cnnMs;
  final int? scoreCamMs;

  const ScanResult({
    required this.id,
    required this.imagePath,
    required this.bodyPart,
    required this.diagnosis,
    required this.confidence,
    required this.classProbabilities,
    required this.timestamp,
    this.notes,
    this.heatmapPath,
    this.classHeatmapPaths = const {},
    this.xaiRationale,
    this.anomalyRatio,
    this.vaeHeatmapPath,
    this.preprocessMs,
    this.vaeMs,
    this.cnnMs,
    this.scoreCamMs,
  });

  ScanResult copyWith({
    String? id,
    String? imagePath,
    String? bodyPart,
    String? diagnosis,
    double? confidence,
    Map<String, double>? classProbabilities,
    DateTime? timestamp,
    String? notes,
    String? heatmapPath,
    Map<String, String>? classHeatmapPaths,
    String? xaiRationale,
    double? anomalyRatio,
    String? vaeHeatmapPath,
    int? preprocessMs,
    int? vaeMs,
    int? cnnMs,
    int? scoreCamMs,
  }) {
    return ScanResult(
      id: id ?? this.id,
      imagePath: imagePath ?? this.imagePath,
      bodyPart: bodyPart ?? this.bodyPart,
      diagnosis: diagnosis ?? this.diagnosis,
      confidence: confidence ?? this.confidence,
      classProbabilities: classProbabilities ?? this.classProbabilities,
      timestamp: timestamp ?? this.timestamp,
      notes: notes ?? this.notes,
      heatmapPath: heatmapPath ?? this.heatmapPath,
      classHeatmapPaths: classHeatmapPaths ?? this.classHeatmapPaths,
      xaiRationale: xaiRationale ?? this.xaiRationale,
      anomalyRatio: anomalyRatio ?? this.anomalyRatio,
      vaeHeatmapPath: vaeHeatmapPath ?? this.vaeHeatmapPath,
      preprocessMs: preprocessMs ?? this.preprocessMs,
      vaeMs: vaeMs ?? this.vaeMs,
      cnnMs: cnnMs ?? this.cnnMs,
      scoreCamMs: scoreCamMs ?? this.scoreCamMs,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'image_path': imagePath,
      'body_part': bodyPart,
      'diagnosis': diagnosis,
      'confidence': confidence,
      'class_probabilities': classProbabilities,
      'timestamp': timestamp.millisecondsSinceEpoch,
      'notes': notes,
      'heatmap_path': heatmapPath,
      'class_heatmap_paths':
          classHeatmapPaths.isEmpty ? null : jsonEncode(classHeatmapPaths),
      'xai_rationale': xaiRationale,
      'vae_heatmap_path': vaeHeatmapPath,
      'anomaly_ratio': anomalyRatio,
      'preprocess_ms': preprocessMs,
      'vae_ms': vaeMs,
      'cnn_ms': cnnMs,
      'score_cam_ms': scoreCamMs,
    };
  }

  factory ScanResult.fromMap(Map<String, dynamic> map) {
    return ScanResult(
      id: map['id'] as String,
      imagePath: map['image_path'] as String,
      bodyPart: map['body_part'] as String,
      diagnosis: map['diagnosis'] as String,
      confidence: map['confidence'] as double,
      classProbabilities: Map<String, double>.from(
        map['class_probabilities'] as Map,
      ),
      timestamp: DateTime.fromMillisecondsSinceEpoch(map['timestamp'] as int),
      notes: map['notes'] as String?,
      heatmapPath: map['heatmap_path'] as String?,
      classHeatmapPaths: map['class_heatmap_paths'] == null
          ? const {}
          : Map<String, String>.from(
              (jsonDecode(map['class_heatmap_paths'] as String) as Map)
                  .map((k, v) => MapEntry(k as String, v as String)),
            ),
      xaiRationale: map['xai_rationale'] as String?,
      vaeHeatmapPath: map['vae_heatmap_path'] as String?,
      anomalyRatio: (map['anomaly_ratio'] as num?)?.toDouble(),
      preprocessMs: (map['preprocess_ms'] as num?)?.toInt(),
      vaeMs: (map['vae_ms'] as num?)?.toInt(),
      cnnMs: (map['cnn_ms'] as num?)?.toInt(),
      scoreCamMs: (map['score_cam_ms'] as num?)?.toInt(),
    );
  }

  @override
  List<Object?> get props => [
        id,
        imagePath,
        bodyPart,
        diagnosis,
        confidence,
        classProbabilities,
        timestamp,
        notes,
        heatmapPath,
        classHeatmapPaths,
        xaiRationale,
        anomalyRatio,
        vaeHeatmapPath,
        preprocessMs,
        vaeMs,
        cnnMs,
        scoreCamMs,
      ];
}
