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
  final String? heatmapPath;       // Score-CAM overlay image saved to device
  final double? anomalyRatio;      // VAE anomaly ratio — not persisted to DB
  final String? vaeHeatmapPath;    // VAE anomaly heatmap image — not persisted to DB
  final int? vaeMs;                // Per-stage inference timing (ms) — not persisted
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
    this.anomalyRatio,
    this.vaeHeatmapPath,
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
    double? anomalyRatio,
    String? vaeHeatmapPath,
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
      anomalyRatio: anomalyRatio ?? this.anomalyRatio,
      vaeHeatmapPath: vaeHeatmapPath ?? this.vaeHeatmapPath,
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
        anomalyRatio,
        vaeHeatmapPath,
        vaeMs,
        cnnMs,
        scoreCamMs,
      ];
}
