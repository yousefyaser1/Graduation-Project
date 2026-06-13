import 'dart:async';
import 'dart:isolate';
import 'package:flutter/services.dart';
import 'ai_service.dart';

// ── Isolate entry point (must be a top-level function) ───────────────────────

void _analysisIsolateEntry(List<dynamic> args) async {
  final gateBytes = (args[0] as TransferableTypedData).materialize().asUint8List();
  final vaeBytes  = (args[1] as TransferableTypedData).materialize().asUint8List();
  final b2Bytes   = (args[2] as TransferableTypedData).materialize().asUint8List();
  final b3Bytes   = (args[3] as TransferableTypedData).materialize().asUint8List();
  final featBytes = (args[4] as TransferableTypedData).materialize().asUint8List();
  final imagePath = args[5] as String;
  final sendPort  = args[6] as SendPort;
  final rootToken = args[7] as RootIsolateToken;

  // Plugins (path_provider's getTemporaryDirectory, used to save the Score-CAM
  // and VAE heatmap JPEGs) talk over platform channels, which are not wired up
  // in a spawned isolate by default. Without this the heatmap saves throw
  // "BackgroundIsolateBinaryMessenger.instance is invalid ...", the error is
  // swallowed by the heatmap try/catch, and the result comes back with no
  // overlay. This must run before any plugin call in the isolate.
  BackgroundIsolateBinaryMessenger.ensureInitialized(rootToken);

  try {
    final service = AIService();
    service.initializeFromBuffers(
      gateBytes: gateBytes,
      vaeBytes: vaeBytes,
      b2Bytes: b2Bytes,
      b3Bytes: b3Bytes,
      featBytes: featBytes,
    );
    final result = await service.analyzeImage(
      imagePath: imagePath,
      onStepChange: (step) => sendPort.send(step),
    );
    sendPort.send(result.toIsolateMap());
  } catch (e, st) {
    sendPort.send({'error': '$e\n$st'});
  }
}

// ── Public helper ────────────────────────────────────────────────────────────

/// Runs the full AI pipeline in a background isolate so the UI stays responsive.
///
/// [onStepChange] is called (on the main isolate) as each pipeline stage starts.
Future<AnalysisResult> runAnalysisInBackground({
  required String imagePath,
  required void Function(int step) onStepChange,
}) async {
  Future<Uint8List> load(String asset) async {
    final d = await rootBundle.load(asset);
    return d.buffer.asUint8List();
  }

  final gateBytes = await load('assets/models/normal_gate.tflite');
  final vaeBytes  = await load('assets/models/vae_model.tflite');
  final b2Bytes   = await load('assets/models/cnn_b2_model.tflite');
  final b3Bytes   = await load('assets/models/cnn_b3_model.tflite');
  final featBytes = await load('assets/models/b3_feature_extractor.tflite');

  // Required so the spawned isolate can use plugin platform channels
  // (path_provider) to save heatmap files. Captured on the root isolate.
  final rootToken = RootIsolateToken.instance!;

  final receivePort = ReceivePort();
  await Isolate.spawn(
    _analysisIsolateEntry,
    [
      TransferableTypedData.fromList([gateBytes]),
      TransferableTypedData.fromList([vaeBytes]),
      TransferableTypedData.fromList([b2Bytes]),
      TransferableTypedData.fromList([b3Bytes]),
      TransferableTypedData.fromList([featBytes]),
      imagePath,
      receivePort.sendPort,
      rootToken,
    ],
  );

  final completer = Completer<AnalysisResult>();
  receivePort.listen((msg) {
    if (msg is int) {
      onStepChange(msg);
    } else if (msg is Map) {
      receivePort.close();
      if (msg.containsKey('error')) {
        completer.completeError(Exception(msg['error']));
      } else {
        completer.complete(AnalysisResult.fromIsolateMap(msg));
      }
    }
  });

  return completer.future;
}
