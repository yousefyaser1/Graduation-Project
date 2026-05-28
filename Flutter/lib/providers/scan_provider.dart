import 'dart:io';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import '../models/scan_result.dart';
import '../services/database/database_service.dart';

/// Live list of all scans from the local database, newest first.
final scanListProvider = FutureProvider<List<ScanResult>>((ref) async {
  final rows = await DatabaseService().getAllScans();
  return rows.map(ScanResult.fromMap).toList();
});

/// Copy the scan's image + heatmaps out of the OS cache/temp dirs (where
/// image_picker and the AI pipeline write them) into the app's documents
/// directory, so History can still load them after a restart. Returns a
/// ScanResult whose paths point at the durable copies.
Future<ScanResult> _persistScanFiles(ScanResult scan) async {
  final docs = await getApplicationDocumentsDirectory();
  final dir = Directory(p.join(docs.path, 'scans'));
  if (!await dir.exists()) await dir.create(recursive: true);

  Future<String?> copyIfNeeded(String? src, String suffix) async {
    if (src == null) return null;
    if (p.isWithin(dir.path, src)) return src; // already durable
    final f = File(src);
    if (!await f.exists()) return src; // source already gone — keep path as-is
    final dest = p.join(dir.path, '${scan.id}_$suffix${p.extension(src)}');
    await f.copy(dest);
    return dest;
  }

  final newClassPaths = <String, String>{};
  for (final entry in scan.classHeatmapPaths.entries) {
    final copied = await copyIfNeeded(entry.value, 'scorecam_${entry.key}');
    if (copied != null) newClassPaths[entry.key] = copied;
  }

  return scan.copyWith(
    imagePath: await copyIfNeeded(scan.imagePath, 'original') ?? scan.imagePath,
    heatmapPath: await copyIfNeeded(scan.heatmapPath, 'scorecam'),
    classHeatmapPaths: newClassPaths,
    vaeHeatmapPath: await copyIfNeeded(scan.vaeHeatmapPath, 'vae'),
  );
}

/// Delete durable scan files whose `<id>_*` prefix has no matching row in the
/// DB. Run once at startup so swipe-to-delete stays instantly reversible (the
/// 4s Undo re-inserts the row) while orphaned files are still reclaimed later.
/// Best-effort — never throws.
Future<void> cleanOrphanScanFiles() async {
  try {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(docs.path, 'scans'));
    if (!await dir.exists()) return;

    final liveIds = (await DatabaseService().getAllScans())
        .map((row) => row['id'] as String)
        .toSet();

    await for (final entity in dir.list()) {
      if (entity is! File) continue;
      final name = p.basename(entity.path);
      final sep = name.indexOf('_');
      if (sep <= 0) continue;
      final id = name.substring(0, sep);
      if (!liveIds.contains(id)) {
        try {
          await entity.delete();
        } catch (_) {/* file already gone */}
      }
    }
  } catch (_) {/* documents dir / DB unavailable — nothing to clean */}
}

/// Call these helpers after any DB mutation to refresh [scanListProvider].
extension ScanActions on WidgetRef {
  Future<void> saveScan(ScanResult scan) async {
    final durable = await _persistScanFiles(scan);
    await DatabaseService().insertScan(durable.toMap());
    invalidate(scanListProvider);
  }

  Future<void> deleteScan(String id) async {
    // Files are intentionally kept here so the History "Undo" can restore the
    // scan. Orphans are reclaimed by [cleanOrphanScanFiles] on next launch.
    await DatabaseService().deleteScan(id);
    invalidate(scanListProvider);
  }
}

/// Holds the scan currently being viewed (set before navigating to results).
final currentScanProvider = StateProvider<ScanResult?>((ref) => null);
