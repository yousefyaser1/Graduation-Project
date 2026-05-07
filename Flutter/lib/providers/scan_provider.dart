import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/scan_result.dart';
import '../services/database/database_service.dart';

/// Live list of all scans from the local database, newest first.
final scanListProvider = FutureProvider<List<ScanResult>>((ref) async {
  final rows = await DatabaseService().getAllScans();
  return rows.map(ScanResult.fromMap).toList();
});

/// Call these helpers after any DB mutation to refresh [scanListProvider].
extension ScanActions on WidgetRef {
  Future<void> saveScan(ScanResult scan) async {
    await DatabaseService().insertScan(scan.toMap());
    invalidate(scanListProvider);
  }

  Future<void> deleteScan(String id) async {
    await DatabaseService().deleteScan(id);
    invalidate(scanListProvider);
  }
}

/// Holds the scan currently being viewed (set before navigating to results).
final currentScanProvider = StateProvider<ScanResult?>((ref) => null);
