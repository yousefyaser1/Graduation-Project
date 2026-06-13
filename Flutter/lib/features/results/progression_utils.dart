import '../../models/scan_result.dart';

/// Pure helpers for the lesion-progression feature. Kept free of Flutter
/// imports so they can be unit-tested without a widget harness.

/// Group [scans] by their `bodyPart`, with each group sorted oldest → newest
/// so a timeline reads top-to-bottom in chronological order.
///
/// Scans with a blank body part are bucketed under "Unspecified". The returned
/// map preserves insertion order of first appearance.
Map<String, List<ScanResult>> groupScansByBodyPart(List<ScanResult> scans) {
  final groups = <String, List<ScanResult>>{};
  for (final s in scans) {
    final key = s.bodyPart.trim().isEmpty ? 'Unspecified' : s.bodyPart.trim();
    groups.putIfAbsent(key, () => <ScanResult>[]).add(s);
  }
  for (final list in groups.values) {
    list.sort((a, b) => a.timestamp.compareTo(b.timestamp));
  }
  return groups;
}

/// Body parts that have at least [minScans] scans, ordered by scan count
/// (descending) then alphabetically — these are the parts worth tracking
/// over time.
List<String> trackableBodyParts(
  Map<String, List<ScanResult>> groups, {
  int minScans = 1,
}) {
  final keys = groups.entries
      .where((e) => e.value.length >= minScans)
      .map((e) => e.key)
      .toList();
  keys.sort((a, b) {
    final byCount = groups[b]!.length.compareTo(groups[a]!.length);
    return byCount != 0 ? byCount : a.compareTo(b);
  });
  return keys;
}

/// Confidence series (0–1) for a part's scans in chronological order.
/// Normal ("No Disease Detected") results contribute 0 so the trend reflects
/// disease likelihood rather than the model's certainty that skin is healthy.
List<double> confidenceTrend(List<ScanResult> chronological) {
  return [
    for (final s in chronological)
      s.diagnosis == 'No Disease Detected' ? 0.0 : s.confidence,
  ];
}
