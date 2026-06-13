import 'package:flutter_test/flutter_test.dart';
import 'package:dermatology_ai_app/models/scan_result.dart';
import 'package:dermatology_ai_app/features/results/progression_utils.dart';

ScanResult _scan({
  required String id,
  required String bodyPart,
  required String diagnosis,
  required double confidence,
  required int day,
}) {
  return ScanResult(
    id: id,
    imagePath: '/tmp/$id.jpg',
    bodyPart: bodyPart,
    diagnosis: diagnosis,
    confidence: confidence,
    classProbabilities: const {},
    timestamp: DateTime(2026, 1, day),
  );
}

void main() {
  group('groupScansByBodyPart', () {
    test('groups by body part and sorts each group oldest → newest', () {
      final scans = [
        _scan(id: 'b', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.7, day: 3),
        _scan(id: 'a', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.5, day: 1),
        _scan(id: 'c', bodyPart: 'Face', diagnosis: 'Eczema', confidence: 0.9, day: 2),
      ];
      final groups = groupScansByBodyPart(scans);
      expect(groups.keys.toSet(), {'Arm', 'Face'});
      expect(groups['Arm']!.map((s) => s.id).toList(), ['a', 'b']);
      expect(groups['Face']!.length, 1);
    });

    test('blank body part buckets under Unspecified', () {
      final groups = groupScansByBodyPart([
        _scan(id: 'a', bodyPart: '', diagnosis: 'Acne', confidence: 0.6, day: 1),
      ]);
      expect(groups.keys, contains('Unspecified'));
    });
  });

  group('trackableBodyParts', () {
    test('orders by scan count descending', () {
      final groups = groupScansByBodyPart([
        _scan(id: '1', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.6, day: 1),
        _scan(id: '2', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.6, day: 2),
        _scan(id: '3', bodyPart: 'Leg', diagnosis: 'Tinea', confidence: 0.8, day: 1),
      ]);
      expect(trackableBodyParts(groups), ['Arm', 'Leg']);
    });

    test('minScans filters single-scan parts', () {
      final groups = groupScansByBodyPart([
        _scan(id: '1', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.6, day: 1),
        _scan(id: '2', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.6, day: 2),
        _scan(id: '3', bodyPart: 'Leg', diagnosis: 'Tinea', confidence: 0.8, day: 1),
      ]);
      expect(trackableBodyParts(groups, minScans: 2), ['Arm']);
    });
  });

  group('confidenceTrend', () {
    test('normal results contribute 0, diseases contribute confidence', () {
      final series = [
        _scan(id: '1', bodyPart: 'Arm', diagnosis: 'Acne', confidence: 0.8, day: 1),
        _scan(id: '2', bodyPart: 'Arm', diagnosis: 'No Disease Detected', confidence: 1.0, day: 2),
      ];
      expect(confidenceTrend(series), [0.8, 0.0]);
    });
  });
}
