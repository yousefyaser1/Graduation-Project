import 'package:flutter_test/flutter_test.dart';
import 'package:dermatology_ai_app/models/appointment.dart';

void main() {
  group('Appointment', () {
    final appt = Appointment(
      id: 'a1',
      userId: 'u1',
      patientName: 'Jane Doe',
      doctorName: 'Dr. Sarah Johnson',
      specialty: 'Dermatologist',
      dateLabel: 'Tomorrow',
      timeLabel: '10:30 AM',
      scanId: 'scan-9',
      diagnosis: 'Eczema',
      createdAt: DateTime.fromMillisecondsSinceEpoch(1700000000000),
    );

    test('round-trips through toMap/fromMap', () {
      final restored = Appointment.fromMap(appt.toMap());
      expect(restored, equals(appt));
    });

    test('defaults status to Scheduled', () {
      expect(appt.status, 'Scheduled');
    });

    test('copyWith changes only status', () {
      final cancelled = appt.copyWith(status: 'Cancelled');
      expect(cancelled.status, 'Cancelled');
      expect(cancelled.id, appt.id);
      expect(cancelled.doctorName, appt.doctorName);
    });

    test('fromMap tolerates missing optional fields', () {
      final restored = Appointment.fromMap(const {
        'id': 'x',
        'created_at': 0,
      });
      expect(restored.id, 'x');
      expect(restored.status, 'Scheduled');
      expect(restored.scanId, isNull);
      expect(restored.patientName, 'Patient');
    });
  });
}
