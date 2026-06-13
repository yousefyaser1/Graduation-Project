// Unit tests for the User model. (Replaces the default counter smoke test,
// which never matched this app and always failed.)

import 'package:flutter_test/flutter_test.dart';
import 'package:dermatology_ai_app/models/user.dart';

void main() {
  group('User', () {
    test('round-trips through toMap/fromMap', () {
      final user = User(
        id: 'u1',
        name: 'Jane Doe',
        email: 'jane@example.com',
        role: 'patient',
        age: 30,
        gender: 'Female',
        medicalHistory: 'Skin Type: oily | Skin Tone: Light',
        createdAt: DateTime.fromMillisecondsSinceEpoch(1700000000000),
      );
      expect(User.fromMap(user.toMap()), equals(user));
    });

    test('fromMap applies sensible defaults for missing fields', () {
      final user = User.fromMap(const {
        'id': 'u2',
        'name': 'No Role',
        'created_at': 0,
      });
      expect(user.role, 'patient');
      expect(user.email, '');
      expect(user.age, isNull);
    });
  });
}
