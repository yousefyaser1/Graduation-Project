import 'package:uuid/uuid.dart';

class DemoUsers {
  static final String patientId = const Uuid().v4();
  static final String specialistId = const Uuid().v4();

  static final Map<String, dynamic> patientUser = {
    'id': patientId,
    'name': 'patient',
    'email': 'patient@example.com',
    'password': '123',
    'role': 'patient',
    'age': 28,
    'gender': 'Female',
    'medical_history': 'Sensitive skin, occasional eczema',
    'created_at': DateTime.now().millisecondsSinceEpoch,
  };

  static final Map<String, dynamic> specialistUser = {
    'id': specialistId,
    'name': 'doctor',
    'email': 'doctor@example.com',
    'password': '123',
    'role': 'specialist',
    'age': 45,
    'gender': 'Male',
    'medical_history': null,
    'created_at': DateTime.now().millisecondsSinceEpoch,
  };

  static List<Map<String, dynamic>> getAllDemoUsers() =>
      [patientUser, specialistUser];
}
