import 'package:equatable/equatable.dart';

class User extends Equatable {
  final String id;
  final String name;
  final String email;
  final String role; // 'patient' | 'specialist'
  final int? age;
  final String? gender;
  final String? medicalHistory;
  final DateTime createdAt;

  const User({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
    this.age,
    this.gender,
    this.medicalHistory,
    required this.createdAt,
  });

  User copyWith({
    String? id,
    String? name,
    String? email,
    String? role,
    int? age,
    String? gender,
    String? medicalHistory,
    DateTime? createdAt,
  }) {
    return User(
      id: id ?? this.id,
      name: name ?? this.name,
      email: email ?? this.email,
      role: role ?? this.role,
      age: age ?? this.age,
      gender: gender ?? this.gender,
      medicalHistory: medicalHistory ?? this.medicalHistory,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'email': email,
      'role': role,
      'age': age,
      'gender': gender,
      'medical_history': medicalHistory,
      'created_at': createdAt.millisecondsSinceEpoch,
    };
  }

  factory User.fromMap(Map<String, dynamic> map) {
    return User(
      id: map['id'] as String,
      name: map['name'] as String,
      email: map['email'] as String? ?? '',
      role: map['role'] as String? ?? 'patient',
      age: map['age'] as int?,
      gender: map['gender'] as String?,
      medicalHistory: map['medical_history'] as String?,
      createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at'] as int),
    );
  }

  @override
  List<Object?> get props =>
      [id, name, email, role, age, gender, medicalHistory, createdAt];
}
