import 'package:equatable/equatable.dart';

/// A booked appointment between a patient and a specialist.
///
/// Stored locally in the `appointments` table. [scanId] optionally links the
/// appointment to the scan that prompted it, so the specialist can open the
/// exact result from their schedule.
class Appointment extends Equatable {
  final String id;
  final String userId; // patient who booked
  final String patientName;
  final String doctorName;
  final String specialty;
  final String dateLabel; // e.g. "Tomorrow" / "Nov 2"
  final String timeLabel; // e.g. "10:30 AM"
  final String status; // 'Scheduled' | 'Completed' | 'Cancelled'
  final String? scanId;
  final String? diagnosis; // snapshot of the linked scan's diagnosis
  final DateTime createdAt;

  const Appointment({
    required this.id,
    required this.userId,
    required this.patientName,
    required this.doctorName,
    required this.specialty,
    required this.dateLabel,
    required this.timeLabel,
    this.status = 'Scheduled',
    this.scanId,
    this.diagnosis,
    required this.createdAt,
  });

  Appointment copyWith({String? status}) => Appointment(
        id: id,
        userId: userId,
        patientName: patientName,
        doctorName: doctorName,
        specialty: specialty,
        dateLabel: dateLabel,
        timeLabel: timeLabel,
        status: status ?? this.status,
        scanId: scanId,
        diagnosis: diagnosis,
        createdAt: createdAt,
      );

  Map<String, dynamic> toMap() => {
        'id': id,
        'user_id': userId,
        'patient_name': patientName,
        'doctor_name': doctorName,
        'specialty': specialty,
        'date_label': dateLabel,
        'time_label': timeLabel,
        'status': status,
        'scan_id': scanId,
        'diagnosis': diagnosis,
        'created_at': createdAt.millisecondsSinceEpoch,
      };

  factory Appointment.fromMap(Map<String, dynamic> map) => Appointment(
        id: map['id'] as String,
        userId: map['user_id'] as String? ?? '',
        patientName: map['patient_name'] as String? ?? 'Patient',
        doctorName: map['doctor_name'] as String? ?? '',
        specialty: map['specialty'] as String? ?? '',
        dateLabel: map['date_label'] as String? ?? '',
        timeLabel: map['time_label'] as String? ?? '',
        status: map['status'] as String? ?? 'Scheduled',
        scanId: map['scan_id'] as String?,
        diagnosis: map['diagnosis'] as String?,
        createdAt:
            DateTime.fromMillisecondsSinceEpoch(map['created_at'] as int),
      );

  @override
  List<Object?> get props => [
        id,
        userId,
        patientName,
        doctorName,
        specialty,
        dateLabel,
        timeLabel,
        status,
        scanId,
        diagnosis,
        createdAt,
      ];
}
