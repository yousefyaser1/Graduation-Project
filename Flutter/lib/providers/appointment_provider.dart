import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/appointment.dart';
import '../services/database/database_service.dart';

/// All appointments, newest first (specialist view).
final allAppointmentsProvider = FutureProvider<List<Appointment>>((ref) async {
  final rows = await DatabaseService().getAllAppointments();
  return rows.map(Appointment.fromMap).toList();
});

/// Appointments booked by a specific patient, newest first.
final userAppointmentsProvider =
    FutureProvider.family<List<Appointment>, String>((ref, userId) async {
  final rows = await DatabaseService().getAppointmentsForUser(userId);
  return rows.map(Appointment.fromMap).toList();
});

/// Mutating helpers that refresh both providers after a write.
extension AppointmentActions on WidgetRef {
  Future<void> saveAppointment(Appointment appt) async {
    await DatabaseService().insertAppointment(appt.toMap());
    invalidate(allAppointmentsProvider);
    invalidate(userAppointmentsProvider);
  }

  Future<void> setAppointmentStatus(String id, String status) async {
    await DatabaseService().updateAppointmentStatus(id, status);
    invalidate(allAppointmentsProvider);
    invalidate(userAppointmentsProvider);
  }
}
