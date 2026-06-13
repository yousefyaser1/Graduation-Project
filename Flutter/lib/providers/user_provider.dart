import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import '../models/user.dart';
import '../services/database/database_service.dart';
import '../services/security/password_hasher.dart';
import '../services/session_service.dart';

class UserNotifier extends StateNotifier<User?> {
  UserNotifier() : super(null);

  /// Returns null on success, or an error message string on failure.
  Future<String?> login(String email, String password) async {
    if (email.trim().isEmpty || password.isEmpty) {
      return 'Please enter your email and password.';
    }
    final row = await DatabaseService().getUserByEmail(email.trim());
    if (row == null) return 'No account found with this email.';

    final stored = row['password'] as String? ?? '';
    if (PasswordHasher.isHashed(stored)) {
      if (!PasswordHasher.verify(password, stored)) {
        return 'Incorrect password.';
      }
    } else {
      // Legacy plaintext row (created before hashing was introduced).
      // Verify against plaintext, then transparently upgrade to a hash.
      if (stored != password) return 'Incorrect password.';
      final upgraded = Map<String, dynamic>.from(row)
        ..['password'] = PasswordHasher.hash(password);
      await DatabaseService().updateUser(upgraded);
    }

    state = User.fromMap(row);
    await SessionService().saveUserId(state!.id);
    return null;
  }

  /// Returns null on success, or an error message string on failure.
  Future<String?> signup({
    required String name,
    required String email,
    required String password,
    required String role,
  }) async {
    if (name.trim().isEmpty) return 'Please enter your full name.';
    if (email.trim().isEmpty) return 'Please enter your email.';
    if (password.length < 6) return 'Password must be at least 6 characters.';

    final existing = await DatabaseService().getUserByEmail(email.trim());
    if (existing != null) return 'An account with this email already exists.';

    final user = User(
      id: const Uuid().v4(),
      name: name.trim(),
      email: email.trim(),
      role: role,
      createdAt: DateTime.now(),
    );

    final row = {
      ...user.toMap(),
      'password': PasswordHasher.hash(password),
    };

    try {
      await DatabaseService().insertUser(row);
      state = user;
      await SessionService().saveUserId(user.id);
      return null;
    } catch (_) {
      return 'Failed to create account. Please try again.';
    }
  }

  void setUser(User user) => state = user;

  void updateUser(User updatedUser) {
    if (state?.id == updatedUser.id) state = updatedUser;
  }

  /// Update profile fields in DB and in-memory state.
  Future<void> updateProfile({
    String? name,
    String? gender,
    int? age,
    String? medicalHistory,
    String? role,
  }) async {
    if (state == null) return;
    final updated = state!.copyWith(
      name: name,
      gender: gender,
      age: age,
      medicalHistory: medicalHistory,
      role: role,
    );
    await DatabaseService().updateUser(updated.toMap());
    state = updated;
  }

  /// Look up email in DB and update password. Returns null on success or error message.
  Future<String?> resetPassword(String email, String newPassword) async {
    if (email.trim().isEmpty) return 'Please enter your email.';
    if (newPassword.length < 6) return 'Password must be at least 6 characters.';
    final row = await DatabaseService().getUserByEmail(email.trim());
    if (row == null) return 'No account found with this email.';
    final updated = Map<String, dynamic>.from(row);
    updated['password'] = PasswordHasher.hash(newPassword);
    await DatabaseService().updateUser(updated);
    return null;
  }

  /// Restore session from SharedPreferences. Returns true if a session was found.
  Future<bool> restoreSession() async {
    final userId = await SessionService().getSavedUserId();
    if (userId == null) return false;
    final row = await DatabaseService().getUserById(userId);
    if (row == null) {
      await SessionService().clearSession();
      return false;
    }
    state = User.fromMap(row);
    return true;
  }

  void clearUser() {
    SessionService().clearSession();
    state = null;
  }

  bool get isLoggedIn => state != null;
}

final userProvider = StateNotifierProvider<UserNotifier, User?>((ref) {
  return UserNotifier();
});

final userRoleProvider = StateProvider<String?>((ref) => null);
