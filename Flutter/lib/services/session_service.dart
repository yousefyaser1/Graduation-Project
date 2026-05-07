import 'package:shared_preferences/shared_preferences.dart';

class SessionService {
  static const _keyUserId = 'session_user_id';
  static const _onboardingPrefix = 'onboarding_complete_';

  Future<void> saveUserId(String id) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyUserId, id);
  }

  Future<String?> getSavedUserId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyUserId);
  }

  Future<void> clearSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyUserId);
  }

  Future<void> markOnboardingComplete(String userId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('$_onboardingPrefix$userId', true);
  }

  Future<bool> isOnboardingComplete(String userId) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool('$_onboardingPrefix$userId') ?? false;
  }
}
