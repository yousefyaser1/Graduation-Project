import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../features/splash/splash_screen.dart';
import '../../features/onboarding/screens/welcome_screen.dart';
import '../../features/onboarding/screens/how_it_works_screen.dart';
import '../../features/onboarding/screens/login_screen.dart';
import '../../features/onboarding/screens/signup_screen.dart';
import '../../features/onboarding/screens/forgot_password_screen.dart';
import '../../features/onboarding/screens/role_selection_screen.dart';
import '../../features/onboarding/screens/personal_info_screen.dart';
import '../../features/onboarding/screens/skin_type_screen.dart';
import '../../features/onboarding/screens/medical_history_screen.dart';
import '../../features/onboarding/screens/all_set_screen.dart';
import '../../features/core_workflow/screens/home_screen.dart';
import '../../features/core_workflow/screens/body_part_selection_screen.dart';
import '../../features/core_workflow/screens/capture_screen.dart';
import '../../features/core_workflow/screens/analyzing_screen.dart';
import '../../features/core_workflow/screens/history_screen.dart';
import '../../features/core_workflow/screens/profile_screen.dart';
import '../../features/core_workflow/screens/notifications_screen.dart';
import '../../features/core_workflow/screens/edit_profile_screen.dart';
import '../../features/core_workflow/screens/help_support_screen.dart';
import '../../features/core_workflow/screens/specialist_dashboard_screen.dart';
import '../../features/results/screens/analysis_results_screen.dart';
import '../../features/results/screens/book_appointment_screen.dart';
import '../../models/scan_result.dart';

class AppRoutes {
  // Launch
  static const String splash         = '/';
  // Onboarding
  static const String welcome        = '/welcome';
  static const String howItWorks     = '/how-it-works';
  static const String login          = '/login';
  static const String signup         = '/signup';
  static const String forgotPassword = '/forgot-password';
  static const String roleSelection  = '/role-selection';
  static const String personalInfo   = '/personal-info';
  static const String skinType       = '/skin-type';
  static const String medicalHistory = '/medical-history';
  static const String allSet         = '/all-set';
  // Main app
  static const String home                = '/home';
  static const String bodyPartSelection   = '/body-part-selection';
  static const String capture             = '/capture';
  static const String analyzing           = '/analyzing';
  static const String analysisResults     = '/analysis-results';
  static const String bookAppointment     = '/book-appointment';
  static const String history             = '/history';
  static const String profile             = '/profile';
  static const String notifications       = '/notifications';
  static const String editProfile         = '/edit-profile';
  static const String helpSupport         = '/help-support';
  static const String specialistDashboard = '/specialist-dashboard';
}

final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: AppRoutes.splash,
    routes: [
      GoRoute(path: AppRoutes.splash,
          builder: (c, s) => const SplashScreen()),
      GoRoute(path: AppRoutes.welcome,
          builder: (c, s) => const WelcomeScreen()),
      GoRoute(path: AppRoutes.howItWorks,
          builder: (c, s) => const HowItWorksScreen()),
      GoRoute(path: AppRoutes.login,
          builder: (c, s) => const LoginScreen()),
      GoRoute(path: AppRoutes.signup,
          builder: (c, s) => const SignUpScreen()),
      GoRoute(path: AppRoutes.forgotPassword,
          builder: (c, s) => const ForgotPasswordScreen()),
      GoRoute(path: AppRoutes.roleSelection,
          builder: (c, s) => const RoleSelectionScreen()),
      GoRoute(path: AppRoutes.personalInfo,
          builder: (c, s) => const PersonalInfoScreen()),
      GoRoute(path: AppRoutes.skinType,
          builder: (c, s) => const SkinTypeScreen()),
      GoRoute(path: AppRoutes.medicalHistory,
          builder: (c, s) => const MedicalHistoryScreen()),
      GoRoute(path: AppRoutes.allSet,
          builder: (c, s) => const AllSetScreen()),
      GoRoute(path: AppRoutes.home,
          builder: (c, s) => const HomeScreen()),
      GoRoute(path: AppRoutes.bodyPartSelection,
          builder: (c, s) => const BodyPartSelectionScreen()),

      // Capture receives the selected body part as a String extra.
      GoRoute(
        path: AppRoutes.capture,
        builder: (c, s) {
          final bodyPart = s.extra as String? ?? '';
          return CaptureScreen(bodyPart: bodyPart);
        },
      ),

      // Analyzing receives Map<String, dynamic> {bodyPart, imagePath}.
      GoRoute(
        path: AppRoutes.analyzing,
        builder: (c, s) {
          final extra = s.extra as Map<String, dynamic>? ?? {};
          return AnalyzingScreen(
            bodyPart: extra['bodyPart'] as String? ?? '',
            imagePath: extra['imagePath'] as String? ?? '',
          );
        },
      ),

      // AnalysisResults receives a ScanResult extra (from scan flow)
      // or a ScanResult passed from history / home.
      GoRoute(
        path: AppRoutes.analysisResults,
        builder: (c, s) {
          final scan = s.extra as ScanResult?;
          return AnalysisResultsScreen(scan: scan);
        },
      ),

      GoRoute(
        path: AppRoutes.bookAppointment,
        builder: (c, s) {
          final scan = s.extra as ScanResult?;
          return BookAppointmentScreen(scan: scan);
        },
      ),
      GoRoute(path: AppRoutes.history,
          builder: (c, s) => const HistoryScreen()),
      GoRoute(path: AppRoutes.profile,
          builder: (c, s) => const ProfileScreen()),
      GoRoute(path: AppRoutes.notifications,
          builder: (c, s) => const NotificationsScreen()),
      GoRoute(path: AppRoutes.editProfile,
          builder: (c, s) => const EditProfileScreen()),
      GoRoute(path: AppRoutes.helpSupport,
          builder: (c, s) => const HelpSupportScreen()),
      GoRoute(path: AppRoutes.specialistDashboard,
          builder: (c, s) => const SpecialistDashboardScreen()),
    ],
    errorBuilder: (context, state) => Scaffold(
      body: Center(child: Text('Page not found: ${state.uri}')),
    ),
  );
});
