import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/routing/app_router.dart';
import '../../providers/user_provider.dart';
import '../../providers/scan_provider.dart';
import '../../services/session_service.dart';

class SplashScreen extends ConsumerStatefulWidget {
  const SplashScreen({super.key});

  @override
  ConsumerState<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends ConsumerState<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _fadeAnim;
  late final Animation<double> _scaleAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    _fadeAnim = CurvedAnimation(parent: _controller, curve: Curves.easeIn);
    _scaleAnim = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
    );
    _controller.forward();

    _initSession();
  }

  Future<void> _initSession() async {
    // Reclaim heatmap/image files left over from previously-deleted scans.
    // Fire-and-forget so it never delays navigation.
    unawaited(cleanOrphanScanFiles());

    // Minimum splash duration: long enough for the logo animation to land,
    // short enough not to feel sluggish on launch.
    await Future.delayed(const Duration(milliseconds: 2200));
    if (!mounted) return;

    final restored = await ref.read(userProvider.notifier).restoreSession();

    if (!mounted) return;

    if (restored) {
      final userId = ref.read(userProvider)!.id;
      final onboardingDone =
          await SessionService().isOnboardingComplete(userId);
      if (!mounted) return;
      if (onboardingDone) {
        context.go(AppRoutes.home);
      } else {
        context.go(AppRoutes.roleSelection);
      }
    } else {
      context.go(AppRoutes.welcome);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: FadeTransition(
          opacity: _fadeAnim,
          child: ScaleTransition(
            scale: _scaleAnim,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Image.asset(
                  'assets/images/logo.png',
                  width: 200,
                  height: 200,
                  fit: BoxFit.contain,
                ),
                const SizedBox(height: 8),
                const Text(
                  'SkinScan AI',
                  style: TextStyle(
                    color: Color(0xFF1E3A8A),
                    fontSize: 32,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'AI-Powered Skin Analysis',
                  style: TextStyle(
                    color: const Color(0xFF3B82F6).withValues(alpha: 0.85),
                    fontSize: 15,
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const SizedBox(height: 56),
                const SizedBox(
                  width: 28,
                  height: 28,
                  child: CircularProgressIndicator(
                    strokeWidth: 2.5,
                    color: Color(0xFF3B82F6),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
