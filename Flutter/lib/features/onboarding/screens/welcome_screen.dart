import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            // Illustration area
            Expanded(
              flex: 5,
              child: Container(
                width: double.infinity,
                color: const Color(0xFFEEF4FF),
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    Positioned(
                      top: -50,
                      right: -50,
                      child: Container(
                        width: 220,
                        height: 220,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: AppColors.primary.withOpacity(0.07),
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: -30,
                      left: -40,
                      child: Container(
                        width: 180,
                        height: 180,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: AppColors.primary.withOpacity(0.05),
                        ),
                      ),
                    ),
                    Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            _PersonFigure(
                              skinColor: const Color(0xFFE8B4A0),
                              shirtColor: const Color(0xFFD4845C),
                              size: 72,
                              withHijab: true,
                            ),
                            const SizedBox(width: 6),
                            _PersonFigure(
                              skinColor: const Color(0xFF8B6552),
                              shirtColor: const Color(0xFF5B8BD0),
                              size: 88,
                            ),
                            const SizedBox(width: 6),
                            _PersonFigure(
                              skinColor: const Color(0xFFF5D5B8),
                              shirtColor: const Color(0xFFE8A0A0),
                              size: 72,
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: const [
                            _TreeWidget(size: 48),
                            _TreeWidget(size: 64),
                            _TreeWidget(size: 48),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            // Content area
            Expanded(
              flex: 4,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(28, 28, 28, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    // Badge
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
                      decoration: BoxDecoration(
                        color: AppColors.primaryLight,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.search_rounded, size: 15, color: AppColors.primary),
                          const SizedBox(width: 5),
                          Text(
                            'SkinScan AI',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              color: AppColors.primary,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 18),

                    const Text(
                      'Your Skin,\nUnderstand It Better',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary,
                        height: 1.25,
                      ),
                    ),
                    const SizedBox(height: 10),

                    const Text(
                      'Receive personalized analysis and build a\nroutine you love.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 14,
                        color: AppColors.textSecondary,
                        height: 1.55,
                      ),
                    ),
                    const Spacer(),

                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: () => context.push(AppRoutes.howItWorks),
                        child: const Text('Get Started'),
                      ),
                    ),
                    const SizedBox(height: 14),

                    TextButton(
                      onPressed: () => context.push(AppRoutes.login),
                      child: Text(
                        'Already have an account?',
                        style: TextStyle(
                          fontSize: 14,
                          color: AppColors.primary,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PersonFigure extends StatelessWidget {
  final Color skinColor;
  final Color shirtColor;
  final double size;
  final bool withHijab;

  const _PersonFigure({
    required this.skinColor,
    required this.shirtColor,
    required this.size,
    this.withHijab = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: size * 0.42,
              height: size * 0.42,
              decoration: BoxDecoration(
                color: skinColor,
                shape: BoxShape.circle,
              ),
            ),
            if (withHijab)
              Container(
                width: size * 0.44,
                height: size * 0.28,
                decoration: BoxDecoration(
                  color: const Color(0xFF7BA3D8),
                  borderRadius: BorderRadius.circular(size * 0.22),
                ),
              ),
          ],
        ),
        Container(
          width: size * 0.5,
          height: size * 0.55,
          decoration: BoxDecoration(
            color: shirtColor,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(10)),
          ),
        ),
      ],
    );
  }
}

class _TreeWidget extends StatelessWidget {
  final double size;
  const _TreeWidget({required this.size});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: size * 0.75,
          height: size * 0.75,
          decoration: const BoxDecoration(
            color: Color(0xFF86EFAC),
            shape: BoxShape.circle,
          ),
        ),
        Container(
          width: 7,
          height: size * 0.28,
          decoration: BoxDecoration(
            color: const Color(0xFF92400E),
            borderRadius: BorderRadius.circular(3),
          ),
        ),
      ],
    );
  }
}
