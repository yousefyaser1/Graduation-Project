import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';

class HowItWorksScreen extends StatelessWidget {
  const HowItWorksScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 48),
              const Center(
                child: Text(
                  'How it works',
                  style: TextStyle(
                    fontSize: 26,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
              const SizedBox(height: 40),

              const Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _StepCard(
                        stepNumber: 1,
                        iconData: Icons.camera_alt_outlined,
                        iconColor: Color(0xFF60A5FA),
                        iconBg: Color(0xFFDBEAFE),
                        title: 'Capture a photo',
                        description:
                            'Take a clear, close-up photo of your skin area in natural light.',
                      ),
                      SizedBox(height: 16),

                      _StepCard(
                        stepNumber: 2,
                        iconData: Icons.search_rounded,
                        iconColor: Color(0xFF34D399),
                        iconBg: Color(0xFFD1FAE5),
                        title: 'Instant Analysis',
                        description:
                            'Our AI technology scans and analyzes your skin condition in seconds.',
                      ),
                      SizedBox(height: 16),

                      _StepCard(
                        stepNumber: 3,
                        iconData: Icons.description_outlined,
                        iconColor: Color(0xFFA78BFA),
                        iconBg: Color(0xFFEDE9FE),
                        title: 'Get your results',
                        description:
                            'Receive a personalized report with insights and recommendations.',
                      ),
                      SizedBox(height: 16),

                      _StepCard(
                        stepNumber: 4,
                        iconData: Icons.lightbulb_outline,
                        iconColor: Color(0xFFF59E0B),
                        iconBg: Color(0xFFFEF3C7),
                        title: 'See why (Explainable AI)',
                        description:
                            'Every result comes with a Score-CAM heatmap showing exactly which areas the AI focused on — no black box.',
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => context.push(AppRoutes.login),
                  child: const Text('Continue'),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}

class _StepCard extends StatelessWidget {
  final int stepNumber;
  final IconData iconData;
  final Color iconColor;
  final Color iconBg;
  final String title;
  final String description;

  const _StepCard({
    required this.stepNumber,
    required this.iconData,
    required this.iconColor,
    required this.iconBg,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: iconBg,
              shape: BoxShape.circle,
            ),
            child: Icon(iconData, color: iconColor, size: 26),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '$stepNumber. $title',
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  description,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textSecondary,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
