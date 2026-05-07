import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../providers/user_provider.dart';

class RoleSelectionScreen extends ConsumerStatefulWidget {
  const RoleSelectionScreen({super.key});

  @override
  ConsumerState<RoleSelectionScreen> createState() =>
      _RoleSelectionScreenState();
}

class _RoleSelectionScreenState extends ConsumerState<RoleSelectionScreen> {
  String _selectedRole = 'patient';

  Future<void> _continue() async {
    await ref.read(userProvider.notifier).updateProfile(role: _selectedRole);
    if (!mounted) return;
    if (_selectedRole == 'specialist') {
      context.go(AppRoutes.specialistDashboard);
    } else {
      context.push(AppRoutes.personalInfo);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () => context.pop(),
        ),
        title: RichText(
          text: TextSpan(
            children: [
              const TextSpan(
                text: 'SkinScan ',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
              TextSpan(
                text: 'AI',
                style: TextStyle(
                  color: AppColors.primary,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
              const TextSpan(
                text: '  |  Role Selection',
                style: TextStyle(
                  color: AppColors.textSecondary,
                  fontWeight: FontWeight.w400,
                  fontSize: 14,
                ),
              ),
            ],
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const Text(
              'Who are you?',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Text(
                  'Step 1 of 5',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: 1 / 5,
                      backgroundColor: AppColors.border,
                      color: AppColors.primary,
                      minHeight: 6,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            Row(
              children: [
                Expanded(
                  child: _RoleCard(
                    icon: Icons.smartphone_outlined,
                    iconBg: const Color(0xFFDBEAFE),
                    iconColor: const Color(0xFF3B82F6),
                    title: 'Patient',
                    description:
                        'Track your skin health and consult with experts',
                    isSelected: _selectedRole == 'patient',
                    onTap: () => setState(() => _selectedRole = 'patient'),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: _RoleCard(
                    icon: Icons.medical_information_outlined,
                    iconBg: const Color(0xFFF3F4F6),
                    iconColor: const Color(0xFF6B7280),
                    title: 'Specialist',
                    description:
                        'Access patient data and provide professional advice.',
                    isSelected: _selectedRole == 'specialist',
                    onTap: () => setState(() => _selectedRole = 'specialist'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: const Color(0xFFFFFBEB),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFFCD34D)),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.warning_amber_rounded,
                      color: Color(0xFFF59E0B), size: 18),
                  const SizedBox(width: 10),
                  Expanded(
                    child: RichText(
                      text: const TextSpan(
                        style: TextStyle(
                          fontSize: 12,
                          color: Color(0xFF92400E),
                          height: 1.5,
                        ),
                        children: [
                          TextSpan(
                            text: 'Disclaimer: ',
                            style: TextStyle(fontWeight: FontWeight.w700),
                          ),
                          TextSpan(
                            text:
                                'This app is for informational purposes only and not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.',
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const Spacer(),

            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => context.pop(),
                    style: OutlinedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 52),
                      side: const BorderSide(color: AppColors.border),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    child: const Text('Previous',
                        style: TextStyle(color: AppColors.textSecondary)),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _continue,
                    child: const Text('Continue'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 28),
          ],
        ),
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  final IconData icon;
  final Color iconBg;
  final Color iconColor;
  final String title;
  final String description;
  final bool isSelected;
  final VoidCallback onTap;

  const _RoleCard({
    required this.icon,
    required this.iconBg,
    required this.iconColor,
    required this.title,
    required this.description,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? AppColors.primary : AppColors.border,
            width: isSelected ? 2 : 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color: isSelected ? AppColors.primaryLight : iconBg,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon,
                  color: isSelected ? AppColors.primary : iconColor, size: 26),
            ),
            const SizedBox(height: 12),
            Text(
              title,
              style: TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: isSelected ? AppColors.primary : AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              description,
              style: const TextStyle(
                fontSize: 12,
                color: AppColors.textSecondary,
                height: 1.45,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
