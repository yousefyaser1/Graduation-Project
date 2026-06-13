import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../providers/user_provider.dart';

class SkinTypeScreen extends ConsumerStatefulWidget {
  const SkinTypeScreen({super.key});

  @override
  ConsumerState<SkinTypeScreen> createState() => _SkinTypeScreenState();
}

class _SkinTypeScreenState extends ConsumerState<SkinTypeScreen> {
  String _selectedSkinType = '';
  int _selectedToneIndex = 2;

  final _skinTypes = [
    {'key': 'oily', 'label': 'Oily', 'icon': Icons.water_drop_outlined},
    {'key': 'dry', 'label': 'Dry', 'icon': Icons.ac_unit_outlined},
    {'key': 'combination', 'label': 'Combination', 'icon': Icons.contrast},
    {
      'key': 'normal',
      'label': 'Normal',
      'icon': Icons.sentiment_satisfied_outlined
    },
  ];

  final _skinTones = [
    const Color(0xFFF5D5B0),
    const Color(0xFFEAB88A),
    const Color(0xFFD4945C),
    const Color(0xFFB87344),
    const Color(0xFF8B5E3C),
    const Color(0xFF4A2D1E),
  ];

  final _toneLabels = ['Fair', 'Light', 'Medium', 'Tan', 'Brown', 'Dark'];

  Future<void> _next() async {
    // Append skin type info to medical_history
    final user = ref.read(userProvider);
    final existing = user?.medicalHistory ?? '';
    // Preserve any existing medical data, prepend skin info
    final skinInfo =
        'Skin Type: ${_selectedSkinType.isNotEmpty ? _selectedSkinType : 'Not specified'} | Skin Tone: ${_toneLabels[_selectedToneIndex]}';
    final combined = existing.contains('Skin Type:')
        ? existing.replaceFirst(RegExp(r'Skin Type:[^|]*\|[^|]*\|?'), '$skinInfo | ')
        : '$skinInfo | $existing'.trim().replaceAll(RegExp(r'\s*\|\s*$'), '');

    await ref.read(userProvider.notifier).updateProfile(medicalHistory: combined);
    if (mounted) context.push(AppRoutes.medicalHistory);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () => context.pop(),
        ),
        title: RichText(
          text: const TextSpan(
            children: [
              TextSpan(
                text: 'SkinScan ',
                style: TextStyle(
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w700,
                    fontSize: 16),
              ),
              TextSpan(
                text: 'AI',
                style: TextStyle(
                    color: AppColors.primary,
                    fontWeight: FontWeight.w700,
                    fontSize: 16),
              ),
            ],
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.fromLTRB(24, 8, 24, 0),
        child: Column(
          children: [
            // Step progress
            Row(
              children: [
                const Text(
                  'Step 3 of 5',
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
                    child: const LinearProgressIndicator(
                      value: 3 / 5,
                      backgroundColor: AppColors.border,
                      color: AppColors.primary,
                      minHeight: 6,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 28),

            const Text(
              'Skin Type Selection',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 24),

            GridView.count(
              crossAxisCount: 2,
              shrinkWrap: true,
              crossAxisSpacing: 14,
              mainAxisSpacing: 14,
              childAspectRatio: 2.2,
              physics: const NeverScrollableScrollPhysics(),
              children: _skinTypes.map((type) {
                final isSelected = _selectedSkinType == type['key'];
                return GestureDetector(
                  onTap: () => setState(
                      () => _selectedSkinType = type['key'] as String),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color:
                            isSelected ? AppColors.primary : AppColors.border,
                        width: isSelected ? 2 : 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.04),
                          blurRadius: 6,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          type['icon'] as IconData,
                          size: 22,
                          color: isSelected
                              ? AppColors.primary
                              : AppColors.textSecondary,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          type['label'] as String,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: isSelected
                                ? AppColors.primary
                                : AppColors.textPrimary,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 28),

            // Skin tone selector
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: List.generate(_skinTones.length, (i) {
                final isSelected = _selectedToneIndex == i;
                return GestureDetector(
                  onTap: () => setState(() => _selectedToneIndex = i),
                  child: Column(
                    children: [
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 180),
                        width: isSelected ? 44 : 36,
                        height: isSelected ? 44 : 36,
                        decoration: BoxDecoration(
                          color: _skinTones[i],
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: isSelected
                                ? AppColors.primary
                                : Colors.transparent,
                            width: 3,
                          ),
                          boxShadow: isSelected
                              ? [
                                  BoxShadow(
                                    color: AppColors.primary.withValues(alpha: 0.35),
                                    blurRadius: 8,
                                    spreadRadius: 1,
                                  ),
                                ]
                              : [],
                        ),
                      ),
                      if (isSelected) ...[
                        const SizedBox(height: 6),
                        Text(
                          _toneLabels[i],
                          style: const TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: AppColors.primary),
                        ),
                      ],
                    ],
                  ),
                );
              }),
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
                    onPressed: _next,
                    child: const Text('Next'),
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
