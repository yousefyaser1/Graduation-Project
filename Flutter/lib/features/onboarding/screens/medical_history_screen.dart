import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../providers/user_provider.dart';

class MedicalHistoryScreen extends ConsumerStatefulWidget {
  const MedicalHistoryScreen({super.key});

  @override
  ConsumerState<MedicalHistoryScreen> createState() =>
      _MedicalHistoryScreenState();
}

class _MedicalHistoryScreenState extends ConsumerState<MedicalHistoryScreen> {
  final Set<String> _selectedConditions = {};
  final Set<String> _selectedAllergies = {};
  final _otherController = TextEditingController();

  final _conditions = [
    'Eczema',
    'Psoriasis',
    'Acne',
    'Rosacea',
    'Dermatitis',
    'Melanoma history',
    'Diabetes',
    'Autoimmune disorder',
  ];

  final _allergies = [
    'Latex',
    'Nickel',
    'Fragrances',
    'Sunscreen',
    'Antibiotics',
    'Preservatives',
  ];

  @override
  void dispose() {
    _otherController.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    // Read existing skin type info (saved by previous screen)
    final user = ref.read(userProvider);
    final existing = user?.medicalHistory ?? '';
    // Extract skin info prefix (before any '|' related to conditions)
    final skinPrefix = existing.contains('Skin Type:')
        ? existing.split('|').take(2).join('|').trim()
        : '';

    final condStr = _selectedConditions.isNotEmpty
        ? 'Conditions: ${_selectedConditions.join(', ')}'
        : '';
    final allergyStr = _selectedAllergies.isNotEmpty
        ? 'Allergies: ${_selectedAllergies.join(', ')}'
        : '';
    final noteStr = _otherController.text.trim().isNotEmpty
        ? 'Notes: ${_otherController.text.trim()}'
        : '';

    final parts = [skinPrefix, condStr, allergyStr, noteStr]
        .where((s) => s.isNotEmpty)
        .toList();
    final combined = parts.join(' | ');

    await ref
        .read(userProvider.notifier)
        .updateProfile(medicalHistory: combined);
    if (mounted) context.push(AppRoutes.allSet);
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
          text: TextSpan(
            children: [
              const TextSpan(
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
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Step progress
                  Row(
                    children: [
                      const Text(
                        'Step 4 of 5',
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
                            value: 4 / 5,
                            backgroundColor: AppColors.border,
                            color: AppColors.primary,
                            minHeight: 6,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),

                  const Text(
                    'Medical History',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'This helps us personalize your analysis. Select all that apply.',
                    style: TextStyle(
                        fontSize: 13,
                        color: AppColors.textSecondary,
                        height: 1.5),
                  ),
                  const SizedBox(height: 24),

                  _buildSectionTitle('Skin conditions (if any)'),
                  const SizedBox(height: 12),
                  _buildChipGroup(_conditions, _selectedConditions),
                  const SizedBox(height: 24),

                  _buildSectionTitle('Known allergies'),
                  const SizedBox(height: 12),
                  _buildChipGroup(_allergies, _selectedAllergies),
                  const SizedBox(height: 24),

                  _buildSectionTitle('Other notes (optional)'),
                  const SizedBox(height: 10),
                  TextField(
                    controller: _otherController,
                    maxLines: 3,
                    decoration: InputDecoration(
                      hintText:
                          'Any medications, recent treatments, or other relevant info...',
                      hintStyle: const TextStyle(
                          color: AppColors.textSecondary, fontSize: 13),
                      filled: true,
                      fillColor: AppColors.primaryLight,
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide:
                              const BorderSide(color: AppColors.border)),
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide:
                              const BorderSide(color: AppColors.border)),
                      focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: const BorderSide(
                              color: AppColors.primary, width: 1.5)),
                    ),
                  ),
                ],
              ),
            ),
          ),

          Padding(
            padding: const EdgeInsets.fromLTRB(24, 0, 24, 28),
            child: Row(
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
                    onPressed: _save,
                    child: const Text('Next'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title) => Text(
        title,
        style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary),
      );

  Widget _buildChipGroup(List<String> items, Set<String> selected) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: items.map((item) {
        final isSelected = selected.contains(item);
        return GestureDetector(
          onTap: () => setState(() {
            isSelected ? selected.remove(item) : selected.add(item);
          }),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            padding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: isSelected ? AppColors.primary : Colors.white,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                  color:
                      isSelected ? AppColors.primary : AppColors.border),
            ),
            child: Text(
              item,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: isSelected ? Colors.white : AppColors.textPrimary,
              ),
            ),
          ),
        );
      }).toList(),
    );
  }
}
