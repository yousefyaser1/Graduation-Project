import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../providers/user_provider.dart';

class PersonalInfoScreen extends ConsumerStatefulWidget {
  const PersonalInfoScreen({super.key});

  @override
  ConsumerState<PersonalInfoScreen> createState() => _PersonalInfoScreenState();
}

class _PersonalInfoScreenState extends ConsumerState<PersonalInfoScreen> {
  String _selectedGender = 'female';
  DateTime? _selectedDate;

  final _genders = [
    {'key': 'female', 'label': 'Female', 'icon': Icons.person_outline},
    {'key': 'male', 'label': 'Male', 'icon': Icons.person_outline},
    {'key': 'nonbinary', 'label': 'Non-binary', 'icon': Icons.people_outline},
  ];

  @override
  void initState() {
    super.initState();
    // Pre-fill from existing user data if available
    final user = ref.read(userProvider);
    if (user != null) {
      if (user.gender != null) _selectedGender = user.gender!;
      if (user.age != null) {
        _selectedDate = DateTime(DateTime.now().year - user.age!);
      }
    }
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate ?? DateTime(1995),
      firstDate: DateTime(1920),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.light(primary: AppColors.primary),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) setState(() => _selectedDate = picked);
  }

  Future<void> _next() async {
    final age = _selectedDate != null
        ? DateTime.now().year - _selectedDate!.year
        : null;
    await ref.read(userProvider.notifier).updateProfile(
          gender: _selectedGender,
          age: age,
        );
    if (mounted) context.push(AppRoutes.skinType);
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
      body: Padding(
        padding: const EdgeInsets.fromLTRB(24, 8, 24, 0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Step progress
            Row(
              children: [
                const Text(
                  'Step 2 of 5',
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
                      value: 2 / 5,
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
              'Personal Info Setup',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 24),

            const Text(
              'Gender Identity',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 12),

            Row(
              children: _genders.map((g) {
                final isSelected = _selectedGender == g['key'];
                return Expanded(
                  child: GestureDetector(
                    onTap: () =>
                        setState(() => _selectedGender = g['key'] as String),
                    child: Container(
                      margin: EdgeInsets.only(
                        right: g['key'] != 'nonbinary' ? 10 : 0,
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? AppColors.primaryLight
                            : const Color(0xFFF8FAFC),
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(
                          color: isSelected
                              ? AppColors.primary
                              : AppColors.border,
                          width: isSelected ? 2 : 1,
                        ),
                      ),
                      child: Column(
                        children: [
                          Stack(
                            alignment: Alignment.topRight,
                            children: [
                              Container(
                                width: 44,
                                height: 44,
                                decoration: BoxDecoration(
                                  color: isSelected
                                      ? AppColors.primary.withOpacity(0.15)
                                      : AppColors.border.withOpacity(0.4),
                                  shape: BoxShape.circle,
                                ),
                                child: Icon(
                                  g['icon'] as IconData,
                                  color: isSelected
                                      ? AppColors.primary
                                      : AppColors.textSecondary,
                                  size: 22,
                                ),
                              ),
                              if (isSelected)
                                Container(
                                  width: 16,
                                  height: 16,
                                  decoration: BoxDecoration(
                                    color: AppColors.primary,
                                    shape: BoxShape.circle,
                                    border: Border.all(
                                        color: Colors.white, width: 2),
                                  ),
                                  child: const Icon(Icons.check,
                                      color: Colors.white, size: 10),
                                ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            g['label'] as String,
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: isSelected
                                  ? AppColors.primary
                                  : AppColors.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 24),

            const Text(
              'Date of Birth',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 10),

            GestureDetector(
              onTap: _pickDate,
              child: Container(
                width: double.infinity,
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                decoration: BoxDecoration(
                  color: AppColors.primaryLight,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        _selectedDate == null
                            ? 'MM / DD / YYYY'
                            : '${_selectedDate!.month.toString().padLeft(2, '0')} / ${_selectedDate!.day.toString().padLeft(2, '0')} / ${_selectedDate!.year}',
                        style: TextStyle(
                          fontSize: 15,
                          color: _selectedDate == null
                              ? AppColors.textSecondary
                              : AppColors.textPrimary,
                        ),
                      ),
                    ),
                    const Icon(Icons.calendar_month_outlined,
                        color: AppColors.textSecondary, size: 20),
                  ],
                ),
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
