import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../core/widgets/app_bottom_nav.dart';
import '../../../providers/user_provider.dart';
import '../../../providers/scan_provider.dart';

class ProfileScreen extends ConsumerWidget {
  const ProfileScreen({super.key});

  static bool _isHighRisk(scan) {
    const hrd = ['Melanoma', 'Carcinoma', 'Lymphoma'];
    return hrd.any((d) => (scan.diagnosis as String).contains(d));
  }

  // Parse "Skin Type: oily | Skin Tone: Light | ..." from medicalHistory
  static String _parseSkinType(String? medicalHistory) {
    if (medicalHistory == null) return 'Not set';
    final match =
        RegExp(r'Skin Type:\s*([^|]+)').firstMatch(medicalHistory);
    final val = match?.group(1)?.trim() ?? '';
    if (val.isEmpty || val == 'Not specified') return 'Not set';
    return val[0].toUpperCase() + val.substring(1);
  }

  static String _parseSkinTone(String? medicalHistory) {
    if (medicalHistory == null) return 'Not set';
    final match =
        RegExp(r'Skin Tone:\s*([^|]+)').firstMatch(medicalHistory);
    return match?.group(1)?.trim() ?? 'Not set';
  }

  static void _showPrivacyDialog(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Row(children: [
          Icon(Icons.lock_outline, color: AppColors.primary, size: 20),
          SizedBox(width: 8),
          Text('Privacy & Security',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
        ]),
        content: const SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _PrivacyPoint(
                icon: Icons.smartphone_outlined,
                text:
                    'On-device only: every AI stage (screening gate, CNN, Score-CAM) runs locally. Your photos never leave this phone.',
              ),
              _PrivacyPoint(
                icon: Icons.storage_outlined,
                text:
                    'Local storage: scans and results are saved in this app\'s private database — not in the cloud.',
              ),
              _PrivacyPoint(
                icon: Icons.key_outlined,
                text:
                    'Passwords are salted and hashed (PBKDF2-HMAC-SHA256) — never stored in plain text.',
              ),
              _PrivacyPoint(
                icon: Icons.delete_outline,
                text:
                    'Deleting a scan removes its image and heatmaps from local storage.',
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Got it',
                style: TextStyle(
                    color: AppColors.primary, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }

  static Color _toneColor(String tone) {
    const colors = {
      'Fair': Color(0xFFF5D5B0),
      'Light': Color(0xFFEAB88A),
      'Medium': Color(0xFFD4945C),
      'Tan': Color(0xFFB87344),
      'Brown': Color(0xFF8B5E3C),
      'Dark': Color(0xFF4A2D1E),
    };
    return colors[tone] ?? const Color(0xFFD4945C);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(userProvider);
    final scanAsync = ref.watch(scanListProvider);

    final skinType = _parseSkinType(user?.medicalHistory);
    final skinTone = _parseSkinTone(user?.medicalHistory);

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        automaticallyImplyLeading: false,
        title: const Text(
          'Profile',
          style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
        child: Column(
          children: [
            // Avatar + name card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.border),
              ),
              child: Column(
                children: [
                  Stack(
                    children: [
                      Container(
                        width: 84,
                        height: 84,
                        decoration: const BoxDecoration(
                          gradient: LinearGradient(
                            colors: [Color(0xFF2563EB), Color(0xFF3B82F6)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.person_outline,
                            color: Colors.white, size: 40),
                      ),
                      Positioned(
                        bottom: 0,
                        right: 0,
                        child: GestureDetector(
                          onTap: () => context.push(AppRoutes.editProfile),
                          child: Container(
                            width: 26,
                            height: 26,
                            decoration: BoxDecoration(
                              color: AppColors.primary,
                              shape: BoxShape.circle,
                              border:
                                  Border.all(color: Colors.white, width: 2),
                            ),
                            child: const Icon(Icons.edit,
                                color: Colors.white, size: 13),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Text(
                    user?.name ?? 'Guest',
                    style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    user?.email ?? '',
                    style: const TextStyle(
                        fontSize: 13, color: AppColors.textSecondary),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 5),
                    decoration: BoxDecoration(
                      color: AppColors.primaryLight,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.verified_user_outlined,
                            color: AppColors.primary, size: 14),
                        const SizedBox(width: 5),
                        Text(
                          user?.role == 'specialist'
                              ? 'Specialist'
                              : 'Patient',
                          style: const TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: AppColors.primary),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Skin profile card
            _SectionCard(
              title: 'Skin Profile',
              children: [
                _InfoRow(
                  icon: Icons.water_drop_outlined,
                  label: 'Skin Type',
                  value: skinType,
                  valueColor: skinType == 'Not set'
                      ? AppColors.textSecondary
                      : AppColors.primary,
                ),
                const Divider(color: AppColors.border, height: 20),
                _InfoRow(
                  icon: Icons.circle,
                  label: 'Skin Tone',
                  value: skinTone,
                  valueColor: AppColors.textPrimary,
                  leadingWidget: skinTone != 'Not set'
                      ? Container(
                          width: 20,
                          height: 20,
                          decoration: BoxDecoration(
                            color: _toneColor(skinTone),
                            shape: BoxShape.circle,
                          ),
                        )
                      : null,
                ),
                const Divider(color: AppColors.border, height: 20),
                _InfoRow(
                  icon: Icons.cake_outlined,
                  label: 'Age',
                  value: user?.age != null ? '${user!.age} years' : 'Not set',
                  valueColor: AppColors.textPrimary,
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Stats card
            scanAsync.when(
              loading: () => const SizedBox.shrink(),
              error: (_, __) => const SizedBox.shrink(),
              data: (scans) {
                final lowRisk = scans.where((s) => !_isHighRisk(s)).length;
                final highRisk = scans.where(_isHighRisk).length;
                return _SectionCard(
                  title: 'My Stats',
                  children: [
                    Row(
                      children: [
                        _MiniStat(
                            value: '${scans.length}',
                            label: 'Total Scans'),
                        _VertDivider(),
                        _MiniStat(
                            value: '$lowRisk',
                            label: 'Low Risk',
                            color: AppColors.success),
                        _VertDivider(),
                        _MiniStat(
                            value: '$highRisk',
                            label: 'High Risk',
                            color: AppColors.error),
                      ],
                    ),
                  ],
                );
              },
            ),
            const SizedBox(height: 16),

            // Settings list
            _SectionCard(
              title: 'Account',
              children: [
                _SettingsTile(
                    icon: Icons.edit_outlined,
                    label: 'Edit Profile',
                    onTap: () => context.push(AppRoutes.editProfile)),
                const Divider(color: AppColors.border, height: 1),
                _SettingsTile(
                    icon: Icons.notifications_outlined,
                    label: 'Notifications',
                    onTap: () => context.push(AppRoutes.notifications)),
                const Divider(color: AppColors.border, height: 1),
                _SettingsTile(
                    icon: Icons.lock_outline,
                    label: 'Privacy & Security',
                    onTap: () => _showPrivacyDialog(context)),
                const Divider(color: AppColors.border, height: 1),
                _SettingsTile(
                    icon: Icons.help_outline,
                    label: 'Help & Support',
                    onTap: () => context.push(AppRoutes.helpSupport)),
              ],
            ),
            const SizedBox(height: 16),

            // Logout
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () {
                  ref.read(userProvider.notifier).clearUser();
                  context.go(AppRoutes.welcome);
                },
                icon: const Icon(Icons.logout, color: AppColors.error),
                label: const Text(
                  'Log Out',
                  style: TextStyle(
                      color: AppColors.error, fontWeight: FontWeight.w600),
                ),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  side: const BorderSide(color: AppColors.error),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
      bottomNavigationBar: const AppBottomNav(currentIndex: 3),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final List<Widget> children;

  const _SectionCard({required this.title, required this.children});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary),
          ),
          const SizedBox(height: 14),
          ...children,
        ],
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color valueColor;
  final Widget? leadingWidget;

  const _InfoRow({
    required this.icon,
    required this.label,
    required this.value,
    required this.valueColor,
    this.leadingWidget,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        leadingWidget ??
            Icon(icon, color: AppColors.textSecondary, size: 18),
        const SizedBox(width: 10),
        Text(label,
            style: const TextStyle(
                fontSize: 13, color: AppColors.textSecondary)),
        const Spacer(),
        Text(value,
            style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: valueColor)),
      ],
    );
  }
}

class _MiniStat extends StatelessWidget {
  final String value;
  final String label;
  final Color color;

  const _MiniStat({
    required this.value,
    required this.label,
    this.color = AppColors.primary,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Text(value,
              style: TextStyle(
                  fontSize: 22, fontWeight: FontWeight.w800, color: color)),
          const SizedBox(height: 2),
          Text(label,
              style: const TextStyle(
                  fontSize: 11, color: AppColors.textSecondary)),
        ],
      ),
    );
  }
}

class _VertDivider extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(width: 1, height: 36, color: AppColors.border);
  }
}

class _PrivacyPoint extends StatelessWidget {
  final IconData icon;
  final String text;
  const _PrivacyPoint({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(icon, color: AppColors.primary, size: 18),
        const SizedBox(width: 10),
        Expanded(
          child: Text(text,
              style: const TextStyle(
                  fontSize: 13, color: AppColors.textPrimary, height: 1.5)),
        ),
      ]),
    );
  }
}

class _SettingsTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _SettingsTile({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: Row(
          children: [
            Icon(icon, color: AppColors.textSecondary, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Text(label,
                  style: const TextStyle(
                      fontSize: 14, color: AppColors.textPrimary)),
            ),
            const Icon(Icons.chevron_right,
                color: AppColors.textSecondary, size: 20),
          ],
        ),
      ),
    );
  }
}
