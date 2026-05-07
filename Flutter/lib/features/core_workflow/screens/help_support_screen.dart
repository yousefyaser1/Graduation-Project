import 'package:flutter/material.dart';
import '../../../core/theme/app_theme.dart';

class HelpSupportScreen extends StatefulWidget {
  const HelpSupportScreen({super.key});

  @override
  State<HelpSupportScreen> createState() => _HelpSupportScreenState();
}

class _HelpSupportScreenState extends State<HelpSupportScreen> {
  final _searchController = TextEditingController();
  int? _expandedFaq;
  String _searchQuery = '';

  final _faqs = [
    {
      'q': 'How accurate is the AI diagnosis?',
      'a':
          'Our AI model achieves over 85% accuracy on validated datasets. However, it is designed as a screening tool and not a medical diagnosis. Always consult a dermatologist for confirmation.',
    },
    {
      'q': 'Is my data private and secure?',
      'a':
          'Yes. All images and results are encrypted and stored securely. We never share your personal data with third parties without your explicit consent.',
    },
    {
      'q': 'How do I get the best scan results?',
      'a':
          'Use natural lighting, hold the camera 10–15 cm from the skin area, ensure the skin is clean, and keep the camera steady. Avoid flash photography.',
    },
    {
      'q': 'Can I use this app for a child?',
      'a':
          'The app can be used for children under adult supervision. However, always consult a pediatric dermatologist for any skin concerns in children.',
    },
    {
      'q': 'How do I book an appointment?',
      'a':
          'After viewing your analysis results, tap "Book Appointment". You can choose a specialist, select a date and time, and confirm the booking directly in the app.',
    },
    {
      'q': 'What skin conditions can the app detect?',
      'a':
          'The app can detect common skin conditions including moles, eczema, psoriasis, acne, rosacea, seborrheic keratosis, and potential early signs of melanoma.',
    },
  ];

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Help & Support',
          style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Hero
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF2563EB), Color(0xFF3B82F6)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('How can we help?',
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w700)),
                  const SizedBox(height: 6),
                  Text('Search our knowledge base',
                      style: TextStyle(
                          color: Colors.white.withOpacity(0.8), fontSize: 13)),
                  const SizedBox(height: 14),
                  TextField(
                    controller: _searchController,
                    onChanged: (v) => setState(() {
                      _searchQuery = v.toLowerCase();
                      _expandedFaq = null;
                    }),
                    decoration: InputDecoration(
                      hintText: 'Search FAQs...',
                      hintStyle:
                          const TextStyle(color: AppColors.textSecondary),
                      prefixIcon: const Icon(Icons.search,
                          color: AppColors.textSecondary),
                      filled: true,
                      fillColor: Colors.white,
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none),
                      contentPadding:
                          const EdgeInsets.symmetric(vertical: 0),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Contact options
            const Text('Contact Us',
                style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary)),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _ContactCard(
                    icon: Icons.email_outlined,
                    label: 'Email',
                    value: 'support@\nskinscan.ai',
                    color: AppColors.primary,
                    bg: AppColors.primaryLight,
                    onTap: () {},
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _ContactCard(
                    icon: Icons.chat_bubble_outline,
                    label: 'Live Chat',
                    value: 'Available\n9AM–6PM',
                    color: AppColors.success,
                    bg: const Color(0xFFDCFCE7),
                    onTap: () {},
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _ContactCard(
                    icon: Icons.phone_outlined,
                    label: 'Call',
                    value: '+1 800\nSKINSCAN',
                    color: const Color(0xFF8B5CF6),
                    bg: const Color(0xFFEDE9FE),
                    onTap: () {},
                  ),
                ),
              ],
            ),
            const SizedBox(height: 28),

            // FAQs
            const Text('Frequently Asked Questions',
                style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary)),
            const SizedBox(height: 12),
            ...() {
                final filtered = _searchQuery.isEmpty
                    ? _faqs
                    : _faqs
                        .where((f) =>
                            f['q']!.toLowerCase().contains(_searchQuery) ||
                            f['a']!.toLowerCase().contains(_searchQuery))
                        .toList();
                if (filtered.isEmpty) {
                  return [
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 20),
                      child: Center(
                        child: Text('No results found',
                            style: TextStyle(
                                fontSize: 14,
                                color: AppColors.textSecondary)),
                      ),
                    )
                  ];
                }
                return List.generate(
                    filtered.length, (i) => _buildFaqTile(i, filtered));
              }(),
            const SizedBox(height: 24),

            // Footer
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info_outline,
                      color: AppColors.textSecondary, size: 20),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Text(
                      'SkinScan AI v1.0.0\nFor medical emergencies, call 911.',
                      style: TextStyle(
                          fontSize: 12,
                          color: AppColors.textSecondary,
                          height: 1.5),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFaqTile(int index, [List<Map<String, String>>? list]) {
    final faq = (list ?? _faqs)[index];
    final isExpanded = _expandedFaq == index;

    return GestureDetector(
      onTap: () =>
          setState(() => _expandedFaq = isExpanded ? null : index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.only(bottom: 10),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: isExpanded ? AppColors.primary : AppColors.border,
              width: isExpanded ? 1.5 : 1),
        ),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(14),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      faq['q']!,
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: isExpanded
                              ? AppColors.primary
                              : AppColors.textPrimary),
                    ),
                  ),
                  Icon(
                    isExpanded
                        ? Icons.keyboard_arrow_up
                        : Icons.keyboard_arrow_down,
                    color: isExpanded
                        ? AppColors.primary
                        : AppColors.textSecondary,
                  ),
                ],
              ),
            ),
            if (isExpanded)
              Padding(
                padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
                child: Text(
                  faq['a']!,
                  style: const TextStyle(
                      fontSize: 13,
                      color: AppColors.textSecondary,
                      height: 1.55),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _ContactCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;
  final Color bg;
  final VoidCallback onTap;

  const _ContactCard({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
    required this.bg,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Column(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(color: bg, shape: BoxShape.circle),
              child: Icon(icon, color: color, size: 22),
            ),
            const SizedBox(height: 8),
            Text(label,
                style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary)),
            const SizedBox(height: 3),
            Text(value,
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 10, color: AppColors.textSecondary)),
          ],
        ),
      ),
    );
  }
}
