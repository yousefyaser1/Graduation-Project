import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../core/widgets/app_bottom_nav.dart';

class TipsScreen extends StatelessWidget {
  const TipsScreen({super.key});

  static const List<({IconData icon, Color color, String title, List<String> tips})>
      _categories = [
    (
      icon: Icons.wb_sunny_outlined,
      color: Color(0xFFF59E0B),
      title: 'Sun Protection',
      tips: [
        'Apply broad-spectrum SPF 30+ sunscreen every day, even when cloudy.',
        'Reapply sunscreen every 2 hours when outdoors.',
        'Avoid direct sun between 10 a.m. and 4 p.m. when UV is strongest.',
        'Wear a wide-brimmed hat and UV-blocking sunglasses.',
      ],
    ),
    (
      icon: Icons.water_drop_outlined,
      color: Color(0xFF3B82F6),
      title: 'Daily Skincare',
      tips: [
        'Cleanse your face twice daily with a gentle, fragrance-free cleanser.',
        'Moisturize within 3 minutes of washing to lock in hydration.',
        'Drink plenty of water to keep skin hydrated from the inside.',
        'Avoid hot showers — lukewarm water is gentler on the skin barrier.',
      ],
    ),
    (
      icon: Icons.healing_outlined,
      color: Color(0xFF22C55E),
      title: 'Acne Care',
      tips: [
        'Never pick or squeeze pimples — it worsens scarring and inflammation.',
        'Use non-comedogenic (won\'t-clog-pores) products.',
        'Change pillowcases regularly and keep phones clean.',
        'Avoid over-washing, which strips oils and triggers more breakouts.',
      ],
    ),
    (
      icon: Icons.spa_outlined,
      color: Color(0xFF8B5CF6),
      title: 'Eczema & Dry Skin',
      tips: [
        'Moisturize frequently with thick, fragrance-free creams or ointments.',
        'Identify and avoid triggers like harsh soaps, wool, and stress.',
        'Use a humidifier in dry environments.',
        'Pat skin dry instead of rubbing after bathing.',
      ],
    ),
    (
      icon: Icons.coronavirus_outlined,
      color: Color(0xFFEF4444),
      title: 'Fungal Infections (Tinea)',
      tips: [
        'Keep skin clean and dry, especially in folds and between toes.',
        'Don\'t share towels, clothing, combs, or footwear.',
        'Wear breathable fabrics and change sweaty clothes promptly.',
        'Complete the full course of any antifungal treatment.',
      ],
    ),
    (
      icon: Icons.favorite_outline,
      color: Color(0xFFEC4899),
      title: 'Healthy Habits',
      tips: [
        'Eat a balanced diet rich in fruits, vegetables, and omega-3s.',
        'Get enough sleep — skin repairs itself overnight.',
        'Manage stress, which can trigger flare-ups of many conditions.',
        'See a dermatologist for any new, changing, or concerning spots.',
      ],
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text(
          'Skin Care Tips',
          style: TextStyle(
            fontSize: 17,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
        children: [
          // Header banner
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF1E3A8A), Color(0xFF3B82F6)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.lightbulb_outline,
                    color: Colors.white, size: 28),
              ),
              const SizedBox(width: 16),
              const Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Keep Your Skin Healthy',
                          style: TextStyle(
                              color: Colors.white,
                              fontSize: 17,
                              fontWeight: FontWeight.w700)),
                      SizedBox(height: 4),
                      Text('Simple daily habits for healthier skin.',
                          style: TextStyle(
                              color: Colors.white70, fontSize: 12)),
                    ]),
              ),
            ]),
          ),
          const SizedBox(height: 16),

          ..._categories.map((c) => _TipCategoryCard(category: c)),
        ],
      ),
      bottomNavigationBar: const AppBottomNav(currentIndex: 4),
    );
  }
}

class _TipCategoryCard extends StatelessWidget {
  final ({IconData icon, Color color, String title, List<String> tips}) category;
  const _TipCategoryCard({required this.category});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          leading: Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: category.color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(category.icon, color: category.color, size: 22),
          ),
          title: Text(
            category.title,
            style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary),
          ),
          children: category.tips
              .map((t) => Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(top: 2, right: 10),
                            child: Icon(Icons.check_circle,
                                size: 16, color: category.color),
                          ),
                          Expanded(
                            child: Text(t,
                                style: const TextStyle(
                                    fontSize: 13,
                                    color: AppColors.textPrimary,
                                    height: 1.45)),
                          ),
                        ]),
                  ))
              .toList(),
        ),
      ),
    );
  }
}
