import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';

/// A tappable region mapped onto the body illustration as fractional bounds
/// (left, top, width, height) relative to the rendered image box.
class _Hotspot {
  final String id;
  final Rect rect;
  const _Hotspot(this.id, this.rect);
}

class BodyPartSelectionScreen extends StatefulWidget {
  const BodyPartSelectionScreen({super.key});

  @override
  State<BodyPartSelectionScreen> createState() =>
      _BodyPartSelectionScreenState();
}

class _BodyPartSelectionScreenState extends State<BodyPartSelectionScreen> {
  bool _showFront = true;

  // Single selection — one scan captures one body part.
  String? _selected;

  // The illustration's intrinsic aspect ratio (width / height).
  static const double _bodyAspect = 390 / 782;

  // Fractional hotspots over the FRONT figure.
  static const List<_Hotspot> _frontSpots = [
    _Hotspot('FACE', Rect.fromLTWH(0.36, 0.00, 0.28, 0.16)),
    _Hotspot('NECK', Rect.fromLTWH(0.42, 0.15, 0.16, 0.05)),
    _Hotspot('CHEST', Rect.fromLTWH(0.30, 0.19, 0.40, 0.14)),
    _Hotspot('STOMACH', Rect.fromLTWH(0.32, 0.33, 0.36, 0.14)),
    _Hotspot('LEFT ARM', Rect.fromLTWH(0.00, 0.18, 0.22, 0.34)),
    _Hotspot('RIGHT ARM', Rect.fromLTWH(0.78, 0.18, 0.22, 0.34)),
    _Hotspot('LEFT LEG', Rect.fromLTWH(0.30, 0.50, 0.20, 0.48)),
    _Hotspot('RIGHT LEG', Rect.fromLTWH(0.50, 0.50, 0.20, 0.48)),
  ];

  // Fractional hotspots over the BACK figure.
  static const List<_Hotspot> _backSpots = [
    _Hotspot('LEFT SHOULDER', Rect.fromLTWH(0.15, 0.16, 0.23, 0.10)),
    _Hotspot('RIGHT SHOULDER', Rect.fromLTWH(0.62, 0.16, 0.23, 0.10)),
    _Hotspot('UPPER BACK', Rect.fromLTWH(0.34, 0.20, 0.32, 0.14)),
    _Hotspot('MID BACK', Rect.fromLTWH(0.34, 0.34, 0.32, 0.12)),
    _Hotspot('LOWER BACK', Rect.fromLTWH(0.34, 0.46, 0.32, 0.06)),
    _Hotspot('BUTTOCKS', Rect.fromLTWH(0.32, 0.52, 0.36, 0.08)),
  ];

  void _selectPart(String part) {
    HapticFeedback.selectionClick();
    setState(() {
      _selected = (_selected == part) ? null : part;
    });
  }

  String _prettify(String part) {
    return part
        .split(' ')
        .map((w) => w.isEmpty ? w : w[0] + w.substring(1).toLowerCase())
        .join(' ');
  }

  @override
  Widget build(BuildContext context) {
    final spots = _showFront ? _frontSpots : _backSpots;

    return Scaffold(
      // Dimmed backdrop so the white card reads as a floating step card.
      backgroundColor: const Color(0xFF8E99A6),
      body: SafeArea(
        child: Center(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(28),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.18),
                  blurRadius: 24,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 22),
              child: Column(
                children: [
                  // ── Header: "Step 3" + close ──
                  Stack(
                    alignment: Alignment.center,
                    children: [
                      const Text(
                        'Step 3',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      Align(
                        alignment: Alignment.centerRight,
                        child: GestureDetector(
                          onTap: () => context.pop(),
                          child: Container(
                            width: 36,
                            height: 36,
                            decoration: const BoxDecoration(
                              color: Color(0xFFF1F5F9),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.close,
                                size: 20, color: AppColors.textSecondary),
                          ),
                        ),
                      ),
                    ],
                  ),

                  // ── Body figure + chevrons ──
                  Expanded(
                    child: Row(
                      children: [
                        _Chevron(
                          icon: Icons.chevron_left,
                          onTap: () => setState(() => _showFront = !_showFront),
                        ),
                        Expanded(
                          child: Center(
                            child: AspectRatio(
                              aspectRatio: _bodyAspect,
                              child: LayoutBuilder(
                                builder: (context, c) {
                                  final w = c.maxWidth;
                                  final h = c.maxHeight;
                                  return Stack(
                                    children: [
                                      AnimatedSwitcher(
                                        duration:
                                            const Duration(milliseconds: 250),
                                        child: Image.asset(
                                          _showFront
                                              ? 'assets/images/body_front.png'
                                              : 'assets/images/body_back.png',
                                          key: ValueKey(_showFront),
                                          fit: BoxFit.contain,
                                        ),
                                      ),
                                      // Selection highlight + tap targets.
                                      for (final s in spots)
                                        Positioned(
                                          left: s.rect.left * w,
                                          top: s.rect.top * h,
                                          width: s.rect.width * w,
                                          height: s.rect.height * h,
                                          child: GestureDetector(
                                            behavior:
                                                HitTestBehavior.translucent,
                                            onTap: () => _selectPart(s.id),
                                            child: AnimatedContainer(
                                              duration: const Duration(
                                                  milliseconds: 180),
                                              decoration: BoxDecoration(
                                                color: _selected == s.id
                                                    ? AppColors.primary
                                                        .withValues(alpha: 0.28)
                                                    : Colors.transparent,
                                                borderRadius:
                                                    BorderRadius.circular(14),
                                              ),
                                            ),
                                          ),
                                        ),
                                    ],
                                  );
                                },
                              ),
                            ),
                          ),
                        ),
                        _Chevron(
                          icon: Icons.chevron_right,
                          onTap: () => setState(() => _showFront = !_showFront),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 16),

                  // ── Footer: hint or Scan button ──
                  if (_selected == null)
                    const Column(
                      children: [
                        Icon(Icons.arrow_upward,
                            size: 28, color: AppColors.textPrimary),
                        SizedBox(height: 8),
                        Text(
                          'Tap to choose body part',
                          style: TextStyle(
                            fontSize: 16,
                            color: AppColors.textSecondary,
                          ),
                        ),
                      ],
                    )
                  else
                    SizedBox(
                      width: double.infinity,
                      height: 56,
                      child: ElevatedButton(
                        onPressed: () => context.push(
                          AppRoutes.capture,
                          extra: _selected,
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.textPrimary,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                        child: Text(
                          'Scan: ${_prettify(_selected!)}',
                          style: const TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Front/back navigation chevron
// ──────────────────────────────────────────────────────────────────────────────
class _Chevron extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;

  const _Chevron({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Icon(icon, size: 34, color: AppColors.textSecondary),
      ),
    );
  }
}
