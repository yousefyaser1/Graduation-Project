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

/// Clips to a hotspot's fractional rect scaled up to the rendered image box.
class _FractionalRectClipper extends CustomClipper<Rect> {
  final Rect fraction;
  const _FractionalRectClipper(this.fraction);

  @override
  Rect getClip(Size size) => Rect.fromLTWH(
        fraction.left * size.width,
        fraction.top * size.height,
        fraction.width * size.width,
        fraction.height * size.height,
      );

  @override
  bool shouldReclip(_FractionalRectClipper oldClipper) =>
      oldClipper.fraction != fraction;
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

  void _flip() {
    HapticFeedback.selectionClick();
    setState(() => _showFront = !_showFront);
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
    final asset = _showFront
        ? 'assets/images/body_front.png'
        : 'assets/images/body_back.png';

    // The selected hotspot, if it belongs to the side currently shown.
    _Hotspot? selectedSpot;
    for (final s in spots) {
      if (s.id == _selected) selectedSpot = s;
    }

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
                  // ── Header: title + close ──
                  Stack(
                    alignment: Alignment.center,
                    children: [
                      const Text(
                        'Select Body Part',
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
                          key: const ValueKey('flip-left'),
                          icon: Icons.chevron_left,
                          onTap: _flip,
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
                                      // Base figure.
                                      AnimatedSwitcher(
                                        duration:
                                            const Duration(milliseconds: 250),
                                        child: Image.asset(
                                          asset,
                                          key: ValueKey(_showFront),
                                          width: w,
                                          height: h,
                                          fit: BoxFit.contain,
                                        ),
                                      ),
                                      // Selection highlight. The illustration
                                      // is redrawn tinted (srcIn keeps color
                                      // only where the PNG is opaque) and
                                      // clipped to the selected hotspot, so
                                      // the shade hugs the chosen body part
                                      // instead of filling a whole rectangle.
                                      IgnorePointer(
                                        child: AnimatedSwitcher(
                                          duration:
                                              const Duration(milliseconds: 180),
                                          child: selectedSpot == null
                                              ? SizedBox(
                                                  key: const ValueKey(
                                                      'no-selection'),
                                                  width: w,
                                                  height: h,
                                                )
                                              : ClipRect(
                                                  key: ValueKey(
                                                      'highlight-${selectedSpot.id}'),
                                                  clipper:
                                                      _FractionalRectClipper(
                                                          selectedSpot.rect),
                                                  child: Image.asset(
                                                    asset,
                                                    width: w,
                                                    height: h,
                                                    fit: BoxFit.contain,
                                                    color: AppColors.primary
                                                        .withValues(
                                                            alpha: 0.45),
                                                    colorBlendMode:
                                                        BlendMode.srcIn,
                                                  ),
                                                ),
                                        ),
                                      ),
                                      // Invisible tap targets.
                                      for (final s in spots)
                                        Positioned(
                                          left: s.rect.left * w,
                                          top: s.rect.top * h,
                                          width: s.rect.width * w,
                                          height: s.rect.height * h,
                                          child: Semantics(
                                            button: true,
                                            selected: _selected == s.id,
                                            label: _prettify(s.id),
                                            child: GestureDetector(
                                              key: ValueKey('hotspot_${s.id}'),
                                              behavior:
                                                  HitTestBehavior.translucent,
                                              onTap: () => _selectPart(s.id),
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
                          key: const ValueKey('flip-right'),
                          icon: Icons.chevron_right,
                          onTap: _flip,
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 10),

                  // ── Front/back caption ──
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 200),
                    child: Text(
                      _showFront ? 'FRONT VIEW' : 'BACK VIEW',
                      key: ValueKey(_showFront),
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.2,
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ),

                  const SizedBox(height: 14),

                  // ── Footer: hint or Scan button ──
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 200),
                    child: _selected == null
                        ? const Column(
                            key: ValueKey('hint'),
                            children: [
                              Icon(Icons.touch_app_outlined,
                                  size: 26, color: AppColors.textSecondary),
                              SizedBox(height: 6),
                              Text(
                                'Tap the body part you want to scan',
                                style: TextStyle(
                                  fontSize: 15,
                                  color: AppColors.textSecondary,
                                ),
                              ),
                            ],
                          )
                        : SizedBox(
                            key: const ValueKey('scan'),
                            width: double.infinity,
                            height: 56,
                            child: ElevatedButton(
                              onPressed: () => context.push(
                                AppRoutes.capture,
                                extra: _prettify(_selected!),
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

  const _Chevron({super.key, required this.icon, required this.onTap});

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
