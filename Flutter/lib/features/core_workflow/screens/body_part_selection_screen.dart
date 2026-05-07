import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';

class BodyPartSelectionScreen extends StatefulWidget {
  const BodyPartSelectionScreen({super.key});

  @override
  State<BodyPartSelectionScreen> createState() =>
      _BodyPartSelectionScreenState();
}

class _BodyPartSelectionScreenState extends State<BodyPartSelectionScreen> {
  bool _showFront = true;
  final Set<String> _selectedParts = {};

  void _togglePart(String part) {
    setState(() {
      if (_selectedParts.contains(part)) {
        _selectedParts.remove(part);
      } else {
        _selectedParts.add(part);
      }
    });
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
        title: Text(
          _showFront
              ? 'Select Body Part - Front'
              : 'Select Body Part - Back',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () {},
            child: Text(
              'Help',
              style: TextStyle(color: AppColors.primary, fontSize: 14),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Info banner
          Container(
            margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: const Color(0xFFDBEAFE),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              children: [
                GestureDetector(
                  onTap: () {},
                  child: const Icon(Icons.close,
                      size: 16, color: AppColors.primary),
                ),
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    'Tap on the areas where you have skin concerns. You can select multiple regions.',
                    style: TextStyle(
                      fontSize: 12,
                      color: AppColors.primary,
                      height: 1.4,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),

          // Front / Back toggle
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xFFF1F5F9),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: GestureDetector(
                      onTap: () => setState(() => _showFront = true),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        margin: const EdgeInsets.all(4),
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        decoration: BoxDecoration(
                          color: _showFront ? Colors.white : Colors.transparent,
                          borderRadius: BorderRadius.circular(8),
                          boxShadow: _showFront
                              ? [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.08),
                                    blurRadius: 4,
                                    offset: const Offset(0, 1),
                                  ),
                                ]
                              : [],
                        ),
                        child: Text(
                          'Front',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: _showFront
                                ? AppColors.textPrimary
                                : AppColors.textSecondary,
                          ),
                        ),
                      ),
                    ),
                  ),
                  Expanded(
                    child: GestureDetector(
                      onTap: () => setState(() => _showFront = false),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        margin: const EdgeInsets.all(4),
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        decoration: BoxDecoration(
                          color:
                              !_showFront ? Colors.white : Colors.transparent,
                          borderRadius: BorderRadius.circular(8),
                          boxShadow: !_showFront
                              ? [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.08),
                                    blurRadius: 4,
                                    offset: const Offset(0, 1),
                                  ),
                                ]
                              : [],
                        ),
                        child: Text(
                          'Back',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: !_showFront
                                ? AppColors.textPrimary
                                : AppColors.textSecondary,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Body diagram
          Expanded(
            child: _showFront
                ? _FrontBodyDiagram(
                    selectedParts: _selectedParts,
                    onToggle: _togglePart,
                  )
                : _BackBodyDiagram(
                    selectedParts: _selectedParts,
                    onToggle: _togglePart,
                  ),
          ),

          // Confirm button
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 28),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _selectedParts.isNotEmpty
                    ? () => context.push(
                          AppRoutes.capture,
                          extra: _selectedParts.join(', '),
                        )
                    : null,
                icon: const Icon(Icons.check_circle_outline, size: 18),
                label: const Text('Confirm'),
                style: ElevatedButton.styleFrom(
                  disabledBackgroundColor: AppColors.border,
                  disabledForegroundColor: AppColors.textSecondary,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Front body diagram
// ──────────────────────────────────────────────────────────────────────────────
class _FrontBodyDiagram extends StatelessWidget {
  final Set<String> selectedParts;
  final void Function(String) onToggle;

  const _FrontBodyDiagram({
    required this.selectedParts,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      final w = constraints.maxWidth;
      final h = constraints.maxHeight;
      final cx = w / 2;

      return Stack(
        children: [
          // Body silhouette
          Center(
            child: CustomPaint(
              size: Size(w * 0.45, h * 0.92),
              painter: _FrontBodyPainter(),
            ),
          ),

          // FACE label — top center
          _PartLabel(
            label: 'FACE',
            x: cx - 28,
            y: h * 0.01,
            selected: selectedParts.contains('FACE'),
            onTap: () => onToggle('FACE'),
            showPlus: true,
          ),

          // NECK
          _PartLabel(
            label: 'NECK',
            x: cx - 25,
            y: h * 0.14,
            selected: selectedParts.contains('NECK'),
            onTap: () => onToggle('NECK'),
          ),

          // LEFT ARM label (user's left = screen right)
          _PartLabel(
            label: 'LEFT ARM',
            x: w * 0.03,
            y: h * 0.27,
            selected: selectedParts.contains('LEFT ARM'),
            onTap: () => onToggle('LEFT ARM'),
            alignRight: false,
          ),

          // CHEST
          _PartLabel(
            label: 'CHEST',
            x: cx - 26,
            y: h * 0.24,
            selected: selectedParts.contains('CHEST'),
            onTap: () => onToggle('CHEST'),
          ),

          // RIGHT ARM
          _PartLabel(
            label: 'RIGHT ARM',
            x: w * 0.55,
            y: h * 0.27,
            selected: selectedParts.contains('RIGHT ARM'),
            onTap: () => onToggle('RIGHT ARM'),
            alignRight: false,
          ),

          // STOMACH
          _PartLabel(
            label: 'STOMACH',
            x: cx - 34,
            y: h * 0.42,
            selected: selectedParts.contains('STOMACH'),
            onTap: () => onToggle('STOMACH'),
          ),

          // LEFT LEG
          _PartLabel(
            label: 'LEFT LEG',
            x: w * 0.07,
            y: h * 0.65,
            selected: selectedParts.contains('LEFT LEG'),
            onTap: () => onToggle('LEFT LEG'),
            alignRight: false,
          ),

          // RIGHT LEG
          _PartLabel(
            label: 'RIGHT LEG',
            x: w * 0.55,
            y: h * 0.65,
            selected: selectedParts.contains('RIGHT LEG'),
            onTap: () => onToggle('RIGHT LEG'),
            alignRight: false,
          ),
        ],
      );
    });
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Back body diagram
// ──────────────────────────────────────────────────────────────────────────────
class _BackBodyDiagram extends StatelessWidget {
  final Set<String> selectedParts;
  final void Function(String) onToggle;

  const _BackBodyDiagram({
    required this.selectedParts,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      final w = constraints.maxWidth;
      final h = constraints.maxHeight;
      final cx = w / 2;

      return Stack(
        children: [
          Center(
            child: CustomPaint(
              size: Size(w * 0.45, h * 0.92),
              painter: _BackBodyPainter(),
            ),
          ),

          // LEFT SHOULDER
          _PartLabel(
            label: 'Left Shoulder',
            x: w * 0.04,
            y: h * 0.12,
            selected: selectedParts.contains('LEFT SHOULDER'),
            onTap: () => onToggle('LEFT SHOULDER'),
            alignRight: false,
          ),

          // RIGHT SHOULDER
          _PartLabel(
            label: 'Right Shoulder',
            x: w * 0.56,
            y: h * 0.12,
            selected: selectedParts.contains('RIGHT SHOULDER'),
            onTap: () => onToggle('RIGHT SHOULDER'),
            alignRight: false,
          ),

          // UPPER BACK
          _PartLabel(
            label: 'Upper Back',
            x: cx - 36,
            y: h * 0.22,
            selected: selectedParts.contains('UPPER BACK'),
            onTap: () => onToggle('UPPER BACK'),
          ),

          // MID BACK
          _PartLabel(
            label: 'Mid Back',
            x: cx - 28,
            y: h * 0.36,
            selected: selectedParts.contains('MID BACK'),
            onTap: () => onToggle('MID BACK'),
          ),

          // LOWER BACK
          _PartLabel(
            label: 'Lower Back',
            x: cx - 34,
            y: h * 0.49,
            selected: selectedParts.contains('LOWER BACK'),
            onTap: () => onToggle('LOWER BACK'),
          ),

          // BUTTOCKS
          _PartLabel(
            label: 'Buttocks',
            x: cx - 28,
            y: h * 0.60,
            selected: selectedParts.contains('BUTTOCKS'),
            onTap: () => onToggle('BUTTOCKS'),
          ),
        ],
      );
    });
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tappable part label
// ──────────────────────────────────────────────────────────────────────────────
class _PartLabel extends StatelessWidget {
  final String label;
  final double x;
  final double y;
  final bool selected;
  final VoidCallback onTap;
  final bool alignRight;
  final bool showPlus;

  const _PartLabel({
    required this.label,
    required this.x,
    required this.y,
    required this.selected,
    required this.onTap,
    this.alignRight = true,
    this.showPlus = false,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: x,
      top: y,
      child: GestureDetector(
        onTap: onTap,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (showPlus) ...[
              Container(
                width: 18,
                height: 18,
                decoration: BoxDecoration(
                  color: selected ? AppColors.primary : const Color(0xFF60A5FA),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.add, color: Colors.white, size: 12),
              ),
              const SizedBox(width: 4),
            ],
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
              decoration: BoxDecoration(
                color: selected
                    ? AppColors.primary
                    : Colors.white.withOpacity(0.9),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(
                  color: selected ? AppColors.primary : AppColors.border,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.06),
                    blurRadius: 4,
                  ),
                ],
              ),
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  color: selected ? Colors.white : AppColors.textSecondary,
                ),
              ),
            ),
            if (!showPlus) ...[
              const SizedBox(width: 4),
              Container(
                width: 16,
                height: 16,
                decoration: BoxDecoration(
                  color: selected ? AppColors.primary : const Color(0xFF60A5FA),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  selected ? Icons.check : Icons.add,
                  color: Colors.white,
                  size: 10,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// CustomPainter — front silhouette
// ──────────────────────────────────────────────────────────────────────────────
class _FrontBodyPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFFBFDBFE)
      ..style = PaintingStyle.fill;

    final w = size.width;
    final h = size.height;

    // Head
    canvas.drawCircle(Offset(w / 2, h * 0.07), w * 0.14, paint);

    // Neck
    final neck = RRect.fromRectAndRadius(
      Rect.fromCenter(
          center: Offset(w / 2, h * 0.155), width: w * 0.18, height: h * 0.06),
      const Radius.circular(6),
    );
    canvas.drawRRect(neck, paint);

    // Torso
    final torso = RRect.fromRectAndRadius(
      Rect.fromLTWH(w * 0.1, h * 0.18, w * 0.8, h * 0.32),
      const Radius.circular(10),
    );
    canvas.drawRRect(torso, paint);

    // Left arm
    final leftArm = RRect.fromRectAndRadius(
      Rect.fromLTWH(-w * 0.02, h * 0.18, w * 0.15, h * 0.32),
      const Radius.circular(8),
    );
    canvas.drawRRect(leftArm, paint);

    // Right arm
    final rightArm = RRect.fromRectAndRadius(
      Rect.fromLTWH(w * 0.87, h * 0.18, w * 0.15, h * 0.32),
      const Radius.circular(8),
    );
    canvas.drawRRect(rightArm, paint);

    // Left leg
    final leftLeg = RRect.fromRectAndRadius(
      Rect.fromLTWH(w * 0.13, h * 0.5, w * 0.31, h * 0.48),
      const Radius.circular(8),
    );
    canvas.drawRRect(leftLeg, paint);

    // Right leg
    final rightLeg = RRect.fromRectAndRadius(
      Rect.fromLTWH(w * 0.56, h * 0.5, w * 0.31, h * 0.48),
      const Radius.circular(8),
    );
    canvas.drawRRect(rightLeg, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// ──────────────────────────────────────────────────────────────────────────────
// CustomPainter — back silhouette
// ──────────────────────────────────────────────────────────────────────────────
class _BackBodyPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFFBFDBFE)
      ..style = PaintingStyle.fill;

    final w = size.width;
    final h = size.height;

    // Head
    canvas.drawCircle(Offset(w / 2, h * 0.07), w * 0.14, paint);

    // Neck
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
            center: Offset(w / 2, h * 0.155), width: w * 0.18, height: h * 0.06),
        const Radius.circular(6),
      ),
      paint,
    );

    // Torso (wider shoulders)
    final torso = Path()
      ..moveTo(w * 0.05, h * 0.185)
      ..lineTo(w * 0.95, h * 0.185)
      ..lineTo(w * 0.88, h * 0.5)
      ..lineTo(w * 0.12, h * 0.5)
      ..close();
    canvas.drawPath(torso, paint);

    // Lower torso
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(w * 0.15, h * 0.49, w * 0.7, h * 0.18),
        const Radius.circular(8),
      ),
      paint,
    );

    // Left arm
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(-w * 0.02, h * 0.18, w * 0.1, h * 0.3),
        const Radius.circular(8),
      ),
      paint,
    );

    // Right arm
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(w * 0.92, h * 0.18, w * 0.1, h * 0.3),
        const Radius.circular(8),
      ),
      paint,
    );

    // Left leg
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(w * 0.16, h * 0.66, w * 0.3, h * 0.32),
        const Radius.circular(8),
      ),
      paint,
    );

    // Right leg
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(w * 0.54, h * 0.66, w * 0.3, h * 0.32),
        const Radius.circular(8),
      ),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
