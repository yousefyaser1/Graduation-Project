import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';

class CaptureScreen extends StatefulWidget {
  final String bodyPart;

  const CaptureScreen({super.key, required this.bodyPart});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final _picker = ImagePicker();
  String? _imagePath;
  bool _checkingQuality = false;

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? file = await _picker.pickImage(
        source: source,
        imageQuality: 90,
        maxWidth: 1024,
      );
      if (file != null && mounted) {
        setState(() => _imagePath = file.path);
      }
    } on PlatformException catch (e) {
      if (!mounted) return;
      final isPermission = e.code == 'camera_access_denied' ||
          e.code == 'photo_access_denied' ||
          (e.message?.toLowerCase().contains('permission') ?? false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(isPermission
              ? 'Permission denied. Enable ${source == ImageSource.camera ? 'camera' : 'photo'} access in Settings.'
              : 'Could not open ${source == ImageSource.camera ? 'camera' : 'gallery'}. Please try again.'),
          behavior: SnackBarBehavior.floating,
          backgroundColor: AppColors.error,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          action: isPermission
              ? SnackBarAction(
                  label: 'Open Settings',
                  textColor: Colors.white,
                  onPressed: () => openAppSettings(),
                )
              : null,
        ),
      );
    }
  }

  // Returns a problem description string, or null if quality is acceptable.
  Future<String?> _getQualityIssue(String path) async {
    try {
      final bytes = await File(path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return null;

      final small = img.copyResize(
        decoded.format == img.Format.uint8 ? decoded : decoded.convert(format: img.Format.uint8, numChannels: 3),
        width: 100,
        height: 100,
        interpolation: img.Interpolation.linear,
      );

      // Laplacian-based sharpness (average absolute Laplacian per pixel)
      double pixLum(img.Pixel p) =>
          p.r.toDouble() * 0.299 + p.g.toDouble() * 0.587 + p.b.toDouble() * 0.114;

      double lapSum = 0;
      int lapCount = 0;
      for (int y = 1; y < 99; y++) {
        for (int x = 1; x < 99; x++) {
          final c = pixLum(small.getPixel(x, y));
          final lap = (4 * c
              - pixLum(small.getPixel(x, y - 1))
              - pixLum(small.getPixel(x, y + 1))
              - pixLum(small.getPixel(x - 1, y))
              - pixLum(small.getPixel(x + 1, y))).abs();
          lapSum += lap;
          lapCount++;
        }
      }
      final sharpness = lapCount > 0 ? lapSum / lapCount : 100.0;

      // Average luminance for brightness check
      double lumSum = 0;
      for (int y = 0; y < 100; y++) {
        for (int x = 0; x < 100; x++) {
          final p = small.getPixel(x, y);
          lumSum += p.r.toDouble() * 0.299 + p.g.toDouble() * 0.587 + p.b.toDouble() * 0.114;
        }
      }
      final brightness = lumSum / 10000.0;

      // Skin-coverage check: the AI is trained on close-up skin photos, so a
      // frame that is mostly background (wall, desk, wide room shot) is
      // out-of-distribution and produces unreliable results. Same lenient
      // warm-foreground heuristic as the training patch extractor: skin of any
      // tone is warm (R >= B), not near-black, not blown out, and not flat
      // bright grey. Lenient on purpose — it must not reject dark skin tones.
      int skinCount = 0;
      for (int y = 0; y < 100; y++) {
        for (int x = 0; x < 100; x++) {
          final p = small.getPixel(x, y);
          final r = p.r.toDouble(), g = p.g.toDouble(), b = p.b.toDouble();
          final v = (r + g + b) / 3.0;
          final mx = math.max(r, math.max(g, b));
          final mn = math.min(r, math.min(g, b));
          final sat = mx > 0 ? (mx - mn) / mx : 0.0;
          final isSkin = v > 30 &&
              v < 248 &&
              r >= b - 5 &&
              !(sat < 0.05 && v > 153);
          if (isSkin) skinCount++;
        }
      }
      final skinFraction = skinCount / 10000.0;

      final issues = <String>[];
      if (sharpness < 3.0) issues.add('Image appears blurry — hold the camera steady');
      if (brightness < 40.0) issues.add('Image is too dark — use better lighting');
      if (brightness > 220.0) issues.add('Image is overexposed — reduce glare');
      if (skinFraction < 0.55) {
        issues.add(
            'Much of the frame does not look like skin — move closer so the skin area fills the photo');
      }

      return issues.isEmpty ? null : issues.join('\n');
    } catch (_) {
      return null; // never block on error
    }
  }

  Future<void> _analyzeWithQualityCheck() async {
    if (_imagePath == null || _checkingQuality) return;
    setState(() => _checkingQuality = true);

    final issue = await _getQualityIssue(_imagePath!);
    if (!mounted) return;
    setState(() => _checkingQuality = false);

    if (issue != null) {
      final proceed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: const Row(children: [
            Icon(Icons.warning_amber_rounded, color: Color(0xFFF59E0B), size: 22),
            SizedBox(width: 8),
            Text('Image Quality Warning',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
          ]),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ...issue.split('\n').map((line) => Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Text('• $line',
                        style: const TextStyle(
                            fontSize: 13, color: AppColors.textPrimary, height: 1.4)),
                  )),
              const SizedBox(height: 10),
              const Text(
                'Poor image quality can reduce diagnosis accuracy. For best results, retake the photo in good lighting.',
                style: TextStyle(fontSize: 12, color: AppColors.textSecondary, height: 1.45),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(ctx, false);
                setState(() => _imagePath = null);
              },
              child: const Text('Retake'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.pop(ctx, true),
              style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1E3A5F),
                  foregroundColor: Colors.white),
              child: const Text('Continue Anyway'),
            ),
          ],
        ),
      );
      if (proceed != true || !mounted) return;
    }

    if (mounted) {
      context.push(AppRoutes.analyzing, extra: {
        'bodyPart': widget.bodyPart,
        'imagePath': _imagePath!,
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool imageSelected = _imagePath != null;

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
          widget.bodyPart.isNotEmpty
              ? 'Capture – ${widget.bodyPart}'
              : 'Capture Photo',
          style: const TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
        child: Column(
          children: [
            // Preview area
            Expanded(
              flex: 5,
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: imageSelected
                      ? Colors.black
                      : const Color(0xFFF1F5F9),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: imageSelected ? AppColors.primary : AppColors.border,
                    width: imageSelected ? 2 : 1,
                  ),
                ),
                child: imageSelected
                    ? _buildImagePreview()
                    : _buildEmptyViewfinder(),
              ),
            ),
            const SizedBox(height: 16),

            // Tip
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFFFFFBEB),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: const Color(0xFFFCD34D)),
              ),
              child: const Row(
                children: [
                  Icon(Icons.lightbulb_outline,
                      color: Color(0xFFF59E0B), size: 16),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Fill the frame with the affected area, use natural light, and hold steady for best results.',
                      style: TextStyle(
                          fontSize: 12,
                          color: Color(0xFF92400E),
                          height: 1.4),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Camera & Gallery buttons
            Row(
              children: [
                Expanded(
                  child: _ActionButton(
                    icon: Icons.camera_alt_outlined,
                    label: 'Take Photo',
                    onTap: () => _pickImage(ImageSource.camera),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: _ActionButton(
                    icon: Icons.photo_library_outlined,
                    label: 'Gallery',
                    onTap: () => _pickImage(ImageSource.gallery),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),

            // Analyze button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: (imageSelected && !_checkingQuality)
                    ? _analyzeWithQualityCheck
                    : null,
                icon: _checkingQuality
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.search_rounded, size: 20),
                label: Text(_checkingQuality ? 'Checking quality...' : 'Analyze Scan'),
                style: ElevatedButton.styleFrom(
                  disabledBackgroundColor: AppColors.border,
                  disabledForegroundColor: AppColors.textSecondary,
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyViewfinder() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 80,
          height: 80,
          decoration: const BoxDecoration(
            color: AppColors.primaryLight,
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.camera_alt_outlined,
              color: AppColors.primary, size: 36),
        ),
        const SizedBox(height: 16),
        const Text(
          'No image selected',
          style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: AppColors.textPrimary),
        ),
        const SizedBox(height: 6),
        const Text(
          'Tap "Take Photo" or "Gallery" below',
          style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
        ),
        const SizedBox(height: 24),
        SizedBox(
          width: 120,
          height: 120,
          child: CustomPaint(painter: _ViewfinderPainter()),
        ),
      ],
    );
  }

  Widget _buildImagePreview() {
    return Stack(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(19),
          child: SizedBox.expand(
            child: Image.file(
              File(_imagePath!),
              fit: BoxFit.cover,
            ),
          ),
        ),
        // Ready badge
        Positioned(
          top: 14,
          right: 14,
          child: Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: AppColors.success,
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.check_circle, color: Colors.white, size: 14),
                SizedBox(width: 4),
                Text('Ready',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w600)),
              ],
            ),
          ),
        ),
        // Retake button
        Positioned(
          bottom: 14,
          right: 14,
          child: GestureDetector(
            onTap: () => setState(() => _imagePath = null),
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.refresh, color: Colors.white, size: 14),
                  SizedBox(width: 4),
                  Text('Retake',
                      style:
                          TextStyle(color: Colors.white, fontSize: 12)),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          color: AppColors.primaryLight,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: AppColors.primary, size: 20),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.primary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ViewfinderPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.primary
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    const len = 20.0;
    final w = size.width;
    final h = size.height;

    canvas.drawLine(const Offset(0, len), const Offset(0, 0), paint);
    canvas.drawLine(const Offset(0, 0), const Offset(len, 0), paint);
    canvas.drawLine(Offset(w - len, 0), Offset(w, 0), paint);
    canvas.drawLine(Offset(w, 0), Offset(w, len), paint);
    canvas.drawLine(Offset(0, h - len), Offset(0, h), paint);
    canvas.drawLine(Offset(0, h), Offset(len, h), paint);
    canvas.drawLine(Offset(w - len, h), Offset(w, h), paint);
    canvas.drawLine(Offset(w, h - len), Offset(w, h), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
