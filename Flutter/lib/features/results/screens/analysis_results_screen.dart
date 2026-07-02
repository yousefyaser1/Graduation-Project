import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:share_plus/share_plus.dart';
import 'package:intl/intl.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../core/widgets/app_bottom_nav.dart';
import '../../../core/widgets/xai_attention_legend.dart';
import '../../../models/scan_result.dart';
import '../../../providers/scan_provider.dart';
import '../../../providers/user_provider.dart';
import '../../../models/user.dart';

class AnalysisResultsScreen extends ConsumerStatefulWidget {
  final ScanResult? scan;

  const AnalysisResultsScreen({super.key, this.scan});

  @override
  ConsumerState<AnalysisResultsScreen> createState() =>
      _AnalysisResultsScreenState();
}

class _AnalysisResultsScreenState
    extends ConsumerState<AnalysisResultsScreen> {
  bool _saved = false;
  bool _saving = false;
  bool _exportingPdf = false;
  bool _showPipelineDetails = true;

  // When top-class confidence is below this, show the "Other" slot
  static const double _otherThreshold = 0.60;

  // Sequential pipeline reveal flags
  bool _showVae = false;
  bool _showCnn = false;
  bool _showScoreCam = false;
  bool _showVerdict = false;
  bool _showExtras = false;

  // Which class's Score-CAM heatmap the specialist is viewing (null = predicted)
  String? _selectedHeatmapClass;

  // ── Disease information database ─────────────────────────────────────────────
  static const _diseaseData = <String, Map<String, Object>>{
    'Acne': {
      'description':
          'Acne occurs when hair follicles become clogged with oil and dead skin cells, producing whiteheads, blackheads, or pimples. It is most common among teenagers but can affect people of all ages.',
      'symptoms': <String>[
        'Pimples and pustules on face, back, or chest',
        'Blackheads and whiteheads',
        'Oily, shiny skin surface',
        'Possible scarring in severe cases',
      ],
      'advice':
          'Consult a dermatologist if over-the-counter treatments are ineffective, or if acne causes significant distress or scarring. Avoid squeezing lesions as this worsens inflammation.',
    },
    'Eczema': {
      'description':
          'Eczema (atopic dermatitis) is a chronic inflammatory condition causing dry, intensely itchy patches of skin. It often flares in response to allergens, irritants, sweat, or stress.',
      'symptoms': <String>[
        'Intense itching, especially at night',
        'Dry, cracked, or thickened skin',
        'Red or brownish-grey patches',
        'Small raised bumps that may weep fluid when scratched',
      ],
      'advice':
          'Avoid known triggers, moisturize immediately after bathing, and use mild unscented soaps. See a doctor for prescription antihistamines, topical corticosteroids, or biologics for severe cases.',
    },
    'Tinea': {
      'description':
          'Tinea (ringworm) is a contagious fungal infection of the skin caused by dermatophytes. Despite the name, no worm is involved. It can affect the body, scalp, feet (athlete\'s foot), or nails.',
      'symptoms': <String>[
        'Ring-shaped, red, scaly rash with clear centre',
        'Persistent itching and inflammation',
        'Hair loss or bald patches if the scalp is affected',
        'Nail thickening or discoloration (onychomycosis)',
      ],
      'advice':
          'Topical antifungal creams are effective for most body infections. Avoid sharing towels, clothing, combs, or footwear. Scalp or nail infections may require oral antifungal medication.',
    },
    'No Disease Detected': {
      'description':
          'The AI screening found no patterns consistent with Acne, Eczema, or Tinea. This system screens for these three conditions only — other skin conditions are outside its detection range.',
      'symptoms': <String>[
        'Disease-screening gate scored below its action threshold',
        'No region looked disease-like enough to classify',
      ],
      'advice':
          'Continue regular self-examination and see a dermatologist annually, or sooner if you notice new, changing, or concerning skin changes.',
    },
    'Inconclusive': {
      'description':
          'The screening could not make a confident call on this photo. It likely needs better framing or lighting, or it shows skin the AI cannot read reliably (it screens for Acne, Eczema, and Tinea only).',
      'symptoms': <String>[
        'Disease probability landed in the uncertain mid-range',
        'Not clearly normal, not clearly one of the three conditions',
      ],
      'advice':
          'Retake the photo: fill the frame with the skin area, use good even lighting, and hold the camera steady and in focus. If it stays inconclusive or you are concerned, see a dermatologist.',
    },
  };

  @override
  void initState() {
    super.initState();
    // Every navigation path now hands us a scan that's already in the DB
    // (fresh scans are auto-saved in AnalyzingScreen; History/Home/etc. load
    // from the DB). Reflect that so the button reads "Saved!" rather than
    // implying the scan still needs to be saved.
    _saved = widget.scan != null;
    _startReveal();
  }

  Future<void> _startReveal() async {
    await Future.delayed(const Duration(milliseconds: 350));
    if (mounted) setState(() => _showVae = true);
    await Future.delayed(const Duration(milliseconds: 700));
    if (mounted) setState(() => _showCnn = true);
    await Future.delayed(const Duration(milliseconds: 700));
    if (mounted) setState(() => _showScoreCam = true);
    await Future.delayed(const Duration(milliseconds: 500));
    if (mounted) setState(() => _showVerdict = true);
    await Future.delayed(const Duration(milliseconds: 400));
    if (mounted) setState(() => _showExtras = true);
  }

  bool get _isHighRisk {
    final dx = widget.scan?.diagnosis ?? '';
    return dx.contains('Melanoma') ||
        dx.contains('Carcinoma') ||
        dx.contains('Lymphoma');
  }

  void _share() {
    final scan = widget.scan;
    if (scan == null) return;
    final confidence = (scan.confidence * 100).round();
    final risk = _isHighRisk ? 'High Risk' : 'Low Risk';
    final text =
        'SkinScan AI Result\n\nDiagnosis: ${scan.diagnosis}\nConfidence: $confidence%\nRisk Level: $risk\nBody Part: ${scan.bodyPart}\nDate: ${scan.timestamp.toLocal()}\n\nThis is an AI-assisted screening result. Please consult a dermatologist for a professional diagnosis.';
    Share.share(text, subject: 'My SkinScan AI Result');
  }

  Future<void> _saveToHistory() async {
    if (_saved || _saving || widget.scan == null) return;
    setState(() => _saving = true);
    await ref.saveScan(widget.scan!);
    if (mounted) {
      setState(() {
        _saved = true;
        _saving = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Saved to history'),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  // ── PDF export ────────────────────────────────────────────────────────────

  Future<void> _exportPdf() async {
    final scan = widget.scan;
    if (scan == null) return;
    setState(() => _exportingPdf = true);
    try {
      final pdfBytes = await _generatePdf(scan, ref.read(userProvider));
      final filename =
          'SkinScan_${scan.diagnosis.replaceAll(' ', '_')}_${DateFormat('yyyyMMdd').format(scan.timestamp)}.pdf';
      await Printing.sharePdf(bytes: pdfBytes, filename: filename);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text('PDF export failed: $e'),
              behavior: SnackBarBehavior.floating),
        );
      }
    } finally {
      if (mounted) setState(() => _exportingPdf = false);
    }
  }

  Future<Uint8List> _generatePdf(ScanResult scan, User? user) async {
    final pdf = pw.Document();
    final isNormal = scan.diagnosis == 'No Disease Detected';
    final confidence = (scan.confidence * 100).round();
    final info = _diseaseData[scan.diagnosis];

    pw.MemoryImage? originalImg;
    pw.MemoryImage? heatmapImg;
    if (File(scan.imagePath).existsSync()) {
      originalImg = pw.MemoryImage(await File(scan.imagePath).readAsBytes());
    }
    if (scan.heatmapPath != null && File(scan.heatmapPath!).existsSync()) {
      heatmapImg = pw.MemoryImage(await File(scan.heatmapPath!).readAsBytes());
    }

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(40),
        header: (_) => pw.Container(
          padding: const pw.EdgeInsets.only(bottom: 10),
          decoration: const pw.BoxDecoration(
              border: pw.Border(
                  bottom: pw.BorderSide(color: PdfColors.grey300))),
          child: pw.Row(
            mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
            children: [
              pw.Text('SkinScan AI',
                  style: pw.TextStyle(
                      fontSize: 17,
                      fontWeight: pw.FontWeight.bold,
                      color: PdfColors.blue700)),
              pw.Text('Dermatological Screening Report',
                  style: const pw.TextStyle(
                      fontSize: 9, color: PdfColors.grey600)),
            ],
          ),
        ),
        footer: (ctx) => pw.Container(
          padding: const pw.EdgeInsets.only(top: 8),
          decoration: const pw.BoxDecoration(
              border: pw.Border(
                  top: pw.BorderSide(color: PdfColors.grey300))),
          child: pw.Row(
            mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
            children: [
              pw.Text(
                  'AI-assisted screening only — not a medical diagnosis.',
                  style: const pw.TextStyle(
                      fontSize: 7, color: PdfColors.grey500)),
              pw.Text('Page ${ctx.pageNumber} of ${ctx.pagesCount}',
                  style: const pw.TextStyle(
                      fontSize: 7, color: PdfColors.grey500)),
            ],
          ),
        ),
        build: (pw.Context ctx) => [
          // Patient/scan header
          pw.Container(
            padding: const pw.EdgeInsets.all(14),
            decoration: const pw.BoxDecoration(
              color: PdfColors.blue50,
              borderRadius:
                  pw.BorderRadius.all(pw.Radius.circular(8)),
            ),
            child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  if (user != null) ...[
                    pw.Text(user.name,
                        style: pw.TextStyle(
                            fontSize: 14,
                            fontWeight: pw.FontWeight.bold)),
                    if (user.age != null || user.gender != null)
                      pw.Text(
                          [
                            if (user.age != null) 'Age: ${user.age}',
                            if (user.gender != null)
                              'Gender: ${user.gender}',
                          ].join('  •  '),
                          style: const pw.TextStyle(
                              fontSize: 10,
                              color: PdfColors.grey600)),
                    pw.SizedBox(height: 6),
                  ],
                  pw.Row(children: [
                    pw.Text('Body Part: ',
                        style: pw.TextStyle(
                            fontSize: 10,
                            fontWeight: pw.FontWeight.bold)),
                    pw.Text(scan.bodyPart,
                        style: const pw.TextStyle(fontSize: 10)),
                    pw.SizedBox(width: 16),
                    pw.Text('Date: ',
                        style: pw.TextStyle(
                            fontSize: 10,
                            fontWeight: pw.FontWeight.bold)),
                    pw.Text(
                        DateFormat('dd MMM yyyy, HH:mm')
                            .format(scan.timestamp.toLocal()),
                        style: const pw.TextStyle(fontSize: 10)),
                  ]),
                ]),
          ),
          pw.SizedBox(height: 14),

          // Images
          if (originalImg != null || heatmapImg != null) ...[
            pw.Row(children: [
              if (originalImg != null)
                pw.Expanded(
                    child: pw.Column(children: [
                  pw.Text('Original Image',
                      style: pw.TextStyle(
                          fontSize: 9,
                          fontWeight: pw.FontWeight.bold,
                          color: PdfColors.grey700)),
                  pw.SizedBox(height: 4),
                  pw.Image(originalImg,
                      height: 150, fit: pw.BoxFit.cover),
                ])),
              if (originalImg != null && heatmapImg != null)
                pw.SizedBox(width: 10),
              if (heatmapImg != null)
                pw.Expanded(
                    child: pw.Column(children: [
                  pw.Text('Score-CAM Heatmap',
                      style: pw.TextStyle(
                          fontSize: 9,
                          fontWeight: pw.FontWeight.bold,
                          color: PdfColors.grey700)),
                  pw.SizedBox(height: 4),
                  pw.Image(heatmapImg,
                      height: 150, fit: pw.BoxFit.cover),
                ])),
            ]),
            pw.SizedBox(height: 14),
          ],

          // Diagnosis card
          pw.Container(
            padding: const pw.EdgeInsets.all(14),
            decoration: pw.BoxDecoration(
              border: pw.Border.all(
                  color: isNormal
                      ? PdfColors.green400
                      : PdfColors.blue300),
              borderRadius:
                  const pw.BorderRadius.all(pw.Radius.circular(8)),
            ),
            child: pw.Row(children: [
              pw.Expanded(
                  child: pw.Column(
                      crossAxisAlignment:
                          pw.CrossAxisAlignment.start,
                      children: [
                    pw.Text('AI Diagnosis',
                        style: const pw.TextStyle(
                            fontSize: 9,
                            color: PdfColors.grey600)),
                    pw.SizedBox(height: 4),
                    pw.Text(scan.diagnosis,
                        style: pw.TextStyle(
                            fontSize: 18,
                            fontWeight: pw.FontWeight.bold,
                            color: isNormal
                                ? PdfColors.green800
                                : PdfColors.blue900)),
                    pw.SizedBox(height: 3),
                    pw.Text('Confidence: $confidence%',
                        style: const pw.TextStyle(
                            fontSize: 11,
                            color: PdfColors.grey700)),
                  ])),
              pw.Container(
                padding: const pw.EdgeInsets.symmetric(
                    horizontal: 10, vertical: 5),
                decoration: pw.BoxDecoration(
                  color: isNormal
                      ? PdfColors.green100
                      : PdfColors.lightBlue50,
                  borderRadius: const pw.BorderRadius.all(
                      pw.Radius.circular(20)),
                ),
                child: pw.Text(
                    isNormal ? 'Normal' : 'Low Risk',
                    style: pw.TextStyle(
                        fontSize: 10,
                        fontWeight: pw.FontWeight.bold,
                        color: isNormal
                            ? PdfColors.green800
                            : PdfColors.blue800)),
              ),
            ]),
          ),
          pw.SizedBox(height: 12),

          // Class probabilities
          if (scan.classProbabilities.isNotEmpty) ...[
            pw.Text('Class Probabilities',
                style: pw.TextStyle(
                    fontSize: 10,
                    fontWeight: pw.FontWeight.bold,
                    color: PdfColors.grey700)),
            pw.SizedBox(height: 5),
            ...scan.classProbabilities.entries.map((e) =>
                pw.Padding(
                  padding: const pw.EdgeInsets.only(bottom: 4),
                  child: pw.Row(children: [
                    pw.SizedBox(
                        width: 60,
                        child: pw.Text(e.key,
                            style:
                                const pw.TextStyle(fontSize: 10))),
                    pw.Expanded(
                        child: pw.LinearProgressIndicator(
                      value: e.value.clamp(0.0, 1.0),
                      backgroundColor: PdfColors.grey200,
                      valueColor: PdfColors.blue400,
                    )),
                    pw.SizedBox(width: 6),
                    pw.Text(
                        '${(e.value * 100).round()}%',
                        style: pw.TextStyle(
                            fontSize: 10,
                            fontWeight: pw.FontWeight.bold)),
                  ]),
                )),
            pw.SizedBox(height: 12),
          ],

          // Gate stage
          if (scan.anomalyRatio != null) ...[
            pw.Text('Stage 1 — Normal vs Disease Gate',
                style: pw.TextStyle(
                    fontSize: 10,
                    fontWeight: pw.FontWeight.bold,
                    color: PdfColors.grey700)),
            pw.SizedBox(height: 4),
            pw.Text(
                'Disease probability: ${(scan.anomalyRatio! * 100).toStringAsFixed(1)}%  (normal below 60%; 60–90% inconclusive; 90%+ classified)',
                style: const pw.TextStyle(
                    fontSize: 10, color: PdfColors.grey600)),
            pw.Text(
                isNormal
                    ? 'Result: Below the 60% threshold — classified as normal skin.'
                    : 'Result: Above the 60% threshold — flagged for review.',
                style: pw.TextStyle(
                    fontSize: 10,
                    color: isNormal
                        ? PdfColors.green700
                        : PdfColors.orange800)),
            pw.SizedBox(height: 12),
          ],

          // Per-stage timing
          if (scan.vaeMs != null) ...[
            pw.Text('On-device Inference Timing',
                style: pw.TextStyle(
                    fontSize: 10,
                    fontWeight: pw.FontWeight.bold,
                    color: PdfColors.grey700)),
            pw.SizedBox(height: 4),
            pw.Text(
                'Preprocess: ${scan.preprocessMs ?? 0}ms  |  Gate: ${scan.vaeMs}ms  |  CNN: ${scan.cnnMs ?? 0}ms  |  Score-CAM: ${scan.scoreCamMs ?? 0}ms  |  Total: ${(scan.preprocessMs ?? 0) + (scan.vaeMs ?? 0) + (scan.cnnMs ?? 0) + (scan.scoreCamMs ?? 0)}ms',
                style: const pw.TextStyle(
                    fontSize: 10, color: PdfColors.grey600)),
            pw.SizedBox(height: 12),
          ],

          // About the condition
          if (info != null) ...[
            pw.Container(
              padding: const pw.EdgeInsets.all(12),
              decoration: pw.BoxDecoration(
                color: PdfColors.grey50,
                borderRadius: const pw.BorderRadius.all(
                    pw.Radius.circular(8)),
                border:
                    pw.Border.all(color: PdfColors.grey300),
              ),
              child: pw.Column(
                  crossAxisAlignment:
                      pw.CrossAxisAlignment.start,
                  children: [
                    pw.Text('About: ${scan.diagnosis}',
                        style: pw.TextStyle(
                            fontSize: 11,
                            fontWeight: pw.FontWeight.bold)),
                    pw.SizedBox(height: 5),
                    pw.Text(info['description'] as String,
                        style: const pw.TextStyle(
                            fontSize: 10, lineSpacing: 2)),
                    pw.SizedBox(height: 8),
                    pw.Text('Common Symptoms:',
                        style: pw.TextStyle(
                            fontSize: 10,
                            fontWeight: pw.FontWeight.bold)),
                    ...(info['symptoms'] as List<String>)
                        .map((s) => pw.Padding(
                              padding: const pw.EdgeInsets.only(
                                  left: 8, top: 2),
                              child: pw.Text('• $s',
                                  style: const pw.TextStyle(
                                      fontSize: 10)),
                            )),
                    pw.SizedBox(height: 6),
                    pw.Text('Advice:',
                        style: pw.TextStyle(
                            fontSize: 10,
                            fontWeight: pw.FontWeight.bold)),
                    pw.Text(info['advice'] as String,
                        style: const pw.TextStyle(
                            fontSize: 10, lineSpacing: 2)),
                  ]),
            ),
            pw.SizedBox(height: 12),
          ],

          // Disclaimer
          pw.Container(
            padding: const pw.EdgeInsets.all(10),
            decoration: pw.BoxDecoration(
              color: PdfColors.yellow50,
              borderRadius:
                  const pw.BorderRadius.all(pw.Radius.circular(8)),
              border: pw.Border.all(color: PdfColors.yellow300),
            ),
            child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text('Medical Disclaimer',
                      style: pw.TextStyle(
                          fontSize: 9,
                          fontWeight: pw.FontWeight.bold,
                          color: PdfColors.orange800)),
                  pw.SizedBox(height: 4),
                  pw.Text(
                      'This report was generated by SkinScan AI, an educational screening tool. '
                      'It is NOT a medical diagnosis and must NOT replace consultation with a qualified '
                      'dermatologist or healthcare provider. Measured accuracy is ~90% on a curated '
                      'Acne/Eczema/Tinea test set and is typically lower on real-world photos; it cannot '
                      'detect healthy skin or conditions outside these three. All processing occurred '
                      'on-device — no data was transmitted.',
                      style: const pw.TextStyle(
                          fontSize: 8,
                          color: PdfColors.orange900,
                          lineSpacing: 2)),
                ]),
          ),
        ],
      ),
    );

    return pdf.save();
  }

  // ── Medical history context note ─────────────────────────────────────────

  String _getContextNote(
      String diagnosis, String skinType, List<String> conditions) {
    final diag = diagnosis.toLowerCase();
    final type = skinType.toLowerCase();
    bool hasCond(String c) =>
        conditions.any((x) => x.toLowerCase().contains(c));

    if (diag.contains('acne')) {
      if (type.contains('oily')) {
        return 'Oily skin overproduces sebum, a key contributor to acne flare-ups.';
      }
      if (hasCond('acne')) {
        return 'This diagnosis is consistent with your previously reported acne history.';
      }
    }
    if (diag.contains('eczema')) {
      if (hasCond('eczema') || hasCond('atopic')) {
        return 'This aligns with your previously reported eczema / atopic dermatitis history.';
      }
      if (type.contains('dry') || type.contains('sensitive')) {
        return 'Dry or sensitive skin is a common risk factor for eczema flare-ups.';
      }
    }
    if (diag.contains('tinea') || diag.contains('ringworm')) {
      return 'Tinea is contagious — avoid sharing towels, footwear, or clothing with others.';
    }
    return '';
  }

  // ── Normal verdict section ────────────────────────────────────────────────

  Widget _buildNormalVerdictSection(ScanResult scan) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // Green header card
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF15803D), Color(0xFF22C55E)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(children: [
          Container(
            width: 60,
            height: 60,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.check_circle_outline,
                color: Colors.white, size: 32),
          ),
          const SizedBox(width: 16),
          const Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('No Condition Detected',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700)),
              SizedBox(height: 4),
              Text('Skin appeared normal to the AI screening system.',
                  style: TextStyle(color: Colors.white70, fontSize: 12)),
            ]),
          ),
        ]),
      ),
      const SizedBox(height: 12),

      // Explanation card
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFBBF7D0)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('What this means',
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textSecondary)),
          const SizedBox(height: 8),
          const Text(
            'The normal-vs-disease gate scored this skin below its action threshold — no region was disease-like enough to flag as Acne, Eczema, or Tinea. The disease-evidence map highlights where the most disease-like skin was, if any.',
            style: TextStyle(
                fontSize: 12, color: AppColors.textPrimary, height: 1.55),
          ),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFFFFBEB),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0xFFFCD34D)),
            ),
            child: const Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Icon(Icons.info_outline, size: 14, color: Color(0xFFF59E0B)),
              SizedBox(width: 8),
              Expanded(
                  child: Text(
                'This system screens for Acne, Eczema, and Tinea only. Other skin conditions are outside its detection range.',
                style: TextStyle(fontSize: 11, color: Color(0xFF92400E)),
              )),
            ]),
          ),
        ]),
      ),
      const SizedBox(height: 20),

      // Action buttons
      Row(children: [
        Expanded(
          child: ElevatedButton(
            onPressed: _saved ? null : (_saving ? null : _saveToHistory),
            style: ElevatedButton.styleFrom(
              backgroundColor: _saved ? AppColors.success : const Color(0xFF15803D),
              disabledBackgroundColor:
                  _saved ? AppColors.success : AppColors.border,
              minimumSize: const Size(double.infinity, 48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: _saving
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : Text(_saved ? 'Saved!' : 'Save to History',
                    style: const TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w600)),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton(
            onPressed: () => context.go(AppRoutes.home),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryLight,
              foregroundColor: AppColors.primary,
              elevation: 0,
              minimumSize: const Size(double.infinity, 48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Scan Again',
                style:
                    TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          ),
        ),
      ]),
      const SizedBox(height: 10),
      _buildPdfButton(),
    ]);
  }

  /// Verdict card for the gate "gray zone" — the AI abstained rather than risk a
  /// confident wrong label (the old "normal skin → Tinea" failure). Amber, not
  /// red/green: it's neither a clear condition nor a clean bill of health.
  Widget _buildInconclusiveVerdictSection(ScanResult scan) {
    final pct = ((scan.anomalyRatio ?? 0) * 100).round();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // Amber header card
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFFB45309), Color(0xFFF59E0B)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(children: [
          Container(
            width: 60,
            height: 60,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.help_outline,
                color: Colors.white, size: 32),
          ),
          const SizedBox(width: 16),
          const Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Inconclusive',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700)),
              SizedBox(height: 4),
              Text("Couldn't get a confident read — please retake the photo.",
                  style: TextStyle(color: Colors.white70, fontSize: 12)),
            ]),
          ),
        ]),
      ),
      const SizedBox(height: 12),

      // Explanation + retake guidance
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFFCD34D)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('What this means',
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textSecondary)),
          const SizedBox(height: 8),
          Text(
            'The disease probability ($pct%) landed in the uncertain mid-range — '
            'not clearly normal, but not a confident match for Acne, Eczema, or '
            'Tinea either. Rather than guess, the AI is asking for a clearer photo.',
            style: const TextStyle(
                fontSize: 12, color: AppColors.textPrimary, height: 1.55),
          ),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFFFFBEB),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0xFFFCD34D)),
            ),
            child: const Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Icon(Icons.camera_alt_outlined, size: 14, color: Color(0xFFF59E0B)),
              SizedBox(width: 8),
              Expanded(
                  child: Text(
                'For a better result: fill the frame with the skin area, use good '
                'even lighting, and hold the camera steady and in focus.',
                style: TextStyle(fontSize: 11, color: Color(0xFF92400E)),
              )),
            ]),
          ),
        ]),
      ),
      const SizedBox(height: 20),

      // Action buttons
      Row(children: [
        Expanded(
          child: ElevatedButton(
            onPressed: _saved ? null : (_saving ? null : _saveToHistory),
            style: ElevatedButton.styleFrom(
              backgroundColor:
                  _saved ? AppColors.success : const Color(0xFFB45309),
              disabledBackgroundColor:
                  _saved ? AppColors.success : AppColors.border,
              minimumSize: const Size(double.infinity, 48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: _saving
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : Text(_saved ? 'Saved!' : 'Save to History',
                    style: const TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w600)),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton(
            onPressed: () => context.go(AppRoutes.capture),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryLight,
              foregroundColor: AppColors.primary,
              elevation: 0,
              minimumSize: const Size(double.infinity, 48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Retake Photo',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          ),
        ),
      ]),
      const SizedBox(height: 10),
      _buildPdfButton(),
    ]);
  }

  Widget _buildPdfButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: _exportingPdf ? null : _exportPdf,
        icon: _exportingPdf
            ? const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.picture_as_pdf_outlined, size: 18),
        label: Text(
            _exportingPdf ? 'Generating PDF...' : 'Export PDF Report'),
        style: OutlinedButton.styleFrom(
          minimumSize: const Size(double.infinity, 46),
          side: const BorderSide(color: AppColors.border),
          foregroundColor: AppColors.textSecondary,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12)),
        ),
      ),
    );
  }

  // ── Disease info expandable card ─────────────────────────────────────────

  Widget _buildDiseaseInfoCard(ScanResult scan) {
    final info = _diseaseData[scan.diagnosis];
    if (info == null) return const SizedBox.shrink();
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
          tilePadding: const EdgeInsets.fromLTRB(16, 2, 16, 2),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          leading: const Icon(Icons.medical_information_outlined,
              size: 20, color: AppColors.primary),
          title: Text(
            'About: ${scan.diagnosis}',
            style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary),
          ),
          children: [
            Text(info['description'] as String,
                style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textPrimary,
                    height: 1.55)),
            const SizedBox(height: 10),
            const Text('Common Symptoms',
                style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textSecondary)),
            const SizedBox(height: 6),
            ...(info['symptoms'] as List<String>).map(
              (s) => Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 5, right: 8),
                    child: Icon(Icons.circle,
                        size: 5, color: AppColors.textSecondary),
                  ),
                  Expanded(
                      child: Text(s,
                          style: const TextStyle(
                              fontSize: 12,
                              color: AppColors.textPrimary,
                              height: 1.4))),
                ]),
              ),
            ),
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppColors.primaryLight,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                const Icon(Icons.local_hospital_outlined,
                    size: 14, color: AppColors.primary),
                const SizedBox(width: 8),
                Expanded(
                    child: Text(info['advice'] as String,
                        style: const TextStyle(
                            fontSize: 11,
                            color: AppColors.primary,
                            height: 1.45))),
              ]),
            ),
          ],
        ),
      ),
    );
  }

  // ── Inference timing card ────────────────────────────────────────────────

  Widget _buildTimingCard(ScanResult scan) {
    if (scan.vaeMs == null) return const SizedBox.shrink();
    final pre = scan.preprocessMs ?? 0;
    final vae = scan.vaeMs!;
    final cnn = scan.cnnMs ?? 0;
    final cam = scan.scoreCamMs ?? 0;
    final total = pre + vae + cnn + cam;
    return _stageCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stageLabel(Icons.timer_outlined, 'ON-DEVICE INFERENCE TIMING'),
        const SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _TimingChip(label: 'Preprocess', ms: pre),
            _TimingChip(label: 'Gate', ms: vae),
            _TimingChip(label: 'CNN', ms: cnn),
            _TimingChip(label: 'Explain', ms: cam),
            _TimingChip(label: 'Total', ms: total, isPrimary: true),
          ],
        ),
        const SizedBox(height: 10),
        const Text(
          'All AI ran fully on-device — no data left the phone.',
          style: TextStyle(fontSize: 11, color: AppColors.textSecondary),
        ),
      ]),
    );
  }

  String _parseSkinType(String? mh) {
    if (mh == null) return '';
    final m = RegExp(r'Skin Type:\s*([^|]+)').firstMatch(mh);
    final v = m?.group(1)?.trim() ?? '';
    if (v.isEmpty || v == 'Not specified') return '';
    return v[0].toUpperCase() + v.substring(1);
  }

  String _parseSkinTone(String? mh) {
    if (mh == null) return '';
    final m = RegExp(r'Skin Tone:\s*([^|]+)').firstMatch(mh);
    return m?.group(1)?.trim() ?? '';
  }

  List<String> _parseConditions(String? mh) {
    if (mh == null) return [];
    final m = RegExp(r'Conditions:\s*([^|]+)').firstMatch(mh);
    if (m == null) return [];
    return m.group(1)!.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
  }

  // ── Sequential reveal wrapper ─────────────────────────────────────────────

  Widget _revealSection({required bool show, required Widget child}) {
    return AnimatedSize(
      duration: const Duration(milliseconds: 380),
      curve: Curves.easeOut,
      alignment: Alignment.topCenter,
      child: show
          ? TweenAnimationBuilder<double>(
              tween: Tween(begin: 0.0, end: 1.0),
              duration: const Duration(milliseconds: 430),
              curve: Curves.easeOut,
              builder: (_, v, c) => Opacity(
                opacity: v,
                child: Transform.translate(
                    offset: Offset(0, 18 * (1 - v)), child: c),
              ),
              child: child,
            )
          : const SizedBox.shrink(),
    );
  }

  // ── Stage card shell ──────────────────────────────────────────────────────

  Widget _stageCard({required Widget child}) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: child,
    );
  }

  Widget _stageLabel(IconData icon, String title) {
    return Row(children: [
      Icon(icon, size: 14, color: AppColors.primary),
      const SizedBox(width: 6),
      Text(title,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: AppColors.textSecondary,
            letterSpacing: 0.5,
          )),
    ]);
  }

  // ── Stage 1: Normal-vs-Disease gate ──────────────────────────────────────

  Widget _buildVaeSection(ScanResult scan) {
    final isInconclusive = scan.diagnosis == 'Inconclusive';
    final isAnomaly =
        !isInconclusive && scan.diagnosis != 'No Disease Detected';
    final ratio = scan.anomalyRatio;
    final color = isInconclusive
        ? const Color(0xFFF59E0B)
        : (isAnomaly ? const Color(0xFFF97316) : AppColors.success);
    final label = isInconclusive
        ? 'Inconclusive — needs a clearer photo'
        : (isAnomaly ? 'Skin condition detected' : 'No condition found');
    final hasVaeImg = scan.vaeHeatmapPath != null &&
        File(scan.vaeHeatmapPath!).existsSync();

    return _stageCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stageLabel(Icons.radar, 'STAGE 1 · NORMAL vs DISEASE GATE'),
        const SizedBox(height: 12),

        // Gate disease-evidence (occlusion) map — shown for normal/inconclusive
        if (hasVaeImg) ...[
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: Image.file(
              File(scan.vaeHeatmapPath!),
              width: double.infinity,
              height: 180,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(height: 8),
          const AttentionLegendBar(
            highLabel: 'More disease-like',
            lowLabel: 'Less disease-like',
          ),
          const SizedBox(height: 6),
          const Text(
            'Occlusion sensitivity: regions that, when covered, lowered the '
            "gate's disease score the most — i.e. what made this look (un)healthy.",
            style: TextStyle(fontSize: 11, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 12),
        ],

        // Decision badge
        Row(children: [
          Container(
            width: 9,
            height: 9,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(label,
              style: TextStyle(
                  fontSize: 14, fontWeight: FontWeight.w700, color: color)),
        ]),

        // Anomaly ratio bar
        if (ratio != null) ...[
          const SizedBox(height: 10),
          Row(children: [
            const Text('Disease probability',
                style: TextStyle(
                    fontSize: 12, color: AppColors.textSecondary)),
            const SizedBox(width: 10),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: ratio.clamp(0.0, 1.0),
                  backgroundColor: AppColors.border,
                  color: color,
                  minHeight: 7,
                ),
              ),
            ),
            const SizedBox(width: 8),
            Text('${(ratio * 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w700)),
          ]),
          const SizedBox(height: 4),
          Text(
            isAnomaly
                ? 'At/above 90% — passed to the CNN classifier.'
                : isInconclusive
                    ? '60–90% — too uncertain to commit; reported inconclusive.'
                    : 'Below 60% — classified as normal skin.',
            style: const TextStyle(
                fontSize: 11, color: AppColors.textSecondary),
          ),
        ],
      ]),
    );
  }

  // ── Stage 2: CNN ──────────────────────────────────────────────────────────

  Widget _buildCnnSection(ScanResult scan) {
    if (scan.classProbabilities.isEmpty) {
      return _stageCard(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _stageLabel(Icons.psychology_outlined,
              'STAGE 2 · CLASSIFICATION  (EFFICIENTNET)'),
          const SizedBox(height: 12),
          const Text(
            'Skipped — the screening gate classified this as normal skin.',
            style:
                TextStyle(fontSize: 13, color: AppColors.textSecondary),
          ),
        ]),
      );
    }

    final sorted = scan.classProbabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final showOther = scan.confidence < _otherThreshold;
    const otherColor = Color(0xFFF97316); // amber

    return _stageCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stageLabel(Icons.psychology_outlined,
            'STAGE 2 · CLASSIFICATION  (EFFICIENTNET)'),
        const SizedBox(height: 12),

        // Known disease bars
        ...sorted.map((e) {
          final isTop = e.key == scan.diagnosis;
          return Padding(
            padding: const EdgeInsets.only(bottom: 9),
            child: Row(children: [
              SizedBox(
                width: 56,
                child: Text(e.key,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight:
                          isTop && !showOther ? FontWeight.w700 : FontWeight.w400,
                      color: isTop && !showOther
                          ? AppColors.primary
                          : AppColors.textPrimary,
                    )),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: e.value.clamp(0.0, 1.0),
                    backgroundColor: AppColors.border,
                    color: showOther
                        ? AppColors.textSecondary.withValues(alpha: 0.35)
                        : isTop
                            ? AppColors.primary
                            : AppColors.textSecondary.withValues(alpha: 0.35),
                    minHeight: 7,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 36,
                child: Text(
                  '${(e.value * 100).round()}%',
                  textAlign: TextAlign.right,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: isTop && !showOther
                        ? FontWeight.w700
                        : FontWeight.w400,
                    color: isTop && !showOther
                        ? AppColors.primary
                        : AppColors.textSecondary,
                  ),
                ),
              ),
            ]),
          );
        }),

        // "Other" slot — shown when no known class is confidently predicted
        if (showOther) ...[
          const Divider(height: 16, color: AppColors.border),
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(children: [
              const SizedBox(
                width: 56,
                child: Text('Other',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: otherColor,
                    )),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: (1 - scan.confidence).clamp(0.0, 1.0),
                    backgroundColor: AppColors.border,
                    color: otherColor,
                    minHeight: 7,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 36,
                child: Text(
                  '${((1 - scan.confidence) * 100).round()}%',
                  textAlign: TextAlign.right,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: otherColor,
                  ),
                ),
              ),
            ]),
          ),
          const Text(
            'Low confidence across all known categories — condition may not be in the training set.',
            style: TextStyle(fontSize: 11, color: otherColor),
          ),
        ],
      ]),
    );
  }

  // ── Stage 3: Score-CAM ────────────────────────────────────────────────────

  Widget _buildScoreCamSection(ScanResult scan) {
    // Classes that actually have a per-class heatmap, ordered by probability.
    final available = scan.classHeatmapPaths.entries
        .where((e) => File(e.value).existsSync())
        .map((e) => e.key)
        .toList()
      ..sort((a, b) => (scan.classProbabilities[b] ?? 0)
          .compareTo(scan.classProbabilities[a] ?? 0));

    final selected = (_selectedHeatmapClass != null &&
            available.contains(_selectedHeatmapClass))
        ? _selectedHeatmapClass!
        : (available.contains(scan.diagnosis)
            ? scan.diagnosis
            : (available.isNotEmpty ? available.first : null));

    final displayPath = selected != null
        ? scan.classHeatmapPaths[selected]
        : scan.heatmapPath;
    final hasHeatmap = displayPath != null && File(displayPath).existsSync();

    return _stageCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stageLabel(Icons.thermostat_outlined,
            'STAGE 3 · EXPLAINABILITY MAP  (SCORE-CAM)'),
        const SizedBox(height: 12),

        // Per-class selector — compare why the model favoured each class.
        if (available.length > 1) ...[
          Wrap(
            spacing: 8,
            children: available.map((cls) {
              final isSel = cls == selected;
              final pct = ((scan.classProbabilities[cls] ?? 0) * 100).round();
              return GestureDetector(
                onTap: () => setState(() => _selectedHeatmapClass = cls),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color:
                        isSel ? AppColors.primary : AppColors.primaryLight,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                        color: isSel ? AppColors.primary : AppColors.border),
                  ),
                  child: Text('$cls · $pct%',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: isSel ? Colors.white : AppColors.primary,
                      )),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 4),
          Text(
            selected == scan.diagnosis
                ? 'Showing the predicted class. Tap another to see its counterfactual map.'
                : 'Counterfactual: where the model would look to support $selected.',
            style: const TextStyle(fontSize: 11, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 10),
        ],

        GestureDetector(
          onTap: hasHeatmap
              ? () => context.push(AppRoutes.xaiHeatmap, extra: scan)
              : null,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: hasHeatmap
                ? Image.file(File(displayPath),
                    key: ValueKey(displayPath),
                    width: double.infinity,
                    height: 190,
                    fit: BoxFit.cover)
                : Container(
                    width: double.infinity,
                    height: 120,
                    color: const Color(0xFFF1F5F9),
                    alignment: Alignment.center,
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    child: const Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.grid_off,
                            size: 28, color: AppColors.textSecondary),
                        SizedBox(height: 8),
                        Text(
                          'Score-CAM runs only once a disease is classified.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                              fontSize: 12, color: AppColors.textSecondary),
                        ),
                      ],
                    ),
                  ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          hasHeatmap
              ? (selected == scan.diagnosis
                  ? 'Highlighted regions drove the $selected prediction.'
                  : 'Regions that would support $selected.')
              : (scan.diagnosis == 'No Disease Detected' ||
                      scan.diagnosis == 'Inconclusive')
                  ? 'No disease was classified, so there is no class to attribute. '
                      'See the disease-evidence map in Stage 1 for where the gate looked.'
                  : 'Score-CAM heatmap unavailable.',
          style: const TextStyle(fontSize: 11, color: AppColors.textSecondary),
        ),
        const SizedBox(height: 8),
        const AttentionLegendBar(),
        if (scan.xaiRationale != null) ...[
          const SizedBox(height: 10),
          Text(scan.xaiRationale!,
              style: const TextStyle(
                  fontSize: 12, color: AppColors.textPrimary, height: 1.5)),
        ],
      ]),
    );
  }

  // ── Shared XAI helpers (uncertainty + limitations) ───────────────────────

  /// Explains the gap between the top-two classes and warns when the result
  /// is poorly separated.
  Widget? _buildUncertaintyNote(ScanResult scan) {
    if (scan.classProbabilities.length < 2) return null;
    final sorted = scan.classProbabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final top = sorted[0], second = sorted[1];
    final margin = top.value - second.value;
    final (label, color) = margin >= 0.40
        ? ('Well separated', AppColors.success)
        : margin >= 0.20
            ? ('Moderate separation', const Color(0xFFF59E0B))
            : ('Low separation', AppColors.error);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.balance_outlined, size: 14, color: color),
          const SizedBox(width: 6),
          Text('CONFIDENCE MARGIN · ${label.toUpperCase()}',
              style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: color,
                  letterSpacing: 0.5)),
        ]),
        const SizedBox(height: 8),
        Text(
          '${top.key} (${(top.value * 100).round()}%) vs ${second.key} '
          '(${(second.value * 100).round()}%) — a ${(margin * 100).round()}-point gap.'
          '${margin < 0.20 ? ' The top two are close, so treat this result with extra caution and seek a clinician\'s opinion.' : ''}',
          style: const TextStyle(
              fontSize: 12, color: AppColors.textPrimary, height: 1.5),
        ),
      ]),
    );
  }

  /// Honest disclosure of when the heatmap / result can mislead.
  Widget _buildLimitationsNote(ScanResult scan) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBEB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFCD34D)),
      ),
      child: const Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(Icons.info_outline, size: 15, color: Color(0xFFF59E0B)),
        SizedBox(width: 8),
        Expanded(
          child: Text(
            'How to read this honestly: the heatmap shows where the model looked, '
            'not proof it is right. Poor lighting, framing, hair, or conditions '
            'outside Acne/Eczema/Tinea can shift attention onto the wrong area. '
            'Use it as a discussion aid with a dermatologist, not a diagnosis.',
            style: TextStyle(
                fontSize: 11, color: Color(0xFF92400E), height: 1.45),
          ),
        ),
      ]),
    );
  }

  // ── Patient-facing explainability (Score-CAM, accessible framing) ─────────

  Widget _buildPatientXaiSection(ScanResult scan) {
    final isNormal = scan.diagnosis == 'No Disease Detected';
    final isInconclusive = scan.diagnosis == 'Inconclusive';
    // Normal / Inconclusive carry the gate occlusion map (vaeHeatmapPath); a
    // classified disease carries the Score-CAM map (heatmapPath).
    final isGateMap = isNormal || isInconclusive;
    final hasScoreCam =
        scan.heatmapPath != null && File(scan.heatmapPath!).existsSync();
    final hasGate = scan.vaeHeatmapPath != null &&
        File(scan.vaeHeatmapPath!).existsSync();
    final imgPath = hasScoreCam
        ? scan.heatmapPath
        : hasGate
            ? scan.vaeHeatmapPath
            : null;
    final hasImg = imgPath != null;

    // Accessible explanation text, matched to the actual XAI method used.
    final String explanation;
    if (isNormal) {
      explanation =
          'Because no condition was found, there is no class to attribute. '
          'Instead, the disease-evidence map below uses occlusion sensitivity: '
          'small parts of your photo were covered one at a time and the gate was '
          're-checked. Warmer regions are where covering the skin lowered the '
          'disease score the most — the most disease-like areas. Overall it '
          'stayed below the threshold, so no Acne, Eczema, or Tinea was flagged.';
    } else if (isInconclusive) {
      explanation =
          'The screening could not commit to a confident result. The map below '
          'uses occlusion sensitivity — covering parts of the photo and '
          're-checking the gate — to show which regions looked the most '
          'disease-like. Warmer means more disease-like. A clearer, well-lit '
          'photo of the area will usually give a more definite result.';
    } else {
      explanation =
          'This map was produced by Score-CAM, an explainable-AI method. The '
          'warmer (red) regions are the areas the classifier weighted most '
          'heavily when predicting ${scan.diagnosis}; cooler (green/blue) '
          'regions had little influence on the result. Use it to check the AI '
          'focused on the actual lesion rather than the background.';
    }

    return _stageCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stageLabel(Icons.lightbulb_outline,
            'WHY THIS RESULT?  (EXPLAINABLE AI)'),
        const SizedBox(height: 12),
        if (hasImg) ...[
          GestureDetector(
            onTap: () => context.push(AppRoutes.xaiHeatmap, extra: scan),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Stack(
                alignment: Alignment.bottomRight,
                children: [
                  Image.file(File(imgPath),
                      width: double.infinity, height: 200, fit: BoxFit.cover),
                  Container(
                    margin: const EdgeInsets.all(8),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.55),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(Icons.fullscreen, color: Colors.white, size: 14),
                      SizedBox(width: 4),
                      Text('Tap to expand',
                          style: TextStyle(color: Colors.white, fontSize: 11)),
                    ]),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
          AttentionLegendBar(
            highLabel: isGateMap ? 'More disease-like' : 'High attention',
            lowLabel: isGateMap ? 'Less disease-like' : 'Low attention',
          ),
          const SizedBox(height: 12),
        ],
        Text(
          explanation,
          style: const TextStyle(
              fontSize: 12, color: AppColors.textPrimary, height: 1.55),
        ),
        if (scan.xaiRationale != null) ...[
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AppColors.primaryLight,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppColors.border),
            ),
            child: Text(scan.xaiRationale!,
                style: const TextStyle(
                    fontSize: 12, color: AppColors.primary, height: 1.45)),
          ),
        ],
        const SizedBox(height: 12),
        _buildLimitationsNote(scan),
      ]),
    );
  }

  // ── Stage 4: XAI Explanation ─────────────────────────────────────────────

  Widget _buildXaiExplanationSection(ScanResult scan) {
    final isNormal = scan.diagnosis == 'No Disease Detected';
    final isInconclusive = scan.diagnosis == 'Inconclusive';
    final isGateMap = isNormal || isInconclusive;

    final String explanation;
    if (isGateMap) {
      explanation =
          'No disease was classified, so there is no CNN class to attribute with '
          'Score-CAM. Instead, the disease-evidence map uses occlusion '
          'sensitivity on the gate: a neutral patch is slid across the image and '
          "the gate is re-run each time. Where covering the skin drops the gate's "
          'disease score the most, that region was the strongest evidence — warmer '
          'colours mean a larger drop.';
    } else {
      explanation =
          'Score-CAM (Score-weighted Class Activation Mapping) explains the CNN '
          'decision by selectively masking the top feature-map channels and '
          'measuring how each channel raises or lowers the predicted probability. '
          'The resulting heatmap (Stage 3) shows WHICH pixels drove the '
          '${scan.diagnosis} prediction — warmer colours mean higher influence.';
    }

    return _stageCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stageLabel(Icons.lightbulb_outline, 'STAGE 4 · EXPLAINABLE AI  (XAI)'),
        const SizedBox(height: 12),
        Text(explanation,
            style: const TextStyle(
                fontSize: 12, color: AppColors.textPrimary, height: 1.6)),
        if (!isNormal && scan.xaiRationale != null) ...[
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.primaryLight,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Where the model looked',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: AppColors.primary)),
                const SizedBox(height: 6),
                Text(scan.xaiRationale!,
                    style: const TextStyle(
                        fontSize: 12, color: AppColors.primary, height: 1.5)),
              ],
            ),
          ),
        ],
        const SizedBox(height: 12),
        _buildLimitationsNote(scan),
      ]),
    );
  }

  // ── Verdict (confidence + diagnosis + actions) ────────────────────────────

  Widget _buildVerdictSection(
    ScanResult scan, {
    required String skinType,
    required String skinTone,
    required List<String> conditions,
  }) {
    if (scan.diagnosis == 'No Disease Detected') {
      return _buildNormalVerdictSection(scan);
    }
    if (scan.diagnosis == 'Inconclusive') {
      return _buildInconclusiveVerdictSection(scan);
    }

    final confidence = (scan.confidence * 100).round();

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // Confidence card
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF1E3A5F), Color(0xFF2B5FA0)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(children: [
          SizedBox(
            width: 72,
            height: 72,
            child: CustomPaint(
              painter: _CircleProgressPainter(progress: confidence / 100),
              child: Center(
                child: RichText(
                  textAlign: TextAlign.center,
                  text: TextSpan(children: [
                    TextSpan(
                      text: '$confidence',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const TextSpan(
                      text: '%',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ]),
                ),
              ),
            ),
          ),
          const SizedBox(width: 20),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Confidence Score',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text(
              confidence >= 80
                  ? 'High Confidence'
                  : confidence >= 60
                      ? 'Moderate Confidence'
                      : 'Low Confidence',
              style:
                  const TextStyle(color: Colors.white70, fontSize: 13),
            ),
          ]),
        ]),
      ),
      const SizedBox(height: 12),

      // Diagnosis summary
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFDBEAFE)),
        ),
        child: Row(children: [
          Expanded(
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Diagnosis',
                      style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textSecondary)),
                  const SizedBox(height: 6),
                  Text(
                      scan.confidence < _otherThreshold
                          ? 'Unknown Condition'
                          : scan.diagnosis,
                      style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary)),
                  if (scan.bodyPart.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(scan.bodyPart,
                        style: const TextStyle(
                            fontSize: 13,
                            color: AppColors.textSecondary)),
                  ],
                ]),
          ),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
            decoration: BoxDecoration(
              color: _isHighRisk
                  ? const Color(0xFFFEE2E2)
                  : const Color(0xFFDCFCE7),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              _isHighRisk ? 'High Risk' : 'Low Risk',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: _isHighRisk
                    ? const Color(0xFFDC2626)
                    : const Color(0xFF16A34A),
              ),
            ),
          ),
        ]),
      ),

      // Skin profile context
      if (skinType.isNotEmpty ||
          skinTone.isNotEmpty ||
          conditions.isNotEmpty) ...[
        const SizedBox(height: 12),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Your Skin Profile',
                    style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textSecondary)),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    if (skinType.isNotEmpty)
                      _ProfileChip(
                          label: skinType,
                          icon: Icons.water_drop_outlined),
                    if (skinTone.isNotEmpty)
                      _ProfileChip(
                          label: skinTone, icon: Icons.circle_outlined),
                    ...conditions.map((c) => _ProfileChip(
                        label: c,
                        icon: Icons.medical_information_outlined)),
                  ],
                ),
                // Contextual note based on profile + diagnosis
                Builder(builder: (_) {
                  final note = _getContextNote(
                      scan.diagnosis, skinType, conditions);
                  if (note.isEmpty) return const SizedBox.shrink();
                  return Padding(
                    padding: const EdgeInsets.only(top: 10),
                    child: Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFFF7ED),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                            color: const Color(0xFFFED7AA)),
                      ),
                      child: Row(
                          crossAxisAlignment:
                              CrossAxisAlignment.start,
                          children: [
                        const Icon(Icons.info_outline,
                            size: 14, color: Color(0xFFF97316)),
                        const SizedBox(width: 6),
                        Expanded(
                            child: Text(note,
                                style: const TextStyle(
                                    fontSize: 11,
                                    color: Color(0xFF92400E),
                                    height: 1.4))),
                      ]),
                    ),
                  );
                }),
              ]),
        ),
      ],
      const SizedBox(height: 20),

      // Action buttons
      Row(children: [
        Expanded(
          child: ElevatedButton(
            onPressed:
                _saved ? null : (_saving ? null : _saveToHistory),
            style: ElevatedButton.styleFrom(
              backgroundColor:
                  _saved ? AppColors.success : const Color(0xFF1E3A5F),
              disabledBackgroundColor:
                  _saved ? AppColors.success : AppColors.border,
              minimumSize: const Size(double.infinity, 48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: _saving
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : Text(
                    _saved ? 'Saved!' : 'Save to History',
                    style: const TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w600),
                  ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton(
            onPressed: () => context.push(
                AppRoutes.bookAppointment,
                extra: widget.scan),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.success,
              minimumSize: const Size(double.infinity, 48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Book Appointment',
                style: TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w600)),
          ),
        ),
      ]),
      const SizedBox(height: 10),
      _buildPdfButton(),
    ]);
  }

  // ── Scope / medical disclaimer (shown for every result) ──────────────────

  /// Honest scope statement. The model is a 3-class differential classifier with
  /// no "healthy/normal" class, and on out-of-distribution real-world photos it
  /// is forced to pick one of Acne/Eczema/Tinea. This banner sets that
  /// expectation so a confident label on non-target or healthy skin is read as
  /// a known scope limit, not a diagnosis.
  Widget _buildScopeDisclaimer() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBEB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFCD34D)),
      ),
      child: const Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(Icons.info_outline, size: 15, color: Color(0xFFF59E0B)),
        SizedBox(width: 8),
        Expanded(
          child: Text(
            'This tool screens among three conditions only — Acne, Eczema, and '
            'Tinea — and assumes a skin lesion is present in the photo. It does '
            'not determine whether skin is healthy, and it is not a medical '
            'diagnosis. Always consult a dermatologist.',
            style: TextStyle(
                fontSize: 11, color: Color(0xFF92400E), height: 1.45),
          ),
        ),
      ]),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final scan = widget.scan;
    final imagePath = scan?.imagePath ?? '';
    final user = ref.watch(userProvider);
    final skinType = _parseSkinType(user?.medicalHistory);
    final skinTone = _parseSkinTone(user?.medicalHistory);
    final conditions = _parseConditions(user?.medicalHistory);

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () =>
              context.canPop() ? context.pop() : context.go(AppRoutes.home),
        ),
        title: const Text(
          'Analysis Results',
          style: TextStyle(
            fontSize: 17,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        actions: [
          if (widget.scan != null)
            IconButton(
              icon: const Icon(Icons.share_outlined,
                  color: AppColors.textPrimary),
              onPressed: _share,
              tooltip: 'Share result',
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Captured image — always visible
            if (imagePath.isNotEmpty && File(imagePath).existsSync()) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: SizedBox(
                  width: double.infinity,
                  height: 200,
                  child: Image.file(File(imagePath), fit: BoxFit.cover),
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Scope / medical disclaimer — shown for every result, both views.
            if (scan != null) _buildScopeDisclaimer(),

            if (user?.role == 'specialist') ...[
              // ── SPECIALIST VIEW: full pipeline ──────────────────────────

              // Pipeline details toggle (controls timing card)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.primaryLight,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.science_outlined,
                            size: 18, color: AppColors.primary),
                        SizedBox(width: 10),
                        Text(
                          'AI Pipeline Details',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: AppColors.primary,
                          ),
                        ),
                      ],
                    ),
                    Switch(
                      value: _showPipelineDetails,
                      onChanged: (v) =>
                          setState(() => _showPipelineDetails = v),
                      activeThumbColor: AppColors.primary,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Stage 1: Normal-vs-Disease gate
              if (scan != null && _showPipelineDetails)
                _revealSection(show: _showVae, child: _buildVaeSection(scan)),

              // Stage 2: CNN
              if (scan != null && _showPipelineDetails)
                _revealSection(show: _showCnn, child: _buildCnnSection(scan)),

              // Stage 3: Score-CAM
              if (scan != null && _showPipelineDetails)
                _revealSection(
                    show: _showScoreCam,
                    child: _buildScoreCamSection(scan)),

              // Stage 4: XAI explanation
              if (scan != null && _showPipelineDetails)
                _revealSection(
                    show: _showScoreCam,
                    child: _buildXaiExplanationSection(scan)),

              // Uncertainty note
              if (scan != null)
                Builder(builder: (_) {
                  final note = _buildUncertaintyNote(scan);
                  if (note == null) return const SizedBox.shrink();
                  return _revealSection(show: _showVerdict, child: note);
                }),

              // Verdict + actions
              if (scan != null)
                _revealSection(
                  show: _showVerdict,
                  child: _buildVerdictSection(scan,
                      skinType: skinType,
                      skinTone: skinTone,
                      conditions: conditions),
                ),

              // Disease info + timing
              if (scan != null) ...[
                _revealSection(
                    show: _showExtras,
                    child: _buildDiseaseInfoCard(scan)),
                if (_showPipelineDetails)
                  _revealSection(
                      show: _showExtras, child: _buildTimingCard(scan)),
              ],
            ] else ...[
              // ── GUEST / PATIENT VIEW: original clean layout ─────────────

              // Uncertainty note
              if (scan != null)
                Builder(builder: (_) {
                  final note = _buildUncertaintyNote(scan);
                  if (note == null) return const SizedBox.shrink();
                  return _revealSection(show: _showVerdict, child: note);
                }),

              // Verdict + actions
              if (scan != null)
                _revealSection(
                  show: _showVerdict,
                  child: _buildVerdictSection(scan,
                      skinType: skinType,
                      skinTone: skinTone,
                      conditions: conditions),
                ),

              // Patient XAI summary + disease info
              if (scan != null) ...[
                _revealSection(
                    show: _showExtras,
                    child: _buildPatientXaiSection(scan)),
                _revealSection(
                    show: _showExtras,
                    child: _buildDiseaseInfoCard(scan)),
              ],
            ],
          ],
        ),
      ),
      bottomNavigationBar: const AppBottomNav(currentIndex: 2),
    );
  }
}

// ── Helper widgets ────────────────────────────────────────────────────────────

class _ProfileChip extends StatelessWidget {
  final String label;
  final IconData icon;
  const _ProfileChip({required this.label, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: AppColors.primaryLight,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: AppColors.primary),
          const SizedBox(width: 5),
          Text(label,
              style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.primary)),
        ],
      ),
    );
  }
}

class _TimingChip extends StatelessWidget {
  final String label;
  final int ms;
  final bool isPrimary;
  const _TimingChip(
      {required this.label, required this.ms, this.isPrimary = false});

  @override
  Widget build(BuildContext context) {
    final display =
        ms >= 1000 ? '${(ms / 1000).toStringAsFixed(1)}s' : '${ms}ms';
    return Column(children: [
      Text(display,
          style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color:
                  isPrimary ? AppColors.primary : AppColors.textPrimary)),
      const SizedBox(height: 2),
      Text(label,
          style: const TextStyle(
              fontSize: 10, color: AppColors.textSecondary)),
    ]);
  }
}

class _CircleProgressPainter extends CustomPainter {
  final double progress;
  const _CircleProgressPainter({required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 5;
    const strokeWidth = 5.0;

    canvas.drawCircle(
      center,
      radius,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.2)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth,
    );

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      2 * math.pi * progress,
      false,
      Paint()
        ..color = const Color(0xFF60A5FA)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
