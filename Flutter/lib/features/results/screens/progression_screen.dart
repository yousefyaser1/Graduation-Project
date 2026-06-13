import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../models/scan_result.dart';
import '../../../providers/scan_provider.dart';
import '../progression_utils.dart';

/// Lesion progression: compare scans of the same body part over time, with a
/// confidence trend, so users (and specialists) can see whether a condition is
/// improving or worsening between visits.
class ProgressionScreen extends ConsumerStatefulWidget {
  const ProgressionScreen({super.key});

  @override
  ConsumerState<ProgressionScreen> createState() => _ProgressionScreenState();
}

class _ProgressionScreenState extends ConsumerState<ProgressionScreen> {
  String? _selectedPart;

  @override
  Widget build(BuildContext context) {
    final scansAsync = ref.watch(scanListProvider);

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () => context.pop(),
        ),
        title: const Text(
          'Progression',
          style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary),
        ),
      ),
      body: scansAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(
          child: Text('Error loading scans: $e',
              style: const TextStyle(color: AppColors.error)),
        ),
        data: (scans) {
          final groups = groupScansByBodyPart(scans);
          final parts = trackableBodyParts(groups);

          if (parts.isEmpty) return _buildEmptyState();

          final selected =
              (_selectedPart != null && parts.contains(_selectedPart))
                  ? _selectedPart!
                  : parts.first;
          final series = groups[selected]!; // chronological (oldest → newest)

          return Column(
            children: [
              _buildPartSelector(parts, groups, selected),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
                  children: [
                    _buildTrendCard(selected, series),
                    const SizedBox(height: 16),
                    Text(
                      'Timeline — ${series.length} scan${series.length == 1 ? '' : 's'}',
                      style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textSecondary),
                    ),
                    const SizedBox(height: 12),
                    // Newest first in the timeline list.
                    ...series.reversed.toList().asMap().entries.map((e) {
                      final reversedIndex = e.key;
                      final scan = e.value;
                      // Compare against the previous (older) scan for delta.
                      final chronoIndex = series.length - 1 - reversedIndex;
                      final prev =
                          chronoIndex > 0 ? series[chronoIndex - 1] : null;
                      return _buildTimelineCard(scan, prev);
                    }),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildPartSelector(List<String> parts,
      Map<String, List<ScanResult>> groups, String selected) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 14),
      child: SizedBox(
        height: 38,
        child: ListView.separated(
          scrollDirection: Axis.horizontal,
          itemCount: parts.length,
          separatorBuilder: (_, __) => const SizedBox(width: 8),
          itemBuilder: (context, i) {
            final p = parts[i];
            final isSel = p == selected;
            return GestureDetector(
              onTap: () => setState(() => _selectedPart = p),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: isSel ? AppColors.primary : AppColors.primaryLight,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: isSel ? AppColors.primary : AppColors.border),
                ),
                child: Text('$p (${groups[p]!.length})',
                    style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: isSel ? Colors.white : AppColors.primary)),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildTrendCard(String part, List<ScanResult> series) {
    final trend = confidenceTrend(series);
    final first = series.first;
    final last = series.last;
    final improving = last.diagnosis == 'No Disease Detected' ||
        (first.diagnosis != 'No Disease Detected' &&
            last.diagnosis != 'No Disease Detected' &&
            last.confidence < first.confidence);

    final span = series.length > 1
        ? '${DateFormat('MMM d').format(first.timestamp)} – ${DateFormat('MMM d, yyyy').format(last.timestamp)}'
        : DateFormat('MMM d, yyyy').format(first.timestamp);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.show_chart, size: 16, color: AppColors.primary),
          const SizedBox(width: 6),
          Text('$part — confidence trend',
              style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary)),
        ]),
        const SizedBox(height: 4),
        Text(span,
            style:
                const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
        const SizedBox(height: 14),
        if (trend.length > 1)
          SizedBox(
            height: 70,
            width: double.infinity,
            child: CustomPaint(painter: _TrendPainter(trend)),
          )
        else
          const Text('Only one scan so far — add another to see a trend.',
              style: TextStyle(fontSize: 12, color: AppColors.textSecondary)),
        if (series.length > 1) ...[
          const SizedBox(height: 12),
          Row(children: [
            Icon(improving ? Icons.trending_down : Icons.trending_up,
                size: 16,
                color: improving ? AppColors.success : AppColors.warning),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                improving
                    ? 'Latest scan suggests improvement or resolution versus the first scan.'
                    : 'Latest scan does not show improvement — consider a specialist review.',
                style: TextStyle(
                    fontSize: 12,
                    color: improving ? AppColors.success : AppColors.warning,
                    height: 1.4),
              ),
            ),
          ]),
        ],
      ]),
    );
  }

  Widget _buildTimelineCard(ScanResult scan, ScanResult? previous) {
    final isNormal = scan.diagnosis == 'No Disease Detected';
    final conf = (scan.confidence * 100).round();
    final hasImg = File(scan.imagePath).existsSync();

    String? delta;
    if (previous != null &&
        !isNormal &&
        previous.diagnosis != 'No Disease Detected') {
      final d = ((scan.confidence - previous.confidence) * 100).round();
      if (d != 0) delta = '${d > 0 ? '+' : ''}$d% vs previous';
    }

    return GestureDetector(
      onTap: () => context.push(AppRoutes.analysisResults, extra: scan),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: hasImg
                ? Image.file(File(scan.imagePath),
                    width: 56, height: 56, fit: BoxFit.cover)
                : Container(
                    width: 56,
                    height: 56,
                    color: AppColors.primaryLight,
                    child: const Icon(Icons.image_not_supported_outlined,
                        color: AppColors.primary, size: 22),
                  ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(isNormal ? 'No Disease Detected' : scan.diagnosis,
                  style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary)),
              const SizedBox(height: 3),
              Text(DateFormat('MMM d, yyyy • h:mm a').format(scan.timestamp),
                  style: const TextStyle(
                      fontSize: 12, color: AppColors.textSecondary)),
              if (delta != null) ...[
                const SizedBox(height: 4),
                Text(delta,
                    style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: delta.startsWith('+')
                            ? AppColors.warning
                            : AppColors.success)),
              ],
            ]),
          ),
          if (!isNormal)
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: AppColors.primaryLight,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text('$conf%',
                  style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: AppColors.primary)),
            ),
        ]),
      ),
    );
  }

  Widget _buildEmptyState() {
    return const Center(
      child: Padding(
        padding: EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.timeline_outlined,
                size: 48, color: AppColors.textSecondary),
            SizedBox(height: 12),
            Text('No scans to track yet',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary)),
            SizedBox(height: 6),
            Text(
              'Scan the same body part more than once to watch how a condition changes over time.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
            ),
          ],
        ),
      ),
    );
  }
}

/// Simple confidence sparkline (values in 0–1) with dots at each scan.
class _TrendPainter extends CustomPainter {
  final List<double> values;
  _TrendPainter(this.values);

  @override
  void paint(Canvas canvas, Size size) {
    if (values.length < 2) return;
    final n = values.length;
    final dx = size.width / (n - 1);

    Offset pointAt(int i) =>
        Offset(i * dx, size.height - values[i].clamp(0.0, 1.0) * size.height);

    // Baseline grid (0%, 50%, 100%).
    final grid = Paint()
      ..color = AppColors.border
      ..strokeWidth = 1;
    for (final f in [0.0, 0.5, 1.0]) {
      final y = size.height - f * size.height;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }

    final line = Paint()
      ..color = AppColors.primary
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeJoin = StrokeJoin.round;
    final path = Path()..moveTo(pointAt(0).dx, pointAt(0).dy);
    for (var i = 1; i < n; i++) {
      path.lineTo(pointAt(i).dx, pointAt(i).dy);
    }
    canvas.drawPath(path, line);

    final dot = Paint()..color = AppColors.primary;
    for (var i = 0; i < n; i++) {
      canvas.drawCircle(pointAt(i), 3.5, dot);
    }
  }

  @override
  bool shouldRepaint(_TrendPainter old) => old.values != values;
}
