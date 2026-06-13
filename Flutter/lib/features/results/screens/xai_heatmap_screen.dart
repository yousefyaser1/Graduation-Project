import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/widgets/xai_attention_legend.dart';
import '../../../models/scan_result.dart';

/// Full-screen explainability viewer.
///
/// Lets the user compare the original photo against the AI attention overlay
/// (Score-CAM for a diagnosis, or the VAE reconstruction-error map for a
/// normal result) with an interactive opacity slider.
class XAIHeatmapScreen extends StatefulWidget {
  final ScanResult? scan;

  const XAIHeatmapScreen({super.key, this.scan});

  @override
  State<XAIHeatmapScreen> createState() => _XAIHeatmapScreenState();
}

class _XAIHeatmapScreenState extends State<XAIHeatmapScreen> {
  double _overlay = 1.0;

  @override
  Widget build(BuildContext context) {
    final scan = widget.scan;
    final isNormal = scan?.diagnosis == 'No Disease Detected';
    final heatmapPath =
        (scan?.heatmapPath != null && File(scan!.heatmapPath!).existsSync())
            ? scan.heatmapPath
            : (scan?.vaeHeatmapPath != null &&
                    File(scan!.vaeHeatmapPath!).existsSync())
                ? scan.vaeHeatmapPath
                : null;
    final hasOriginal =
        scan != null && File(scan.imagePath).existsSync();
    final isInconclusive = scan?.diagnosis == 'Inconclusive';
    final isGateMap = isNormal || isInconclusive;
    final method = isGateMap ? 'Disease-Evidence Map' : 'Score-CAM';

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        elevation: 0,
        title: Text(method,
            style: const TextStyle(
                fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white)),
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: () => context.pop(),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: heatmapPath == null
                  ? _emptyState()
                  : Stack(
                      alignment: Alignment.center,
                      children: [
                        if (hasOriginal)
                          Image.file(File(scan.imagePath), fit: BoxFit.contain),
                        Opacity(
                          opacity: _overlay,
                          child: Image.file(File(heatmapPath),
                              fit: BoxFit.contain),
                        ),
                      ],
                    ),
            ),
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
            decoration: const BoxDecoration(
              color: Color(0xFF111317),
              borderRadius:
                  BorderRadius.vertical(top: Radius.circular(20)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (hasOriginal && heatmapPath != null) ...[
                  Row(children: [
                    const Icon(Icons.opacity, color: Colors.white70, size: 16),
                    const SizedBox(width: 8),
                    const Text('Overlay strength',
                        style:
                            TextStyle(color: Colors.white70, fontSize: 13)),
                    const Spacer(),
                    Text('${(_overlay * 100).round()}%',
                        style: const TextStyle(
                            color: Colors.white, fontSize: 13)),
                  ]),
                  Slider(
                    value: _overlay,
                    onChanged: (v) => setState(() => _overlay = v),
                    activeColor: AppColors.primary,
                    inactiveColor: Colors.white24,
                  ),
                  const SizedBox(height: 4),
                ],
                AttentionLegendBar(
                  highLabel: isGateMap ? 'More disease-like' : 'High attention',
                  lowLabel: isGateMap ? 'Less disease-like' : 'Low attention',
                ),
                const SizedBox(height: 14),
                Text(
                  isGateMap
                      ? 'Warmer regions are where covering the skin lowered the '
                          "gate's disease score the most — the most disease-like "
                          'areas (occlusion sensitivity). Drag the slider to '
                          'compare against your original photo.'
                      : 'Warmer regions are where the classifier focused most '
                          'when predicting ${scan?.diagnosis ?? 'this result'}. '
                          'Drag the slider to compare against your original photo.',
                  style: const TextStyle(
                      color: Colors.white60, fontSize: 12, height: 1.5),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _emptyState() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.grid_off, size: 80, color: Colors.grey[600]),
        const SizedBox(height: 16),
        Text('No heatmap available for this scan',
            style: TextStyle(fontSize: 15, color: Colors.grey[400])),
      ],
    );
  }
}
