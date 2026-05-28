import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Jet-colormap gradient bar with "High / Low attention" end labels.
///
/// Shared by every surface that renders a Score-CAM or VAE heatmap so the
/// colour-to-meaning mapping is described identically across the app.
class AttentionLegendBar extends StatelessWidget {
  /// Label for the warm (red) end. Defaults to attention semantics.
  final String highLabel;

  /// Label for the cool (green) end.
  final String lowLabel;

  const AttentionLegendBar({
    super.key,
    this.highLabel = 'High attention',
    this.lowLabel = 'Low attention',
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(highLabel,
                style: const TextStyle(
                    fontSize: 11, color: AppColors.textSecondary)),
            Text(lowLabel,
                style: const TextStyle(
                    fontSize: 11, color: AppColors.textSecondary)),
          ],
        ),
        const SizedBox(height: 4),
        Container(
          height: 7,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(4),
            gradient: const LinearGradient(
              colors: [
                Color(0xFFEF4444),
                Color(0xFFF97316),
                Color(0xFFFACC15),
                Color(0xFF86EFAC),
                Color(0xFF22C55E),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
