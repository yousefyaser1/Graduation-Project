import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class XAIHeatmapScreen extends StatelessWidget {
  const XAIHeatmapScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('XAI Heatmap'),
      ),
      body: Column(
        children: [
          // Heatmap Display Area
          Expanded(
            child: Container(
              color: Colors.grey[200],
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.grid_on,
                      size: 100,
                      color: Colors.grey[400],
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'XAI Heatmap Visualization',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.grey[700],
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'This will show the explainable AI heatmap',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'highlighting regions of interest',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          
          // Controls and Information
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Theme.of(context).scaffoldBackgroundColor,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 10,
                  offset: const Offset(0, -5),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Heatmap Explanation',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 16),
                _HeatmapLegend(),
                const SizedBox(height: 16),
                Text(
                  'The heatmap shows which areas of the image contributed most to the AI\'s diagnosis. Red areas indicate high importance, while blue areas indicate low importance.',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[700],
                  ),
                ),
                const SizedBox(height: 24),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => context.pop(),
                        icon: const Icon(Icons.arrow_back),
                        label: const Text('Back to Results'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HeatmapLegend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        _LegendItem(
          color: Colors.red,
          label: 'High',
        ),
        _LegendItem(
          color: Colors.orange,
          label: 'Medium-High',
        ),
        _LegendItem(
          color: Colors.yellow,
          label: 'Medium',
        ),
        _LegendItem(
          color: Colors.green,
          label: 'Low',
        ),
        _LegendItem(
          color: Colors.blue,
          label: 'Very Low',
        ),
      ],
    );
  }
}

class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendItem({
    required this.color,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Container(
            width: 40,
            height: 20,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
