import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:uuid/uuid.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../models/scan_result.dart';
import '../../../services/ai/ai_isolate_runner.dart';

class AnalyzingScreen extends StatefulWidget {
  final String bodyPart;
  final String imagePath;

  const AnalyzingScreen({
    super.key,
    required this.bodyPart,
    required this.imagePath,
  });

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen>
    with TickerProviderStateMixin {
  late final AnimationController _pulseController;
  late final AnimationController _rotateController;

  int _currentStep = 0;
  final List<({String title, String subtitle})> _steps = const [
    (
      title: 'Preprocessing image',
      subtitle: 'Normalising and resizing your photo for the models.',
    ),
    (
      title: 'Detecting anomalies (VAE)',
      subtitle: 'Checking which areas look unusual versus healthy skin.',
    ),
    (
      title: 'Classifying condition (EfficientNet)',
      subtitle: 'Weighing the evidence for Acne, Eczema, and Tinea.',
    ),
    (
      title: 'Explaining the result (Score-CAM)',
      subtitle: 'Mapping which regions drove the prediction.',
    ),
  ];

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);

    _rotateController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();

    _runAnalysis();
  }

  Future<void> _runAnalysis() async {
    try {
      final result = await runAnalysisInBackground(
        imagePath: widget.imagePath,
        onStepChange: (step) {
          if (mounted) setState(() => _currentStep = step);
        },
      );

      if (!mounted) return;

      final ScanResult scan;
      if (result.isNormal) {
        scan = ScanResult(
          id: const Uuid().v4(),
          imagePath: widget.imagePath,
          bodyPart: widget.bodyPart,
          diagnosis: 'No Disease Detected',
          confidence: 1.0,
          classProbabilities: const {},
          timestamp: DateTime.now(),
          anomalyRatio: result.anomalyRatio,
          vaeHeatmapPath: result.vaeHeatmapPath,
          preprocessMs: result.preprocessMs,
          vaeMs: result.vaeMs,
          cnnMs: result.cnnMs,
          scoreCamMs: result.scoreCamMs,
        );
      } else {
        scan = ScanResult(
          id: const Uuid().v4(),
          imagePath: widget.imagePath,
          bodyPart: widget.bodyPart,
          diagnosis: result.diagnosis,
          confidence: result.confidence,
          classProbabilities: result.classProbabilities,
          timestamp: DateTime.now(),
          heatmapPath: result.heatmapPath,
          classHeatmapPaths: result.classHeatmapPaths,
          xaiRationale: result.xaiRationale,
          anomalyRatio: result.anomalyRatio,
          vaeHeatmapPath: result.vaeHeatmapPath,
          preprocessMs: result.preprocessMs,
          vaeMs: result.vaeMs,
          cnnMs: result.cnnMs,
          scoreCamMs: result.scoreCamMs,
        );
      }

      // Mark all steps done, then navigate
      if (mounted) setState(() => _currentStep = _steps.length);
      await Future.delayed(const Duration(milliseconds: 400));
      if (mounted) {
        context.pushReplacement(AppRoutes.analysisResults, extra: scan);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Analysis failed: $e')),
      );
      if (mounted) context.pop();
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _rotateController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Spacer(flex: 2),

              // Animated scanning circle
              SizedBox(
                width: 160,
                height: 160,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    AnimatedBuilder(
                      animation: _pulseController,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: 1.0 + _pulseController.value * 0.12,
                          child: Container(
                            width: 160,
                            height: 160,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: AppColors.primary.withValues(
                                  alpha: 0.08 * (1 - _pulseController.value)),
                            ),
                          ),
                        );
                      },
                    ),
                    AnimatedBuilder(
                      animation: _rotateController,
                      builder: (context, child) {
                        return Transform.rotate(
                          angle: _rotateController.value * 2 * 3.14159,
                          child: SizedBox(
                            width: 130,
                            height: 130,
                            child: CircularProgressIndicator(
                              value: null,
                              strokeWidth: 4,
                              color: AppColors.primary,
                              backgroundColor:
                                  AppColors.primary.withValues(alpha: 0.12),
                            ),
                          ),
                        );
                      },
                    ),
                    Container(
                      width: 88,
                      height: 88,
                      decoration: const BoxDecoration(
                        color: AppColors.primaryLight,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.biotech_outlined,
                          color: AppColors.primary, size: 40),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 36),

              const Text(
                'Analyzing your skin...',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                widget.bodyPart.isNotEmpty
                    ? 'Scanning: ${widget.bodyPart}'
                    : 'Our AI is carefully examining your image.',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 14,
                  color: AppColors.textSecondary,
                  height: 1.55,
                ),
              ),
              const SizedBox(height: 40),

              // Steps
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFFF8FAFC),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  children: List.generate(_steps.length, (i) {
                    final isDone = _currentStep > i;
                    final isActive = _currentStep == i;

                    return Padding(
                      padding: EdgeInsets.only(
                          bottom: i < _steps.length - 1 ? 14 : 0),
                      child: Row(
                        children: [
                          AnimatedContainer(
                            duration: const Duration(milliseconds: 300),
                            width: 28,
                            height: 28,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: isDone
                                  ? AppColors.success
                                  : isActive
                                      ? AppColors.primary
                                      : AppColors.border,
                            ),
                            child: Center(
                              child: AnimatedSwitcher(
                                duration: const Duration(milliseconds: 250),
                                child: isDone
                                    ? const Icon(Icons.check,
                                        key: ValueKey('done'),
                                        color: Colors.white,
                                        size: 16)
                                    : isActive
                                        ? const SizedBox(
                                            key: ValueKey('active'),
                                            width: 14,
                                            height: 14,
                                            child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                              color: Colors.white,
                                            ),
                                          )
                                        : const Icon(Icons.circle,
                                            key: ValueKey('pending'),
                                            color: Colors.white,
                                            size: 8),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                AnimatedDefaultTextStyle(
                                  duration: const Duration(milliseconds: 250),
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: isActive || isDone
                                        ? FontWeight.w600
                                        : FontWeight.w400,
                                    color: isDone
                                        ? AppColors.success
                                        : isActive
                                            ? AppColors.primary
                                            : AppColors.textSecondary,
                                  ),
                                  child: Text(_steps[i].title),
                                ),
                                AnimatedSize(
                                  duration: const Duration(milliseconds: 250),
                                  alignment: Alignment.topLeft,
                                  child: isActive
                                      ? Padding(
                                          padding:
                                              const EdgeInsets.only(top: 2),
                                          child: Text(
                                            _steps[i].subtitle,
                                            style: const TextStyle(
                                              fontSize: 11,
                                              color: AppColors.textSecondary,
                                              height: 1.35,
                                            ),
                                          ),
                                        )
                                      : const SizedBox.shrink(),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                ),
              ),

              const Spacer(flex: 3),

              const Text(
                'Running on-device AI — no data leaves your phone.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 11, color: AppColors.textSecondary),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
