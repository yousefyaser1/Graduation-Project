import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../core/widgets/app_bottom_nav.dart';
import '../../../models/scan_result.dart';
import '../../../providers/scan_provider.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> {
  String _filter = 'All';
  String _searchQuery = '';
  bool _searching = false;
  final _searchController = TextEditingController();

  final _filters = ['All', 'Low Risk', 'High Risk'];

  static bool _isHighRisk(ScanResult s) {
    const hrd = ['Melanoma', 'Carcinoma', 'Lymphoma'];
    return hrd.any((d) => s.diagnosis.contains(d));
  }

  List<ScanResult> _applyFilters(List<ScanResult> scans) {
    var result = scans;
    if (_filter == 'Low Risk') result = result.where((s) => !_isHighRisk(s)).toList();
    if (_filter == 'High Risk') result = result.where(_isHighRisk).toList();
    if (_searchQuery.isNotEmpty) {
      final q = _searchQuery.toLowerCase();
      result = result
          .where((s) =>
              s.diagnosis.toLowerCase().contains(q) ||
              s.bodyPart.toLowerCase().contains(q))
          .toList();
    }
    return result;
  }

  Future<bool?> _confirmDelete(BuildContext context) {
    return showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Delete scan?',
            style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
        content: const Text(
          'This will permanently remove this scan from your history.',
          style: TextStyle(fontSize: 14, color: AppColors.textSecondary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel',
                style: TextStyle(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete',
                style: TextStyle(
                    color: AppColors.error, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }

  void _showUndoSnackbar(ScanResult scan) {
    ScaffoldMessenger.of(context).clearSnackBars();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Scan deleted'),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        duration: const Duration(seconds: 4),
        action: SnackBarAction(
          label: 'Undo',
          textColor: Colors.white,
          onPressed: () async {
            await ref.saveScan(scan);
          },
        ),
      ),
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final scanAsync = ref.watch(scanListProvider);

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        automaticallyImplyLeading: false,
        title: _searching
            ? TextField(
                controller: _searchController,
                autofocus: true,
                decoration: const InputDecoration(
                  hintText: 'Search diagnosis or body part...',
                  border: InputBorder.none,
                  hintStyle:
                      TextStyle(fontSize: 14, color: AppColors.textSecondary),
                ),
                onChanged: (v) => setState(() => _searchQuery = v),
              )
            : const Text(
                'Scan History',
                style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary),
              ),
        actions: [
          if (!_searching)
            IconButton(
              tooltip: 'Progression',
              icon: const Icon(Icons.timeline_outlined,
                  color: AppColors.textPrimary),
              onPressed: () => context.push(AppRoutes.progression),
            ),
          IconButton(
            icon: Icon(
              _searching ? Icons.close : Icons.search_rounded,
              color: AppColors.textPrimary,
            ),
            onPressed: () {
              setState(() {
                _searching = !_searching;
                if (!_searching) {
                  _searchQuery = '';
                  _searchController.clear();
                }
              });
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Filter chips
          Container(
            color: Colors.white,
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
            child: Row(
              children: _filters.map((f) {
                final isActive = _filter == f;
                return GestureDetector(
                  onTap: () => setState(() => _filter = f),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    margin: const EdgeInsets.only(right: 10),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 7),
                    decoration: BoxDecoration(
                      color:
                          isActive ? AppColors.primary : AppColors.primaryLight,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: isActive ? AppColors.primary : AppColors.border,
                      ),
                    ),
                    child: Text(
                      f,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: isActive ? Colors.white : AppColors.primary,
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          ),

          // Body
          Expanded(
            child: scanAsync.when(
              loading: () => const _HistorySkeletonList(),
              error: (e, _) => Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.error_outline,
                        color: AppColors.error, size: 40),
                    const SizedBox(height: 12),
                    const Text('Error loading history',
                        style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textPrimary)),
                    const SizedBox(height: 8),
                    ElevatedButton(
                      onPressed: () => ref.invalidate(scanListProvider),
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              ),
              data: (allScans) {
                final scans = _applyFilters(allScans);

                if (allScans.isNotEmpty) {
                  return Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.fromLTRB(20, 14, 20, 8),
                        child: Row(
                          children: [
                            Text(
                              '${scans.length} scan${scans.length == 1 ? '' : 's'}',
                              style: const TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600,
                                  color: AppColors.textSecondary),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: scans.isEmpty
                            ? _buildEmptyState('No matching scans')
                            : RefreshIndicator(
                                color: AppColors.primary,
                                onRefresh: () async =>
                                    ref.invalidate(scanListProvider),
                                child: ListView.separated(
                                padding:
                                    const EdgeInsets.fromLTRB(16, 0, 16, 20),
                                physics: const AlwaysScrollableScrollPhysics(),
                                itemCount: scans.length,
                                separatorBuilder: (_, __) =>
                                    const SizedBox(height: 12),
                                itemBuilder: (context, index) => _ScanCard(
                                  scan: scans[index],
                                  isHighRisk: _isHighRisk(scans[index]),
                                  onTap: () => context.push(
                                    AppRoutes.analysisResults,
                                    extra: scans[index],
                                  ),
                                  onConfirmDelete: () =>
                                      _confirmDelete(context),
                                  onDeleted: () async {
                                    final scan = scans[index];
                                    await ref.deleteScan(scan.id);
                                    _showUndoSnackbar(scan);
                                  },
                                ),
                              ),
                            ),
                      ),
                    ],
                  );
                }
                return _buildEmptyState('No scans yet');
              },
            ),
          ),
        ],
      ),
      bottomNavigationBar: const AppBottomNav(currentIndex: 2),
    );
  }

  Widget _buildEmptyState(String message) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: const BoxDecoration(
              color: AppColors.primaryLight,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.history_outlined,
                color: AppColors.primary, size: 36),
          ),
          const SizedBox(height: 16),
          Text(
            message,
            style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary),
          ),
          const SizedBox(height: 6),
          const Text(
            'Your scan history will appear here',
            style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
          ),
        ],
      ),
    );
  }
}

class _HistorySkeletonList extends StatefulWidget {
  const _HistorySkeletonList();

  @override
  State<_HistorySkeletonList> createState() => _HistorySkeletonListState();
}

class _HistorySkeletonListState extends State<_HistorySkeletonList>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Widget _box(double h, {double? w, double r = 8}) {
    return AnimatedBuilder(
      animation: _ctrl,
      builder: (_, __) => Container(
        width: w,
        height: h,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(r),
          gradient: LinearGradient(
            begin: Alignment(_ctrl.value * 4 - 2, 0),
            end: Alignment(_ctrl.value * 4 - 0.5, 0),
            colors: const [
              Color(0xFFE8EDF2),
              Color(0xFFF5F8FA),
              Color(0xFFE8EDF2),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListView.separated(
      physics: const NeverScrollableScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
      itemCount: 5,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (_, __) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          children: [
            _box(48, w: 48, r: 12),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _box(16, w: 140, r: 6),
                  const SizedBox(height: 8),
                  _box(12, w: 100, r: 6),
                ],
              ),
            ),
            _box(24, w: 60, r: 12),
          ],
        ),
      ),
    );
  }
}

class _ScanCard extends StatelessWidget {
  final ScanResult scan;
  final bool isHighRisk;
  final VoidCallback onTap;
  final Future<bool?> Function() onConfirmDelete;
  final VoidCallback onDeleted;

  const _ScanCard({
    required this.scan,
    required this.isHighRisk,
    required this.onTap,
    required this.onConfirmDelete,
    required this.onDeleted,
  });

  @override
  Widget build(BuildContext context) {
    final dateStr = DateFormat('MMM d, yyyy').format(scan.timestamp);
    final confidence = (scan.confidence * 100).round();

    return Dismissible(
      key: Key(scan.id),
      direction: DismissDirection.endToStart,
      confirmDismiss: (_) => onConfirmDelete(),
      onDismissed: (_) => onDeleted(),
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        decoration: BoxDecoration(
          color: AppColors.error,
          borderRadius: BorderRadius.circular(14),
        ),
        child: const Icon(Icons.delete_outline, color: Colors.white),
      ),
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: isHighRisk
                      ? const Color(0xFFFEE2E2)
                      : const Color(0xFFDCFCE7),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  isHighRisk
                      ? Icons.warning_amber_outlined
                      : Icons.check_circle_outline,
                  color: isHighRisk ? AppColors.error : AppColors.success,
                  size: 24,
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      scan.diagnosis,
                      style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${scan.bodyPart}  •  $dateStr',
                      style: const TextStyle(
                          fontSize: 12, color: AppColors.textSecondary),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: isHighRisk
                          ? const Color(0xFFFEE2E2)
                          : const Color(0xFFDCFCE7),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      isHighRisk ? 'High Risk' : 'Low Risk',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: isHighRisk
                            ? const Color(0xFFDC2626)
                            : const Color(0xFF16A34A),
                      ),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '$confidence%',
                    style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textSecondary),
                  ),
                ],
              ),
              const SizedBox(width: 6),
              const Icon(Icons.chevron_right,
                  color: AppColors.textSecondary, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}
