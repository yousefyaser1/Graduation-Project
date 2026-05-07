import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../models/scan_result.dart';
import '../../../providers/scan_provider.dart';

class SpecialistDashboardScreen extends ConsumerStatefulWidget {
  const SpecialistDashboardScreen({super.key});

  @override
  ConsumerState<SpecialistDashboardScreen> createState() =>
      _SpecialistDashboardScreenState();
}

class _SpecialistDashboardScreenState
    extends ConsumerState<SpecialistDashboardScreen> {
  int _selectedTab = 0;
  String _searchQuery = '';
  final _searchController = TextEditingController();
  bool _showSearch = false;

  // Hardcoded schedule (appointments need a backend — kept as demo)
  final _appointments = [
    {'patient': 'Ahmed Yasser', 'time': '09:00 AM', 'type': 'Follow-up', 'color': 0xFF3B82F6},
    {'patient': 'Sara Ali', 'time': '10:30 AM', 'type': 'First Consultation', 'color': 0xFFEF4444},
    {'patient': 'Omar Khalid', 'time': '12:00 PM', 'type': 'Results Review', 'color': 0xFF8B5CF6},
    {'patient': 'Lina Mahmoud', 'time': '02:30 PM', 'type': 'Follow-up', 'color': 0xFF10B981},
  ];

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  // Map a ScanResult to the patient card data format.
  Map<String, dynamic> _scanToCard(ScanResult scan, int index) {
    final now = DateTime.now();
    final isPending = now.difference(scan.timestamp).inDays < 3;
    final isNormal = scan.diagnosis == 'No Disease Detected';
    return {
      'name': scan.bodyPart.isNotEmpty ? '${scan.bodyPart} Scan' : 'Skin Scan #${index + 1}',
      'date': _relativeDate(scan.timestamp),
      'diagnosis': isNormal ? 'Normal Skin' : scan.diagnosis,
      'risk': 'Low Risk',
      'confidence': (scan.confidence * 100).round(),
      'status': isPending ? 'Pending Review' : 'Reviewed',
      'scan': scan,
    };
  }

  String _relativeDate(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Yesterday';
    if (diff.inDays < 7) return DateFormat('EEEE').format(dt);
    return DateFormat('MMM d').format(dt);
  }

  List<Map<String, dynamic>> _buildPatientList(List<ScanResult> scans) {
    final cards = scans.asMap().entries
        .map((e) => _scanToCard(e.value, e.key))
        .toList();

    List<Map<String, dynamic>> filtered;
    if (_selectedTab == 0) {
      filtered = List.from(cards);
    } else if (_selectedTab == 1) {
      filtered = cards.where((p) => p['status'] == 'Pending Review').toList();
    } else if (_selectedTab == 2) {
      filtered = cards.where((p) => p['status'] == 'Reviewed').toList();
    } else {
      filtered = [];
    }

    if (_searchQuery.isNotEmpty) {
      final q = _searchQuery.toLowerCase();
      filtered = filtered
          .where((p) =>
              (p['name'] as String).toLowerCase().contains(q) ||
              (p['diagnosis'] as String).toLowerCase().contains(q))
          .toList();
    }
    return filtered;
  }

  @override
  Widget build(BuildContext context) {
    final scansAsync = ref.watch(scanListProvider);

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        automaticallyImplyLeading: false,
        title: _showSearch
            ? TextField(
                controller: _searchController,
                autofocus: true,
                decoration: const InputDecoration(
                  hintText: 'Search scans or diagnosis...',
                  border: InputBorder.none,
                  hintStyle: TextStyle(fontSize: 14, color: AppColors.textSecondary),
                ),
                onChanged: (v) => setState(() => _searchQuery = v),
              )
            : RichText(
                text: const TextSpan(
                  children: [
                    TextSpan(
                      text: 'SkinScan ',
                      style: TextStyle(
                          color: AppColors.textPrimary,
                          fontWeight: FontWeight.w800,
                          fontSize: 18),
                    ),
                    TextSpan(
                      text: 'AI',
                      style: TextStyle(
                          color: AppColors.primary,
                          fontWeight: FontWeight.w800,
                          fontSize: 18),
                    ),
                  ],
                ),
              ),
        actions: [
          IconButton(
            icon: Icon(
              _showSearch ? Icons.close : Icons.search_rounded,
              color: AppColors.textPrimary,
            ),
            onPressed: () => setState(() {
              _showSearch = !_showSearch;
              if (!_showSearch) {
                _searchQuery = '';
                _searchController.clear();
              }
            }),
          ),
          IconButton(
            icon: const Icon(Icons.notifications_outlined, color: AppColors.textPrimary),
            onPressed: () => context.go(AppRoutes.notifications),
          ),
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0xFFEDE9FE),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Text('Specialist',
                style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF7C3AED))),
          ),
        ],
      ),
      body: scansAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(
          child: Text('Error loading scans: $e',
              style: const TextStyle(color: AppColors.error)),
        ),
        data: (scans) {
          if (_selectedTab == 3) return _buildScheduleTab();

          final allCards = scans.asMap().entries
              .map((e) => _scanToCard(e.value, e.key))
              .toList();
          final pendingCount =
              allCards.where((p) => p['status'] == 'Pending Review').length;
          final avgConf = scans.isEmpty
              ? 'N/A'
              : '${(scans.map((s) => s.confidence).reduce((a, b) => a + b) / scans.length * 100).round()}%';

          final filtered = _buildPatientList(scans);

          return Column(
            children: [
              // Stats row (real data)
              Container(
                color: Colors.white,
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
                child: Row(
                  children: [
                    _SpecialistStat(
                        value: '${scans.length}',
                        label: 'Total\nScans',
                        color: AppColors.primary),
                    _SpecialistStat(
                        value: '$pendingCount',
                        label: 'Pending\nReview',
                        color: AppColors.warning),
                    _SpecialistStat(
                        value: '${_appointments.length}',
                        label: 'Scheduled\nToday',
                        color: const Color(0xFF8B5CF6)),
                    _SpecialistStat(
                        value: avgConf,
                        label: 'Avg\nConfidence',
                        color: AppColors.success),
                  ],
                ),
              ),

              // Tabs
              Container(
                color: Colors.white,
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      _Tab(label: 'All', isActive: _selectedTab == 0,
                          onTap: () => setState(() => _selectedTab = 0)),
                      const SizedBox(width: 8),
                      _Tab(label: 'Pending', isActive: _selectedTab == 1,
                          onTap: () => setState(() => _selectedTab = 1)),
                      const SizedBox(width: 8),
                      _Tab(label: 'Reviewed', isActive: _selectedTab == 2,
                          onTap: () => setState(() => _selectedTab = 2)),
                      const SizedBox(width: 8),
                      _Tab(label: 'Schedule', isActive: _selectedTab == 3,
                          onTap: () => setState(() => _selectedTab = 3)),
                    ],
                  ),
                ),
              ),

              // Scan list
              Expanded(
                child: filtered.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.biotech_outlined,
                                size: 48, color: AppColors.textSecondary),
                            const SizedBox(height: 12),
                            Text(
                              scans.isEmpty
                                  ? 'No scans recorded yet'
                                  : _searchQuery.isNotEmpty
                                      ? 'No scans match "$_searchQuery"'
                                      : 'No scans in this category',
                              style: const TextStyle(
                                  fontSize: 14, color: AppColors.textSecondary),
                            ),
                            if (scans.isEmpty) ...[
                              const SizedBox(height: 8),
                              const Text(
                                'Run the AI pipeline to see scan results here.',
                                style: TextStyle(
                                    fontSize: 12, color: AppColors.textSecondary),
                              ),
                            ],
                          ],
                        ),
                      )
                    : ListView.separated(
                        padding: const EdgeInsets.all(16),
                        itemCount: filtered.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 12),
                        itemBuilder: (context, i) => _PatientCard(
                          patient: filtered[i],
                          onTap: () {
                            final scan = filtered[i]['scan'] as ScanResult;
                            context.push(AppRoutes.analysisResults, extra: scan);
                          },
                        ),
                      ),
              ),
            ],
          );
        },
      ),
      bottomNavigationBar: _buildSpecialistNav(context),
    );
  }

  Widget _buildScheduleTab() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
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
                decoration: const BoxDecoration(
                    color: AppColors.primaryLight, shape: BoxShape.circle),
                child: const Icon(Icons.calendar_today_outlined,
                    color: AppColors.primary, size: 22),
              ),
              const SizedBox(width: 14),
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Today',
                    style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary)),
                Text(
                  '${_appointments.length} appointments scheduled',
                  style: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
                ),
              ]),
            ],
          ),
        ),
        const SizedBox(height: 16),
        const Text('Upcoming Appointments',
            style: TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary)),
        const SizedBox(height: 12),
        ..._appointments.map((appt) {
          final c = Color(appt['color'] as int);
          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
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
                    width: 52,
                    height: 52,
                    decoration: BoxDecoration(
                        color: c.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(10)),
                    child: Icon(Icons.access_time_rounded, color: c, size: 24),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(appt['patient'] as String,
                          style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: AppColors.textPrimary)),
                      const SizedBox(height: 3),
                      Text(appt['type'] as String,
                          style: const TextStyle(
                              fontSize: 12, color: AppColors.textSecondary)),
                    ]),
                  ),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                        color: c.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(20)),
                    child: Text(appt['time'] as String,
                        style: TextStyle(
                            fontSize: 12, fontWeight: FontWeight.w700, color: c)),
                  ),
                ],
              ),
            ),
          );
        }),
      ],
    );
  }

  Widget _buildSpecialistNav(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: AppColors.border)),
      ),
      child: SafeArea(
        child: SizedBox(
          height: 64,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _NavBtn(
                  icon: Icons.dashboard_outlined,
                  label: 'Dashboard',
                  isActive: true,
                  onTap: () {}),
              _NavBtn(
                  icon: Icons.biotech_outlined,
                  label: 'Scans',
                  isActive: false,
                  onTap: () => setState(() => _selectedTab = 0)),
              _NavBtn(
                  icon: Icons.calendar_month_outlined,
                  label: 'Schedule',
                  isActive: false,
                  onTap: () => setState(() => _selectedTab = 3)),
              _NavBtn(
                  icon: Icons.person_outline,
                  label: 'Profile',
                  isActive: false,
                  onTap: () => context.go(AppRoutes.profile)),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Subwidgets ────────────────────────────────────────────────────────────────

class _SpecialistStat extends StatelessWidget {
  final String value;
  final String label;
  final Color color;
  const _SpecialistStat({required this.value, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(children: [
        Text(value,
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: color)),
        const SizedBox(height: 3),
        Text(label,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 10, color: AppColors.textSecondary)),
      ]),
    );
  }
}

class _Tab extends StatelessWidget {
  final String label;
  final bool isActive;
  final VoidCallback onTap;
  const _Tab({required this.label, required this.isActive, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
        decoration: BoxDecoration(
          color: isActive ? AppColors.primary : AppColors.primaryLight,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: isActive ? AppColors.primary : AppColors.border),
        ),
        child: Text(label,
            style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: isActive ? Colors.white : AppColors.primary)),
      ),
    );
  }
}

class _PatientCard extends StatelessWidget {
  final Map<String, dynamic> patient;
  final VoidCallback onTap;
  const _PatientCard({required this.patient, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final isLowRisk = patient['risk'] == 'Low Risk';
    final isPending = patient['status'] == 'Pending Review';

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
              color: isPending
                  ? const Color(0xFFFCD34D)
                  : AppColors.border,
              width: isPending ? 1.5 : 1),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: AppColors.primaryLight,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.biotech_outlined,
                  color: AppColors.primary, size: 22),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(patient['name'] as String,
                    style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary)),
                Text(patient['date'] as String,
                    style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
              ]),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: isPending
                    ? const Color(0xFFFFFBEB)
                    : const Color(0xFFDCFCE7),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                    color: isPending
                        ? const Color(0xFFFCD34D)
                        : AppColors.success),
              ),
              child: Text(
                patient['status'] as String,
                style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: isPending
                        ? const Color(0xFF92400E)
                        : AppColors.success),
              ),
            ),
          ]),
          const SizedBox(height: 12),
          const Divider(color: AppColors.border, height: 1),
          const SizedBox(height: 12),
          Row(children: [
            _InfoChip(
                label: patient['diagnosis'] as String,
                color: AppColors.primaryLight,
                textColor: AppColors.primary),
            const SizedBox(width: 8),
            _InfoChip(
                label: patient['risk'] as String,
                color: isLowRisk
                    ? const Color(0xFFDCFCE7)
                    : const Color(0xFFFEE2E2),
                textColor: isLowRisk
                    ? const Color(0xFF16A34A)
                    : const Color(0xFFDC2626)),
            const Spacer(),
            Text('${patient['confidence']}% confidence',
                style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
            const SizedBox(width: 8),
            const Icon(Icons.chevron_right, color: AppColors.textSecondary, size: 18),
          ]),
        ]),
      ),
    );
  }
}

class _InfoChip extends StatelessWidget {
  final String label;
  final Color color;
  final Color textColor;
  const _InfoChip({required this.label, required this.color, required this.textColor});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration:
          BoxDecoration(color: color, borderRadius: BorderRadius.circular(20)),
      child: Text(label,
          style: TextStyle(
              fontSize: 11, fontWeight: FontWeight.w600, color: textColor)),
    );
  }
}

class _NavBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isActive;
  final VoidCallback onTap;
  const _NavBtn({required this.icon, required this.label, required this.isActive, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: SizedBox(
        width: 64,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon,
                color: isActive ? AppColors.primary : AppColors.textSecondary,
                size: 24),
            const SizedBox(height: 3),
            Text(label,
                style: TextStyle(
                    fontSize: 11,
                    color: isActive ? AppColors.primary : AppColors.textSecondary,
                    fontWeight: isActive ? FontWeight.w600 : FontWeight.w400)),
          ],
        ),
      ),
    );
  }
}
