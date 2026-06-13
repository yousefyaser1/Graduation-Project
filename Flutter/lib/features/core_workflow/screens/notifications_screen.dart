import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../models/appointment.dart';
import '../../../models/scan_result.dart';
import '../../../providers/appointment_provider.dart';
import '../../../providers/scan_provider.dart';
import '../../../providers/user_provider.dart';

/// A single notification, derived from real app data (scans + appointments).
class _Notif {
  final String id;
  final String type; // 'result' | 'appointment' | 'tip'
  final String title;
  final String body;
  final DateTime time;
  final ScanResult? scan; // set for result notifications

  const _Notif({
    required this.id,
    required this.type,
    required this.title,
    required this.body,
    required this.time,
    this.scan,
  });
}

class NotificationsScreen extends ConsumerStatefulWidget {
  const NotificationsScreen({super.key});

  @override
  ConsumerState<NotificationsScreen> createState() =>
      _NotificationsScreenState();
}

class _NotificationsScreenState extends ConsumerState<NotificationsScreen> {
  // Ids the user has opened/marked read this session.
  final Set<String> _read = {};

  /// Build the notification feed from scans + this user's appointments.
  List<_Notif> _buildFeed(
      List<ScanResult> scans, List<Appointment> appointments) {
    final items = <_Notif>[];

    // Scan results (most recent first, capped to keep the feed focused).
    for (final s in scans.take(15)) {
      final isNormal = s.diagnosis == 'No Disease Detected';
      items.add(_Notif(
        id: 'scan_${s.id}',
        type: 'result',
        title: 'Analysis Complete',
        body: isNormal
            ? 'Your ${_part(s.bodyPart)} scan looked normal — no condition detected.'
            : 'Your ${_part(s.bodyPart)} scan was analyzed: ${s.diagnosis} (${(s.confidence * 100).round()}%).',
        time: s.timestamp,
        scan: s,
      ));
    }

    // Appointments booked by this user.
    for (final a in appointments) {
      items.add(_Notif(
        id: 'appt_${a.id}',
        type: 'appointment',
        title: a.status == 'Scheduled'
            ? 'Appointment Confirmed'
            : 'Appointment ${a.status}',
        body:
            '${a.doctorName} (${a.specialty}) — ${a.dateLabel} at ${a.timeLabel}.',
        time: a.createdAt,
      ));
    }

    // App-generated reminder: prompt a check-up if it's been over a week.
    if (scans.isNotEmpty) {
      final daysSince = DateTime.now().difference(scans.first.timestamp).inDays;
      if (daysSince >= 7) {
        items.add(_Notif(
          id: 'tip_weekly_${scans.first.id}',
          type: 'tip',
          title: 'Weekly Skin Check',
          body:
              "It's been $daysSince days since your last scan. Consider a quick check-up.",
          time: DateTime.now().subtract(const Duration(hours: 1)),
        ));
      }
    }

    items.sort((a, b) => b.time.compareTo(a.time));
    return items;
  }

  String _part(String bodyPart) =>
      bodyPart.trim().isEmpty ? 'skin' : bodyPart.toLowerCase();

  String _relativeTime(DateTime t) {
    final d = DateTime.now().difference(t);
    if (d.inMinutes < 1) return 'Just now';
    if (d.inMinutes < 60) return '${d.inMinutes}m ago';
    if (d.inHours < 24) return '${d.inHours}h ago';
    if (d.inDays == 1) return 'Yesterday';
    if (d.inDays < 7) return '${d.inDays} days ago';
    return '${(d.inDays / 7).floor()}w ago';
  }

  @override
  Widget build(BuildContext context) {
    final user = ref.watch(userProvider);
    final scans = ref.watch(scanListProvider).maybeWhen(
          data: (s) => s,
          orElse: () => const <ScanResult>[],
        );
    final allAppts = ref.watch(allAppointmentsProvider).maybeWhen(
          data: (a) => a,
          orElse: () => const <Appointment>[],
        );
    // Specialists see all bookings; patients see only their own.
    final appts = (user?.role == 'specialist')
        ? allAppts
        : allAppts.where((a) => a.userId == user?.id).toList();

    final feed = _buildFeed(scans, appts);
    final unread = feed.where((n) => !_read.contains(n.id)).length;

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
          'Notifications',
          style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary),
        ),
        actions: [
          if (unread > 0)
            TextButton(
              onPressed: () =>
                  setState(() => _read.addAll(feed.map((n) => n.id))),
              child: const Text('Mark all read',
                  style: TextStyle(color: AppColors.primary, fontSize: 13)),
            ),
        ],
      ),
      body: feed.isEmpty
          ? _buildEmptyState()
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: feed.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, i) {
                final n = feed[i];
                final isRead = _read.contains(n.id);
                return Semantics(
                  button: true,
                  label:
                      '${n.title}. ${n.body}. ${_relativeTime(n.time)}. ${isRead ? 'Read' : 'Unread'}',
                  child: GestureDetector(
                    onTap: () {
                      setState(() => _read.add(n.id));
                      if (n.type == 'result' && n.scan != null) {
                        context.push(AppRoutes.analysisResults, extra: n.scan);
                      } else if (n.type == 'appointment') {
                        context.push(AppRoutes.bookAppointment);
                      }
                    },
                    child: Container(
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: isRead ? Colors.white : const Color(0xFFEFF6FF),
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(
                            color: isRead
                                ? AppColors.border
                                : const Color(0xFFBFDBFE)),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            width: 42,
                            height: 42,
                            decoration: BoxDecoration(
                              color: _iconBg(n.type),
                              shape: BoxShape.circle,
                            ),
                            child: Icon(
                              _iconData(n.type),
                              color: _iconColor(n.type),
                              size: 20,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      child: Text(
                                        n.title,
                                        style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: isRead
                                                ? FontWeight.w600
                                                : FontWeight.w700,
                                            color: AppColors.textPrimary),
                                      ),
                                    ),
                                    if (!isRead)
                                      Container(
                                        width: 8,
                                        height: 8,
                                        decoration: const BoxDecoration(
                                          color: AppColors.primary,
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  n.body,
                                  style: const TextStyle(
                                      fontSize: 12,
                                      color: AppColors.textSecondary,
                                      height: 1.45),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  _relativeTime(n.time),
                                  style: const TextStyle(
                                      fontSize: 11,
                                      color: AppColors.textSecondary),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
    );
  }

  IconData _iconData(String type) {
    switch (type) {
      case 'result':      return Icons.biotech_outlined;
      case 'appointment': return Icons.calendar_month_outlined;
      default:            return Icons.lightbulb_outline;
    }
  }

  Color _iconBg(String type) {
    switch (type) {
      case 'result':      return AppColors.primaryLight;
      case 'appointment': return const Color(0xFFDCFCE7);
      default:            return const Color(0xFFFFFBEB);
    }
  }

  Color _iconColor(String type) {
    switch (type) {
      case 'result':      return AppColors.primary;
      case 'appointment': return AppColors.success;
      default:            return const Color(0xFFF59E0B);
    }
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: const BoxDecoration(
                color: AppColors.primaryLight, shape: BoxShape.circle),
            child: const Icon(Icons.notifications_none_outlined,
                color: AppColors.primary, size: 36),
          ),
          const SizedBox(height: 16),
          const Text('No notifications',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary)),
          const SizedBox(height: 6),
          const Text('Run a scan or book an appointment to see updates here.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
        ],
      ),
    );
  }
}
