import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  final _notifications = [
    {
      'type': 'result',
      'title': 'Analysis Complete',
      'body': 'Your scan from today has been analyzed. View results.',
      'time': 'Just now',
      'read': false,
    },
    {
      'type': 'appointment',
      'title': 'Appointment Reminder',
      'body': 'Dr. Sarah Johnson — Tomorrow at 10:30 AM.',
      'time': '1h ago',
      'read': false,
    },
    {
      'type': 'tip',
      'title': 'Skin Care Tip',
      'body': 'Remember to apply sunscreen daily, even on cloudy days.',
      'time': '3h ago',
      'read': true,
    },
    {
      'type': 'result',
      'title': 'Specialist Reviewed Your Scan',
      'body': 'Dr. Mohammed Ali has reviewed your Oct 18 scan.',
      'time': 'Yesterday',
      'read': true,
    },
    {
      'type': 'tip',
      'title': 'Weekly Skin Check',
      'body': "It's been 7 days since your last scan. Time for a check-up?",
      'time': '2 days ago',
      'read': true,
    },
    {
      'type': 'appointment',
      'title': 'Appointment Confirmed',
      'body': 'Your appointment with Dr. Emily Chen on Nov 2 is confirmed.',
      'time': '3 days ago',
      'read': true,
    },
  ];

  @override
  Widget build(BuildContext context) {
    final unread = _notifications.where((n) => n['read'] == false).length;

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
              onPressed: () => setState(() {
                for (final n in _notifications) {
                  n['read'] = true;
                }
              }),
              child: Text('Mark all read',
                  style: TextStyle(
                      color: AppColors.primary, fontSize: 13)),
            ),
        ],
      ),
      body: _notifications.isEmpty
          ? _buildEmptyState()
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: _notifications.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, i) {
                final n = _notifications[i];
                final isRead = n['read'] as bool;
                return GestureDetector(
                  onTap: () {
                    setState(() => n['read'] = true);
                    if (n['type'] == 'result') {
                      context.push(AppRoutes.history);
                    } else if (n['type'] == 'appointment') {
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
                            color: _iconBg(n['type'] as String),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            _iconData(n['type'] as String),
                            color: _iconColor(n['type'] as String),
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
                                      n['title'] as String,
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
                                      decoration: BoxDecoration(
                                        color: AppColors.primary,
                                        shape: BoxShape.circle,
                                      ),
                                    ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              Text(
                                n['body'] as String,
                                style: const TextStyle(
                                    fontSize: 12,
                                    color: AppColors.textSecondary,
                                    height: 1.45),
                              ),
                              const SizedBox(height: 6),
                              Text(
                                n['time'] as String,
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
            decoration: BoxDecoration(
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
          const Text("You're all caught up!",
              style:
                  TextStyle(fontSize: 13, color: AppColors.textSecondary)),
        ],
      ),
    );
  }
}
