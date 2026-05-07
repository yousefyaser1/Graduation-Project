import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../models/scan_result.dart';

class BookAppointmentScreen extends StatefulWidget {
  final ScanResult? scan;

  const BookAppointmentScreen({super.key, this.scan});

  @override
  State<BookAppointmentScreen> createState() => _BookAppointmentScreenState();
}

class _BookAppointmentScreenState extends State<BookAppointmentScreen> {
  int _selectedDoctor  = 0;
  int _selectedDate    = 1;
  int _selectedTime    = 0;
  bool _confirmed      = false;

  final _doctors = [
    {
      'name': 'Dr. Sarah Johnson',
      'specialty': 'Dermatologist',
      'rating': '4.9',
      'reviews': '124',
      'available': 'Today',
    },
    {
      'name': 'Dr. Mohammed Ali',
      'specialty': 'Skin Specialist',
      'rating': '4.7',
      'reviews': '98',
      'available': 'Tomorrow',
    },
    {
      'name': 'Dr. Emily Chen',
      'specialty': 'Dermatologist',
      'rating': '4.8',
      'reviews': '211',
      'available': 'Nov 2',
    },
  ];

  final _dates = ['Today', 'Tomorrow', 'Nov 2', 'Nov 3', 'Nov 4'];
  final _times = ['9:00 AM', '10:30 AM', '12:00 PM', '2:00 PM', '3:30 PM', '5:00 PM'];

  @override
  Widget build(BuildContext context) {
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
          'Book Appointment',
          style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary),
        ),
      ),
      body: _confirmed ? _buildConfirmedState() : _buildBookingForm(),
    );
  }

  Widget _buildBookingForm() {
    return Column(
      children: [
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Diagnosis context card
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: const Color(0xFFDBEAFE),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline,
                          color: AppColors.primary, size: 18),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          widget.scan != null
                              ? 'Booking for: ${widget.scan!.diagnosis} — ${(widget.scan!.confidence * 100).round()}% confidence'
                              : 'Book an appointment with a specialist',
                          style: const TextStyle(
                              fontSize: 13,
                              color: AppColors.primary,
                              fontWeight: FontWeight.w500),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                _sectionTitle('Select Specialist'),
                const SizedBox(height: 12),
                ...List.generate(_doctors.length, (i) => _buildDoctorCard(i)),
                const SizedBox(height: 24),

                _sectionTitle('Select Date'),
                const SizedBox(height: 12),
                SizedBox(
                  height: 50,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: _dates.length,
                    separatorBuilder: (_, __) => const SizedBox(width: 10),
                    itemBuilder: (context, i) {
                      final isSelected = _selectedDate == i;
                      return GestureDetector(
                        onTap: () => setState(() => _selectedDate = i),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 180),
                          padding: const EdgeInsets.symmetric(horizontal: 20),
                          decoration: BoxDecoration(
                            color: isSelected
                                ? AppColors.primary
                                : Colors.white,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                                color: isSelected
                                    ? AppColors.primary
                                    : AppColors.border),
                          ),
                          child: Center(
                            child: Text(
                              _dates[i],
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                color: isSelected
                                    ? Colors.white
                                    : AppColors.textPrimary,
                              ),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 24),

                _sectionTitle('Select Time'),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: List.generate(_times.length, (i) {
                    final isSelected = _selectedTime == i;
                    return GestureDetector(
                      onTap: () => setState(() => _selectedTime = i),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 180),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 10),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? AppColors.primary
                              : Colors.white,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                              color: isSelected
                                  ? AppColors.primary
                                  : AppColors.border),
                        ),
                        child: Text(
                          _times[i],
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: isSelected
                                ? Colors.white
                                : AppColors.textPrimary,
                          ),
                        ),
                      ),
                    );
                  }),
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
        ),

        // Summary + confirm
        Container(
          padding: const EdgeInsets.fromLTRB(20, 14, 20, 28),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(top: BorderSide(color: AppColors.border)),
          ),
          child: Column(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Appointment with',
                      style: TextStyle(
                          fontSize: 12, color: AppColors.textSecondary)),
                  Text(_doctors[_selectedDoctor]['name']!,
                      style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textPrimary)),
                ],
              ),
              const SizedBox(height: 6),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Date & Time',
                      style: TextStyle(
                          fontSize: 12, color: AppColors.textSecondary)),
                  Text(
                      '${_dates[_selectedDate]}, ${_times[_selectedTime]}',
                      style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textPrimary)),
                ],
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => setState(() => _confirmed = true),
                  style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.success),
                  child: const Text('Confirm Appointment'),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildDoctorCard(int index) {
    final doc = _doctors[index];
    final isSelected = _selectedDoctor == index;
    return GestureDetector(
      onTap: () => setState(() => _selectedDoctor = index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
              color: isSelected ? AppColors.primary : AppColors.border,
              width: isSelected ? 2 : 1),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: isSelected
                    ? AppColors.primaryLight
                    : const Color(0xFFF1F5F9),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.person_outline,
                  color: isSelected
                      ? AppColors.primary
                      : AppColors.textSecondary,
                  size: 24),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(doc['name']!,
                      style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary)),
                  const SizedBox(height: 2),
                  Text(doc['specialty']!,
                      style: const TextStyle(
                          fontSize: 12, color: AppColors.textSecondary)),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      const Icon(Icons.star_rounded,
                          color: Color(0xFFF59E0B), size: 14),
                      const SizedBox(width: 3),
                      Text(
                          '${doc['rating']} (${doc['reviews']} reviews)',
                          style: const TextStyle(
                              fontSize: 11,
                              color: AppColors.textSecondary)),
                    ],
                  ),
                ],
              ),
            ),
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: const Color(0xFFDCFCE7),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(doc['available']!,
                  style: const TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF16A34A))),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConfirmedState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 28),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 96,
              height: 96,
              decoration: const BoxDecoration(
                  color: Color(0xFFDCFCE7), shape: BoxShape.circle),
              child: const Icon(Icons.check_rounded,
                  color: AppColors.success, size: 52),
            ),
            const SizedBox(height: 24),
            const Text('Appointment Confirmed!',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary)),
            const SizedBox(height: 10),
            Text(
              'Your appointment with\n${_doctors[_selectedDoctor]['name']} is booked for\n${_dates[_selectedDate]} at ${_times[_selectedTime]}.',
              textAlign: TextAlign.center,
              style: const TextStyle(
                  fontSize: 14,
                  color: AppColors.textSecondary,
                  height: 1.6),
            ),
            const SizedBox(height: 36),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => context.go(AppRoutes.home),
                child: const Text('Back to Home'),
              ),
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: () => context.go(AppRoutes.history),
              child: Text('View in History',
                  style: TextStyle(
                      color: AppColors.primary,
                      fontWeight: FontWeight.w600)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _sectionTitle(String title) => Text(
        title,
        style: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary),
      );
}
