import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../providers/user_provider.dart';

class ForgotPasswordScreen extends ConsumerStatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  ConsumerState<ForgotPasswordScreen> createState() =>
      _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends ConsumerState<ForgotPasswordScreen> {
  final _emailController = TextEditingController();
  final _newPasswordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();

  // 0 = email step, 1 = new password step, 2 = success
  int _step = 0;
  bool _loading = false;
  bool _obscureNew = true;
  bool _obscureConfirm = true;
  String? _error;

  @override
  void dispose() {
    _emailController.dispose();
    _newPasswordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  Future<void> _verifyEmail() async {
    final email = _emailController.text.trim();
    if (email.isEmpty) {
      setState(() => _error = 'Please enter your email address.');
      return;
    }
    // Just move to password step; resetPassword will validate email existence
    setState(() {
      _error = null;
      _step = 1;
    });
  }

  Future<void> _resetPassword() async {
    final newPw = _newPasswordController.text;
    final confirmPw = _confirmPasswordController.text;

    if (newPw.length < 6) {
      setState(() => _error = 'Password must be at least 6 characters.');
      return;
    }
    if (newPw != confirmPw) {
      setState(() => _error = 'Passwords do not match.');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });
    final error = await ref
        .read(userProvider.notifier)
        .resetPassword(_emailController.text.trim(), newPw);
    setState(() => _loading = false);

    if (error != null) {
      setState(() => _error = error);
      return;
    }
    setState(() => _step = 2);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () {
            if (_step == 1) {
              setState(() {
                _step = 0;
                _error = null;
              });
            } else {
              context.pop();
            }
          },
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: _step == 2
              ? _buildSuccessState()
              : _step == 1
                  ? _buildNewPasswordStep()
                  : _buildEmailStep(),
        ),
      ),
    );
  }

  Widget _buildEmailStep() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 16),
        Container(
          width: 64,
          height: 64,
          decoration: const BoxDecoration(
            color: AppColors.primaryLight,
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.lock_reset_outlined,
              color: AppColors.primary, size: 30),
        ),
        const SizedBox(height: 20),
        const Text(
          'Forgot password?',
          style: TextStyle(
            fontSize: 26,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          "Enter the email linked to your account and we'll verify it.",
          style: TextStyle(
            fontSize: 14,
            color: AppColors.textSecondary,
            height: 1.5,
          ),
        ),
        const SizedBox(height: 32),

        const Text(
          'Email address',
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AppColors.textPrimary),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _emailController,
          keyboardType: TextInputType.emailAddress,
          decoration: InputDecoration(
            hintText: 'you@example.com',
            hintStyle: const TextStyle(color: AppColors.textSecondary),
            prefixIcon: const Icon(Icons.email_outlined,
                color: AppColors.textSecondary, size: 20),
            filled: true,
            fillColor: AppColors.primaryLight,
            border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppColors.border)),
            enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppColors.border)),
            focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide:
                    const BorderSide(color: AppColors.primary, width: 1.5)),
          ),
        ),

        if (_error != null) ...[
          const SizedBox(height: 10),
          Text(_error!,
              style: const TextStyle(color: AppColors.error, fontSize: 13)),
        ],

        const SizedBox(height: 28),

        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: _loading ? null : _verifyEmail,
            child: _loading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : const Text('Continue'),
          ),
        ),
        const SizedBox(height: 16),

        Center(
          child: TextButton.icon(
            onPressed: () => context.go(AppRoutes.login),
            icon: const Icon(Icons.arrow_back, size: 16),
            label: const Text('Back to Sign In'),
            style:
                TextButton.styleFrom(foregroundColor: AppColors.textSecondary),
          ),
        ),
      ],
    );
  }

  Widget _buildNewPasswordStep() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 16),
        Container(
          width: 64,
          height: 64,
          decoration: const BoxDecoration(
            color: AppColors.primaryLight,
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.lock_outline,
              color: AppColors.primary, size: 30),
        ),
        const SizedBox(height: 20),
        const Text(
          'Create new password',
          style: TextStyle(
            fontSize: 26,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          'Your new password must be at least 6 characters.',
          style: TextStyle(
            fontSize: 14,
            color: AppColors.textSecondary,
            height: 1.5,
          ),
        ),
        const SizedBox(height: 32),

        const Text(
          'New password',
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AppColors.textPrimary),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _newPasswordController,
          obscureText: _obscureNew,
          decoration: InputDecoration(
            hintText: 'Enter new password',
            hintStyle: const TextStyle(color: AppColors.textSecondary),
            prefixIcon: const Icon(Icons.lock_outline,
                color: AppColors.textSecondary, size: 20),
            suffixIcon: IconButton(
              icon: Icon(
                  _obscureNew ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                  color: AppColors.textSecondary,
                  size: 20),
              onPressed: () => setState(() => _obscureNew = !_obscureNew),
            ),
            filled: true,
            fillColor: AppColors.primaryLight,
            border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppColors.border)),
            enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppColors.border)),
            focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide:
                    const BorderSide(color: AppColors.primary, width: 1.5)),
          ),
        ),
        const SizedBox(height: 16),

        const Text(
          'Confirm password',
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AppColors.textPrimary),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _confirmPasswordController,
          obscureText: _obscureConfirm,
          decoration: InputDecoration(
            hintText: 'Re-enter new password',
            hintStyle: const TextStyle(color: AppColors.textSecondary),
            prefixIcon: const Icon(Icons.lock_outline,
                color: AppColors.textSecondary, size: 20),
            suffixIcon: IconButton(
              icon: Icon(
                  _obscureConfirm
                      ? Icons.visibility_off_outlined
                      : Icons.visibility_outlined,
                  color: AppColors.textSecondary,
                  size: 20),
              onPressed: () =>
                  setState(() => _obscureConfirm = !_obscureConfirm),
            ),
            filled: true,
            fillColor: AppColors.primaryLight,
            border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppColors.border)),
            enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppColors.border)),
            focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide:
                    const BorderSide(color: AppColors.primary, width: 1.5)),
          ),
        ),

        if (_error != null) ...[
          const SizedBox(height: 10),
          Text(_error!,
              style: const TextStyle(color: AppColors.error, fontSize: 13)),
        ],

        const SizedBox(height: 28),

        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: _loading ? null : _resetPassword,
            child: _loading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : const Text('Reset Password'),
          ),
        ),
      ],
    );
  }

  Widget _buildSuccessState() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Container(
          width: 88,
          height: 88,
          decoration: const BoxDecoration(
            color: Color(0xFFDCFCE7),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.check_circle_outline,
              color: AppColors.success, size: 46),
        ),
        const SizedBox(height: 24),
        const Text(
          'Password reset!',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 10),
        const Text(
          'Your password has been updated successfully.\nYou can now sign in with your new password.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 14,
            color: AppColors.textSecondary,
            height: 1.55,
          ),
        ),
        const SizedBox(height: 36),

        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: () => context.go(AppRoutes.login),
            child: const Text('Back to Sign In'),
          ),
        ),
      ],
    );
  }
}
