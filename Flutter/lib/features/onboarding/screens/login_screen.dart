import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/routing/app_router.dart';
import '../../../providers/user_provider.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _emailController    = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;
  bool _loading = false;
  String? _emailError;
  String? _passwordError;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  bool _validate() {
    String? emailErr;
    String? passErr;
    final email = _emailController.text.trim();
    final pass = _passwordController.text;
    if (email.isEmpty) {
      emailErr = 'Email is required';
    } else if (!email.contains('@') || !email.contains('.')) {
      emailErr = 'Enter a valid email address';
    }
    if (pass.isEmpty) {
      passErr = 'Password is required';
    }
    setState(() {
      _emailError = emailErr;
      _passwordError = passErr;
    });
    return emailErr == null && passErr == null;
  }

  Future<void> _signIn() async {
    if (!_validate()) return;
    setState(() => _loading = true);
    final error = await ref.read(userProvider.notifier).login(
          _emailController.text,
          _passwordController.text,
        );
    if (!mounted) return;
    setState(() => _loading = false);

    if (error != null) {
      if (error.contains('email')) {
        setState(() => _emailError = error);
      } else {
        setState(() => _passwordError = error);
      }
    } else {
      context.go(AppRoutes.home);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: Column(
            children: [
              const SizedBox(height: 48),

              Container(
                width: 96,
                height: 96,
                decoration: const BoxDecoration(
                  color: Color(0xFFDCEFFE),
                  shape: BoxShape.circle,
                ),
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    const Icon(Icons.face_retouching_natural,
                        size: 52, color: Color(0xFF60A5FA)),
                    Positioned(
                      bottom: 14,
                      right: 14,
                      child: Container(
                        padding: const EdgeInsets.all(3),
                        decoration: const BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.search,
                            size: 16, color: Color(0xFF3B82F6)),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              const Text(
                'Welcome back',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: 32),

              // Email
              TextField(
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                onChanged: (_) { if (_emailError != null) setState(() => _emailError = null); },
                decoration: InputDecoration(
                  hintText: 'Email',
                  hintStyle: const TextStyle(color: AppColors.textSecondary),
                  errorText: _emailError,
                  prefixIcon: const Icon(Icons.email_outlined,
                      color: AppColors.textSecondary),
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
                      borderSide: const BorderSide(
                          color: AppColors.primary, width: 1.5)),
                  errorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(color: AppColors.error)),
                  focusedErrorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(color: AppColors.error, width: 1.5)),
                ),
              ),
              const SizedBox(height: 14),

              // Password
              TextField(
                controller: _passwordController,
                obscureText: _obscurePassword,
                onChanged: (_) { if (_passwordError != null) setState(() => _passwordError = null); },
                decoration: InputDecoration(
                  hintText: 'Password',
                  hintStyle: const TextStyle(color: AppColors.textSecondary),
                  errorText: _passwordError,
                  prefixIcon: const Icon(Icons.lock_outline,
                      color: AppColors.textSecondary),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _obscurePassword
                          ? Icons.visibility_off_outlined
                          : Icons.visibility_outlined,
                      color: AppColors.textSecondary,
                      size: 20,
                    ),
                    onPressed: () =>
                        setState(() => _obscurePassword = !_obscurePassword),
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
                      borderSide: const BorderSide(
                          color: AppColors.primary, width: 1.5)),
                  errorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(color: AppColors.error)),
                  focusedErrorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(color: AppColors.error, width: 1.5)),
                ),
              ),
              const SizedBox(height: 8),

              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => context.push(AppRoutes.forgotPassword),
                  child: Text(
                    'Forgot password?',
                    style: TextStyle(
                        fontSize: 13,
                        color: AppColors.primary,
                        fontWeight: FontWeight.w500),
                  ),
                ),
              ),
              const SizedBox(height: 8),

              // Sign In button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _loading ? null : _signIn,
                  child: _loading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white))
                      : const Text('Sign In'),
                ),
              ),
              const SizedBox(height: 14),

              Row(
                children: [
                  const Expanded(child: Divider(color: AppColors.border)),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: Text('or',
                        style: TextStyle(
                            color: AppColors.textSecondary, fontSize: 13)),
                  ),
                  const Expanded(child: Divider(color: AppColors.border)),
                ],
              ),
              const SizedBox(height: 14),

              // Quick demo login (skip auth for demo purposes)
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _loading
                      ? null
                      : () => context.go(AppRoutes.home),
                  icon: const Icon(Icons.play_circle_outline,
                      color: AppColors.primary),
                  label: const Text(
                    'Continue as Guest',
                    style: TextStyle(
                        color: AppColors.textPrimary,
                        fontWeight: FontWeight.w500),
                  ),
                  style: OutlinedButton.styleFrom(
                    minimumSize: const Size(double.infinity, 52),
                    side: const BorderSide(color: AppColors.border),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
              const SizedBox(height: 28),

              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    "Don't have an account? ",
                    style: TextStyle(
                        fontSize: 14, color: AppColors.textSecondary),
                  ),
                  GestureDetector(
                    onTap: () => context.push(AppRoutes.signup),
                    child: Text(
                      'Sign up',
                      style: TextStyle(
                          fontSize: 14,
                          color: AppColors.primary,
                          fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
