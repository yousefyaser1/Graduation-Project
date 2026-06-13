import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

/// Salted PBKDF2-HMAC-SHA256 password hashing.
///
/// Stored format (4 `$`-separated fields):
///   `pbkdf2_sha256$<iterations>$<base64 salt>$<base64 derived key>`
///
/// This replaces the previous plaintext password storage. Verification is
/// constant-time, and a per-user random salt means identical passwords produce
/// different stored values. [isHashed] lets the auth layer detect legacy
/// plaintext rows and transparently upgrade them on next successful login.
class PasswordHasher {
  static const String _prefix = 'pbkdf2_sha256';
  static const int _iterations = 100000; // OWASP-recommended floor for SHA-256
  static const int _keyLen = 32; // bytes (256-bit derived key)
  static const int _saltLen = 16; // bytes

  /// Hash [password] with a fresh random salt. Pass [salt] only for tests.
  static String hash(String password, {Uint8List? salt}) {
    final s = salt ?? _randomSalt(_saltLen);
    final dk = _pbkdf2(utf8.encode(password), s, _iterations, _keyLen);
    return '$_prefix\$$_iterations\$${base64.encode(s)}\$${base64.encode(dk)}';
  }

  /// True if [password] matches the [stored] hash. Returns false for malformed
  /// or non-hashed input rather than throwing.
  static bool verify(String password, String stored) {
    final parts = stored.split('\$');
    if (parts.length != 4 || parts[0] != _prefix) return false;
    final iterations = int.tryParse(parts[1]);
    if (iterations == null || iterations < 1) return false;
    final Uint8List salt;
    final Uint8List expected;
    try {
      salt = base64.decode(parts[2]);
      expected = base64.decode(parts[3]);
    } catch (_) {
      return false;
    }
    final dk = _pbkdf2(utf8.encode(password), salt, iterations, expected.length);
    return _constantTimeEquals(dk, expected);
  }

  /// True if [stored] is already in the PBKDF2 format (vs. legacy plaintext).
  static bool isHashed(String stored) => stored.startsWith('$_prefix\$');

  // ── Internals ──────────────────────────────────────────────────────────────

  static Uint8List _randomSalt(int n) {
    final rng = Random.secure();
    final out = Uint8List(n);
    for (var i = 0; i < n; i++) {
      out[i] = rng.nextInt(256);
    }
    return out;
  }

  static bool _constantTimeEquals(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    var diff = 0;
    for (var i = 0; i < a.length; i++) {
      diff |= a[i] ^ b[i];
    }
    return diff == 0;
  }

  /// PBKDF2 (RFC 2898) with HMAC-SHA256 as the PRF.
  static Uint8List _pbkdf2(
      List<int> password, List<int> salt, int iterations, int keyLen) {
    final hmac = Hmac(sha256, password);
    const hLen = 32; // SHA-256 output size
    final numBlocks = (keyLen / hLen).ceil();
    final out = Uint8List(numBlocks * hLen);

    for (var block = 1; block <= numBlocks; block++) {
      // U_1 = PRF(password, salt || INT_32_BE(block))
      final msg = Uint8List(salt.length + 4)
        ..setRange(0, salt.length, salt)
        ..[salt.length] = (block >> 24) & 0xff
        ..[salt.length + 1] = (block >> 16) & 0xff
        ..[salt.length + 2] = (block >> 8) & 0xff
        ..[salt.length + 3] = block & 0xff;

      var u = Uint8List.fromList(hmac.convert(msg).bytes);
      final t = Uint8List.fromList(u);
      for (var i = 1; i < iterations; i++) {
        u = Uint8List.fromList(hmac.convert(u).bytes);
        for (var j = 0; j < hLen; j++) {
          t[j] ^= u[j];
        }
      }
      out.setRange((block - 1) * hLen, block * hLen, t);
    }
    return Uint8List.sublistView(out, 0, keyLen);
  }
}
