import 'package:flutter_test/flutter_test.dart';
import 'package:dermatology_ai_app/services/security/password_hasher.dart';

void main() {
  group('PasswordHasher', () {
    test('hash output is in the expected PBKDF2 format', () {
      final h = PasswordHasher.hash('correct horse');
      expect(PasswordHasher.isHashed(h), isTrue);
      expect(h.split(r'$').length, 4);
      expect(h, startsWith('pbkdf2_sha256\$'));
    });

    test('verify accepts the correct password', () {
      final h = PasswordHasher.hash('s3cret!');
      expect(PasswordHasher.verify('s3cret!', h), isTrue);
    });

    test('verify rejects an incorrect password', () {
      final h = PasswordHasher.hash('s3cret!');
      expect(PasswordHasher.verify('wrong', h), isFalse);
    });

    test('same password hashes differently (random salt)', () {
      final a = PasswordHasher.hash('repeat');
      final b = PasswordHasher.hash('repeat');
      expect(a, isNot(equals(b)));
      // ...but both still verify.
      expect(PasswordHasher.verify('repeat', a), isTrue);
      expect(PasswordHasher.verify('repeat', b), isTrue);
    });

    test('isHashed distinguishes legacy plaintext from hashes', () {
      expect(PasswordHasher.isHashed('123'), isFalse);
      expect(PasswordHasher.isHashed(PasswordHasher.hash('123')), isTrue);
    });

    test('verify returns false for malformed stored values', () {
      expect(PasswordHasher.verify('x', 'not-a-hash'), isFalse);
      expect(PasswordHasher.verify('x', 'pbkdf2_sha256\$bad'), isFalse);
      expect(PasswordHasher.verify('x', ''), isFalse);
    });
  });
}
