import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();

  Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final Directory dir = await getApplicationDocumentsDirectory();
    final String path = join(dir.path, 'dermatology_ai.db');

    return openDatabase(
      path,
      version: 3,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE scans (
        id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL,
        body_part TEXT NOT NULL,
        diagnosis TEXT NOT NULL,
        confidence REAL NOT NULL,
        class_probabilities TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        notes TEXT,
        heatmap_path TEXT
      )
    ''');

    await db.execute('''
      CREATE TABLE users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'patient',
        age INTEGER,
        gender TEXT,
        medical_history TEXT,
        created_at INTEGER NOT NULL
      )
    ''');
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    if (oldVersion < 3) {
      try {
        await db.execute('ALTER TABLE scans ADD COLUMN heatmap_path TEXT');
      } catch (_) {
        // Column may already exist
      }
    }
  }

  // ── Scans ──────────────────────────────────────────────────────────────────

  Future<void> insertScan(Map<String, dynamic> scan) async {
    final db = await database;
    final row = Map<String, dynamic>.from(scan);
    if (row['class_probabilities'] is Map) {
      row['class_probabilities'] = jsonEncode(row['class_probabilities']);
    }
    await db.insert('scans', row, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<List<Map<String, dynamic>>> getAllScans() async {
    final db = await database;
    final rows = await db.query('scans', orderBy: 'timestamp DESC');
    return rows.map((row) {
      final m = Map<String, dynamic>.from(row);
      if (m['class_probabilities'] is String) {
        m['class_probabilities'] = Map<String, double>.from(
          (jsonDecode(m['class_probabilities'] as String) as Map)
              .map((k, v) => MapEntry(k as String, (v as num).toDouble())),
        );
      }
      return m;
    }).toList();
  }

  Future<void> deleteScan(String id) async {
    final db = await database;
    await db.delete('scans', where: 'id = ?', whereArgs: [id]);
  }

  // ── Users ──────────────────────────────────────────────────────────────────

  Future<void> insertUser(Map<String, dynamic> user) async {
    final db = await database;
    await db.insert('users', user, conflictAlgorithm: ConflictAlgorithm.fail);
  }

  Future<Map<String, dynamic>?> getUserByEmail(String email) async {
    final db = await database;
    final rows =
        await db.query('users', where: 'email = ?', whereArgs: [email]);
    return rows.isNotEmpty ? rows.first : null;
  }

  Future<Map<String, dynamic>?> getUserById(String id) async {
    final db = await database;
    final rows = await db.query('users', where: 'id = ?', whereArgs: [id]);
    return rows.isNotEmpty ? rows.first : null;
  }

  Future<void> updateUser(Map<String, dynamic> user) async {
    final db = await database;
    await db.update('users', user,
        where: 'id = ?', whereArgs: [user['id']]);
  }
}
