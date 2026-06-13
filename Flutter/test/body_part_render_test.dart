import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:dermatology_ai_app/core/theme/app_theme.dart';
import 'package:dermatology_ai_app/features/core_workflow/screens/body_part_selection_screen.dart';

// Behavioural tests for the body-part selection screen. The screen renders body
// parts as transparent positioned hotspots over a silhouette image (keyed
// 'hotspot_<ID>'), with a chevron toggle between the front and back figures and
// single-selection semantics. (These replace the earlier golden snapshots, which
// were captured against a prior text-button design and broke when the screen was
// redesigned.)
void main() {
  Future<void> pump(WidgetTester tester) async {
    tester.view.physicalSize = const Size(1080, 2160);
    tester.view.devicePixelRatio = 3.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(MaterialApp(
      theme: AppTheme.lightTheme,
      home: const BodyPartSelectionScreen(),
    ));
    await tester.pumpAndSettle();
  }

  Finder hotspot(String id) => find.byKey(ValueKey('hotspot_$id'));

  testWidgets('renders the header and empty-state hint', (tester) async {
    await pump(tester);
    expect(find.text('Select Body Part'), findsOneWidget);
    expect(find.text('Tap the body part you want to scan'), findsOneWidget);
    expect(find.textContaining('Scan:'), findsNothing);
  });

  testWidgets('tapping a front hotspot selects that part', (tester) async {
    await pump(tester);
    await tester.tap(hotspot('CHEST'));
    await tester.pumpAndSettle();
    expect(find.text('Scan: Chest'), findsOneWidget);
    expect(find.text('Tap the body part you want to scan'), findsNothing);
  });

  testWidgets('tapping the selected hotspot again clears the selection',
      (tester) async {
    await pump(tester);
    await tester.tap(hotspot('CHEST'));
    await tester.pumpAndSettle();
    expect(find.text('Scan: Chest'), findsOneWidget);

    await tester.tap(hotspot('CHEST'));
    await tester.pumpAndSettle();
    expect(find.text('Scan: Chest'), findsNothing);
    expect(find.text('Tap the body part you want to scan'), findsOneWidget);
  });

  testWidgets('selecting a second part replaces the first (single selection)',
      (tester) async {
    await pump(tester);
    await tester.tap(hotspot('CHEST'));
    await tester.pumpAndSettle();
    // Selecting another area must move the selection, not add to it.
    await tester.tap(hotspot('FACE'));
    await tester.pumpAndSettle();
    expect(find.text('Scan: Face'), findsOneWidget);
    expect(find.text('Scan: Chest'), findsNothing);
  });

  testWidgets('chevron switches to the back figure and its hotspots',
      (tester) async {
    await pump(tester);
    // A back-only region is absent on the front figure.
    expect(hotspot('UPPER BACK'), findsNothing);
    expect(find.text('FRONT VIEW'), findsOneWidget);

    await tester.tap(find.byIcon(Icons.chevron_right));
    await tester.pumpAndSettle();
    expect(hotspot('UPPER BACK'), findsOneWidget);
    expect(find.text('BACK VIEW'), findsOneWidget);

    await tester.tap(hotspot('UPPER BACK'));
    await tester.pumpAndSettle();
    expect(find.text('Scan: Upper Back'), findsOneWidget);
  });

  testWidgets('a front selection is not highlighted on the back view',
      (tester) async {
    await pump(tester);
    await tester.tap(hotspot('CHEST'));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('highlight-CHEST')), findsOneWidget);

    await tester.tap(find.byIcon(Icons.chevron_left));
    await tester.pumpAndSettle();
    // The scan button keeps the choice, but no back-side region is shaded.
    expect(find.text('Scan: Chest'), findsOneWidget);
    expect(find.byKey(const ValueKey('highlight-CHEST')), findsNothing);
    expect(find.byKey(const ValueKey('no-selection')), findsOneWidget);
  });
}
