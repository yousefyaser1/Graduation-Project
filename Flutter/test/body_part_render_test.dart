import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:dermatology_ai_app/core/theme/app_theme.dart';
import 'package:dermatology_ai_app/features/core_workflow/screens/body_part_selection_screen.dart';

// Renders the body-part selection screen to PNG goldens so the layout and the
// single-selection silhouette highlight can be visually inspected. Run with:
//   flutter test --update-goldens test/body_part_render_test.dart
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

  testWidgets('front — empty', (tester) async {
    await pump(tester);
    await expectLater(find.byType(BodyPartSelectionScreen),
        matchesGoldenFile('goldens/body_part_front_empty.png'));
  });

  testWidgets('front — one region selected', (tester) async {
    await pump(tester);
    await tester.tap(find.text('CHEST'));
    await tester.pumpAndSettle();
    await expectLater(find.byType(BodyPartSelectionScreen),
        matchesGoldenFile('goldens/body_part_front_selected.png'));
  });

  testWidgets('front — selection replaces previous', (tester) async {
    await pump(tester);
    await tester.tap(find.text('CHEST'));
    await tester.pumpAndSettle();
    // Selecting a second area must move the selection, not add to it.
    await tester.tap(find.text('FACE'));
    await tester.pumpAndSettle();
    await expectLater(find.byType(BodyPartSelectionScreen),
        matchesGoldenFile('goldens/body_part_front_replaced.png'));
  });

  testWidgets('back — one region selected', (tester) async {
    await pump(tester);
    await tester.tap(find.text('Back'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Upper Back'));
    await tester.pumpAndSettle();
    await expectLater(find.byType(BodyPartSelectionScreen),
        matchesGoldenFile('goldens/body_part_back_selected.png'));
  });

  testWidgets('help sheet', (tester) async {
    await pump(tester);
    await tester.tap(find.text('Help'));
    await tester.pumpAndSettle();
    await expectLater(find.byType(MaterialApp),
        matchesGoldenFile('goldens/body_part_help_sheet.png'));
  });
}
