# Design QA — Focus Queue

Reference: `work/focus-queue-reference.png`  
Implementation capture: `work/lada-mac-final.png`

## Comparison

- Three-column structure matches the selected direction: navigation, queue table, and inspector.
- Primary queue actions are visible and labeled.
- Queue rows preserve the reference hierarchy for thumbnail, metadata, profile, state, progress, and ETA.
- Inspector contains preview, video information, restoration controls, output format, and destination.
- Native macOS spacing, typography, dividers, selection, and controls remain consistent.
- Empty, waiting, processing, completed, selected, and inspector states are implemented.

## Findings

- P0: none.
- P1: none.
- P2: none.
- P3: the capture follows the Mac's current dark appearance, while the generated reference used light appearance. The interface intentionally supports both through system-native colors.
- P3: demo filenames are shortened because QA uses repository screenshots rather than distributable sample videos.

final result: passed
