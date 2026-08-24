# GNP — High-Level To-Do List

## 1. Fix test_producer.py fixture mismatch
7 pre-existing test failures — tests reference old sample characters (ada/marcus) instead of the real ones (alyssa/hood). Low-hanging fruit to get CI fully green (123 → 130).

## 2. Add fixture PanelSpecs to tests/fixtures/
Compiler tests that load PanelSpecs from `output/` skip on CI because `output/` is gitignored. Ship a small fixture PanelSpec in `tests/fixtures/` so those tests actually run in CI instead of silently skipping — this already masked a broken assertion once.

## 3. Costume edge case
Alyssa's `morning_routine` variant exists in the character YAML but the panel schema has no way to select it. Chapter 1 opens with her waking up, so this bites immediately on real generation. Needs a design decision: add `costume` field to panel schema, or have the Parser infer it from panel description.

## 4. More reference images
More reference images coming soon. Pipeline is ready — each new character/environment needs YAML + PNGs + manifest regeneration via `scripts/sync_assets.py generate`.

## 5. Redesign continuity

## 6. Redesign regeneration control
