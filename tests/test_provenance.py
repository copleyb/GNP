"""
test_provenance.py — Test suite for the Provenance Store.

Tests:
1. Basic append and read_all
2. Multiple attempts (append-only, ordering)
3. Attempt numbering
4. Latest record retrieval
5. has_records check
6. Record validation (missing fields, invalid types)
7. Page-level queries
8. Chapter-level queries
9. query_failed
10. query_latest_per_panel
11. Empty file handling
12. Nonexistent panel reads
13. Output directory auto-creation
14. JSONL format verification (one record per line)
15. Unicode/special characters in records
16. Regenerated flag and override flag preservation
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.provenance import ProvenanceStore


# -- Test helpers -----------------------------------------------------------

def make_minimal_record(panel_id: str, attempt: int, **overrides) -> dict:
    """Create a minimal valid Generation Record for testing."""
    record = {
        "record_id": f"{panel_id}_attempt_{attempt:03d}",
        "panel_id": panel_id,
        "attempt_number": attempt,
        "timestamp_utc": "2026-07-25T09:30:00Z",
        "compiler": {
            "version": "1.0.0",
            "prompt_hash": f"sha256:abc{attempt}123",
        },
        "scene_prompt": {
            "model": "gpt-4o-mini",
            "context_profile": "default_v1",
            "inputs": {
                "style_tokens": "noir style...",
                "user_feedback": None,
            },
            "output": "A detailed scene prompt...",
            "regenerated": True,
        },
        "reference_selection": {
            "budget": 8,
            "allocation_algorithm": "proportional_primary_priority_v1",
            "available": [],
            "selected": [],
        },
        "generation_request": {
            "model": "gpt-image-2",
            "prompt": "The full prompt string...",
            "size": "1024x1536",
            "quality": "high",
            "thinking": "medium",
            "seed": None,
            "n": 1,
        },
        "outcome": {
            "status": "success",
            "output_file": f"output/{panel_id}_attempt_{attempt:03d}.png",
            "api_response_id": f"img-abc{attempt}xyz",
        },
        "validation": {
            "layout_compliance": 1.0,
            "character_consistency": {
                "score": 0.85,
                "observations": ["Looks good"],
                "confidence": "high",
            },
            "style_adherence": {
                "score": 0.90,
                "observations": ["Style matches"],
                "confidence": "high",
            },
            "composite_score": 0.87,
            "threshold": 0.80,
            "weights_snapshot": {"character_consistency": 0.6, "style_adherence": 0.4},
            "accepted_for_production": True,
        },
    }
    record.update(overrides)
    return record


def make_failed_record(panel_id: str, attempt: int) -> dict:
    """Create a record with a failed validation."""
    record = make_minimal_record(panel_id, attempt)
    record["validation"]["accepted_for_production"] = False
    record["validation"]["composite_score"] = 0.72
    return record


def make_generation_failure_record(panel_id: str, attempt: int) -> dict:
    """Create a record where the image generation itself failed."""
    record = make_minimal_record(panel_id, attempt)
    record["outcome"]["status"] = "content_filtered"
    record["outcome"]["output_file"] = None
    return record


# -- Test runner ------------------------------------------------------------

def run_tests():
    """Run all tests and report results."""
    tests = [
        ("Basic append and read_all", test_basic_append_read),
        ("Multiple attempts ordering", test_multiple_attempts),
        ("Attempt numbering", test_attempt_numbering),
        ("Latest record retrieval", test_latest_record),
        ("has_records check", test_has_records),
        ("Record validation - missing fields", test_validation_missing_fields),
        ("Record validation - invalid types", test_validation_invalid_types),
        ("Page-level queries", test_query_by_page),
        ("Chapter-level queries", test_query_by_chapter),
        ("Query failed panels", test_query_failed),
        ("query_latest_per_panel", test_query_latest_per_panel),
        ("Empty file handling", test_empty_file),
        ("Nonexistent panel reads", test_nonexistent_panel),
        ("Output directory auto-creation", test_auto_create_dirs),
        ("JSONL format verification", test_jsonl_format),
        ("Unicode in records", test_unicode),
        ("Regenerated flag preservation", test_regenerated_flag),
        ("Override flag preservation", test_override_flag),
        ("Backend-only reuse flag", test_backend_only_reuse),
        ("List all panels", test_list_all_panels),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        print("FAILURES DETECTED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


# -- Individual tests -------------------------------------------------------

def test_basic_append_read():
    """Test that a single record can be appended and read back."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        record = make_minimal_record("c01_pg01_l01_pn01", 1)
        store.append(record)
        records = store.read_all("c01_pg01_l01_pn01")
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        assert records[0]["panel_id"] == "c01_pg01_l01_pn01"
        assert records[0]["attempt_number"] == 1
        assert records[0]["record_id"] == "c01_pg01_l01_pn01_attempt_001"


def test_multiple_attempts():
    """Test that multiple attempts are stored and returned in order."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c02_pg03_l02_pn01"
        for i in range(1, 4):
            store.append(make_minimal_record(panel, i))
        records = store.read_all(panel)
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        assert records[0]["attempt_number"] == 1
        assert records[1]["attempt_number"] == 2
        assert records[2]["attempt_number"] == 3


def test_attempt_numbering():
    """Test get_latest_attempt_number and get_next_attempt_number."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        assert store.get_latest_attempt_number(panel) == 0, "Empty panel should return 0"
        assert store.get_next_attempt_number(panel) == 1, "Empty panel should return next=1"

        store.append(make_minimal_record(panel, 1))
        assert store.get_latest_attempt_number(panel) == 1
        assert store.get_next_attempt_number(panel) == 2

        store.append(make_minimal_record(panel, 2))
        store.append(make_minimal_record(panel, 3))
        assert store.get_latest_attempt_number(panel) == 3
        assert store.get_next_attempt_number(panel) == 4


def test_latest_record():
    """Test get_latest_record returns the most recent attempt."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        assert store.get_latest_record(panel) is None, "Empty panel should return None"

        store.append(make_minimal_record(panel, 1))
        store.append(make_minimal_record(panel, 2))
        latest = store.get_latest_record(panel)
        assert latest is not None
        assert latest["attempt_number"] == 2, "Should return attempt 2 (latest)"


def test_has_records():
    """Test has_records correctly detects existing and non-existing records."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        assert store.has_records(panel) is False, "Empty panel should have no records"

        store.append(make_minimal_record(panel, 1))
        assert store.has_records(panel) is True

        assert store.has_records("c99_pg99_l99_pn99") is False, "Nonexistent panel should have no records"


def test_validation_missing_fields():
    """Test that appending a record with missing required fields raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        # Missing record_id
        bad_record = make_minimal_record("c01_pg01_l01_pn01", 1)
        del bad_record["record_id"]
        try:
            store.append(bad_record)
            raise AssertionError("Should have raised ValueError for missing record_id")
        except ValueError:
            pass

        # Missing compiler
        bad_record = make_minimal_record("c01_pg01_l01_pn01", 1)
        del bad_record["compiler"]
        try:
            store.append(bad_record)
            raise AssertionError("Should have raised ValueError for missing compiler")
        except ValueError:
            pass

        # Missing outcome
        bad_record = make_minimal_record("c01_pg01_l01_pn01", 1)
        del bad_record["outcome"]
        try:
            store.append(bad_record)
            raise AssertionError("Should have raised ValueError for missing outcome")
        except ValueError:
            pass


def test_validation_invalid_types():
    """Test that appending a record with invalid types raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        # attempt_number as string
        bad_record = make_minimal_record("c01_pg01_l01_pn01", 1)
        bad_record["attempt_number"] = "1"
        try:
            store.append(bad_record)
            raise AssertionError("Should have raised ValueError for string attempt_number")
        except ValueError:
            pass

        # attempt_number = 0
        bad_record = make_minimal_record("c01_pg01_l01_pn01", 0)
        try:
            store.append(bad_record)
            raise AssertionError("Should have raised ValueError for attempt_number=0")
        except ValueError:
            pass

        # panel_id as empty string
        bad_record = make_minimal_record("c01_pg01_l01_pn01", 1)
        bad_record["panel_id"] = ""
        try:
            store.append(bad_record)
            raise AssertionError("Should have raised ValueError for empty panel_id")
        except ValueError:
            pass

        # Non-dict record
        try:
            store.append("not a dict")
            raise AssertionError("Should have raised TypeError for non-dict record")
        except TypeError:
            pass


def test_query_by_page():
    """Test page-level queries return the correct panels."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        # Create panels on page 2_3 (chapter 2, page 3)
        store.append(make_minimal_record("c02_pg03_l01_pn01", 1))
        store.append(make_minimal_record("c02_pg03_l02_pn01", 1))
        store.append(make_minimal_record("c02_pg03_l02_pn02", 1))

        # Create panels on a different page
        store.append(make_minimal_record("c02_pg04_l01_pn01", 1))
        store.append(make_minimal_record("c01_pg01_l01_pn01", 1))

        result = store.query_by_page(2, 3)
        assert len(result) == 3, f"Expected 3 panels on page 2_3, got {len(result)}"
        assert "c02_pg03_l01_pn01" in result
        assert "c02_pg03_l02_pn01" in result
        assert "c02_pg03_l02_pn02" in result
        assert "c02_pg04_l01_pn01" not in result

        # Verify each panel has its records
        for panel_id, records in result.items():
            assert len(records) == 1, f"Expected 1 record for {panel_id}"


def test_query_by_chapter():
    """Test chapter-level queries return all panels in the chapter."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        # Chapter 2 panels
        store.append(make_minimal_record("c02_pg01_l01_pn01", 1))
        store.append(make_minimal_record("c02_pg03_l02_pn01", 1))
        store.append(make_minimal_record("c02_pg05_l01_pn01", 1))

        # Chapter 1 panel
        store.append(make_minimal_record("c01_pg01_l01_pn01", 1))

        # Chapter 3 panel
        store.append(make_minimal_record("c03_pg01_l01_pn01", 1))

        result = store.query_by_chapter(2)
        assert len(result) == 3, f"Expected 3 panels in chapter 2, got {len(result)}"

        result_ch1 = store.query_by_chapter(1)
        assert len(result_ch1) == 1

        result_ch4 = store.query_by_chapter(4)
        assert len(result_ch4) == 0, "Chapter 4 should have no panels"


def test_query_failed():
    """Test query_failed returns panels with failed validation or generation."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        # Accepted panel
        store.append(make_minimal_record("c01_pg01_l01_pn01", 1))  # accepted=True

        # Failed validation panel
        store.append(make_failed_record("c01_pg01_l01_pn02", 1))  # accepted=False

        # Content filtered panel
        store.append(make_generation_failure_record("c01_pg01_l02_pn01", 1))

        # Panel that failed then succeeded (latest is accepted)
        store.append(make_failed_record("c01_pg01_l02_pn02", 1))
        store.append(make_minimal_record("c01_pg01_l02_pn02", 2))  # latest is accepted

        failed = store.query_failed()
        assert len(failed) == 2, f"Expected 2 failed panels, got {len(failed)}"
        failed_ids = [r["panel_id"] for r in failed]
        assert "c01_pg01_l01_pn02" in failed_ids, "Validation-failed panel should be in failed list"
        assert "c01_pg01_l02_pn01" in failed_ids, "Generation-failed panel should be in failed list"
        assert "c01_pg01_l01_pn01" not in failed_ids, "Accepted panel should not be in failed list"
        assert "c01_pg01_l02_pn02" not in failed_ids, "Panel with latest accepted should not be in failed list"


def test_query_latest_per_panel():
    """Test query_latest_per_panel returns the latest record for each panel."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        store.append(make_minimal_record("c01_pg01_l01_pn01", 1))
        store.append(make_minimal_record("c01_pg01_l01_pn01", 2))
        store.append(make_minimal_record("c01_pg01_l02_pn01", 1))

        result = store.query_latest_per_panel(["c01_pg01_l01_pn01", "c01_pg01_l02_pn01", "c01_pg01_l03_pn01"])
        assert len(result) == 3
        assert result["c01_pg01_l01_pn01"]["attempt_number"] == 2, "Should return attempt 2 (latest)"
        assert result["c01_pg01_l02_pn01"]["attempt_number"] == 1
        assert result["c01_pg01_l03_pn01"] is None, "Nonexistent panel should return None"


def test_empty_file():
    """Test that an empty provenance file is handled correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        # Create an empty file
        path = store._provenance_path(panel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        assert store.has_records(panel) is False, "Empty file should have no records"
        records = store.read_all(panel)
        assert records == [], "Empty file should return empty list"
        assert store.get_latest_attempt_number(panel) == 0
        assert store.get_latest_record(panel) is None


def test_nonexistent_panel():
    """Test reading a panel that has no provenance file."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        assert store.read_all("c99_pg99_l99_pn99") == []
        assert store.get_latest_attempt_number("c99_pg99_l99_pn99") == 0
        assert store.get_latest_record("c99_pg99_l99_pn99") is None
        assert store.has_records("c99_pg99_l99_pn99") is False


def test_auto_create_dirs():
    """Test that the store creates output and archive directories."""
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "nested", "output")
        store = ProvenanceStore(output)
        assert os.path.isdir(output), "Output directory should be created"
        assert os.path.isdir(os.path.join(output, "archive")), "Archive directory should be created"


def test_jsonl_format():
    """Test that the provenance file is valid JSONL (one JSON object per line)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"
        store.append(make_minimal_record(panel, 1))
        store.append(make_minimal_record(panel, 2))

        path = store._provenance_path(panel)
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        for i, line in enumerate(lines):
            line = line.strip()
            assert line, f"Line {i+1} is empty"
            obj = json.loads(line)  # Should not raise
            assert obj["panel_id"] == panel
            assert obj["attempt_number"] == i + 1


def test_unicode():
    """Test that unicode characters in records are preserved."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"
        record = make_minimal_record(panel, 1)
        record["scene_prompt"]["output"] = "Ada stands in the café, rain pouring — «c'est la vie»"
        record["validation"]["character_consistency"]["observations"] = ["Hair matches ✓", "Coat slightly different — café scene"]

        store.append(record)
        records = store.read_all(panel)
        assert len(records) == 1
        assert "café" in records[0]["scene_prompt"]["output"]
        assert "✓" in records[0]["validation"]["character_consistency"]["observations"][0]
        assert "«c'est la vie»" in records[0]["scene_prompt"]["output"]


def test_regenerated_flag():
    """Test that the regenerated flag is preserved correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        # First attempt: regenerated=True (fresh scene prompt)
        r1 = make_minimal_record(panel, 1)
        r1["scene_prompt"]["regenerated"] = True
        store.append(r1)

        # Second attempt: regenerated=False (backend-only, reused prompt)
        r2 = make_minimal_record(panel, 2)
        r2["scene_prompt"]["regenerated"] = False
        store.append(r2)

        records = store.read_all(panel)
        assert records[0]["scene_prompt"]["regenerated"] is True
        assert records[1]["scene_prompt"]["regenerated"] is False


def test_override_flag():
    """Test that the override flag is preserved correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        # Human override: accepted despite low score
        record = make_failed_record(panel, 1)  # accepted_for_production=False
        record["validation"]["accepted_for_production"] = True
        record["validation"]["override"] = True
        store.append(record)

        records = store.read_all(panel)
        assert records[0]["validation"]["accepted_for_production"] is True
        assert records[0]["validation"]["override"] is True

        # This panel should NOT appear in query_failed
        failed = store.query_failed()
        assert len(failed) == 0, "Overridden panel should not be in failed list"


def test_backend_only_reuse():
    """Test the scenario from DESIGN.md §13: backend-only regeneration reuses the compiled prompt."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)
        panel = "c01_pg01_l01_pn01"

        # Attempt 1: full pipeline
        r1 = make_minimal_record(panel, 1)
        r1["scene_prompt"]["regenerated"] = True
        r1["generation_request"]["prompt"] = "ORIGINAL PROMPT STRING"
        r1["validation"]["accepted_for_production"] = False
        r1["validation"]["composite_score"] = 0.72
        store.append(r1)

        # Attempt 2: backend-only (reuses prompt from attempt 1)
        r2 = make_minimal_record(panel, 2)
        r2["scene_prompt"]["regenerated"] = False
        r2["generation_request"]["prompt"] = "ORIGINAL PROMPT STRING"  # same prompt
        r2["validation"]["accepted_for_production"] = True
        r2["validation"]["composite_score"] = 0.88
        store.append(r2)

        records = store.read_all(panel)
        assert len(records) == 2

        # Verify the prompt was reused
        assert records[0]["generation_request"]["prompt"] == records[1]["generation_request"]["prompt"]
        assert records[0]["scene_prompt"]["regenerated"] is True
        assert records[1]["scene_prompt"]["regenerated"] is False

        # Latest record should be accepted
        latest = store.get_latest_record(panel)
        assert latest["validation"]["accepted_for_production"] is True

        # Should not appear in failed list (latest is accepted)
        assert len(store.query_failed()) == 0


def test_list_all_panels():
    """Test list_all_panels returns all panels with records, sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ProvenanceStore(tmp)

        store.append(make_minimal_record("c02_pg03_l02_pn01", 1))
        store.append(make_minimal_record("c01_pg01_l01_pn01", 1))
        store.append(make_minimal_record("c01_pg01_l02_pn01", 1))

        panels = store.list_all_panels()
        assert len(panels) == 3
        assert panels == sorted(panels), "Panels should be sorted"
        assert "c01_pg01_l01_pn01" in panels
        assert "c02_pg03_l02_pn01" in panels


if __name__ == "__main__":
    run_tests()
