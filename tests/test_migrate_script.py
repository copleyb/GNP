"""Tests for scripts/migrate_scene_output.py — one-time chapter_tag prefix migration."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "migrate_scene_output.py"


@pytest.fixture
def migrate():
    spec = importlib.util.spec_from_file_location("migrate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_output(tmp_path):
    """Miniature scene output state: 2 panels, 1 archived attempt, provenance records."""
    out = tmp_path / "output"
    (out / "archive").mkdir(parents=True)
    # PNGs in the pre-migration (unprefixed) state
    (out / "s702_l01_st01_pn01_attempt_001.png").write_bytes(b"p1")
    (out / "s702_l02_st01_pn01_attempt_001.png").write_bytes(b"p2")
    (out / "archive" / "s702_l01_st01_pn01_attempt_000.png").write_bytes(b"p0")
    # Identity files that must NOT change
    (out / "s702_l01_st01_pn01.panelspec.json").write_text('{"panel_id": "s702_l01_st01_pn01"}')
    # Provenance records with path refs at two nesting depths
    rec1 = {
        "record_id": "s702_l01_st01_pn01_attempt_001",
        "panel_id": "s702_l01_st01_pn01",
        "outcome": {"status": "success", "output_file": "output/s702_l01_st01_pn01_attempt_001.png"},
    }
    rec2 = {
        "record_id": "s702_l01_st01_pn01_attempt_002",
        "panel_id": "s702_l01_st01_pn01",
        "surgical_context": {
            "source": {"file": "output/archive/s702_l01_st01_pn01_attempt_000.png"}
        },
        "outcome": {"output_file": "output/s702_l01_st01_pn01_attempt_002.png"},
    }
    (out / "s702_l01_st01_pn01.provenance.jsonl").write_text(
        json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n"
    )
    return out


class TestPatchRecord:
    def test_patches_output_file_at_any_depth(self, migrate):
        rec = {"outcome": {"output_file": "output/s702_l01_st01_pn01_attempt_001.png"}}
        assert migrate._patch_record(rec, "c01", "s702") == 1
        assert rec["outcome"]["output_file"] == "output/c01_s702_l01_st01_pn01_attempt_001.png"

    def test_patches_archive_paths(self, migrate):
        rec = {"src": "output/archive/s702_l01_st01_pn01_attempt_000.png"}
        assert migrate._patch_record(rec, "c01", "s702") == 1
        assert rec["src"] == "output/archive/c01_s702_l01_st01_pn01_attempt_000.png"

    def test_never_touches_identity_fields(self, migrate):
        rec = {"panel_id": "s702_l01_st01_pn01", "record_id": "s702_l01_st01_pn01_attempt_001"}
        assert migrate._patch_record(rec, "c01", "s702") == 0
        assert rec["panel_id"] == "s702_l01_st01_pn01"

    def test_already_prefixed_is_idempotent(self, migrate):
        rec = {"outcome": {"output_file": "output/c01_s702_l01_st01_pn01_attempt_001.png"}}
        assert migrate._patch_record(rec, "c01", "s702") == 0


class TestDryRun:
    def test_dry_run_changes_nothing(self, migrate, fake_output, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["m", "--scene", "s702", "--tag", "c01",
                                         "--project", str(fake_output.parent)])
        rc = migrate.main()
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY-RUN" in out
        assert (fake_output / "s702_l01_st01_pn01_attempt_001.png").exists()
        assert not (fake_output / "c01_s702_l01_st01_pn01_attempt_001.png").exists()
        rec = json.loads((fake_output / "s702_l01_st01_pn01.provenance.jsonl")
                         .read_text().splitlines()[0])
        assert rec["outcome"]["output_file"] == "output/s702_l01_st01_pn01_attempt_001.png"


class TestApply:
    def test_apply_renames_and_patches(self, migrate, fake_output, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["m", "--scene", "s702", "--tag", "c01", "--apply",
                                         "--project", str(fake_output.parent)])
        rc = migrate.main()
        assert rc == 0
        # Renames (flat + archive)
        assert (fake_output / "c01_s702_l01_st01_pn01_attempt_001.png").exists()
        assert (fake_output / "c01_s702_l02_st01_pn01_attempt_001.png").exists()
        assert (fake_output / "archive" / "c01_s702_l01_st01_pn01_attempt_000.png").exists()
        assert not (fake_output / "s702_l01_st01_pn01_attempt_001.png").exists()
        # Provenance patched, identity untouched
        lines = (fake_output / "s702_l01_st01_pn01.provenance.jsonl").read_text().splitlines()
        rec1, rec2 = json.loads(lines[0]), json.loads(lines[1])
        assert rec1["outcome"]["output_file"] == "output/c01_s702_l01_st01_pn01_attempt_001.png"
        assert rec1["record_id"] == "s702_l01_st01_pn01_attempt_001"
        assert rec2["surgical_context"]["source"]["file"] == \
            "output/archive/c01_s702_l01_st01_pn01_attempt_000.png"
        # Store filename + panelspec untouched
        assert (fake_output / "s702_l01_st01_pn01.provenance.jsonl").exists()
        assert (fake_output / "s702_l01_st01_pn01.panelspec.json").exists()
