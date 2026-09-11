"""Tests for display-stem output naming (chapter_tag prefix on scene panels).

Per the provenance-key decision: output PNGs carry the chapter tag as a
display prefix (c01_s702_l01_st01_pn01_attempt_001.png); provenance identity
(record keys, store filenames, panelspec filenames) stays tag-free.
"""

import pytest

from pipeline.orchestrator import Orchestrator, _display_stem

SCENE_SPEC = {
    "panel_id": "s702_l01_st01_pn01",
    "chapter_tag": "c01",
}
CHAPTER_SPEC = {
    "panel_id": "c01_pg1_l02_pn01",  # tag lives inside the ID in chapter mode
}


class TestDisplayStem:
    def test_scene_spec_gets_tag_prefix(self):
        assert _display_stem(SCENE_SPEC) == "c01_s702_l01_st01_pn01"

    def test_chapter_spec_unchanged(self):
        assert _display_stem(CHAPTER_SPEC) == "c01_pg1_l02_pn01"

    def test_no_tag_means_identity(self):
        assert _display_stem({"panel_id": "s702_l01_st01_pn01"}) == "s702_l01_st01_pn01"


def _bare_orchestrator(tmp_path):
    """Orchestrator instance with only what _write_output/_next_attempt_number need."""
    orch = object.__new__(Orchestrator)
    orch.config = type("C", (), {"output_dir": tmp_path})()
    return orch


class TestWriteOutputNaming:
    def test_scene_panel_writes_prefixed_filename(self, tmp_path):
        orch = _bare_orchestrator(tmp_path)
        stem = _display_stem(SCENE_SPEC)
        path = orch._write_output(b"png", stem, 1, 100, 200)
        assert path.name == "c01_s702_l01_st01_pn01_attempt_001.png"
        assert path.exists()

    def test_chapter_panel_writes_legacy_filename(self, tmp_path):
        orch = _bare_orchestrator(tmp_path)
        stem = _display_stem(CHAPTER_SPEC)
        path = orch._write_output(b"png", stem, 1, 100, 200)
        assert path.name == "c01_pg1_l02_pn01_attempt_001.png"

    def test_next_attempt_number_sees_prefixed_files(self, tmp_path):
        orch = _bare_orchestrator(tmp_path)
        stem = _display_stem(SCENE_SPEC)
        orch._write_output(b"a", stem, 1, 100, 200)
        orch._write_output(b"b", stem, 2, 100, 200)
        assert orch._next_attempt_number(stem) == 3

    def test_prior_attempts_move_to_archive_by_stem(self, tmp_path):
        orch = _bare_orchestrator(tmp_path)
        stem = _display_stem(SCENE_SPEC)
        orch._write_output(b"a", stem, 1, 100, 200)
        orch._write_output(b"b", stem, 2, 100, 200)  # moves attempt 1 to archive
        assert not (tmp_path / "c01_s702_l01_st01_pn01_attempt_001.png").exists()
        assert (tmp_path / "archive" / "c01_s702_l01_st01_pn01_attempt_001.png").exists()
        assert (tmp_path / "c01_s702_l01_st01_pn01_attempt_002.png").exists()
