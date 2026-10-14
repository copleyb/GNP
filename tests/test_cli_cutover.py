"""
Tests for the Phase 3 cutover (DESIGN.md §16.9): pipeline_mode default =
"scene", and the CLI is scene-first with --chapter as the explicit
legacy opt-in.

All tests are offline. Scope-dispatch wiring (produce/parse/validate) is
verified via monkeypatched downstream handlers — no LLM calls, no parsing
side effects. Scene-scoped list/status run the real Scene Parser against
fixture scene files with config.output_dir redirected to tmp, so no
repo pollution.
"""

import dataclasses
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import cli
from pipeline.config import load_config

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_SCENE = FIXTURES_DIR / "fixture_scene_minimal.yaml"


def tmp_config(tmp_path):
    """Real project config with output redirected to tmp."""
    cfg = load_config(str(PROJECT_ROOT))
    return dataclasses.replace(cfg, output_dir=tmp_path / "output")


# -- Flag flip ----------------------------------------------------------------

class TestPipelineModeDefault:

    def test_repo_project_runs_scene_mode(self):
        assert load_config(str(PROJECT_ROOT)).pipeline_mode == "scene"

    def test_config_fallback_is_scene(self):
        """ProjectConfig's declared default is scene (Phase 3 cutover)."""
        import pipeline.config as cfg_mod
        field = next(f for f in cfg_mod.ProjectConfig.__dataclass_fields__.values()
                     if f.name == "pipeline_mode")
        assert field.default == "scene"


# -- produce dispatch ----------------------------------------------------------

class TestProduceDispatch:

    def test_bare_produce_errors(self, capsys):
        rc = cli.cmd_produce(SimpleNamespace(project=str(PROJECT_ROOT),
                                             chapter=None, input=None))
        assert rc == 1
        assert "legacy" in capsys.readouterr().out

    def test_scene_input_dispatches_to_scene_producer(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "cmd_produce_scene",
                            lambda args: seen.setdefault("input", args.input) and 0)
        rc = cli.cmd_produce(SimpleNamespace(project=str(PROJECT_ROOT),
                                             chapter=None, input="s702", model=None))
        assert rc == 0
        assert seen["input"] == "s702"

    def test_chapter_flag_dispatches_to_chapter_producer(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "_produce_chapter",
                            lambda args: seen.setdefault("chapter", args.chapter) and 0)
        rc = cli.cmd_produce(SimpleNamespace(project=str(PROJECT_ROOT),
                                             chapter=1, input=None))
        assert rc == 0
        assert seen["chapter"] == 1

    def test_chapter_and_input_mutually_exclusive(self, capsys):
        rc = cli.cmd_produce(SimpleNamespace(project=str(PROJECT_ROOT),
                                             chapter=1, input="s702"))
        assert rc == 1
        assert "mutually exclusive" in capsys.readouterr().out

    def test_chapter_path_requires_synopsis(self, capsys):
        """Legacy chapter path with no synopsis fails before any LLM call."""
        rc = cli._produce_chapter(SimpleNamespace(project=str(PROJECT_ROOT),
                                                  chapter=1, synopsis=None,
                                                  synopsis_file=None, model=None))
        assert rc == 1
        assert "synopsis" in capsys.readouterr().out


# -- parse dispatch ------------------------------------------------------------

class TestParseDispatch:

    def test_bare_parse_errors(self, capsys):
        rc = cli.cmd_parse(SimpleNamespace(project=str(PROJECT_ROOT),
                                           chapter=None, scene=None))
        assert rc == 1
        assert "legacy" in capsys.readouterr().out

    def test_scene_arg_dispatches_to_scene_parser(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "cmd_parse_scene",
                            lambda args: seen.setdefault("scene", args.scene) and 0)
        rc = cli.cmd_parse(SimpleNamespace(project=str(PROJECT_ROOT),
                                           chapter=None, scene="s702"))
        assert rc == 0
        assert seen["scene"] == "s702"

    def test_chapter_flag_dispatches_to_chapter_parser(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "_parse_chapter",
                            lambda args: seen.setdefault("chapter", args.chapter) and 0)
        rc = cli.cmd_parse(SimpleNamespace(project=str(PROJECT_ROOT),
                                           chapter=1, scene=None))
        assert rc == 0
        assert seen["chapter"] == 1

    def test_bare_parse_real_fixture_scene(self, tmp_path, monkeypatch, capsys):
        """End-to-end: `parse <fixture scene file>` runs the real scene parser
        (offline) through the new default path, writing to tmp output."""
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))
        rc = cli.cmd_parse(SimpleNamespace(project=".", chapter=None,
                                           scene=str(FIXTURE_SCENE)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Parsed" in out and "panels" in out

    def test_chapter_and_scene_mutually_exclusive(self, capsys):
        rc = cli.cmd_parse(SimpleNamespace(project=str(PROJECT_ROOT),
                                           chapter=1, scene="s702"))
        assert rc == 1
        assert "mutually exclusive" in capsys.readouterr().out


# -- scene-scoped list / status ------------------------------------------------

class TestSceneListStatus:

    def test_list_panels_scene(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))
        rc = cli.cmd_list(SimpleNamespace(project=".", what="panels",
                                          scene=str(FIXTURE_SCENE),
                                          chapter=None, panel=None))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Scene" in out and "panels" in out
        assert "Element" in out  # grouped by element, not page

    def test_list_panels_requires_scope(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))
        rc = cli.cmd_list(SimpleNamespace(project=".", what="panels",
                                          scene=None, chapter=None, panel=None))
        assert rc == 1
        assert "legacy" in capsys.readouterr().out

    def test_status_scene(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))
        rc = cli.cmd_status(SimpleNamespace(project=".", scene=str(FIXTURE_SCENE),
                                            chapter=None))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Status" in out and "Not generated:" in out

    def test_status_requires_scope(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))
        rc = cli.cmd_status(SimpleNamespace(project=".", scene=None, chapter=None))
        assert rc == 1
        assert "legacy" in capsys.readouterr().out


# -- validate dispatch ---------------------------------------------------------

class TestValidateDispatch:

    def test_bare_validate_errors(self, capsys):
        rc = cli.cmd_validate(SimpleNamespace(project=str(PROJECT_ROOT),
                                              scene=None, chapter=None,
                                              page=None, panel=None))
        assert rc == 1
        assert "legacy" in capsys.readouterr().out

    def test_scene_validate_rejects_page(self, capsys):
        rc = cli.cmd_validate(SimpleNamespace(project=str(PROJECT_ROOT),
                                              scene="s702", chapter=None,
                                              page="1", panel=None))
        assert rc == 1
        assert "unpaginated" in capsys.readouterr().out

    def test_scene_validate_dispatches_to_shared_loop(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))

        def fake_validate(config, panels):
            seen["n"] = len(panels)
            return 0
        monkeypatch.setattr(cli, "_validate_panels", fake_validate)
        rc = cli.cmd_validate(SimpleNamespace(project=".", scene=str(FIXTURE_SCENE),
                                              chapter=None, page=None, panel=None))
        assert rc == 0
        assert seen["n"] > 0

    def test_scene_validate_panel_filter_miss(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_config", lambda *_: tmp_config(tmp_path))
        rc = cli.cmd_validate(SimpleNamespace(project=".", scene=str(FIXTURE_SCENE),
                                              chapter=None, page=None,
                                              panel="s999_l99_st99_pn99"))
        assert rc == 1
        assert "not found" in capsys.readouterr().out


# -- generate / regenerate scope errors ------------------------------------------

class TestScopeErrorText:

    def test_generate_without_scope_mentions_scene_first(self, capsys):
        rc = cli.cmd_generate(SimpleNamespace(project=str(PROJECT_ROOT),
                                              scene=None, chapter=None,
                                              page=None, panel=None))
        assert rc == 1
        out = capsys.readouterr().out
        assert "--scene" in out and "legacy" in out


class TestProduceSceneModelDefault:
    """Regression (cutover): bare `produce <input>` carries --model=None into
    cmd_produce_scene (the chapter parser has no default for the flag). The
    handler must fall back to gpt-4o — live failure was OpenAI 400
    'you must provide a model parameter'."""

    def test_produce_scene_falls_back_when_model_none(self, monkeypatch):
        import scene_producer as sp

        seen = {}

        class FakeProducer:
            def __init__(self, config, model=None, progress_callback=None):
                seen["model"] = model

            def produce_scene(self, target):
                seen["target"] = target
                return {
                    "parse_result": SimpleNamespace(scene_id="s999", total_panels=9),
                    "attempts": {"chunker": 1, "elements": [2, 1]},
                    "total_llm_calls": 2,
                    "file_path": "scenes/fake.yaml",
                }

        monkeypatch.setattr(sp, "SceneProducer", FakeProducer)
        monkeypatch.setattr(sp, "SceneProducerError", Exception)
        monkeypatch.setattr(sp, "SceneInputError", Exception)

        rc = cli.cmd_produce_scene(
            SimpleNamespace(project=".", input="s707", model=None)
        )
        assert rc == 0
        assert seen["model"] == "gpt-4o", "model=None must not reach the API"
        assert seen["target"] == "s707"
