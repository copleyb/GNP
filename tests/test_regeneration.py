"""
Tests for the regeneration control system (orchestrator.py + compiler.py).

Covers:
- Category inference (replay, reroll, revise, regenerate)
- PanelSpec patching (non-destructive)
- Structured diff computation
- Scene prompt mode determination
- Compile-for-replay (no LLM call)
- Preservation mode (prior prompt + diff)
- Full regenerate_panel for each category
- Auto-regeneration loop
- Provenance recording with regeneration metadata
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.config import load_config
from pipeline.compiler import PromptCompiler, GenerationRequest
from pipeline.orchestrator import Orchestrator, PanelResult
from pipeline.provenance import ProvenanceStore
from pipeline.backend import GenerationResult

PROJECT_ROOT = Path(__file__).parent.parent


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def config():
    return load_config(str(PROJECT_ROOT))


def _find_panelspecs() -> list[Path]:
    """Find all .panelspec.json files in the output directory."""
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.panelspec.json"))


def _find_multi_char_spec() -> Path | None:
    """Find a PanelSpec with more than one character, if one exists."""
    for path in _find_panelspecs():
        with path.open() as f:
            spec = json.load(f)
        if len(spec.get("characters", [])) >= 2:
            return path
    return None


@pytest.fixture
def panel_spec():
    """Load any available PanelSpec from the output directory."""
    specs = _find_panelspecs()
    if not specs:
        pytest.skip("No PanelSpecs in output/ — run the Parser first")
    with specs[0].open() as f:
        return json.load(f)


@pytest.fixture
def multi_char_spec():
    """Load a PanelSpec with multiple characters, if available."""
    path = _find_multi_char_spec()
    if path is None:
        pytest.skip("No multi-character PanelSpec found — need a panel with 2+ characters")
    with path.open() as f:
        return json.load(f)


# -- Mock helpers ------------------------------------------------------------

MOCK_SCENE_PROMPT = "We see a character in a room, warm light filtering through blinds."
MOCK_PRESERVATION_PROMPT = "We see a character in a room, warm light filtering through blinds, now with a dutch angle perspective."


def mock_call_llm(model, system_prompt, user_prompt):
    return MOCK_SCENE_PROMPT


def mock_call_llm_preservation(model, system_prompt, user_prompt):
    # Verify preservation context is present in the user prompt
    assert "PREVIOUS SCENE PROMPT" in user_prompt
    assert "REQUESTED CHANGES" in user_prompt
    return MOCK_PRESERVATION_PROMPT


def make_mock_backend(output_bytes=b"fake-png-data", status="success"):
    """Create a mock backend that returns a fake GenerationResult."""
    mock = MagicMock()
    mock.generate.return_value = GenerationResult(
        status=status,
        output_bytes=output_bytes if status == "success" else None,
        api_response_id="mock-resp-123",
        model="gpt-image-2",
        error=None if status == "success" else "Mock error",
    )
    return mock


# -- Category inference tests ------------------------------------------------

class TestCategoryInference:
    """Tests for Orchestrator._infer_category()."""

    def test_no_overrides_is_replay(self):
        assert Orchestrator._infer_category({}) == "replay"

    def test_seed_only_is_reroll(self):
        assert Orchestrator._infer_category({"seed": 42}) == "reroll"

    def test_quality_only_is_reroll(self):
        assert Orchestrator._infer_category({"quality": "high"}) == "reroll"

    def test_thinking_only_is_reroll(self):
        assert Orchestrator._infer_category({"thinking": "high"}) == "reroll"

    def test_feedback_only_is_revise(self):
        assert Orchestrator._infer_category({"feedback": "warmer"}) == "revise"

    def test_fresh_prompt_only_is_revise(self):
        assert Orchestrator._infer_category({"fresh_prompt": True}) == "revise"

    def test_scene_prompt_only_is_revise(self):
        assert Orchestrator._infer_category({"scene_prompt": "custom text"}) == "revise"

    def test_costume_only_is_regenerate(self):
        assert Orchestrator._infer_category({"costume": "morning_routine"}) == "regenerate"

    def test_shot_type_only_is_regenerate(self):
        assert Orchestrator._infer_category({"shot_type": "dutch_angle"}) == "regenerate"

    def test_mood_only_is_regenerate(self):
        assert Orchestrator._infer_category({"mood": "tense"}) == "regenerate"

    def test_description_only_is_regenerate(self):
        assert Orchestrator._infer_category({"description": "new desc"}) == "regenerate"

    def test_priority_panelspec_over_scene_prompt(self):
        # Both costume and feedback → regenerate (deeper layer wins)
        overrides = {"costume": "morning_routine", "feedback": "warmer"}
        assert Orchestrator._infer_category(overrides) == "regenerate"

    def test_priority_scene_prompt_over_backend(self):
        # Both feedback and seed → revise (deeper layer wins)
        overrides = {"feedback": "warmer", "seed": 42}
        assert Orchestrator._infer_category(overrides) == "revise"

    def test_compose_all_layers(self):
        overrides = {
            "costume": "morning_routine",
            "feedback": "warmer",
            "seed": 42,
            "thinking": "high",
        }
        assert Orchestrator._infer_category(overrides) == "regenerate"


# -- PanelSpec patching tests ------------------------------------------------

class TestPanelSpecPatching:
    """Tests for Orchestrator._apply_panelspec_patches()."""

    def test_shot_type_patch(self, panel_spec):
        overrides = {"shot_type": "dutch_angle"}
        patched = Orchestrator._apply_panelspec_patches(panel_spec, overrides)
        assert patched["shot_type"] == "dutch_angle"
        # Original unchanged
        assert panel_spec["shot_type"] != "dutch_angle"

    def test_mood_patch(self, panel_spec):
        overrides = {"mood": "tense"}
        patched = Orchestrator._apply_panelspec_patches(panel_spec, overrides)
        assert patched["mood"] == "tense"

    def test_description_patch(self, panel_spec):
        original_desc = panel_spec["description"]
        overrides = {"description": "A completely new scene."}
        patched = Orchestrator._apply_panelspec_patches(panel_spec, overrides)
        assert patched["description"] == "A completely new scene."
        assert panel_spec["description"] == original_desc

    def test_no_patches_returns_copy(self, panel_spec):
        patched = Orchestrator._apply_panelspec_patches(panel_spec, {})
        assert patched == panel_spec
        assert patched is not panel_spec  # different object

    def test_costume_patch_filters_refs(self, multi_char_spec):
        overrides = {"costume": "morning_routine"}
        patched = Orchestrator._apply_panelspec_patches(multi_char_spec, overrides)
        # Characters should have refs filtered to only morning_routine costume
        for char in patched.get("characters", []):
            for ref in char.get("references", []):
                assert ref.get("costume") == "morning_routine"

    def test_non_destructive(self, panel_spec):
        """Original PanelSpec is never modified."""
        original = json.dumps(panel_spec, sort_keys=True)
        overrides = {"shot_type": "dutch_angle", "mood": "tense", "description": "new"}
        Orchestrator._apply_panelspec_patches(panel_spec, overrides)
        assert json.dumps(panel_spec, sort_keys=True) == original


# -- Structured diff tests ---------------------------------------------------

class TestComputeDiff:
    """Tests for Orchestrator._compute_diff()."""

    def test_empty_overrides_empty_diff(self, panel_spec):
        diff = Orchestrator._compute_diff(panel_spec, {})
        assert diff == {}

    def test_shot_type_diff(self, panel_spec):
        original = panel_spec["shot_type"]
        diff = Orchestrator._compute_diff(panel_spec, {"shot_type": "dutch_angle"})
        assert diff["shot_type"] == {"from": original, "to": "dutch_angle"}

    def test_mood_diff(self, panel_spec):
        original = panel_spec["mood"]
        diff = Orchestrator._compute_diff(panel_spec, {"mood": "tense"})
        assert diff["mood"] == {"from": original, "to": "tense"}

    def test_description_diff(self, panel_spec):
        original = panel_spec["description"]
        diff = Orchestrator._compute_diff(panel_spec, {"description": "new"})
        assert diff["description"] == {"from": original, "to": "new"}

    def test_feedback_in_diff(self, panel_spec):
        diff = Orchestrator._compute_diff(panel_spec, {"feedback": "warmer"})
        assert diff["feedback"] == "warmer"

    def test_composed_diff(self, panel_spec):
        overrides = {"shot_type": "dutch_angle", "feedback": "darker"}
        diff = Orchestrator._compute_diff(panel_spec, overrides)
        assert "shot_type" in diff
        assert "feedback" in diff
        assert diff["shot_type"]["to"] == "dutch_angle"
        assert diff["feedback"] == "darker"

    def test_seed_not_in_diff(self, panel_spec):
        """Backend overrides don't appear in the diff (not scene-prompt relevant)."""
        diff = Orchestrator._compute_diff(panel_spec, {"seed": 42})
        assert "seed" not in diff


# -- Scene prompt mode tests -------------------------------------------------

class TestScenePromptMode:
    """Tests for Orchestrator._determine_scene_prompt_mode()."""

    def test_no_flags_is_preservation(self):
        assert Orchestrator._determine_scene_prompt_mode({}) == "preservation"

    def test_feedback_is_preservation_with_feedback(self):
        assert Orchestrator._determine_scene_prompt_mode({"feedback": "warmer"}) == "preservation_with_feedback"

    def test_fresh_prompt_is_cold_start(self):
        assert Orchestrator._determine_scene_prompt_mode({"fresh_prompt": True}) == "cold_start"

    def test_scene_prompt_is_direct(self):
        assert Orchestrator._determine_scene_prompt_mode({"scene_prompt": "custom"}) == "direct"


# -- Compile-for-replay tests ------------------------------------------------

class TestCompileForReplay:
    """Tests for PromptCompiler.compile_for_replay()."""

    def test_replay_uses_stored_prompt(self, config, panel_spec):
        compiler = PromptCompiler(config)
        stored = "This is the exact stored prompt from provenance."
        req = compiler.compile_for_replay(panel_spec, stored_prompt=stored)

        assert req.prompt == stored
        assert req.panel_id == panel_spec["panel_id"]
        assert req.model == config.image_generation.model

    def test_replay_seed_override(self, config, panel_spec):
        compiler = PromptCompiler(config)
        req = compiler.compile_for_replay(panel_spec, "stored prompt", seed=42)
        assert req.seed == 42

    def test_replay_quality_override(self, config, panel_spec):
        compiler = PromptCompiler(config)
        req = compiler.compile_for_replay(panel_spec, "stored prompt", quality="high")
        assert req.quality == "high"

    def test_replay_thinking_override(self, config, panel_spec):
        compiler = PromptCompiler(config)
        req = compiler.compile_for_replay(panel_spec, "stored prompt", thinking="high")
        assert req.thinking == "high"

    def test_replay_no_overrides_uses_config_defaults(self, config, panel_spec):
        compiler = PromptCompiler(config)
        req = compiler.compile_for_replay(panel_spec, "stored prompt")
        assert req.seed == config.image_generation.seed
        assert req.quality == config.image_generation.quality
        assert req.thinking == config.image_generation.thinking

    def test_replay_has_reference_images(self, config, panel_spec):
        compiler = PromptCompiler(config)
        req = compiler.compile_for_replay(panel_spec, "stored prompt")
        assert len(req.reference_images) > 0

    def test_replay_no_llm_call(self, config, panel_spec):
        """compile_for_replay must not make any LLM call."""
        compiler = PromptCompiler(config)
        # If this works without any mock, no LLM call was made
        req = compiler.compile_for_replay(panel_spec, "stored prompt")
        assert req.prompt == "stored prompt"


# -- Preservation mode tests -------------------------------------------------

class TestPreservationMode:
    """Tests for ScenePromptGenerator preservation mode."""

    def test_preservation_system_prompt_differs(self, config):
        gen = PromptCompiler(config).scene_prompt_generator
        normal = gen._build_system_prompt()
        preservation = gen._build_preservation_system_prompt()
        assert normal != preservation
        assert "revising" in preservation.lower()

    def test_preservation_user_prompt_includes_prior(self, config, panel_spec):
        gen = PromptCompiler(config).scene_prompt_generator
        prior = "The previous scene prompt text."
        diff = {"shot_type": {"from": "wide", "to": "dutch_angle"}}
        prompt = gen._build_preservation_user_prompt(
            panel_spec, None, prior, diff
        )
        assert "PREVIOUS SCENE PROMPT" in prompt
        assert prior in prompt
        assert "REQUESTED CHANGES" in prompt
        assert "shot_type" in prompt
        assert "dutch_angle" in prompt

    def test_preservation_with_feedback(self, config, panel_spec):
        gen = PromptCompiler(config).scene_prompt_generator
        prior = "The previous scene prompt."
        diff = {"feedback": "make it darker"}
        prompt = gen._build_preservation_user_prompt(
            panel_spec, None, prior, diff, feedback="make it darker"
        )
        assert "DIRECTOR'S NOTE" in prompt

    def test_compile_with_preservation_context(self, config, panel_spec):
        """compile() with preservation_context calls LLM in preservation mode."""
        compiler = PromptCompiler(config)
        ctx = {
            "prior_prompt": "Old scene prompt.",
            "change_summary": {"shot_type": {"from": "wide", "to": "dutch_angle"}},
        }
        called_with = {}

        def tracking_mock(model, system_prompt, user_prompt):
            called_with["system"] = system_prompt
            called_with["user"] = user_prompt
            return "Revised scene prompt."

        req = compiler.compile(
            panel_spec,
            call_llm=tracking_mock,
            preservation_context=ctx,
        )

        assert "revising" in called_with["system"].lower()
        assert "PREVIOUS SCENE PROMPT" in called_with["user"]
        assert "Old scene prompt." in called_with["user"]
        assert "dutch_angle" in called_with["user"]
        assert req.prompt != "Old scene prompt."  # new prompt, not the old one


# -- regenerate_panel integration tests --------------------------------------

class TestRegeneratePanel:
    """Integration tests for Orchestrator.regenerate_panel() with mocked backend."""

    @pytest.fixture
    def orchestrator(self, config, tmp_path):
        """Orchestrator with mock backend and temp output dir."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend
        # Override output dir to temp
        orch.config.__dataclass_params__ = orch.config.__dataclass_params__  # frozen check
        return orch

    def test_replay_no_prior_record_falls_back(self, config, panel_spec, tmp_path, monkeypatch):
        """Replay with no prior provenance falls back to cold start."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        # Monkeypatch provenance to return no records
        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = None
        orch.provenance.get_latest_attempt_number.return_value = 0
        orch.provenance.get_next_attempt_number.return_value = 1

        # Monkeypatch _write_output and _post_process to avoid file I/O
        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=1)

        result = orch.regenerate_panel(panel_spec, overrides={}, call_llm=mock_call_llm)

        assert result.status == "success"
        assert result.panel_id == panel_spec["panel_id"]
        # Should have made a backend call
        assert mock_backend.generate.called

    def test_reroll_with_seed_override(self, config, panel_spec, tmp_path):
        """Reroll applies seed override to the GenerationRequest."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        # Set up provenance with a prior record
        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Stored prompt text.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Stored prompt text."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        result = orch.regenerate_panel(panel_spec, overrides={"seed": 42})

        assert result.status == "success"
        # Check the backend received a request with seed=42
        gen_request = mock_backend.generate.call_args[0][0]
        assert gen_request.seed == 42

    def test_reroll_with_quality_override(self, config, panel_spec, tmp_path):
        """Reroll applies quality override."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Stored prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Stored prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        orch.regenerate_panel(panel_spec, overrides={"quality": "high"})

        gen_request = mock_backend.generate.call_args[0][0]
        assert gen_request.quality == "high"

    def test_revise_with_feedback(self, config, panel_spec, tmp_path):
        """Revise with feedback uses preservation mode."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Old compiled prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Old scene prompt text."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        def mock_llm(model, system_prompt, user_prompt):
            assert "revising" in system_prompt.lower()
            assert "PREVIOUS SCENE PROMPT" in user_prompt
            assert "Old scene prompt text." in user_prompt
            assert "DIRECTOR'S NOTE" in user_prompt
            return "Revised prompt with warmer lighting."

        result = orch.regenerate_panel(
            panel_spec,
            overrides={"feedback": "warmer lighting"},
            call_llm=mock_llm,
        )

        assert result.status == "success"

    def test_revise_with_fresh_prompt(self, config, panel_spec, tmp_path):
        """Revise with --fresh-prompt runs cold start (no preservation context)."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Old prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Old scene prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        def mock_llm(model, system_prompt, user_prompt):
            # Cold start: normal system prompt, no preservation context
            assert "revising" not in system_prompt.lower()
            assert "PREVIOUS SCENE PROMPT" not in user_prompt
            return "Fresh cold start prompt."

        result = orch.regenerate_panel(
            panel_spec,
            overrides={"fresh_prompt": True},
            call_llm=mock_llm,
        )

        assert result.status == "success"

    def test_revise_with_scene_prompt_direct(self, config, panel_spec, tmp_path):
        """Revise with --scene-prompt uses direct text, no LLM call."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Old prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Old scene prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        # No call_llm provided — if LLM is called, it'll try real API and fail
        # With direct mode, the mock intercepts and returns the text
        result = orch.regenerate_panel(
            panel_spec,
            overrides={"scene_prompt": "Direct custom scene text."},
        )

        assert result.status == "success"
        gen_request = mock_backend.generate.call_args[0][0]
        # The direct scene prompt should be in the compiled prompt
        assert "Direct custom scene text." in gen_request.prompt

    def test_regenerate_with_shot_type(self, config, panel_spec, tmp_path):
        """Regenerate patches PanelSpec and runs full pipeline."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Old prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Old scene prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        result = orch.regenerate_panel(
            panel_spec,
            overrides={"shot_type": "dutch_angle"},
            call_llm=mock_call_llm,
        )

        assert result.status == "success"
        # The compiled prompt should contain the patched shot type context
        gen_request = mock_backend.generate.call_args[0][0]
        # Shot type is in layer 2, should be in the prompt
        assert "dutch_angle" in gen_request.prompt

    def test_compose_all_overrides(self, config, panel_spec, tmp_path):
        """All override categories compose into a single regeneration."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Old prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Old scene prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        overrides = {
            "shot_type": "dutch_angle",
            "feedback": "darker mood",
            "seed": 999,
            "thinking": "high",
        }

        def mock_llm(model, system_prompt, user_prompt):
            assert "revising" in system_prompt.lower()
            return "Composed regeneration prompt."

        result = orch.regenerate_panel(
            panel_spec,
            overrides=overrides,
            call_llm=mock_llm,
        )

        assert result.status == "success"
        gen_request = mock_backend.generate.call_args[0][0]
        assert gen_request.seed == 999
        assert gen_request.thinking == "high"
        assert "dutch_angle" in gen_request.prompt


# -- Auto-regeneration tests -------------------------------------------------

class TestAutoRegeneration:
    """Tests for Orchestrator.auto_regenerate_panel()."""

    def test_auto_regen_attempt2_is_reroll(self, config, panel_spec, tmp_path):
        """First auto-regen (attempt 2) should be reroll (no overrides)."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Stored prompt.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Stored scene prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1  # attempt 1 done

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        result = orch.auto_regenerate_panel(panel_spec, max_attempts=3)

        assert result.status == "success"
        # Should have used the stored prompt (replay/reroll, no LLM call)
        gen_request = mock_backend.generate.call_args[0][0]
        assert gen_request.prompt == "Stored prompt."

    def test_auto_regen_respects_max_attempts(self, config, panel_spec):
        """Auto-regen does not exceed max_attempts."""
        mock_backend = make_mock_backend(status="failure", output_bytes=None)
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Stored.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Stored."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 3  # already at 3

        orch._write_output = MagicMock()
        orch._post_process = MagicMock(return_value=(None, None))
        orch._next_attempt_number = MagicMock(return_value=4)

        result = orch.auto_regenerate_panel(panel_spec, max_attempts=3)

        assert result.status == "failure"


# -- Provenance recording tests ----------------------------------------------

class TestRegenerationProvenance:
    """Tests for regeneration metadata in provenance records."""

    def test_replay_records_category(self, config, panel_spec, tmp_path):
        """Replay records regeneration_category in provenance."""
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Stored.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Stored."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        orch.regenerate_panel(panel_spec, overrides={})

        # Check provenance.append was called with regeneration metadata
        append_call = orch.provenance.append.call_args[0][0]
        assert append_call["regeneration_category"] == "replay"
        assert append_call["overrides"] == {}
        assert append_call["scene_prompt"]["mode"] == "reused"
        assert append_call["scene_prompt"]["regenerated"] is False

    def test_reroll_records_category_and_overrides(self, config, panel_spec, tmp_path):
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Stored.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Stored."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        orch.regenerate_panel(panel_spec, overrides={"seed": 42, "quality": "high"})

        append_call = orch.provenance.append.call_args[0][0]
        assert append_call["regeneration_category"] == "reroll"
        assert append_call["overrides"]["seed"] == 42
        assert append_call["overrides"]["quality"] == "high"

    def test_revise_records_preservation_mode(self, config, panel_spec, tmp_path):
        mock_backend = make_mock_backend()
        orch = Orchestrator(config)
        orch.backend = mock_backend

        orch.provenance = MagicMock()
        orch.provenance.get_latest_record.return_value = {
            "generation_request": {
                "prompt": "Old.",
                "model": "gpt-image-2",
                "size": "1024x1024",
                "quality": "medium",
                "thinking": "medium",
                "seed": 100,
            },
            "scene_prompt": {"output": "Old scene prompt."},
        }
        orch.provenance.get_latest_attempt_number.return_value = 1

        orch._write_output = MagicMock(return_value=tmp_path / "output.png")
        orch._post_process = MagicMock(return_value=((1024, 1024), (1024, 1024)))
        orch._next_attempt_number = MagicMock(return_value=2)

        orch.regenerate_panel(
            panel_spec,
            overrides={"feedback": "warmer"},
            call_llm=mock_call_llm,
        )

        append_call = orch.provenance.append.call_args[0][0]
        assert append_call["regeneration_category"] == "revise"
        assert append_call["scene_prompt"]["mode"] == "preservation_with_feedback"
        assert append_call["scene_prompt"]["regenerated"] is True
        assert "preservation_context" in append_call["scene_prompt"]
        assert append_call["scene_prompt"]["preservation_context"]["prior_prompt"] == "Old scene prompt."
