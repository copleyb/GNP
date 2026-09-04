"""
Tests for the Prompt Compiler (compiler.py).

Tests cover all prompt layers, reference budget allocation, aspect ratio
selection, negative space injection, Scene Prompt Generator (mocked),
and full compilation to GenerationRequest.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Fix import paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.config import load_config
from pipeline.compiler import (
    PromptCompiler,
    GenerationRequest,
    ReferenceSelection,
    ScenePromptGenerator,
    SceneContextStore,
    select_aspect_ratio,
    allocate_reference_budget,
)

PROJECT_ROOT = Path(__file__).parent.parent


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def config():
    return load_config(str(PROJECT_ROOT))


@pytest.fixture
def compiler(config):
    return PromptCompiler(config)


FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def _load_fixture(name):
    """Load a committed fixture PanelSpec from tests/fixtures/."""
    with (FIXTURES_DIR / name).open() as f:
        return json.load(f)


@pytest.fixture
def panel_spec():
    """Single-character fixture PanelSpec (alyssa, default costume)."""
    return _load_fixture("fixture_single_char.panelspec.json")


@pytest.fixture
def multi_char_spec():
    """Multi-character fixture PanelSpec (alyssa + hood)."""
    return _load_fixture("fixture_multi_char.panelspec.json")


@pytest.fixture
def no_char_spec():
    """Zero-character fixture PanelSpec (environment only)."""
    return _load_fixture("fixture_no_char.panelspec.json")


# A mock LLM callable that returns a fixed scene prompt
MOCK_SCENE_PROMPT = "We see Alyssa sitting on the edge of her bed, dawn light filtering through venetian blinds, casting warm striations across the room."

def mock_call_llm(model, system_prompt, user_prompt):
    return MOCK_SCENE_PROMPT


# -- Aspect ratio selector tests --------------------------------------------

class TestAspectRatioSelector:
    def test_landscape_panel(self):
        """Wide panels should select landscape size."""
        # 2420x1714 → ratio ~1.41 → closest to 1.5 (1536x1024)
        result = select_aspect_ratio(2420, 1714)
        assert result == "1536x1024"  # 1.3 is closer to 1.5 than 1.0

    def test_portrait_panel(self):
        """Tall panels should select portrait size."""
        # 600x1136 → ratio ~0.53 → closest to 0.667 (1024x1536)
        result = select_aspect_ratio(600, 1136)
        assert result == "1024x1536"

    def test_square_panel(self):
        """Square panels should select square size."""
        result = select_aspect_ratio(1000, 1000)
        assert result == "1024x1024"

    def test_slightly_wide(self):
        """Slightly wide should still select landscape."""
        result = select_aspect_ratio(1300, 1000)
        assert result == "1536x1024"  # 1.3 is closer to 1.5 than 1.0

    def test_slightly_tall(self):
        """Slightly tall should select portrait."""
        result = select_aspect_ratio(1000, 1200)
        assert result == "1024x1536"

    def test_real_panel_geometries(self):
        """Test against actual panel geometries from our layouts."""
        # layout_02 panel 1: 2420x1700 → ~1.42 → landscape
        assert select_aspect_ratio(2420, 1700) == "1536x1024"
        # layout_02 panel 2: 1200x1728 → ~0.69 → portrait
        assert select_aspect_ratio(1200, 1728) == "1024x1536"
        # layout_01 panel 1: 2420x1714 → ~1.41 → landscape
        assert select_aspect_ratio(2420, 1714) == "1536x1024"
        # layout_03 panel 1: 1800x1136 → ~1.58 → landscape
        assert select_aspect_ratio(1800, 1136) == "1536x1024"
        # layout_03 panel 2: 600x1136 → ~0.53 → portrait
        assert select_aspect_ratio(600, 1136) == "1024x1536"


# -- Reference budget allocation tests ---------------------------------------

class TestReferenceBudgetAllocation:
    def test_single_character_with_environment(self):
        """One character (2 refs) + one environment (1 ref), budget 8."""
        chars = [{"character_id": "alyssa", "references": [
            {"ref_id": "ref_front", "priority": 1},
            {"ref_id": "ref_three_quarter", "priority": 2},
        ]}]
        env = {"environment_id": "apt", "references": [
            {"ref_id": "ref_est", "priority": 1}
        ]}
        result = allocate_reference_budget(chars, env, budget=8)
        # guaranteed: char=1, env=1, remaining=6, extra_per_char=6, all to primary
        # alyssa: 1 + 6 = 7 (but only has 2 refs → capped at 2)
        assert len(result["character:alyssa"]) == 2
        assert len(result["environment:apt"]) == 1

    def test_two_characters_with_environment(self):
        """Two characters + environment, budget 8."""
        chars = [
            {"character_id": "alyssa", "references": [
                {"ref_id": "ref_front", "priority": 1},
                {"ref_id": "ref_three_quarter", "priority": 2},
            ]},
            {"character_id": "hood", "references": [
                {"ref_id": "ref_front", "priority": 1},
                {"ref_id": "ref_right", "priority": 2},
            ]},
        ]
        env = {"environment_id": "city", "references": [
            {"ref_id": "ref_est", "priority": 1}
        ]}
        result = allocate_reference_budget(chars, env, budget=8)
        # guaranteed: 2 chars + 1 env = 3, remaining=5
        # extra_per_char = 5//2 = 2, leftover=1 → primary gets +1
        # alyssa: 1+2+1=4 (but only has 2 refs → capped at 2)
        # hood: 1+2=3 (but only has 2 refs → capped at 2)
        # env: 1
        assert len(result["character:alyssa"]) == 2
        assert len(result["character:hood"]) == 2
        assert len(result["environment:city"]) == 1

    def test_underfill_not_redistributed(self):
        """Unused slots from underfilled characters are NOT redistributed."""
        chars = [
            {"character_id": "a", "references": [{"ref_id": "r1", "priority": 1}]},
            {"character_id": "b", "references": [
                {"ref_id": "r1", "priority": 1},
                {"ref_id": "r2", "priority": 2},
                {"ref_id": "r3", "priority": 3},
            ]},
        ]
        env = None
        result = allocate_reference_budget(chars, env, budget=8)
        # guaranteed: 2, remaining=6, extra_per_char=3, leftover=0
        # a: 1+3=4 (but only has 1 ref → capped at 1, NOT redistributed)
        # b: 1+3=4 (has 3 refs → gets 3)
        assert len(result["character:a"]) == 1
        assert len(result["character:b"]) == 3

    def test_no_characters(self):
        """No characters, only environment."""
        env = {"environment_id": "city", "references": [
            {"ref_id": "ref_est", "priority": 1}
        ]}
        result = allocate_reference_budget([], env, budget=8)
        assert len(result["environment:city"]) == 1

    def test_no_environment(self):
        """Characters only, no environment."""
        chars = [{"character_id": "a", "references": [
            {"ref_id": "r1", "priority": 1},
            {"ref_id": "r2", "priority": 2},
        ]}]
        result = allocate_reference_budget(chars, None, budget=8)
        assert "environment:" not in str(result.keys())
        assert len(result["character:a"]) == 2

    def test_budget_too_small(self):
        """Budget smaller than guaranteed minimums."""
        chars = [
            {"character_id": "a", "references": [{"ref_id": "r1", "priority": 1}]},
            {"character_id": "b", "references": [{"ref_id": "r1", "priority": 1}]},
        ]
        env = {"environment_id": "e", "references": [{"ref_id": "r1", "priority": 1}]}
        result = allocate_reference_budget(chars, env, budget=2)
        # Only 2 slots, 3 guaranteed — first two chars get 1 each, env gets 0
        assert len(result["character:a"]) == 1
        assert len(result["character:b"]) == 1
        assert "environment:" not in str(result.keys())

    def test_priority_ordering(self):
        """References should be selected by priority (lowest = highest)."""
        chars = [{"character_id": "a", "references": [
            {"ref_id": "r3", "priority": 3},
            {"ref_id": "r1", "priority": 1},
            {"ref_id": "r2", "priority": 2},
        ]}]
        result = allocate_reference_budget(chars, None, budget=2)
        # Should get r1 (priority 1) and r2 (priority 2)
        selected_ids = [r["ref_id"] for r in result["character:a"]]
        assert selected_ids == ["r1", "r2"]


# -- Negative space injection tests ------------------------------------------

class TestNegativeSpace:
    def test_even_seed_injects(self, compiler, panel_spec):
        """Even hex seed should inject the speech bubble space directive."""
        spec = dict(panel_spec)
        spec["panel_seed"] = "A2"  # 162, even
        result = compiler._layer_negative_space(spec)
        assert "speech bubble" in result
        assert "low detail" in result

    def test_odd_seed_does_not_inject(self, compiler, panel_spec):
        """Odd hex seed should NOT inject."""
        spec = dict(panel_spec)
        spec["panel_seed"] = "A3"  # 163, odd
        result = compiler._layer_negative_space(spec)
        assert result == ""

    def test_zero_seed_injects(self, compiler, panel_spec):
        """00 is even, should inject."""
        spec = dict(panel_spec)
        spec["panel_seed"] = "00"
        result = compiler._layer_negative_space(spec)
        assert "speech bubble" in result

    def test_ff_seed_does_not_inject(self, compiler, panel_spec):
        """FF = 255, odd, should NOT inject."""
        spec = dict(panel_spec)
        spec["panel_seed"] = "FF"
        result = compiler._layer_negative_space(spec)
        assert result == ""


# -- Layer assembly tests ----------------------------------------------------

class TestLayerAssembly:
    def test_layer_style(self, compiler, panel_spec):
        """Layer [1] should contain style prompt tokens."""
        result = compiler._layer_style(panel_spec)
        assert "graphic novel" in result.lower()
        assert "high contrast" in result.lower()

    def test_layer_shot_mood(self, compiler, panel_spec):
        """Layer [2] should contain shot type and mood."""
        result = compiler._layer_shot_mood(panel_spec)
        assert "wide" in result  # panel_spec has shot_type: wide
        assert "calm" in result.lower() or "anticipation" in result.lower()

    def test_layer_environment(self, compiler, panel_spec):
        """Layer [3] should contain environment identity tokens."""
        result = compiler._layer_environment(panel_spec)
        assert "apartment" in result.lower()

    def test_layer_characters(self, compiler, panel_spec):
        """Layer [4] should contain character identity and costume."""
        result = compiler._layer_characters(panel_spec)
        assert "Alyssa" in result
        assert "jacket" in result  # costume description

    def test_layer_exclusions_dedup(self, compiler, panel_spec):
        """Layer [6] should contain deduplicated exclusions."""
        result = compiler._layer_exclusions(panel_spec)
        # Style forbidden elements should be present
        assert "no photorealistic" in result
        # Should not have duplicates
        assert result.count("no photorealistic") == 1

    def test_layer_exclusions_combines_sources(self, compiler, panel_spec):
        """Layer [6] should combine style, environment, and character exclusions."""
        result = compiler._layer_exclusions(panel_spec)
        # Environment exclusions (from alyssa_apartment)
        assert "no dirt" in result or "no mess" in result or "no rot" in result
        # Character exclusions (from alyssa)
        assert "no alternate hair" in result


# -- Reference image description tests ---------------------------------------

class TestReferenceDescriptions:
    def test_descriptions_generated(self, compiler, panel_spec):
        """Layer [8] should generate descriptions for selected references."""
        selections, _ = compiler._select_references(panel_spec)
        result = compiler._layer_reference_descriptions(selections)
        assert "Reference image" in result
        assert "Alyssa" in result  # character display name

    def test_descriptions_include_purpose(self, compiler, panel_spec):
        """Descriptions should include the purpose of each reference."""
        selections, _ = compiler._select_references(panel_spec)
        result = compiler._layer_reference_descriptions(selections)
        assert "front neutral" in result.lower() or "three quarter" in result.lower()

    def test_environment_reference_description(self, compiler, no_char_spec):
        """Environment-only panels should get environment reference descriptions."""
        selections, _ = compiler._select_references(no_char_spec)
        result = compiler._layer_reference_descriptions(selections)
        assert "Reference image" in result
        # Should mention the bar interior
        assert "Bar" in result or "bar" in result.lower()


# -- Scene Prompt Generator tests (mocked) ----------------------------------

class TestScenePromptGenerator:
    def test_generate_with_mock(self, config):
        """ScenePromptGenerator should return the mock's output."""
        gen = ScenePromptGenerator(
            model=config.scene_prompt.model,
            context_profile=config.scene_prompt.context_profile,
        )
        result = gen.generate(
            panel_spec={},
            call_llm=mock_call_llm,
        )
        assert result == MOCK_SCENE_PROMPT

    def test_system_prompt_content(self, config):
        """System prompt should instruct scene prompt writing."""
        gen = ScenePromptGenerator("gpt-4o-mini", "default_v1")
        sys_prompt = gen._build_system_prompt()
        assert "scene prompt" in sys_prompt.lower()
        assert "graphic novel" in sys_prompt.lower()
        assert "text" in sys_prompt.lower() or "speech" in sys_prompt.lower()

    def test_user_prompt_includes_description(self, config, panel_spec):
        """User prompt should include the panel description."""
        gen = ScenePromptGenerator("gpt-4o-mini", "default_v1")
        user_prompt = gen._build_user_prompt(panel_spec)
        assert panel_spec["description"] in user_prompt

    def test_user_prompt_includes_characters(self, config, panel_spec):
        """User prompt should include character identity tokens."""
        gen = ScenePromptGenerator("gpt-4o-mini", "default_v1")
        user_prompt = gen._build_user_prompt(panel_spec)
        assert "Alyssa" in user_prompt

    def test_user_prompt_includes_surrounding_panels(self, config, panel_spec):
        """User prompt should include surrounding panel descriptions."""
        gen = ScenePromptGenerator("gpt-4o-mini", "default_v1")
        surrounding = ["Previous panel: Alyssa sleeping in bed."]
        user_prompt = gen._build_user_prompt(panel_spec, surrounding)
        assert "Previous panel" in user_prompt
        assert "Alyssa sleeping" in user_prompt

    def test_user_prompt_includes_user_feedback(self, config, panel_spec):
        """User prompt should include user feedback as director's note."""
        gen = ScenePromptGenerator("gpt-4o-mini", "default_v1")
        feedback = "Make the lighting warmer and more golden."
        user_prompt = gen._build_user_prompt(panel_spec, user_feedback=feedback)
        assert "DIRECTOR" in user_prompt
        assert feedback in user_prompt


# -- Full compilation tests --------------------------------------------------

class TestFullCompilation:
    def test_compile_returns_generation_request(self, compiler, panel_spec):
        """compile() should return a GenerationRequest."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        assert isinstance(result, GenerationRequest)
        assert result.panel_id == panel_spec["panel_id"]

    def test_compiled_prompt_contains_all_layers(self, compiler, panel_spec):
        """The compiled prompt should contain content from all layers."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        prompt = result.prompt
        # [1] Style
        assert "graphic novel" in prompt.lower()
        # [2] Shot + mood
        assert "wide" in prompt.lower()
        # [3] Environment
        assert "apartment" in prompt.lower()
        # [4] Characters
        assert "Alyssa" in prompt
        # [5] Scene prompt
        assert MOCK_SCENE_PROMPT in prompt
        # [6] Exclusions
        assert "no photorealistic" in prompt
        # [8] Reference descriptions
        assert "Reference image" in prompt

    def test_size_selection(self, compiler, panel_spec):
        """GenerationRequest size should match the panel aspect ratio."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        geo = panel_spec["panel_geometry"]
        expected = select_aspect_ratio(geo["width_px"], geo["height_px"])
        assert result.size == expected

    def test_reference_images_list(self, compiler, panel_spec):
        """GenerationRequest should include resolved reference image paths."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        assert len(result.reference_images) > 0
        for ref in result.reference_images:
            assert "ref_id" in ref
            assert "file" in ref
            assert "role" in ref
            assert ref["file"].startswith("characters/") or ref["file"].startswith("environments/")

    def test_multi_character_compilation(self, compiler, multi_char_spec):
        """Compilation with two characters should include both."""
        result = compiler.compile(multi_char_spec, call_llm=mock_call_llm)
        prompt = result.prompt
        assert "Alyssa" in prompt
        assert "Hood" in prompt
        # Should have references for both characters
        char_refs = [r for r in result.reference_images if r["role"] == "character"]
        assert len(char_refs) >= 2

    def test_no_character_compilation(self, compiler, no_char_spec):
        """Compilation with no characters should still work."""
        result = compiler.compile(no_char_spec, call_llm=mock_call_llm)
        prompt = result.prompt
        assert "bar" in prompt.lower() or "dive" in prompt.lower()
        # Only environment references
        assert all(r["role"] == "environment" for r in result.reference_images)

    def test_compiler_version_embedded(self, compiler, panel_spec):
        """GenerationRequest should carry the compiler version."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        assert result.compiler_version == "1.0.0"

    def test_model_from_config(self, compiler, panel_spec):
        """GenerationRequest model should come from project config."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        assert result.model == "gpt-image-2"

    def test_quality_and_thinking_from_config(self, compiler, panel_spec):
        """Quality and thinking should come from project config."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        assert result.quality == "high"
        assert result.thinking == "medium"

    def test_to_dict_for_provenance(self, compiler, panel_spec):
        """to_dict should include internal metadata for the Provenance Store."""
        result = compiler.compile(panel_spec, call_llm=mock_call_llm)
        d = result.to_dict()
        assert "_scene_prompt" in d
        assert d["_scene_prompt"] == MOCK_SCENE_PROMPT
        assert "_reference_selections" in d
        assert len(d["_reference_selections"]) > 0
        assert "_context_profile" in d

    def test_user_feedback_passed_to_scene_generator(self, compiler, panel_spec):
        """User feedback should reach the scene prompt generator."""
        feedback = "Make it darker and more moody."
        captured = {}

        def capturing_llm(model, sys_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return "Mock scene prompt."

        result = compiler.compile(
            panel_spec,
            call_llm=capturing_llm,
            user_feedback=feedback,
        )
        assert feedback in captured["user_prompt"]
        assert "DIRECTOR" in captured["user_prompt"]

    def test_surrounding_descriptions_passed(self, compiler, panel_spec):
        """Surrounding panel descriptions should reach the scene generator."""
        surrounding = ["Previous: Alyssa in bed.", "Next: Alyssa puts on jacket."]
        captured = {}

        def capturing_llm(model, sys_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return "Mock scene prompt."

        result = compiler.compile(
            panel_spec,
            call_llm=capturing_llm,
            surrounding_descriptions=surrounding,
        )
        assert "Previous" in captured["user_prompt"]
        assert "Next" in captured["user_prompt"]


# -- SceneContextStore placeholder tests -------------------------------------

class TestSceneContextStore:
    def test_curate_returns_empty(self):
        """v1: curate_scene_context returns empty dict."""
        store = SceneContextStore()
        assert store.curate_scene_context("any_panel") == {}

    def test_update_is_noop(self):
        """v1: update_scene_context is a no-op (should not raise)."""
        store = SceneContextStore()
        store.update_scene_context("panel_1", "some prompt", {"key": "value"})
