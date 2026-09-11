"""
Tests for the Scene Producer (src/scene_producer.py) — Scope Redesign Phase 2
(DESIGN.md §16.11).

All tests are offline: the LLM client is a mock. Uses real project assets
(alyssa, hood, city_exterior, templates/) exactly like the chapter producer
tests do.
"""

import dataclasses
import os
import sys
from pathlib import Path

import pytest
import yaml

# Fix import paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.config import load_config
from pipeline.scene_parser import ScenePlanParser
from pipeline.scene_parser import SceneParserError
from scene_producer import (
    SceneProducer,
    SceneProducerError,
    SceneInputError,
    SceneProducerLLMError,
)


PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture()
def config(tmp_path):
    """Real config with scene dirs redirected to tmp."""
    cfg = load_config(str(PROJECT_ROOT))
    return dataclasses.replace(
        cfg,
        output_dir=tmp_path / "output",
        scenes_dir=tmp_path / "scenes",
        scene_inputs_dir=tmp_path / "scene_inputs",
    )

@pytest.fixture()
def scene_input():
    with (FIXTURES_DIR / "fixture_scene_input.yaml").open("r") as f:
        return yaml.safe_load(f)


NARRATIVE = "Alyssa crosses the waking market; a shadow follows her through it."

def make_panel(i: int, chars=None, env="city_exterior", description=None) -> dict:
    return {
        "description": description or f"Panel {i}: absolute description of the beat.",
        "characters": chars if chars is not None else [{"id": "alyssa", "costume": None}],
        "environment": env,
        "shot_type": "medium",
        "mood": "calm",
    }

def chunker_ok(n_elements: int) -> dict:
    return {
        "narrative": NARRATIVE,
        "chunks": [f"Chunk {i} of the synopsis." for i in range(1, n_elements + 1)],
    }

def element_ok(count: int, **kwargs) -> dict:
    return {"panels": [make_panel(i, **kwargs) for i in range(1, count + 1)]}


class MockLLM:
    """Scripted mock: pops responses per call; records every prompt."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []          # (schema_name, system, user)

    def __call__(self, system, user, schema_name, schema):
        self.prompts.append((schema_name, system, user))
        if not self.responses:
            raise AssertionError("MockLLM: unexpected extra call")
        return self.responses.pop(0)


def standard_mock() -> MockLLM:
    """Happy path: chunker ok, element 1 (4 panels), element 2 (3 panels)."""
    return MockLLM([
        chunker_ok(2),
        element_ok(4),
        element_ok(3, chars=[{"id": "hood", "costume": None}]),
    ])


def make_producer(config, mock) -> SceneProducer:
    return SceneProducer(config, llm_client=mock)


# -- Input loading ----------------------------------------------------------------

class TestInputLoading:

    def test_fixture_input_loads(self, config):
        producer = SceneProducer(config, llm_client=standard_mock())
        scene_input = producer.load_input(FIXTURES_DIR / "fixture_scene_input.yaml")
        assert scene_input["scene_id"] == "s904"
        assert len(scene_input["elements"]) == 2

    def test_example_project_input_loads(self, config):
        """The shipped scene_inputs/c01_s702.yaml starter loads against the real menu."""
        cfg = load_config(str(PROJECT_ROOT))
        producer = SceneProducer(cfg, llm_client=standard_mock())
        scene_input = producer.load_input(PROJECT_ROOT / "scene_inputs" / "c01_s702.input.yaml")
        ids = producer.parser.derive_panel_ids("s702", scene_input["elements"])
        assert len(ids) == 15

    def test_missing_synopsis_rejected(self, config, scene_input, tmp_path):
        del scene_input["synopsis"]
        path = tmp_path / "in.yaml"
        path.write_text(yaml.safe_dump(scene_input))
        producer = SceneProducer(config, llm_client=standard_mock())
        with pytest.raises(SceneInputError, match="synopsis"):
            producer.load_input(path)

    def test_unknown_layout_rejected_before_llm(self, config, scene_input, tmp_path):
        scene_input["elements"][0] = {"layout": "l_ghost"}
        path = tmp_path / "in.yaml"
        path.write_text(yaml.safe_dump(scene_input))
        producer = SceneProducer(config, llm_client=standard_mock())
        with pytest.raises(SceneInputError, match="l_ghost"):
            producer.load_input(path)

    def test_find_input_glob(self, config, scene_input, tmp_path):
        config.scene_inputs_dir.mkdir(parents=True)
        (config.scene_inputs_dir / "c01_s904.input.yaml").write_text(yaml.safe_dump(scene_input))
        producer = SceneProducer(config, llm_client=standard_mock())
        found = producer.find_input("s904")
        assert found.name == "c01_s904.input.yaml"

    def test_find_input_missing(self, config):
        producer = SceneProducer(config, llm_client=standard_mock())
        with pytest.raises(SceneInputError):
            producer.find_input("s999")


# -- Stage 1: chunker ------------------------------------------------------------------

class TestStage1Chunker:

    def test_happy_path(self, config):
        mock = standard_mock()
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        assert result["scene"]["narrative"] == NARRATIVE
        assert result["attempts"]["chunker"] == 1
        assert result["total_llm_calls"] == 3   # 1 chunker + 2 elements

    def test_user_prompt_contains_counts(self, config):
        mock = standard_mock()
        producer = make_producer(config, mock)
        producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        chunker_user = next(u for name, _, u in mock.prompts if name == "scene_chunks")
        assert "EXACTLY 4 panels" in chunker_user
        assert "EXACTLY 3 panels" in chunker_user
        assert "Producer fixture" in chunker_user
        assert "being followed" in chunker_user  # synopsis present

    def test_chunk_count_mismatch_retries_then_succeeds(self, config):
        mock = MockLLM([
            {"narrative": NARRATIVE, "chunks": ["only one"]},   # wrong count
            chunker_ok(2),                                       # retry ok
            element_ok(4),
            element_ok(3),
        ])
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        assert result["attempts"]["chunker"] == 2

    def test_chunker_exhausts_retries(self, config):
        mock = MockLLM([
            {"narrative": NARRATIVE, "chunks": ["one"]} for _ in range(3)
        ])
        producer = make_producer(config, mock)
        with pytest.raises(SceneProducerLLMError, match="chunker"):
            producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        # No element calls were spent.
        assert len(mock.prompts) == 3

    def test_empty_narrative_rejected(self, config):
        mock = MockLLM([
            {"narrative": "   ", "chunks": ["a", "b"]},
            chunker_ok(2), element_ok(4), element_ok(3),
        ])
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        assert result["attempts"]["chunker"] == 2


# -- Stage 2: per-element content ----------------------------------------------------------

class TestStage2ElementCalls:

    def test_first_element_prompt_no_handoff(self, config):
        mock = standard_mock()
        producer = make_producer(config, mock)
        producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        element_prompts = [u for name, _, u in mock.prompts if name == "element_panels"]
        assert "first element" in element_prompts[0]
        assert "WHERE WE LEFT OFF" not in element_prompts[0]
        assert NARRATIVE in element_prompts[0]
        assert "Chunk 1" in element_prompts[0]
        assert "MUST have EXACTLY 4 entries" in element_prompts[0]

    def test_second_element_prompt_has_handoff(self, config):
        mock = standard_mock()
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        element_prompts = [u for name, _, u in mock.prompts if name == "element_panels"]
        second = element_prompts[1]
        assert "WHERE WE LEFT OFF" in second
        # Handoff carries element 1's LAST panel description + characters.
        last_desc = result["scene"]["panels"]["s904_l01_st02_pn02"]["description"]
        assert last_desc in second
        assert "alyssa" in second
        assert "Chunk 2" in second

    def test_handoff_carries_costume(self, config):
        mock = MockLLM([
            chunker_ok(2),
            element_ok(4, chars=[{"id": "alyssa", "costume": "morning_routine"}]),
            element_ok(3),
        ])
        producer = make_producer(config, mock)
        producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        second_user = [u for name, _, u in mock.prompts if name == "element_panels"][1]
        assert "costume: morning_routine" in second_user

    def test_array_length_mismatch_retries_then_succeeds(self, config):
        mock = MockLLM([
            chunker_ok(2),
            element_ok(3),   # element 1 expects 4 — wrong
            element_ok(4),   # retry — right
            element_ok(3),
        ])
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        assert result["attempts"]["elements"] == [2, 1]

    def test_unknown_character_retries_then_fails(self, config):
        mock = MockLLM([
            chunker_ok(2),
            element_ok(4, chars=[{"id": "ghost", "costume": None}]),
            element_ok(4, chars=[{"id": "ghost", "costume": None}]),
            element_ok(4, chars=[{"id": "ghost", "costume": None}]),
        ])
        producer = make_producer(config, mock)
        with pytest.raises(SceneProducerLLMError, match="ghost"):
            producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")

    def test_unknown_environment_fails_validation(self, config):
        mock = MockLLM([
            chunker_ok(2),
            element_ok(4, env="ghost_place"),
        ] * 2)
        producer = make_producer(config, mock)
        with pytest.raises(SceneProducerLLMError, match="ghost_place"):
            producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")

    def test_unknown_costume_fails_validation(self, config):
        mock = MockLLM([
            chunker_ok(2),
            element_ok(4, chars=[{"id": "alyssa", "costume": "space_suit"}]),
        ] * 2)
        producer = make_producer(config, mock)
        with pytest.raises(SceneProducerLLMError, match="space_suit"):
            producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")

    def test_null_costume_normalized_to_default(self, config):
        """OpenAI strict mode emits costume: null; producer drops it (== default)."""
        mock = standard_mock()
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        for panel in result["scene"]["panels"].values():
            for c in panel.get("characters", []):
                assert c.get("costume") is None or isinstance(c["costume"], str)

    def test_sequential_order_element2_gets_element1_output(self, config):
        """Stage 2 is sequential: the handoff is built from element 1's RETURNED
        content, proving call k depends on call k-1's output."""
        mock = standard_mock()
        producer = make_producer(config, mock)
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        first_prompts = [u for name, _, u in mock.prompts if name == "element_panels"]
        last_desc = result["scene"]["panels"]["s904_l01_st02_pn02"]["description"]
        assert last_desc in first_prompts[1]


# -- Assembly + parse gate -------------------------------------------------------------------

class TestAssemblyAndGate:

    def test_produce_commits_scene_and_emits_panelspecs(self, config):
        producer = make_producer(config, standard_mock())
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")

        # Scene file committed atomically with chapter-tagged name.
        scene_path = config.scenes_dir / "c01_s904.yaml"
        assert scene_path.exists()
        assert result["file_path"] == scene_path

        # Header verbatim from input; narrative from stage 1.
        scene = result["scene"]
        assert scene["scene_id"] == "s904"
        assert scene["chapter_tag"] == "c01"
        assert scene["title"] == "Producer fixture"
        assert scene["elements"] == [
            {"layout": "l_quad"}, {"strip": "s_eq3_33"},
        ]
        assert scene["narrative"] == NARRATIVE

        # Panels keyed by derived positional IDs — exact coverage (7 panels).
        expected_ids = {
            "s904_l01_st01_pn01", "s904_l01_st01_pn02",
            "s904_l01_st02_pn01", "s904_l01_st02_pn02",
            "s904_l02_st01_pn01", "s904_l02_st01_pn02", "s904_l02_st01_pn03",
        }
        assert set(scene["panels"]) == expected_ids

        # PanelSpecs emitted to output/ in the same run (parse gate side effect).
        assert result["parse_result"].total_panels == 7
        for panel_id in expected_ids:
            assert (config.output_dir / f"{panel_id}.panelspec.json").exists()

    def test_committed_scene_reparses_with_fresh_parser(self, config):
        producer = make_producer(config, standard_mock())
        result = producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")

        fresh = ScenePlanParser(config)
        reparse = fresh.parse(result["file_path"])
        assert reparse.total_panels == 7
        assert reparse.scene_id == "s904"

    def test_failure_leaves_no_scene_file(self, config):
        mock = MockLLM([
            chunker_ok(2),
            element_ok(3), element_ok(3), element_ok(3),  # element 1 never complies
        ])
        producer = make_producer(config, mock)
        with pytest.raises(SceneProducerLLMError):
            producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        assert not config.scenes_dir.exists() or not list(config.scenes_dir.glob("*.yaml"))
        assert not list(config.output_dir.glob("*.panelspec.json")) if config.output_dir.exists() else True

    def test_no_temp_files_left_behind(self, config):
        producer = make_producer(config, standard_mock())
        producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")
        leftovers = [p for p in config.scenes_dir.iterdir() if p.name.startswith(".")]
        assert leftovers == []

    def test_produce_by_scene_id_convenience(self, config, scene_input):
        config.scene_inputs_dir.mkdir(parents=True)
        (config.scene_inputs_dir / "c01_s904.input.yaml").write_text(yaml.safe_dump(scene_input))
        producer = make_producer(config, standard_mock())
        result = producer.produce_scene("s904")   # scene_id, not a path
        assert result["scene"]["scene_id"] == "s904"

    def test_progress_callback_fires_in_order(self, config):
        """Milestones + retries are reported when a callback is wired."""
        events = []
        mock = MockLLM([
            chunker_ok(2),
            element_ok(3),   # element 1 wrong length -> retry notice
            element_ok(4),   # retry ok
            element_ok(3),
        ])
        producer = SceneProducer(config, llm_client=mock, progress_callback=events.append)
        producer.produce_scene(FIXTURES_DIR / "fixture_scene_input.yaml")

        assert events[0].startswith("Input validated: s904")
        assert events[1].startswith("Stage 1 complete")
        assert "Stage 2 element 1 retry (2/3)" in events[2]
        assert "element 1/2" in events[3]
        assert "element 2/2" in events[4]
        assert events[5].startswith("Parse gate:")
        assert events[-1].startswith("Parse gate passed")

    def test_untagged_input_gets_plain_filename(self, config, scene_input):
        del scene_input["chapter_tag"]
        config.scene_inputs_dir.mkdir(parents=True)
        (config.scene_inputs_dir / "s904.input.yaml").write_text(yaml.safe_dump(scene_input))
        producer = make_producer(config, standard_mock())
        result = producer.produce_scene("s904")
        assert result["file_path"].name == "s904.yaml"
