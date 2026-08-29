"""
test_producer.py — Test suite for the Chapter Plan Producer.

Tests cover:
1. Context assembly (characters, environments, layouts, style)
2. System prompt construction
3. User prompt construction
4. OpenAI schema conversion
5. Reference ID validation
6. Panel count validation
7. Chapter plan writing (YAML output, schema validation)
8. Context management placeholders
9. Full produce() flow with mocked LLM
10. Error handling for invalid LLM output
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import load_config
from producer import ChapterPlanProducer


# -- Test data --------------------------------------------------------------

SAMPLE_SYNOPSIS = """Chapter 1: First Light.

Alyssa wakes before dawn in her apartment, preparing for the day ahead.
She moves through the quiet city streets to a dimly lit bar, where she
meets a contact. Hood watches from the shadows. The chapter ends with
Alyssa walking away, package secured, as Hood observes from a distance."""

VALID_CHAPTER_PLAN = {
    "chapter_id": 1,
    "title": "First Light",
    "notes": "Opening chapter. Establishes Alyssa's world and introduces Hood as observer.",
    "pages": [
        {
            "page_id": "1_1",
            "layout": "layout_02",
            "continuity": {
                "time_of_day": "pre-dawn",
                "location": "alyssa_apartment",
            },
            "panels": [
                {
                    "position": 1,
                    "scene_id": "c01_s01",
                    "characters": [{"character_id": "alyssa"}],
                    "environment": "alyssa_apartment",
                    "shot_type": "wide",
                    "mood": "quiet anticipation",
                    "description": "Alyssa awakens in her small apartment, a hint of dawn light filtering through the window blinds.",
                },
                {
                    "position": 2,
                    "scene_id": "c01_s01",
                    "characters": [{"character_id": "alyssa"}],
                    "environment": "alyssa_apartment",
                    "shot_type": "medium",
                    "mood": "determined",
                    "description": "Alyssa pulls on her technical jacket, readying herself for the day. The clasps click shut.",
                },
                {
                    "position": 3,
                    "scene_id": "c01_s01",
                    "characters": [{"character_id": "alyssa"}],
                    "environment": "alyssa_apartment",
                    "shot_type": "close_up",
                    "mood": "focused",
                    "description": "Close-up of Alyssa's hands securing the distinctive clasps on her jacket collar.",
                },
            ],
        },
        {
            "page_id": "1_2",
            "layout": "layout_01",
            "continuity": {
                "time_of_day": "early morning",
                "location": "city_exterior",
            },
            "panels": [
                {
                    "position": 1,
                    "scene_id": "c01_s01",
                    "characters": [{"character_id": "alyssa"}],
                    "environment": "city_exterior",
                    "shot_type": "wide",
                    "mood": "serene and purposeful",
                    "description": "Alyssa walks swiftly along the cold city streets, the urban canyon stretching before her.",
                },
                {
                    "position": 2,
                    "scene_id": "c01_s02",
                    "characters": [{"character_id": "hood"}],
                    "environment": "city_exterior",
                    "shot_type": "overhead",
                    "mood": "mysterious",
                    "description": "Hood watches Alyssa from above, blending into the shadows of a fire escape.",
                },
            ],
        },
    ],
    "scenes": [
        {
            "scene_id": "c01_s01",
            "panels": [
                {"page": "1_1", "position": 1, "narrative": "Alyssa in bed, dawn light, sleepwear."},
                {"page": "1_1", "position": 2, "narrative": "Alyssa pulling on jacket, same room."},
                {"page": "1_1", "position": 3, "narrative": "Close-up of jacket clasps, same room."},
                {"page": "1_2", "position": 1, "narrative": "Alyssa walking city streets, different environment."},
            ],
        },
        {
            "scene_id": "c01_s02",
            "panels": [
                {"page": "1_2", "position": 2, "narrative": "Hood watching from above, new scene."},
            ],
        },
    ],
}


# -- Mock LLM call ----------------------------------------------------------

class MockProducer(ChapterPlanProducer):
    """ChapterPlanProducer with a mockable LLM call for testing."""

    def __init__(self, config, mock_response=None, model="gpt-4o", **kwargs):
        super().__init__(config, model=model, **kwargs)
        self._mock_response = mock_response or VALID_CHAPTER_PLAN
        self.llm_call_count = 0
        self.last_system_prompt = None
        self.last_user_prompt = None

    def call_llm(self, system_prompt, user_prompt):
        self.llm_call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return json.loads(json.dumps(self._mock_response))  # deep copy


# -- Test runner ------------------------------------------------------------

def run_tests():
    tests = [
        ("Context assembly loads characters", test_context_characters),
        ("Context assembly loads environments", test_context_environments),
        ("Context assembly loads layouts", test_context_layouts),
        ("Context assembly loads style", test_context_style),
        ("System prompt contains critical rules", test_system_prompt),
        ("User prompt contains synopsis and context", test_user_prompt),
        ("OpenAI schema is compatible with strict mode", test_openai_schema),
        ("Reference ID validation catches unknown characters", test_unknown_character),
        ("Reference ID validation catches unknown environments", test_unknown_environment),
        ("Reference ID validation catches unknown layouts", test_unknown_layout),
        ("Panel count validation catches mismatch", test_panel_count_mismatch),
        ("Chapter plan writes valid YAML", test_write_chapter_plan),
        ("Written YAML is readable and valid", test_written_yaml_readable),
        ("Context management placeholders are no-ops", test_context_placeholders),
        ("Full produce() flow with mock LLM", test_full_produce_flow),
        ("Produce() validates LLM output against full schema", test_produce_validates_schema),
        ("Produce() catches invalid shot_type", test_invalid_shot_type),
        ("Produce() overwrites existing chapter plan", test_overwrite),
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

def _get_producer():
    """Create a MockProducer for testing."""
    config = load_config(".")
    return MockProducer(config)


def test_context_characters():
    """Test that context assembly loads all character files."""
    producer = _get_producer()
    context = producer.assemble_context()
    assert "characters" in context
    char_ids = [c["character_id"] for c in context["characters"]]
    assert "alyssa" in char_ids, f"alyssa not found in {char_ids}"
    assert "hood" in char_ids, f"hood not found in {char_ids}"
    # Verify character data is structured correctly
    alyssa = next(c for c in context["characters"] if c["character_id"] == "alyssa")
    assert alyssa["display_name"] == "Alyssa"
    assert "blonde" in alyssa["physical_description"]["hair"]


def test_context_environments():
    """Test that context assembly loads all environment files."""
    producer = _get_producer()
    context = producer.assemble_context()
    env_ids = [e["environment_id"] for e in context["environments"]]
    assert "city_exterior" in env_ids
    assert "city_bar_interior" in env_ids
    assert "alyssa_apartment" in env_ids
    ext = next(e for e in context["environments"] if e["environment_id"] == "city_exterior")
    assert "no greenery" in ext["exclusions"], f"Exclusions not loaded: {ext['exclusions']}"


def test_context_layouts():
    """Test that context assembly loads all layout files with panel counts."""
    producer = _get_producer()
    context = producer.assemble_context()
    layout_ids = [l["layout_id"] for l in context["layouts"]]
    assert "layout_01" in layout_ids
    assert "layout_02" in layout_ids
    l02 = next(l for l in context["layouts"] if l["layout_id"] == "layout_02")
    assert l02["panel_count"] == 3, f"layout_02 should have 3 panels, got {l02['panel_count']}"
    l01 = next(l for l in context["layouts"] if l["layout_id"] == "layout_01")
    assert l01["panel_count"] == 2


def test_context_style():
    """Test that context assembly loads the style file."""
    producer = _get_producer()
    context = producer.assemble_context()
    assert "style" in context
    assert context["style"]["style_id"] == "new_bridgeton_vivid"
    assert "graphic novel" in context["style"]["visual_style"]["label"]
    assert len(context["style"]["forbidden_elements"]) > 0


def test_system_prompt():
    """Test that the system prompt contains the critical rules."""
    producer = _get_producer()
    prompt = producer._build_system_prompt()
    assert "graphic novel chapter planner" in prompt
    assert "Only use character_ids that exist" in prompt
    assert "Only use environment_ids that exist" in prompt
    assert "Only use layout_ids that exist" in prompt
    assert "panel count" in prompt.lower()
    assert "exclusions" in prompt.lower()


def test_user_prompt():
    """Test that the user prompt contains the synopsis and project context."""
    producer = _get_producer()
    prompt = producer._build_user_prompt(SAMPLE_SYNOPSIS, 1)
    assert "First Light" in prompt  # synopsis title
    assert "alyssa" in prompt.lower()
    assert "hood" in prompt.lower()
    assert "city_exterior" in prompt
    assert "city_bar_interior" in prompt
    assert "alyssa_apartment" in prompt
    assert "layout_01" in prompt
    assert "layout_02" in prompt
    assert "graphic novel" in prompt
    assert "Chapter 1" in prompt


def test_openai_schema():
    """Test that the OpenAI schema is compatible with strict mode."""
    producer = _get_producer()
    schema = producer._to_openai_schema()

    # Strict mode requires additionalProperties: false at every object level
    assert schema["additionalProperties"] is False
    assert "notes" in schema["required"]  # all properties must be required

    # Check nested objects
    page_schema = schema["properties"]["pages"]["items"]
    assert page_schema["additionalProperties"] is False
    assert "page_id" in page_schema["required"]

    continuity_schema = page_schema["properties"]["continuity"]
    assert continuity_schema["additionalProperties"] is False

    panel_schema = page_schema["properties"]["panels"]["items"]
    assert panel_schema["additionalProperties"] is False
    assert "position" in panel_schema["required"]
    assert "description" in panel_schema["required"]

    # notes should be nullable
    assert schema["properties"]["notes"]["type"] == ["string", "null"]


def test_unknown_character():
    """Test that the reference ID validator catches unknown character IDs."""
    producer = _get_producer()
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    plan["pages"][0]["panels"][0]["characters"] = [{"character_id": "unknown_char"}]
    try:
        producer._validate_referenced_ids(plan)
        raise AssertionError("Should have caught unknown character ID")
    except ValueError as e:
        assert "unknown_char" in str(e)


def test_unknown_environment():
    """Test that the reference ID validator catches unknown environment IDs."""
    producer = _get_producer()
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    plan["pages"][0]["panels"][0]["environment"] = "unknown_env"
    try:
        producer._validate_referenced_ids(plan)
        raise AssertionError("Should have caught unknown environment ID")
    except ValueError as e:
        assert "unknown_env" in str(e)


def test_unknown_layout():
    """Test that the reference ID validator catches unknown layout IDs."""
    producer = _get_producer()
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    plan["pages"][0]["layout"] = "unknown_layout"
    try:
        producer._validate_referenced_ids(plan)
        raise AssertionError("Should have caught unknown layout ID")
    except ValueError as e:
        assert "unknown_layout" in str(e)


def test_panel_count_mismatch():
    """Test that panel count validation catches mismatches."""
    producer = _get_producer()
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    # layout_02 expects 3 panels, add a 4th
    plan["pages"][0]["panels"].append({
        "position": 4,
        "characters": [{"character_id": "alyssa"}],
        "environment": "alyssa_apartment",
        "shot_type": "wide",
        "mood": "confused",
        "description": "Extra panel that shouldn't be here.",
    })
    try:
        producer._validate_referenced_ids(plan)
        raise AssertionError("Should have caught panel count mismatch")
    except ValueError as e:
        assert "panel count" in str(e).lower()


def test_write_chapter_plan():
    """Test that a valid chapter plan is written as YAML."""
    producer = _get_producer()
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    path = producer.write_chapter_plan(plan, 1)
    assert path.exists(), f"File not written: {path}"
    assert path.name == "chapter_1.yaml"
    assert path.parent == producer.config.chapters_dir


def test_written_yaml_readable():
    """Test that the written YAML is readable and matches the input."""
    import yaml as yaml_lib
    producer = _get_producer()
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    path = producer.write_chapter_plan(plan, 1)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml_lib.safe_load(f)
    assert loaded["chapter_id"] == 1
    assert loaded["title"] == "First Light"
    assert len(loaded["pages"]) == 2
    assert loaded["pages"][0]["page_id"] == "1_1"
    assert loaded["pages"][0]["layout"] == "layout_02"
    assert len(loaded["pages"][0]["panels"]) == 3
    assert loaded["pages"][1]["page_id"] == "1_2"
    assert loaded["pages"][1]["layout"] == "layout_01"
    assert len(loaded["pages"][1]["panels"]) == 2

    # Cleanup
    path.unlink()


def test_context_placeholders():
    """Test that context management placeholders are functional no-ops in v1."""
    producer = _get_producer()
    assert producer.context_store == {}
    result = producer.curate_context()
    assert result == producer.context_store
    # update_context should not raise
    producer.update_context(VALID_CHAPTER_PLAN)
    # context_store should still be empty (v1: no-op)
    assert producer.context_store == {}


def test_full_produce_flow():
    """Test the full produce() flow with a mocked LLM call."""
    producer = _get_producer()
    result = producer.produce(SAMPLE_SYNOPSIS, 1)

    assert "chapter_plan" in result
    assert "file_path" in result
    assert result["model"] == "gpt-4o"

    # Verify the LLM was called once
    assert producer.llm_call_count == 1

    # Verify the system and user prompts were passed
    assert "graphic novel chapter planner" in producer.last_system_prompt
    assert "First Light" in producer.last_user_prompt

    # Verify the chapter plan
    plan = result["chapter_plan"]
    assert plan["chapter_id"] == 1
    assert plan["title"] == "First Light"
    assert len(plan["pages"]) == 2

    # Verify the file was written
    assert result["file_path"].exists()
    assert result["file_path"].name == "chapter_1.yaml"

    # Cleanup
    result["file_path"].unlink()


def test_produce_validates_schema():
    """Test that produce() validates the LLM output against the full schema."""
    # Create a mock with an invalid plan (missing required field)
    bad_plan = {"chapter_id": 1, "title": "Bad Plan"}  # missing "pages"
    producer = MockProducer(load_config("."), mock_response=bad_plan)
    try:
        producer.produce(SAMPLE_SYNOPSIS, 1)
        raise AssertionError("Should have raised schema validation error")
    except Exception as e:
        # jsonschema.ValidationError or ValueError
        assert "pages" in str(e).lower() or "required" in str(e).lower()


def test_invalid_shot_type():
    """Test that an invalid shot_type is caught by schema validation."""
    plan = json.loads(json.dumps(VALID_CHAPTER_PLAN))
    plan["pages"][0]["panels"][0]["shot_type"] = "invalid_angle"
    producer = MockProducer(load_config("."), mock_response=plan)
    try:
        producer.produce(SAMPLE_SYNOPSIS, 1)
        raise AssertionError("Should have caught invalid shot_type")
    except Exception as e:
        assert "shot_type" in str(e) or "invalid_angle" in str(e) or "enum" in str(e).lower()


def test_overwrite():
    """Test that producing a chapter overwrites an existing file."""
    producer = _get_producer()

    # First write
    result1 = producer.produce(SAMPLE_SYNOPSIS, 1)
    assert result1["file_path"].exists()

    # Second write (same chapter)
    result2 = producer.produce("Different synopsis.", 1)
    assert result2["file_path"].exists()
    assert result2["file_path"] == result1["file_path"]  # same path

    # Cleanup
    result2["file_path"].unlink()


if __name__ == "__main__":
    run_tests()
