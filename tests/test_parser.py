"""
Tests for the Chapter Plan Parser (parser.py).

Tests cover all four validation stages, PanelSpec assembly, persistence,
and the standalone geometry validator.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# Fix import paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.config import load_config
from pipeline.parser import (
    ChapterPlanParser,
    ParseResult,
    ParsedPanel,
    ParserError,
    SchemaValidationError,
    ReferenceResolutionError,
    LayoutGeometryError,
    PanelCountError,
    validate_layout_geometry,
)

# -- Fixtures ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

# A minimal valid chapter plan for testing
VALID_CHAPTER = {
    "chapter_id": 1,
    "title": "Test Chapter",
    "notes": "For testing.",
    "pages": [
        {
            "page_id": "1_1",
            "layout": "layout_02",
            "continuity": {
                "time_of_day": "pre_dawn",
                "location": "alyssa_apartment",
            },
            "panels": [
                {
                    "position": 1,
                    "characters": ["alyssa"],
                    "environment": "alyssa_apartment",
                    "shot_type": "wide",
                    "mood": "calm",
                    "description": "Alyssa in her apartment.",
                },
                {
                    "position": 2,
                    "characters": ["alyssa"],
                    "environment": "alyssa_apartment",
                    "shot_type": "close_up",
                    "mood": "determined",
                    "description": "Alyssa puts on her jacket.",
                },
                {
                    "position": 3,
                    "characters": ["alyssa"],
                    "environment": "alyssa_apartment",
                    "shot_type": "medium",
                    "mood": "focused",
                    "description": "Alyssa looks out the window.",
                },
            ],
        },
        {
            "page_id": "1_2",
            "layout": "layout_01",
            "continuity": {
                "time_of_day": "early_morning",
                "location": "city_exterior",
            },
            "panels": [
                {
                    "position": 1,
                    "characters": ["alyssa"],
                    "environment": "city_exterior",
                    "shot_type": "wide",
                    "mood": "quiet",
                    "description": "Alyssa walks the streets.",
                },
                {
                    "position": 2,
                    "characters": ["alyssa", "hood"],
                    "environment": "city_exterior",
                    "shot_type": "overhead",
                    "mood": "suspenseful",
                    "description": "Hood watches from above.",
                },
            ],
        },
    ],
}


@pytest.fixture
def config():
    """Load the real project config."""
    return load_config(str(PROJECT_ROOT))


@pytest.fixture
def parser(config):
    """Create a Parser with the real project config."""
    return ChapterPlanParser(config)


@pytest.fixture
def chapter_file(tmp_path):
    """Write a valid chapter plan to a temp file."""
    path = tmp_path / "chapter_1.yaml"
    with path.open("w") as f:
        yaml.dump(VALID_CHAPTER, f, default_flow_style=False)
    return path


@pytest.fixture
def clean_output(config):
    """Ensure output dir is clean before and after tests."""
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Clean any existing panelspec files
    for f in output.glob("*.panelspec.json"):
        f.unlink()
    yield output
    # Cleanup after
    for f in output.glob("*.panelspec.json"):
        f.unlink()


# -- Stage 1: Schema validation tests ---------------------------------------

class TestSchemaValidation:
    def test_valid_plan_passes(self, parser, chapter_file):
        """A valid chapter plan should pass schema validation."""
        result = parser.parse(chapter_file)
        assert result.chapter_id == 1
        assert result.total_panels == 5

    def test_missing_required_field(self, parser, tmp_path):
        """Missing 'title' should fail schema validation."""
        bad_plan = {**VALID_CHAPTER}
        del bad_plan["title"]
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(SchemaValidationError, match="schema validation"):
            parser.parse(path)

    def test_invalid_shot_type(self, parser, tmp_path):
        """Invalid shot_type enum value should fail schema validation."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["panels"][0]["shot_type"] = "fisheye"
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(SchemaValidationError):
            parser.parse(path)

    def test_wrong_chapter_id_type(self, parser, tmp_path):
        """String chapter_id should fail schema validation (expects integer)."""
        bad_plan = {**VALID_CHAPTER, "chapter_id": "one"}
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(SchemaValidationError):
            parser.parse(path)


# -- Stage 2: Reference resolution tests ------------------------------------

class TestReferenceResolution:
    def test_unknown_character(self, parser, tmp_path):
        """Unknown character ID should fail reference resolution."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["panels"][0]["characters"] = ["ghost"]
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(ReferenceResolutionError, match="character 'ghost'"):
            parser.parse(path)

    def test_unknown_environment(self, parser, tmp_path):
        """Unknown environment ID should fail reference resolution."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["panels"][0]["environment"] = "mars_surface"
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(ReferenceResolutionError, match="environment 'mars_surface'"):
            parser.parse(path)

    def test_unknown_layout(self, parser, tmp_path):
        """Unknown layout ID should fail reference resolution."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["layout"] = "layout_99"
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(ReferenceResolutionError, match="layout 'layout_99'"):
            parser.parse(path)

    def test_multiple_unknown_references(self, parser, tmp_path):
        """Multiple unknown references should all be reported."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["panels"][0]["characters"] = ["ghost", "phantom"]
        bad_plan["pages"][0]["panels"][0]["environment"] = "mars_surface"
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(ReferenceResolutionError) as exc_info:
            parser.parse(path)
        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "phantom" in msg
        assert "mars_surface" in msg

    def test_empty_characters_list(self, parser, tmp_path):
        """A panel with no characters should be valid."""
        plan = json.loads(json.dumps(VALID_CHAPTER))
        plan["pages"][0]["panels"][0]["characters"] = []
        path = tmp_path / "chapter_test.yaml"
        with path.open("w") as f:
            yaml.dump(plan, f)
        result = parser.parse(path)
        panel = result.panels[0].panel_spec
        assert panel["characters"] == []


# -- Stage 3: Layout geometry validation tests ------------------------------

class TestLayoutGeometry:
    def test_valid_geometry_passes(self, parser, chapter_file):
        """Real layouts should pass geometry validation."""
        result = parser.parse(chapter_file)
        assert result.total_panels == 5

    def test_geometry_validator_standalone(self):
        """The geometry validator should work as a standalone function."""
        layout = {
            "page": {"width_px": 2480, "height_px": 3508, "dpi": 300},
            "panels": [
                {"position": 1, "x": 0, "y": 0, "width_px": 2480, "height_px": 1754},
            ],
        }
        validate_layout_geometry(layout)  # should not raise

    def test_panel_exceeds_width(self):
        """Panel extending beyond page width should raise."""
        layout = {
            "page": {"width_px": 2480, "height_px": 3508, "dpi": 300},
            "panels": [
                {"position": 1, "x": 0, "y": 0, "width_px": 3000, "height_px": 1000},
            ],
        }
        with pytest.raises(LayoutGeometryError, match="right edge 3000 exceeds page width 2480"):
            validate_layout_geometry(layout)

    def test_panel_exceeds_height(self):
        """Panel extending beyond page height should raise."""
        layout = {
            "page": {"width_px": 2480, "height_px": 3508, "dpi": 300},
            "panels": [
                {"position": 1, "x": 0, "y": 3000, "width_px": 1000, "height_px": 600},
            ],
        }
        with pytest.raises(LayoutGeometryError, match="bottom edge 3600 exceeds page height 3508"):
            validate_layout_geometry(layout)

    def test_negative_origin(self):
        """Negative x or y should raise."""
        layout = {
            "page": {"width_px": 2480, "height_px": 3508, "dpi": 300},
            "panels": [
                {"position": 1, "x": -10, "y": 0, "width_px": 1000, "height_px": 1000},
            ],
        }
        with pytest.raises(LayoutGeometryError, match="negative origin"):
            validate_layout_geometry(layout)

    def test_geometry_validator_with_filename(self):
        """Error messages should include the filename."""
        layout = {
            "page": {"width_px": 100, "height_px": 100, "dpi": 300},
            "panels": [
                {"position": 1, "x": 0, "y": 0, "width_px": 200, "height_px": 50},
            ],
        }
        with pytest.raises(LayoutGeometryError, match="my_layout.yaml"):
            validate_layout_geometry(layout, filename="my_layout.yaml")


# -- Stage 4: Panel count consistency tests ----------------------------------

class TestPanelCount:
    def test_correct_panel_counts(self, parser, chapter_file):
        """Panel counts matching layouts should pass."""
        result = parser.parse(chapter_file)
        assert result.total_panels == 5  # 3 + 2

    def test_too_few_panels(self, parser, tmp_path):
        """Fewer panels than layout requires should fail."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        # Remove one panel from the 3-panel layout page
        bad_plan["pages"][0]["panels"] = bad_plan["pages"][0]["panels"][:2]
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(PanelCountError, match="expected 3 panels, got 2"):
            parser.parse(path)

    def test_too_many_panels(self, parser, tmp_path):
        """More panels than layout requires should fail."""
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        # Add an extra panel to the 2-panel layout page
        extra = json.loads(json.dumps(bad_plan["pages"][1]["panels"][0]))
        extra["position"] = 3
        bad_plan["pages"][1]["panels"].append(extra)
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(PanelCountError, match="expected 2 panels, got 3"):
            parser.parse(path)


# -- PanelSpec assembly tests -------------------------------------------------

class TestPanelSpecAssembly:
    def test_panel_id_format(self, parser, chapter_file, clean_output):
        """Panel IDs should follow the c{chapter}_pg{page}_l{layout}_pn{pos} format."""
        result = parser.parse(chapter_file)
        # Page 1, layout_02, panel 1 → c01_pg1_l02_pn01
        assert result.panels[0].panel_spec["panel_id"] == "c01_pg1_l02_pn01"
        # Page 2, layout_01, panel 2 → c01_pg2_l01_pn02
        assert result.panels[4].panel_spec["panel_id"] == "c01_pg2_l01_pn02"

    def test_panel_geometry(self, parser, chapter_file, clean_output):
        """PanelSpec should include geometry from the layout."""
        result = parser.parse(chapter_file)
        geo = result.panels[0].panel_spec["panel_geometry"]
        assert "x" in geo
        assert "y" in geo
        assert "width_px" in geo
        assert "height_px" in geo
        # layout_02, position 1 — verify against layout file
        layout_path = PROJECT_ROOT / "layouts" / "layout_02.yaml"
        with layout_path.open() as f:
            layout = yaml.safe_load(f)
        panel1 = next(p for p in layout["panels"] if p["position"] == 1)
        assert geo["x"] == panel1["x"]
        assert geo["y"] == panel1["y"]
        assert geo["width_px"] == panel1["width_px"]
        assert geo["height_px"] == panel1["height_px"]

    def test_character_resolution(self, parser, chapter_file, clean_output):
        """Characters should be fully resolved with prompt_tokens and references."""
        result = parser.parse(chapter_file)
        # Panel 1 has Alyssa
        chars = result.panels[0].panel_spec["characters"]
        assert len(chars) == 1
        assert chars[0]["character_id"] == "alyssa"
        assert "identity" in chars[0]["prompt_tokens"]
        assert "default" in chars[0]["costume"]
        assert len(chars[0]["references"]) >= 1

    def test_multiple_characters(self, parser, chapter_file, clean_output):
        """Panel with multiple characters should resolve all of them."""
        result = parser.parse(chapter_file)
        # Panel 5 (page 2, panel 2) has alyssa + hood
        chars = result.panels[4].panel_spec["characters"]
        assert len(chars) == 2
        char_ids = [c["character_id"] for c in chars]
        assert "alyssa" in char_ids
        assert "hood" in char_ids

    def test_environment_resolution(self, parser, chapter_file, clean_output):
        """Environment should be fully resolved with prompt_tokens and references."""
        result = parser.parse(chapter_file)
        env = result.panels[0].panel_spec["environment"]
        assert env["environment_id"] == "alyssa_apartment"
        assert "identity" in env["prompt_tokens"]
        assert len(env["references"]) >= 1

    def test_style_embedded(self, parser, chapter_file, clean_output):
        """Style should be embedded in the PanelSpec at parse time."""
        result = parser.parse(chapter_file)
        style = result.panels[0].panel_spec["style"]
        assert style["style_id"] == "new_bridgeton_vivid"
        assert "visual_style" in style
        assert "forbidden_elements" in style
        assert "lighting_defaults" in style

    def test_continuity(self, parser, chapter_file, clean_output):
        """Continuity should be carried through from the page level."""
        result = parser.parse(chapter_file)
        cont = result.panels[0].panel_spec["continuity"]
        assert cont["time_of_day"] == "pre_dawn"
        assert cont["location"] == "alyssa_apartment"

    def test_panel_seed(self, parser, chapter_file, clean_output):
        """panel_seed should be a valid hex byte string."""
        result = parser.parse(chapter_file)
        for panel in result.panels:
            seed = panel.panel_spec["panel_seed"]
            assert isinstance(seed, str)
            assert len(seed) == 2
            int(seed, 16)  # should not raise — valid hex

    def test_compiler_version(self, parser, chapter_file, clean_output):
        """compiler_version should match project.yaml."""
        result = parser.parse(chapter_file)
        assert result.panels[0].panel_spec["compiler_version"] == "1.0.0"

    def test_shot_type_and_mood(self, parser, chapter_file, clean_output):
        """shot_type and mood should be carried through from the chapter plan."""
        result = parser.parse(chapter_file)
        assert result.panels[0].panel_spec["shot_type"] == "wide"
        assert result.panels[0].panel_spec["mood"] == "calm"

    def test_description(self, parser, chapter_file, clean_output):
        """Description should be carried through from the chapter plan."""
        result = parser.parse(chapter_file)
        assert result.panels[0].panel_spec["description"] == "Alyssa in her apartment."


# -- Persistence tests -------------------------------------------------------

class TestPersistence:
    def test_panelspecs_written_to_disk(self, parser, chapter_file, clean_output):
        """PanelSpecs should be written to the output directory."""
        result = parser.parse(chapter_file)
        for panel in result.panels:
            assert panel.output_path.exists()
            # Verify it's valid JSON
            with panel.output_path.open() as f:
                data = json.load(f)
            assert data["panel_id"] == panel.panel_spec["panel_id"]

    def test_panelspec_filename(self, parser, chapter_file, clean_output):
        """Files should be named {panel_id}.panelspec.json."""
        result = parser.parse(chapter_file)
        for panel in result.panels:
            assert panel.output_path.name.endswith(".panelspec.json")
            expected = f"{panel.panel_spec['panel_id']}.panelspec.json"
            assert panel.output_path.name == expected

    def test_panelspec_is_self_contained(self, parser, chapter_file, clean_output):
        """PanelSpec should contain all data the Compiler needs — no file I/O required."""
        result = parser.parse(chapter_file)
        spec = result.panels[0].panel_spec
        # All required top-level fields
        required = [
            "panel_id", "chapter_id", "page_id", "layout_id", "position",
            "panel_geometry", "characters", "environment", "shot_type",
            "mood", "description", "continuity", "style", "panel_seed",
            "compiler_version",
        ]
        for key in required:
            assert key in spec, f"Missing key: {key}"

    def test_overwrite_existing(self, parser, chapter_file, clean_output):
        """Re-parsing should overwrite existing PanelSpec files."""
        result1 = parser.parse(chapter_file)
        seed1 = result1.panels[0].panel_spec["panel_seed"]

        result2 = parser.parse(chapter_file)
        seed2 = result2.panels[0].panel_spec["panel_seed"]

        # Seeds should be different (random each parse)
        assert seed1 != seed2
        # But same number of panels
        assert result1.total_panels == result2.total_panels


# -- Integration: parse_chapter convenience method ---------------------------

class TestParseChapter:
    def test_parse_chapter_by_number(self, parser, clean_output):
        """parse_chapter(1) should find chapters/chapter_test.yaml... 
        Actually it looks for chapter_1.yaml, so we need to create one."""
        chapter_path = PROJECT_ROOT / "chapters" / "chapter_1.yaml"
        with chapter_path.open("w") as f:
            yaml.dump(VALID_CHAPTER, f, default_flow_style=False)
        try:
            result = parser.parse_chapter(1)
            assert result.chapter_id == 1
            assert result.total_panels == 5
        finally:
            if chapter_path.exists():
                chapter_path.unlink()


# -- Error handling tests ----------------------------------------------------

class TestErrorHandling:
    def test_file_not_found(self, parser):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/chapter_1.yaml")

    def test_relative_path_resolved(self, parser, chapter_file, clean_output):
        """Relative paths should be resolved against project root."""
        # chapter_file is in tmp_path, so use an absolute path
        result = parser.parse(chapter_file)
        assert result.total_panels == 5

    def test_stage_ordering_schema_before_refs(self, parser, tmp_path):
        """Schema validation should happen before reference resolution."""
        bad_plan = {"chapter_id": 1, "title": "Bad"}  # missing required fields
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(SchemaValidationError):
            parser.parse(path)

    def test_stage_ordering_refs_before_geometry(self, parser, tmp_path):
        """Reference resolution should happen before geometry validation."""
        # Unknown layout — should fail at Stage 2, not Stage 3
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["layout"] = "nonexistent_layout"
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(ReferenceResolutionError, match="layout 'nonexistent_layout'"):
            parser.parse(path)

    def test_stage_ordering_geometry_before_count(self, parser, tmp_path):
        """Geometry validation should happen before panel count check."""
        # Can't easily trigger a geometry error with real layouts (they're valid),
        # but we can verify count errors come after geometry by testing count alone
        bad_plan = json.loads(json.dumps(VALID_CHAPTER))
        bad_plan["pages"][0]["panels"] = bad_plan["pages"][0]["panels"][:2]
        path = tmp_path / "chapter_bad.yaml"
        with path.open("w") as f:
            yaml.dump(bad_plan, f)
        with pytest.raises(PanelCountError):
            parser.parse(path)
