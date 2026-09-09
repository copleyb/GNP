"""
Tests for the Scene Plan Parser (scene_parser.py) and Template Registry
(templates.py) — Scope Redesign Phase 1 (DESIGN.md §16).

All tests are offline: no LLM calls, no API keys. Uses real project assets
(alyssa, hood, city_exterior, style.yaml, templates/) plus a test-only
template menu for geometry edge cases (skyline, floats, overflow).
"""

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Fix import paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.config import load_config
from pipeline.scene_parser import (
    ScenePlanParser,
    SceneParseResult,
    SceneParserError,
    SceneSchemaError,
    SceneReferenceError,
    SceneGeometryError,
    SceneCoverageError,
)
from pipeline.templates import (
    TemplateRegistry,
    TemplateLoadError,
    TemplateValidationError,
)
from pipeline.compiler import select_aspect_ratio


PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_TEMPLATES = FIXTURES_DIR / "test_templates"


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture()
def config():
    return load_config(str(PROJECT_ROOT))


@pytest.fixture()
def parser(config, tmp_path):
    """Real-parser fixture with output redirected to a tmp directory."""
    cfg = dataclasses.replace(config, output_dir=tmp_path / "output")
    return ScenePlanParser(cfg)


@pytest.fixture()
def geo_parser(config, tmp_path):
    """Parser against the test-only template menu (edge-case geometry)."""
    cfg = dataclasses.replace(
        config,
        templates_dir=TEST_TEMPLATES,
        output_dir=tmp_path / "output",
    )
    return ScenePlanParser(cfg)


def write_scene(tmp_path, scene: dict, name: str = "scene.yaml") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scene, f, sort_keys=False)
    return path


def load_fixture_scene(filename: str) -> dict:
    with (FIXTURES_DIR / filename).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expected_seed(panel_id: str) -> str:
    digest = hashlib.sha256(panel_id.encode("utf-8")).hexdigest()
    return f"{int(digest[:2], 16):02X}"


def write_menu(dir_path: Path, panels=None, strips=None, layouts=None) -> Path:
    """Write a test template menu directory."""
    dir_path.mkdir(parents=True, exist_ok=True)
    if panels is not None:
        with (dir_path / "panels.yaml").open("w") as f:
            yaml.safe_dump(panels, f)
    if strips is not None:
        with (dir_path / "strips.yaml").open("w") as f:
            yaml.safe_dump({"strips": strips}, f)
    if layouts is not None:
        with (dir_path / "layouts.yaml").open("w") as f:
            yaml.safe_dump({"layouts": layouts}, f)
    return dir_path


# -- Template Registry -----------------------------------------------------------

class TestTemplateRegistry:

    def test_real_menu_loads(self):
        r = TemplateRegistry(PROJECT_ROOT / "templates")
        assert r.panel_count() == 28
        assert r.strip_count() == 16
        assert r.layout_count() == 10
        assert r.has_panel("p0050")
        assert r.has_strip("s_eq3_33")
        assert r.has_layout("l_quad")
        assert not r.has_layout("nope")

    def test_real_menu_strip_members(self):
        r = TemplateRegistry(PROJECT_ROOT / "templates")
        members = r.get_strip("s_eq3_33")
        assert [m.position for m in members] == [1, 2, 3]
        assert all(m.panel_id == "p3333" for m in members)
        assert all(m.floating is None for m in members)

    def test_real_menu_layout_members(self):
        r = TemplateRegistry(PROJECT_ROOT / "templates")
        members = r.get_layout("l_opening_4_inset")
        kinds = [(m.kind, m.ref_id, m.floating) for m in members]
        assert kinds == [
            ("panel", "p0066", None),
            ("strip", "s_eq3_33", None),
            ("panel", "p900x500", (400, 400)),
        ]
        assert [m.strip_index for m in members] == [1, 2, 3]

    def test_missing_menu_file(self, tmp_path):
        write_menu(tmp_path, panels={"p": {"width_px": 1, "height_px": 1}})
        with pytest.raises(TemplateLoadError):
            TemplateRegistry(tmp_path)

    def test_panel_preset_extra_fields_rejected(self, tmp_path):
        write_menu(
            tmp_path,
            panels={"p": {"width_px": 1, "height_px": 1, "notes": "hi"}},
            strips={"s": {"panels": [{"panel": "p"}]}},
            layouts={"l": {"elements": [{"strip": "s"}]}},
        )
        with pytest.raises(TemplateValidationError, match="nothing more"):
            TemplateRegistry(tmp_path)

    def test_unknown_panel_in_strip(self, tmp_path):
        write_menu(
            tmp_path,
            panels={"p": {"width_px": 1, "height_px": 1}},
            strips={"s": {"panels": [{"panel": "ghost"}]}},
            layouts={"l": {"elements": [{"strip": "s"}]}},
        )
        with pytest.raises(TemplateValidationError, match="unknown panel 'ghost'"):
            TemplateRegistry(tmp_path)

    def test_strip_missing_panels_list(self, tmp_path):
        write_menu(
            tmp_path,
            panels={"p": {"width_px": 1, "height_px": 1}},
            strips={"s": {"description": "no panels here"}},
            layouts={"l": {"elements": [{"strip": "s"}]}},
        )
        with pytest.raises(TemplateValidationError, match="non-empty 'panels' list"):
            TemplateRegistry(tmp_path)

    def test_layout_unknown_strip(self, tmp_path):
        write_menu(
            tmp_path,
            panels={"p": {"width_px": 1, "height_px": 1}},
            strips={"s": {"panels": [{"panel": "p"}]}},
            layouts={"l": {"elements": [{"strip": "ghost"}]}},
        )
        with pytest.raises(TemplateValidationError, match="unknown strip 'ghost'"):
            TemplateRegistry(tmp_path)

    def test_layout_element_without_ref(self, tmp_path):
        write_menu(
            tmp_path,
            panels={"p": {"width_px": 1, "height_px": 1}},
            strips={"s": {"panels": [{"panel": "p"}]}},
            layouts={"l": {"elements": [{"nothing": "here"}]}},
        )
        with pytest.raises(TemplateValidationError, match="must reference"):
            TemplateRegistry(tmp_path)

    def test_floating_bad_shape(self, tmp_path):
        write_menu(
            tmp_path,
            panels={"p": {"width_px": 1, "height_px": 1}},
            strips={"s": {"panels": [{"panel": "p", "floating": {"x": 1}}]}},
            layouts={"l": {"elements": [{"strip": "s"}]}},
        )
        with pytest.raises(TemplateValidationError, match="floating"):
            TemplateRegistry(tmp_path)


# -- Scene schema validation --------------------------------------------------------

class TestSceneSchema:

    def test_missing_narrative(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        del scene["narrative"]
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneSchemaError):
            parser.parse(path)

    def test_bad_scene_id(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        scene["scene_id"] = "702"
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneSchemaError):
            parser.parse(path)

    def test_bad_panel_key_pattern(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        content = scene["panels"].pop("s901_l01_st01_pn01")
        scene["panels"]["just_some_key"] = content
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneSchemaError):
            parser.parse(path)

    def test_element_with_both_refs(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        scene["elements"].append({"layout": "l_quad", "strip": "s_eq3_33"})
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneSchemaError):
            parser.parse(path)

    def test_panel_missing_description(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        del scene["panels"]["s901_l01_st01_pn01"]["description"]
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneSchemaError):
            parser.parse(path)

    def test_panel_additive_field_rejected_until_version_bump(self, parser, tmp_path):
        """§16.7: additive fields are deliberate — schema is strict until bumped."""
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        scene["panels"]["s901_l01_st01_pn01"]["body_language"] = "leaning"
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneSchemaError):
            parser.parse(path)

    def test_missing_scene_file(self, parser, tmp_path):
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "ghost.yaml")


# -- Reference resolution ------------------------------------------------------------

class TestSceneReferenceResolution:

    def test_unknown_layout_ref(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        scene["elements"] = [{"layout": "l_ghost"}]
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneReferenceError, match="l_ghost"):
            parser.parse(path)

    def test_unknown_character(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        scene["panels"]["s901_l01_st01_pn01"]["characters"] = [{"id": "ghost"}]
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneReferenceError, match="ghost"):
            parser.parse(path)

    def test_unknown_environment(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        scene["panels"]["s901_l01_st01_pn01"]["environment"] = "ghost_place"
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneReferenceError, match="ghost_place"):
            parser.parse(path)


# -- ID derivation + coverage gate ------------------------------------------------------

class TestDerivationAndCoverage:

    def test_mixed_scene_derives_expected_ids(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_mixed.yaml")
        ids = [p.panel_spec["panel_id"] for p in result.panels]
        assert ids == [
            "s902_l01_st01_pn01",
            "s902_l01_st01_pn02",
            "s902_l01_st02_pn01",
            "s902_l01_st02_pn02",
            "s902_l02_st01_pn01",
            "s902_l02_st01_pn02",
            "s902_l02_st01_pn03",
            "s902_l03_st01_pn01",
        ]
        assert result.total_panels == 8

    def test_coverage_missing_entry(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        del scene["panels"]["s901_l01_st02_pn01"]
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneCoverageError, match="missing content entries.*s901_l01_st02_pn01"):
            parser.parse(path)

    def test_coverage_orphan_key(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        content = dict(scene["panels"]["s901_l01_st01_pn01"])
        scene["panels"]["s901_l01_st01_pn99"] = content
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneCoverageError, match="orphan.*s901_l01_st01_pn99"):
            parser.parse(path)

    def test_coverage_both_reported(self, parser, tmp_path):
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        del scene["panels"]["s901_l01_st01_pn01"]
        content = dict(scene["panels"]["s901_l01_st01_pn02"])
        scene["panels"]["s901_l01_st01_pn99"] = content
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneCoverageError) as exc:
            parser.parse(path)
        assert "missing content entries" in str(exc.value)
        assert "orphan" in str(exc.value)


# -- Geometry ------------------------------------------------------------------------------

class TestSceneGeometry:

    def test_strip_skyline_flow(self, geo_parser):
        """Mixed-height strip: flow left-to-right, all at y=0, height = max."""
        placements, height = geo_parser._flow_strip("s_mixed")
        assert height == 80
        p1, p2 = placements
        assert (p1.x, p1.y, p1.width_px, p1.height_px) == (0, 0, 100, 50)
        assert (p2.x, p2.y, p2.width_px, p2.height_px) == (120, 0, 100, 80)

    def test_strip_float_does_not_consume_flow(self, geo_parser):
        """Floats take explicit coords and don't shift the flow cursor."""
        placements, height = geo_parser._flow_strip("s_float")
        anchor, fl = placements
        assert (anchor.x, anchor.y) == (0, 0)
        assert (fl.x, fl.y) == (10, 10)
        assert fl.is_float
        assert height == 50  # anchor height only; float is an overlay

    def test_strip_all_floats_rejected(self, geo_parser):
        with pytest.raises(SceneGeometryError, match="flow anchor"):
            geo_parser._flow_strip("s_all_floats")

    def test_strip_float_escape_rejected(self, geo_parser):
        with pytest.raises(SceneGeometryError, match="floating"):
            geo_parser._flow_strip("s_float_escape")

    def test_strip_overflow_width_rejected(self, geo_parser):
        with pytest.raises(SceneGeometryError, match="usable width"):
            geo_parser._flow_strip("s_overflow_w")

    def test_layout_stack(self, geo_parser):
        """Two 80-high strips + 20 gutter: second strip starts at y=100."""
        placements = geo_parser._flow_layout("l_two")
        assert len(placements) == 4
        strip2_ys = {p.y for p in placements if p.strip_slot == 2}
        assert strip2_ys == {100}

    def test_layout_float_panel(self, geo_parser):
        placements = geo_parser._flow_layout("l_inset")
        fl = [p for p in placements if p.is_float]
        assert len(fl) == 1
        assert (fl[0].x, fl[0].y) == (5, 5)
        assert fl[0].strip_slot == 2  # float counts as a strip member

    def test_layout_float_strip(self, geo_parser):
        placements = geo_parser._flow_layout("l_float_strip")
        fl = [p for p in placements if p.strip_slot == 2]
        assert len(fl) == 3
        assert {p.x for p in fl} == {40, 160, 280}

    def test_layout_overflow_rejected(self, geo_parser):
        with pytest.raises(SceneGeometryError, match="overflows usable height"):
            geo_parser._flow_layout("l_overflow")

    def test_layout_float_escape_rejected(self, geo_parser):
        with pytest.raises(SceneGeometryError, match="escapes"):
            geo_parser._flow_layout("l_float_escape")

    def test_scene_bare_panel_too_tall(self, geo_parser, tmp_path):
        scene = {
            "scene_id": "s9",
            "title": "t",
            "narrative": "n",
            "elements": [{"panel": "ptall"}],
            "panels": {"s9_l01_st01_pn01": {
                "description": "d", "environment": "city_exterior",
                "shot_type": "wide", "mood": "m",
            }},
        }
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneGeometryError, match="exceeds usable height"):
            geo_parser.parse(path)

    def test_scene_strip_taller_than_page(self, geo_parser, tmp_path):
        scene = {
            "scene_id": "s9",
            "title": "t",
            "narrative": "n",
            "elements": [{"strip": "s_tall"}],
            "panels": {
                "s9_l01_st01_pn01": {
                    "description": "d", "environment": "city_exterior",
                    "shot_type": "wide", "mood": "m",
                },
            },
        }
        path = write_scene(tmp_path, scene)
        with pytest.raises(SceneGeometryError, match="exceeds usable height"):
            geo_parser.parse(path)


# -- Scene-space flow (real menu) --------------------------------------------------------------

class TestSceneSpaceFlow:

    def test_layout_consumes_full_page(self, parser, tmp_path):
        """A layout element consumes one full page of scene space."""
        result = parser.parse(FIXTURES_DIR / "fixture_scene_mixed.yaml")
        geos = {p.panel_spec["panel_id"]: p.panel_spec["panel_geometry"]
                for p in result.panels}
        # Element 1 (l_quad) fills the page at scene y=0..3448.
        # p5050 = 1200 x 1714; st02 starts at y = 1714 + 20 = 1734.
        assert geos["s902_l01_st01_pn01"] == {"x": 0, "y": 0, "width_px": 1200, "height_px": 1714}
        assert geos["s902_l01_st01_pn02"] == {"x": 1220, "y": 0, "width_px": 1200, "height_px": 1714}
        assert geos["s902_l01_st02_pn01"] == {"x": 0, "y": 1734, "width_px": 1200, "height_px": 1714}
        # Element 2 (loose strip) starts after the page + gutter: 3448 + 20 = 3468.
        assert geos["s902_l02_st01_pn01"] == {"x": 0, "y": 3468, "width_px": 793, "height_px": 1136}
        assert geos["s902_l02_st01_pn02"]["x"] == 813
        assert geos["s902_l02_st01_pn03"]["x"] == 1626
        # Element 3 (bare panel, 2420 x 1136) flows at 3468 + 1136 + 20 = 4624.
        assert geos["s902_l03_st01_pn01"] == {"x": 0, "y": 4624, "width_px": 2420, "height_px": 1136}
        # Scene height: 4624 + 1136 = 5760.
        assert result.scene_height_px == 5760

    def test_inset_float_page_relative(self, parser, tmp_path):
        """l_opening_4_inset float sits at page-margin-relative (400, 400)."""
        result = parser.parse(FIXTURES_DIR / "fixture_scene_inset.yaml")
        geos = {p.panel_spec["panel_id"]: p.panel_spec["panel_geometry"]
                for p in result.panels}
        # Bare p0066 (2420 x 2292) then s_eq3_33 (1136) stacked: exact page.
        assert geos["s903_l01_st01_pn01"]["y"] == 0
        assert geos["s903_l01_st02_pn01"]["y"] == 2312
        assert geos["s903_l01_st02_pn01"]["height_px"] == 1136
        # The float: 900 x 500 at (400, 400), page-relative.
        assert geos["s903_l01_st03_pn01"] == {"x": 400, "y": 400, "width_px": 900, "height_px": 500}


# -- PanelSpec emission ------------------------------------------------------------------------

class TestPanelSpecEmission:

    def test_minimal_scene_emits_four_specs(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        assert isinstance(result, SceneParseResult)
        assert result.scene_id == "s901"
        assert result.total_panels == 4
        for p in result.panels:
            assert p.output_path.exists()
            assert p.output_path.name == f"{p.panel_spec['panel_id']}.panelspec.json"

    def test_panel_spec_fields(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        spec = result.panels[0].panel_spec
        assert spec["panel_id"] == "s901_l01_st01_pn01"
        assert spec["scene_id"] == "s901"
        assert spec["chapter_tag"] == "c01"       # display only
        assert spec["title"] == "Fixture scene — minimal"
        assert "midmorning" in spec["continuity_narrative"]
        assert spec["shot_type"] == "wide"
        assert spec["mood"] == "calm"
        assert spec["description"]
        assert spec["style"]["style_id"]
        assert spec["compiler_version"]
        assert spec["panel_geometry"]["width_px"] == 1200

    def test_character_costume_resolution(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        alyssa = result.panels[0].panel_spec["characters"][0]
        assert alyssa["character_id"] == "alyssa"
        assert alyssa["costume_variant"] == "morning_routine"
        assert "Currently wearing" in alyssa["prompt_tokens"]["identity"]
        # Hood panel uses default costume (no costume field).
        hood = result.panels[3].panel_spec["characters"][0]
        assert hood["character_id"] == "hood"
        assert hood["costume_variant"] == "default"

    def test_environment_resolution(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        env = result.panels[0].panel_spec["environment"]
        assert env["environment_id"] == "city_exterior"
        assert env["prompt_tokens"]
        assert "references" in env

    def test_panel_seed_deterministic(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        for p in result.panels:
            assert p.panel_spec["panel_seed"] == expected_seed(p.panel_spec["panel_id"])

    def test_reparse_byte_identical(self, parser, tmp_path):
        """Parse is a pure function: same inputs -> byte-identical PanelSpecs."""
        first = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        snapshot = {p.output_path.name: p.output_path.read_bytes() for p in first.panels}
        second = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        for p in second.panels:
            assert p.output_path.read_bytes() == snapshot[p.output_path.name]

    def test_empty_characters_list(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_mixed.yaml")
        # s902_l01_st02_pn01 declares an empty characters list.
        spec = next(
            p.panel_spec for p in result.panels
            if p.panel_spec["panel_id"] == "s902_l01_st02_pn01"
        )
        assert spec["characters"] == []

    def test_downstream_aspect_ratio_compat(self, parser, tmp_path):
        """Emitted geometry drives the compiler's aspect-ratio selector."""
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        for p in result.panels:
            geo = p.panel_spec["panel_geometry"]
            # 1200 x 1714 (ratio 0.70) -> closest gpt-image-2 size is 1024x1536.
            assert select_aspect_ratio(geo["width_px"], geo["height_px"]) == "1024x1536"

    def test_panel_specs_json_serializable(self, parser, tmp_path):
        result = parser.parse(FIXTURES_DIR / "fixture_scene_minimal.yaml")
        for p in result.panels:
            json.dumps(p.panel_spec)


# -- parse_scene convenience -----------------------------------------------------------------------

class TestParseScene:

    def test_parse_scene_glob(self, config, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        cfg = dataclasses.replace(
            config, scenes_dir=scenes_dir, output_dir=tmp_path / "output"
        )
        scene = load_fixture_scene("fixture_scene_minimal.yaml")
        with (scenes_dir / "c01_s901.yaml").open("w") as f:
            yaml.safe_dump(scene, f)
        parser = ScenePlanParser(cfg)
        result = parser.parse_scene("s901")
        assert result.total_panels == 4

    def test_parse_scene_missing(self, config, tmp_path):
        cfg = dataclasses.replace(
            config, scenes_dir=tmp_path / "empty", output_dir=tmp_path / "output"
        )
        parser = ScenePlanParser(cfg)
        with pytest.raises(FileNotFoundError):
            parser.parse_scene("s901")


# -- Config guard -----------------------------------------------------------------------------------

class TestSceneConfigGuard:

    def test_parser_requires_scene_block(self, config, tmp_path):
        cfg = dataclasses.replace(config, scene=None)
        with pytest.raises(SceneParserError, match="scene"):
            ScenePlanParser(cfg)
