"""
scene_parser.py — Scene Plan Parser (Scope Redesign, DESIGN.md §16).

Parses a Scene Plan YAML file (100% Producer-generated) into PanelSpec JSON
objects for the shared downstream pipeline (Compiler, Backend, Orchestrator,
Validation, Provenance). Additive module: the chapter-plan path (parser.py)
remains operational and frozen until Phase 3 cutover.

Per DESIGN.md §16, the parser:
  1. Validates the scene file against scene_plan.schema.json.
  2. Resolves element template references via the Template Registry.
  3. Derives the panel set deterministically from the elements list:
     element index -> l, strip index within element -> st,
     position within strip -> pn. Bare panel = one-panel strip.
     Floats count as strip members and DO NOT consume flow space.
  4. Enforces the EXACT-COVERAGE gate: the panels map must cover the derived
     set exactly — missing content entries and orphan keys both fail parse.
  5. Flows geometry in continuous SCENE space (y=0 at scene top, page-slicing
     is post-production's concern):
       - non-floating strip panels flow left-to-right from strip x=0, all at
         strip y=0, heights MAY differ (skyline rows); strip height = max of
         its non-floating panels' heights.
       - layout elements flow top-down inside their page frame and the layout
         element consumes one FULL page of scene space.
       - loose strip/panel elements flow top-down from the scene's top-left.
       - gutters are project-level defaults from project.yaml (scene block).
       - every element must fit the usable page area (bleeds applied).
         Overflow fails; under-fill is legal.
       - floats carry explicit coords: within strips, relative to the strip's
         top-left corner; within layouts, relative to the page's top-left
         margin. Overlap legality and z-order are human-owned.
  6. Emits one self-contained PanelSpec per panel as immutable JSON to
     output/{panel_id}.panelspec.json, with tag-free positional IDs
     (s{scene}_l{NN}_st{NN}_pn{NN}) and the chapter_tag carried as
     display-only metadata.

Parse is a PURE FUNCTION of (scene file, templates, project config): the
panel_seed is derived deterministically from the panel_id (sha256 first byte)
rather than randomly, so re-parsing identical inputs yields byte-identical
PanelSpecs (DESIGN.md §16.5).

Per DESIGN.md §13.5: returns structured data, no print() statements, no CLI logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validate as validate_schema

from pipeline.templates import (
    LayoutMember,
    StripMember,
    TemplateRegistry,
    TemplateValidationError,
)
from pipeline.wardrobe import Wardrobe


# -- Exceptions ---------------------------------------------------------------

class SceneParserError(Exception):
    """Base exception for all Scene Parser errors."""
    pass


class SceneSchemaError(SceneParserError):
    """Stage 1: Scene Plan fails JSON Schema validation."""
    pass


class SceneReferenceError(SceneParserError):
    """Stage 2: A template/character/environment reference doesn't resolve."""
    pass


class SceneGeometryError(SceneParserError):
    """Stage 3: An element overflows the usable page area."""
    pass


class SceneCoverageError(SceneParserError):
    """Stage 4: The panels map doesn't exactly cover the derived panel set."""
    pass


# -- Result types ------------------------------------------------------------

@dataclass(frozen=True)
class ParsedScenePanel:
    """A single parsed panel result — one PanelSpec and its output path."""
    panel_spec: dict[str, Any]
    output_path: Path


@dataclass(frozen=True)
class SceneParseResult:
    """Result of parsing a full Scene Plan."""
    scene_file: Path
    scene_id: str
    total_panels: int
    panels: list[ParsedScenePanel]
    scene_height_px: int
    warnings: list[str] = field(default_factory=list)


# -- Internal geometry types --------------------------------------------------

@dataclass(frozen=True)
class _Placement:
    """A resolved panel box, positioned relative to its parent frame."""
    x: int
    y: int
    width_px: int
    height_px: int
    strip_slot: int           # st index within the parent element
    position: int             # pn index within the strip
    is_float: bool = False


# -- Parser -------------------------------------------------------------------

class ScenePlanParser:
    """
    Parses Scene Plan YAML into PanelSpec JSON objects.

    The parser is project-aware: it loads the Template Registry, characters,
    environments, and style from the project configuration. PanelSpecs are
    fully self-contained — the shared downstream pipeline needs no additional
    file I/O.
    """

    def __init__(self, config: Any):
        """
        Initialise the Parser with a project configuration.

        Args:
            config: A ProjectConfig object (from pipeline.config.load_config)
                    with a scene config block (page size, bleed, gutter).

        Raises:
            SceneParserError: If the project has no scene config block.
        """
        if config.scene is None:
            raise SceneParserError(
                "Project config has no 'scene' block (page size, bleed, "
                "gutter) — required for scene parsing. See DESIGN.md §16."
            )
        self.config = config
        self.registry = TemplateRegistry(config.templates_dir)
        self._scene_schema: dict[str, Any] | None = None
        self._wardrobe = Wardrobe(config.characters_dir)
        self._environment_cache: dict[str, dict[str, Any]] = {}

        sc = config.scene
        self.usable_w = sc.page_width_px - 2 * sc.bleed_px
        self.usable_h = sc.page_height_px - 2 * sc.bleed_px
        self.gutter = sc.gutter_px

    # -- Schema ----------------------------------------------------------------

    def _scene_plan_schema(self) -> dict[str, Any]:
        """Load and cache the scene plan JSON schema."""
        if self._scene_schema is None:
            schema_path = self.config.schemas_dir / "scene_plan.schema.json"
            with schema_path.open("r", encoding="utf-8") as f:
                self._scene_schema = json.load(f)
        return self._scene_schema

    # -- Asset resolution --------------------------------------------------------

    def _load_environment(self, environment_id: str) -> dict[str, Any]:
        """Load and cache an environment YAML by ID."""
        if environment_id not in self._environment_cache:
            env_file = self.config.environments_dir / environment_id / f"{environment_id}.yaml"
            if not env_file.exists():
                raise FileNotFoundError(f"Environment file not found: {env_file}")
            with env_file.open("r", encoding="utf-8") as f:
                self._environment_cache[environment_id] = yaml.safe_load(f)
        return self._environment_cache[environment_id]

    def _resolve_environment(self, environment_id: str) -> dict[str, Any]:
        """Resolve an environment ID to its PanelSpec-ready data."""
        env_data = self._load_environment(environment_id)
        return {
            "environment_id": env_data["environment_id"],
            "display_name": env_data["display_name"],
            "description": env_data["description"],
            "prompt_tokens": env_data["prompt_tokens"],
            "references": env_data.get("references", []),
        }

    def _embed_style(self) -> dict[str, Any]:
        """Extract style fields for embedding (frozen snapshot, §16.2)."""
        style = self.config.style
        return {
            "style_id": style["style_id"],
            "visual_style": style["visual_style"],
            "forbidden_elements": style.get("forbidden_elements", []),
            "lighting_defaults": style.get("lighting_defaults", ""),
        }

    # -- Panel seed (deterministic) -----------------------------------------------

    def _generate_panel_seed(self, panel_id: str) -> str:
        """
        Derive a deterministic hex byte (00-FF) from the panel_id.

        Same file + templates => byte-identical PanelSpecs (§16.5 pure-function
        requirement). Used by the Prompt Compiler for negative-space injection
        decisions, same as the chapter path's (random) panel_seed.
        """
        digest = hashlib.sha256(panel_id.encode("utf-8")).hexdigest()
        return f"{int(digest[:2], 16):02X}"

    # -- Geometry: strip flow ------------------------------------------------------

    def _flow_strip(self, strip_id: str) -> tuple[list[_Placement], int]:
        """
        Flow a strip's panels left-to-right from the strip's top-left corner.

        Non-floating panels start at strip y=0, x accumulates width + gutter.
        Floats take explicit coords relative to the strip origin and do not
        consume flow space. Strip height = max non-floating panel height.

        Float bounds: a float must fit within a page frame anchored at the
        strip's origin (position-independent conservative check; scene space
        is not page-bounded until post-production slices pages).

        Returns (placements, strip_height) in strip-local coordinates.

        Raises:
            SceneGeometryError: If the strip overflows the usable width, a
                float escapes the page frame, or the strip has no flow anchor.
        """
        members: list[StripMember] = self.registry.get_strip(strip_id)
        placements: list[_Placement] = []
        x = 0
        flow_height = 0
        anchored = False

        for member in members:
            preset = self.registry.get_panel(member.panel_id)
            if member.floating is not None:
                fx, fy = member.floating
                if fx + preset.width_px > self.usable_w:
                    raise SceneGeometryError(
                        f"strip '{strip_id}' / panel {member.position} "
                        f"(floating): right edge {fx + preset.width_px} "
                        f"exceeds usable width {self.usable_w}"
                    )
                if fy + preset.height_px > self.usable_h:
                    raise SceneGeometryError(
                        f"strip '{strip_id}' / panel {member.position} "
                        f"(floating): bottom edge {fy + preset.height_px} "
                        f"exceeds usable height {self.usable_h}"
                    )
                placements.append(_Placement(
                    x=fx, y=fy,
                    width_px=preset.width_px, height_px=preset.height_px,
                    strip_slot=1, position=member.position, is_float=True,
                ))
            else:
                if x + preset.width_px > self.usable_w:
                    raise SceneGeometryError(
                        f"strip '{strip_id}' / panel {member.position}: "
                        f"right edge {x + preset.width_px} exceeds usable "
                        f"width {self.usable_w}"
                    )
                placements.append(_Placement(
                    x=x, y=0,
                    width_px=preset.width_px, height_px=preset.height_px,
                    strip_slot=1, position=member.position,
                ))
                x += preset.width_px + self.gutter
                flow_height = max(flow_height, preset.height_px)
                anchored = True

        if not anchored:
            raise SceneGeometryError(
                f"strip '{strip_id}': has no non-floating panel — a strip "
                f"needs at least one flow anchor"
            )
        return placements, flow_height

    # -- Geometry: layout flow -------------------------------------------------------

    def _flow_layout(self, layout_id: str) -> list[_Placement]:
        """
        Flow a layout's elements top-down within the page's usable frame.

        Strips and bare panels stack from the page's top-left margin with
        gutters; each occupies one st slot (floats included, per Decision 1:
        a bare panel is a one-panel strip). Floating elements take explicit
        page-margin-relative coords and do not consume flow space.

        Returns placements in page-local (margin-relative) coordinates.

        Raises:
            SceneGeometryError: If the flow stack overflows the usable page,
                or any float escapes it.
        """
        members: list[LayoutMember] = self.registry.get_layout(layout_id)
        placements: list[_Placement] = []
        y = 0

        for member in members:
            if member.kind == "strip":
                strip_placements, strip_h = self._flow_strip(member.ref_id)
                if member.floating is not None:
                    fx, fy = member.floating
                    if fx + self._strip_width(strip_placements) > self.usable_w or fy + strip_h > self.usable_h:
                        raise SceneGeometryError(
                            f"layout '{layout_id}' / strip '{member.ref_id}' "
                            f"(floating at {fx},{fy}): escapes the usable "
                            f"page frame"
                        )
                    for p in strip_placements:
                        placements.append(_Placement(
                            x=fx + p.x, y=fy + p.y,
                            width_px=p.width_px, height_px=p.height_px,
                            strip_slot=member.strip_index,
                            position=p.position, is_float=p.is_float,
                        ))
                else:
                    if y + strip_h > self.usable_h:
                        raise SceneGeometryError(
                            f"layout '{layout_id}': stack overflows usable "
                            f"height — strip '{member.ref_id}' bottom edge "
                            f"{y + strip_h} exceeds {self.usable_h}"
                        )
                    if strip_h > self.usable_h:
                        raise SceneGeometryError(
                            f"layout '{layout_id}': strip '{member.ref_id}' "
                            f"height {strip_h} exceeds usable height "
                            f"{self.usable_h}"
                        )
                    for p in strip_placements:
                        placements.append(_Placement(
                            x=p.x, y=y + p.y,
                            width_px=p.width_px, height_px=p.height_px,
                            strip_slot=member.strip_index,
                            position=p.position, is_float=p.is_float,
                        ))
                    y += strip_h + self.gutter
            else:  # bare panel = one-panel strip (Decision 1)
                preset = self.registry.get_panel(member.ref_id)
                if member.floating is not None:
                    fx, fy = member.floating
                    if fx + preset.width_px > self.usable_w or fy + preset.height_px > self.usable_h:
                        raise SceneGeometryError(
                            f"layout '{layout_id}' / panel '{member.ref_id}' "
                            f"(floating at {fx},{fy}): escapes the usable "
                            f"page frame"
                        )
                    placements.append(_Placement(
                        x=fx, y=fy,
                        width_px=preset.width_px, height_px=preset.height_px,
                        strip_slot=member.strip_index,
                        position=1, is_float=True,
                    ))
                else:
                    if y + preset.height_px > self.usable_h:
                        raise SceneGeometryError(
                            f"layout '{layout_id}': stack overflows usable "
                            f"height — panel '{member.ref_id}' bottom edge "
                            f"{y + preset.height_px} exceeds {self.usable_h}"
                        )
                    placements.append(_Placement(
                        x=0, y=y,
                        width_px=preset.width_px, height_px=preset.height_px,
                        strip_slot=member.strip_index,
                        position=1,
                    ))
                    y += preset.height_px + self.gutter

        return placements

    @staticmethod
    def _strip_width(placements: list[_Placement]) -> int:
        """Right edge of a strip's flow (max x + width)."""
        return max((p.x + p.width_px for p in placements), default=0)

    # -- Deterministic panel-set derivation ------------------------------------------

    def _derive_panel_placements(self, elements: list[dict[str, Any]]) -> tuple[list[tuple[str, _Placement]], int]:
        """
        Derive every (panel_id, placement) from the scene's elements list.

        Scene-space flow: elements stack from the scene's top (y=0) with
        project gutters. A layout element consumes one FULL page of scene
        space (it is a page, by design); loose strips/panels consume their
        own height. Every element must fit the usable page area.

        Returns (placements-with-ids, total_scene_height).
        """
        out: list[tuple[str, _Placement]] = []
        y = 0

        for element_index, element in enumerate(elements, start=1):
            if "layout" in element:
                layout_id = element["layout"]
                for p in self._flow_layout(layout_id):
                    panel_id = (
                        f"{{scene}}_l{element_index:02d}"
                        f"_st{p.strip_slot:02d}_pn{p.position:02d}"
                    )
                    out.append((panel_id, _Placement(
                        x=p.x, y=y + p.y,
                        width_px=p.width_px, height_px=p.height_px,
                        strip_slot=p.strip_slot, position=p.position,
                        is_float=p.is_float,
                    )))
                y += self.usable_h + self.gutter
            elif "strip" in element:
                strip_id = element["strip"]
                placements, strip_h = self._flow_strip(strip_id)
                if strip_h > self.usable_h:
                    raise SceneGeometryError(
                        f"scene element {element_index}: strip '{strip_id}' "
                        f"height {strip_h} exceeds usable height "
                        f"{self.usable_h} — element must fit a page"
                    )
                for p in placements:
                    panel_id = (
                        f"{{scene}}_l{element_index:02d}"
                        f"_st01_pn{p.position:02d}"
                    )
                    out.append((panel_id, _Placement(
                        x=p.x, y=y + p.y,
                        width_px=p.width_px, height_px=p.height_px,
                        strip_slot=1, position=p.position,
                        is_float=p.is_float,
                    )))
                y += strip_h + self.gutter
            elif "panel" in element:
                preset = self.registry.get_panel(element["panel"])
                if preset.height_px > self.usable_h:
                    raise SceneGeometryError(
                        f"scene element {element_index}: panel "
                        f"'{element['panel']}' height {preset.height_px} "
                        f"exceeds usable height {self.usable_h}"
                    )
                panel_id = (
                    f"{{scene}}_l{element_index:02d}_st01_pn01"
                )
                out.append((panel_id, _Placement(
                    x=0, y=y,
                    width_px=preset.width_px, height_px=preset.height_px,
                    strip_slot=1, position=1,
                )))
                y += preset.height_px + self.gutter
            else:
                # Schema should prevent this; guard anyway.
                raise SceneParserError(
                    f"scene element {element_index}: references neither a "
                    f"layout, strip, nor panel"
                )

        # Strip the trailing gutter from the scene height.
        scene_height = max(0, y - self.gutter) if out else 0
        return out, scene_height

    # -- Coverage gate ----------------------------------------------------------------

    def _check_coverage(
        self,
        scene_id: str,
        derived: list[str],
        panels_map: dict[str, Any],
        filename: str,
    ) -> None:
        """
        Stage 4: the panels map must cover the derived set EXACTLY.

        The Producer's keys are checked, never trusted. Missing content
        entries and orphan keys both fail parse, each reported with its
        precise ID.
        """
        resolved = [pid.format(scene=scene_id) for pid in derived]
        expected = set(resolved)
        provided = set(panels_map.keys())

        missing = sorted(expected - provided)
        orphaned = sorted(provided - expected)

        if missing or orphaned:
            errors: list[str] = []
            if missing:
                errors.append(
                    f"missing content entries ({len(missing)}): "
                    + ", ".join(missing)
                )
            if orphaned:
                errors.append(
                    f"orphan keys with no matching panel ({len(orphaned)}): "
                    + ", ".join(orphaned)
                )
            raise SceneCoverageError(
                f"{filename}: exact-coverage gate failed — "
                + "; ".join(errors)
            )

    # -- PanelSpec assembly -------------------------------------------------------------

    def _build_panel_spec(
        self,
        scene: dict[str, Any],
        panel_id: str,
        placement: _Placement,
        content: dict[str, Any],
        style_embedded: dict[str, Any],
        filename: str,
    ) -> dict[str, Any]:
        """Build one self-contained PanelSpec from resolved placement + content."""
        resolved_chars: list[dict[str, Any]] = []
        for char_entry in content.get("characters", []):
            char_id = char_entry["id"]
            costume = char_entry.get("costume")
            try:
                resolved_chars.append(
                    self._wardrobe.resolve_character(char_id, costume)
                )
            except FileNotFoundError:
                raise SceneReferenceError(
                    f"{filename} / {panel_id}: character '{char_id}' not "
                    f"found in project roster"
                ) from None

        env_id = content.get("environment", "")
        resolved_env: dict[str, Any] | None = None
        if env_id:
            try:
                resolved_env = self._resolve_environment(env_id)
            except FileNotFoundError:
                raise SceneReferenceError(
                    f"{filename} / {panel_id}: environment '{env_id}' not "
                    f"found in project roster"
                ) from None

        spec: dict[str, Any] = {
            "panel_id": panel_id,
            "scene_id": scene["scene_id"],
            "chapter_tag": scene.get("chapter_tag"),   # display only
            "title": scene["title"],
            "panel_geometry": {
                "x": placement.x,
                "y": placement.y,
                "width_px": placement.width_px,
                "height_px": placement.height_px,
            },
            "characters": resolved_chars,
            "environment": resolved_env,
            "shot_type": content["shot_type"],
            "mood": content["mood"],
            "description": content["description"],
            "continuity_narrative": scene["narrative"],
            "style": style_embedded,
            "panel_seed": self._generate_panel_seed(panel_id),
            "compiler_version": self.config.compiler_version,
        }
        return spec

    # -- Public API ----------------------------------------------------------------------

    def parse(self, scene_file: str | Path) -> SceneParseResult:
        """
        Parse a Scene Plan YAML file into PanelSpecs.

        Executes all validation stages sequentially, halting on the first
        failure. On success, emits one PanelSpec per panel and persists each
        to output/{panel_id}.panelspec.json.

        Args:
            scene_file: Path to the scene YAML (e.g. scenes/c01_s702.yaml).

        Returns:
            SceneParseResult containing all emitted PanelSpecs.

        Raises:
            SceneSchemaError, SceneReferenceError, SceneGeometryError,
            SceneCoverageError, SceneParserError.
        """
        path = Path(scene_file)
        if not path.is_absolute():
            path = self.config.project_root / path
        if not path.exists():
            raise FileNotFoundError(f"Scene file not found: {path}")
        filename = path.name

        with path.open("r", encoding="utf-8") as f:
            scene = yaml.safe_load(f)

        # Stage 1: Schema validation
        try:
            validate_schema(instance=scene, schema=self._scene_plan_schema())
        except Exception as e:
            raise SceneSchemaError(
                f"{filename}: schema validation failed — {getattr(e, 'message', e)}"
            ) from e

        scene_id = scene["scene_id"]

        # Stage 2+3: Derive panel placements (template resolution happens
        # inside the registry/flow calls; geometry failures raise here).
        try:
            derived, scene_height = self._derive_panel_placements(scene["elements"])
        except TemplateValidationError as e:
            # A dangling template reference in the scene file (or the menu).
            raise SceneReferenceError(f"{filename}: {e}") from e

        # Stage 4: Exact-coverage gate
        self._check_coverage(scene_id, [pid for pid, _ in derived], scene["panels"], filename)

        # Embed style at parse time (frozen snapshot)
        style_embedded = self._embed_style()

        # Build PanelSpecs
        parsed_panels: list[ParsedScenePanel] = []
        warnings: list[str] = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for pid_template, placement in derived:
            panel_id = pid_template.format(scene=scene_id)
            content = scene["panels"][panel_id]
            spec = self._build_panel_spec(
                scene=scene,
                panel_id=panel_id,
                placement=placement,
                content=content,
                style_embedded=style_embedded,
                filename=filename,
            )
            output_path = self.config.output_dir / f"{panel_id}.panelspec.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
            parsed_panels.append(ParsedScenePanel(
                panel_spec=spec,
                output_path=output_path,
            ))

        return SceneParseResult(
            scene_file=path,
            scene_id=scene_id,
            total_panels=len(parsed_panels),
            panels=parsed_panels,
            scene_height_px=scene_height,
            warnings=warnings,
        )

    def parse_scene(self, scene_id: str) -> SceneParseResult:
        """
        Convenience method: parse scenes/{chapter_}_{scene_id}.yaml or
        scenes/{scene_id}.yaml (chapter-tagged filenames are display-only).

        Args:
            scene_id: The scene identifier (e.g. "s702").

        Returns:
            SceneParseResult containing all emitted PanelSpecs.
        """
        scenes_dir = self.config.scenes_dir
        candidates = [
            scenes_dir / f"{scene_id}.yaml",
            *sorted(scenes_dir.glob(f"*_{scene_id}.yaml")),
        ]
        for candidate in candidates:
            if candidate.exists():
                return self.parse(candidate)
        raise FileNotFoundError(
            f"No scene file for '{scene_id}' in {scenes_dir} "
            f"(looked for {scene_id}.yaml or *_{scene_id}.yaml)"
        )
