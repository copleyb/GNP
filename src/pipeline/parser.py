"""
parser.py — Chapter Plan Parser.

Loads a Chapter Plan YAML file, validates it through four sequential stages,
resolves all references against project assets, and emits one PanelSpec
per panel as immutable JSON.

Per DESIGN.md §8: The Parser is the gateway between human-authored plans
and the automated production pipeline. It must fail loudly and precisely —
never pass ambiguous or unresolvable input downstream.

Per DESIGN.md §13.5: Returns structured data, no print() statements, no CLI logic.

Validation stages (sequential, halt on first failure):
  1. Schema validation against chapter_plan.schema.json
  2. Reference resolution (characters, environments, layouts exist in project)
  3. Layout geometry validation (panels within page bounds)
  4. Panel count consistency (plan panels == layout panels)
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validate as validate_schema
from pipeline.wardrobe import Wardrobe


# -- Exceptions ---------------------------------------------------------------

class ParserError(Exception):
    """Base exception for all Parser errors."""
    pass


class SchemaValidationError(ParserError):
    """Stage 1: Chapter Plan fails JSON Schema validation."""
    pass


class ReferenceResolutionError(ParserError):
    """Stage 2: A referenced ID doesn't exist in the project roster."""
    pass


class LayoutGeometryError(ParserError):
    """Stage 3: A panel extends beyond the page bounds."""
    pass


class PanelCountError(ParserError):
    """Stage 4: Panel count in the plan doesn't match the layout."""
    pass


# -- Geometry Validator (standalone, replaceable) ----------------------------

def validate_layout_geometry(
    layout_data: dict[str, Any],
    filename: str = "<layout>",
) -> None:
    """
    Validate that all panels in a layout fit within the page dimensions.

    This is a standalone, replaceable function per DESIGN.md §8 modularity
    requirement. The Parser calls it; it does not depend on Parser internals.
    Future iterations can add bleed checks, gutter consistency, or overlap
    detection by replacing this function without touching the Parser.

    Args:
        layout_data: Parsed layout YAML (must have 'page' and 'panels' keys).
        filename: Layout filename for error messages.

    Raises:
        LayoutGeometryError: If any panel extends beyond page bounds.
    """
    page = layout_data["page"]
    page_w = page["width_px"]
    page_h = page["height_px"]

    for panel in layout_data["panels"]:
        pos = panel["position"]
        right = panel["x"] + panel["width_px"]
        bottom = panel["y"] + panel["height_px"]

        if panel["x"] < 0 or panel["y"] < 0:
            raise LayoutGeometryError(
                f"{filename} / panel {pos}: negative origin "
                f"(x={panel['x']}, y={panel['y']})"
            )
        if right > page_w:
            raise LayoutGeometryError(
                f"{filename} / panel {pos}: right edge {right} exceeds "
                f"page width {page_w}"
            )
        if bottom > page_h:
            raise LayoutGeometryError(
                f"{filename} / panel {pos}: bottom edge {bottom} exceeds "
                f"page height {page_h}"
            )


# -- Result types ------------------------------------------------------------

@dataclass(frozen=True)
class ParsedPanel:
    """
    A single parsed panel result — one PanelSpec and its output path.
    """
    panel_spec: dict[str, Any]
    output_path: Path


@dataclass(frozen=True)
class ParseResult:
    """
    Result of parsing a full Chapter Plan.

    Contains all emitted PanelSpecs and metadata about the parse.
    """
    chapter_file: Path
    chapter_id: int
    total_panels: int
    panels: list[ParsedPanel]
    warnings: list[str] = field(default_factory=list)


# -- Parser ------------------------------------------------------------------

class ChapterPlanParser:
    """
    Parses Chapter Plan YAML into PanelSpec JSON objects.

    The Parser is project-aware: it loads and resolves characters, environments,
    layouts, and style from the project configuration. It produces self-contained
    PanelSpecs that require no further file I/O for the Prompt Compiler.
    """

    def __init__(self, config: Any):
        """
        Initialise the Parser with a project configuration.

        Args:
            config: A ProjectConfig object (from pipeline.config.load_config).
        """
        self.config = config
        self._chapter_schema: dict[str, Any] | None = None
        self._wardrobe = Wardrobe(config.characters_dir)
        # Share the wardrobe's character cache so both Parser and Wardrobe
        # see the same loaded character data
        self._character_cache = self._wardrobe._character_cache
        self._environment_cache: dict[str, dict[str, Any]] = {}
        self._layout_cache: dict[str, dict[str, Any]] = {}

    # -- Asset loading (cached) ----------------------------------------------

    def _chapter_plan_schema(self) -> dict[str, Any]:
        """Load and cache the chapter plan JSON schema."""
        if self._chapter_schema is None:
            schema_path = self.config.schemas_dir / "chapter_plan.schema.json"
            with schema_path.open("r", encoding="utf-8") as f:
                self._chapter_schema = json.load(f)
        return self._chapter_schema

    def _load_character(self, character_id: str) -> dict[str, Any]:
        """Load and cache a character YAML by ID (delegates to Wardrobe)."""
        return self._wardrobe.load_character(character_id)

    def _load_environment(self, environment_id: str) -> dict[str, Any]:
        """Load and cache an environment YAML by ID."""
        if environment_id not in self._environment_cache:
            env_file = self.config.environments_dir / environment_id / f"{environment_id}.yaml"
            if not env_file.exists():
                raise FileNotFoundError(f"Environment file not found: {env_file}")
            with env_file.open("r", encoding="utf-8") as f:
                self._environment_cache[environment_id] = yaml.safe_load(f)
        return self._environment_cache[environment_id]

    def _load_layout(self, layout_id: str) -> dict[str, Any]:
        """Load and cache a layout YAML by ID."""
        if layout_id not in self._layout_cache:
            layout_file = self.config.layouts_dir / f"{layout_id}.yaml"
            if not layout_file.exists():
                raise FileNotFoundError(f"Layout file not found: {layout_file}")
            with layout_file.open("r", encoding="utf-8") as f:
                self._layout_cache[layout_id] = yaml.safe_load(f)
        return self._layout_cache[layout_id]

    # -- Style embedding ------------------------------------------------------

    def _embed_style(self) -> dict[str, Any]:
        """
        Extract style fields for embedding in PanelSpecs.

        Style is embedded at parse time so in-flight generations are not
        affected by later changes to style.yaml (DESIGN.md §8).
        """
        style = self.config.style
        return {
            "style_id": style["style_id"],
            "visual_style": style["visual_style"],
            "forbidden_elements": style.get("forbidden_elements", []),
            "lighting_defaults": style.get("lighting_defaults", ""),
        }

    # -- Character resolution ------------------------------------------------

    def _resolve_character(self, character_id: str, costume_variant: str | None = None) -> dict[str, Any]:
        """Resolve a character ID to PanelSpec-ready data (delegates to Wardrobe)."""
        return self._wardrobe.resolve_character(character_id, costume_variant)

    # -- Environment resolution ----------------------------------------------

    def _resolve_environment(self, environment_id: str) -> dict[str, Any]:
        """
        Resolve an environment ID to its full data for PanelSpec embedding.

        Includes description, prompt_tokens, and all references.
        """
        env_data = self._load_environment(environment_id)

        return {
            "environment_id": env_data["environment_id"],
            "display_name": env_data["display_name"],
            "description": env_data["description"],
            "prompt_tokens": env_data["prompt_tokens"],
            "references": env_data.get("references", []),
        }

    # -- Panel ID generation --------------------------------------------------

    def _make_panel_id(
        self,
        chapter_id: int,
        page_id: str,
        layout_id: str,
        position: int,
    ) -> str:
        """
        Generate the canonical panel ID.

        Format: c{chapter}_pg{page}_l{layout_id}_pn{position}
        Example: c01_pg01_l02_pn03
        """
        # page_id is "1_1" — extract the page-within-chapter number
        page_num = page_id.split("_")[-1]
        return f"c{chapter_id:02d}_pg{page_num}_l{layout_id.replace('layout_', '')}_pn{position:02d}"

    # -- Panel seed generation ------------------------------------------------

    def _generate_panel_seed(self) -> str:
        """
        Generate a random hex byte (00–FF) for the panel_seed field.

        Used by the Prompt Compiler for deterministic negative-space
        injection decisions. Persisted in the PanelSpec so it's stable
        across re-compilations of the same PanelSpec.
        """
        return f"{random.randint(0, 255):02X}"

    # -- Validation stages ----------------------------------------------------

    def _validate_schema(self, chapter_plan: dict[str, Any], filename: str) -> None:
        """Stage 1: Validate the Chapter Plan against its JSON Schema."""
        try:
            validate_schema(
                instance=chapter_plan,
                schema=self._chapter_plan_schema(),
            )
        except Exception as e:
            raise SchemaValidationError(
                f"{filename}: schema validation failed — {e.message}"
            ) from e

    def _resolve_references(
        self,
        chapter_plan: dict[str, Any],
        filename: str,
    ) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
        """
        Stage 2: Resolve all character, environment, and layout references.

        Returns a tuple of (characters, environments, layouts) dicts keyed
        by ID, with fully loaded data.

        Raises ReferenceResolutionError if any ID is not found.
        """
        errors: list[str] = []
        characters: dict[str, dict] = {}
        environments: dict[str, dict] = {}
        layouts: dict[str, dict] = {}

        # Collect all unique character IDs across all panels
        all_char_ids: set[str] = set()
        all_env_ids: set[str] = set()
        all_layout_ids: set[str] = set()

        for page in chapter_plan.get("pages", []):
            page_id = page.get("page_id", "?")
            layout_id = page.get("layout", "")
            all_layout_ids.add(layout_id)

            for panel in page.get("panels", []):
                pos = panel.get("position", "?")
                for char_entry in panel.get("characters", []):
                    if isinstance(char_entry, dict):
                        char_id = char_entry["character_id"]
                    elif isinstance(char_entry, str):
                        char_id = char_entry
                    else:
                        continue
                    all_char_ids.add(char_id)
                env_id = panel.get("environment", "")
                if env_id:
                    all_env_ids.add(env_id)

        # Resolve characters
        for char_id in sorted(all_char_ids):
            try:
                characters[char_id] = self._load_character(char_id)
            except FileNotFoundError:
                errors.append(
                    f"{filename} / character '{char_id}' not found in project roster"
                )

        # Resolve environments
        for env_id in sorted(all_env_ids):
            try:
                environments[env_id] = self._load_environment(env_id)
            except FileNotFoundError:
                errors.append(
                    f"{filename} / environment '{env_id}' not found in project roster"
                )

        # Resolve layouts
        for layout_id in sorted(all_layout_ids):
            try:
                layouts[layout_id] = self._load_layout(layout_id)
            except FileNotFoundError:
                errors.append(
                    f"{filename} / layout '{layout_id}' not found in project roster"
                )

        if errors:
            raise ReferenceResolutionError(
                "Reference resolution failed:\n  " + "\n  ".join(errors)
            )

        return characters, environments, layouts

    def _validate_all_geometry(self, layouts: dict[str, dict]) -> list[str]:
        """
        Stage 3: Validate geometry for all loaded layouts.

        Returns a list of layout filenames that passed (for informational use).
        Raises LayoutGeometryError on first failure.
        """
        passed: list[str] = []
        for layout_id, layout_data in layouts.items():
            filename = f"{layout_id}.yaml"
            validate_layout_geometry(layout_data, filename)
            passed.append(filename)
        return passed

    def _validate_panel_counts(
        self,
        chapter_plan: dict[str, Any],
        layouts: dict[str, dict],
        filename: str,
    ) -> None:
        """
        Stage 4: Verify panel count in each page matches its layout.

        Raises PanelCountError on first mismatch.
        """
        errors: list[str] = []
        for page in chapter_plan.get("pages", []):
            page_id = page.get("page_id", "?")
            layout_id = page.get("layout", "")
            actual_count = len(page.get("panels", []))

            if layout_id not in layouts:
                # Already caught in Stage 2, skip here
                continue

            expected_count = len(layouts[layout_id]["panels"])
            if actual_count != expected_count:
                errors.append(
                    f"{filename} / page {page_id} (layout {layout_id}): "
                    f"expected {expected_count} panels, got {actual_count}"
                )

        if errors:
            raise PanelCountError(
                "Panel count mismatch:\n  " + "\n  ".join(errors)
            )

    # -- PanelSpec assembly ---------------------------------------------------

    def _build_panel_spec(
        self,
        panel: dict[str, Any],
        page: dict[str, Any],
        chapter_id: int,
        layout_data: dict[str, Any],
        style_embedded: dict[str, Any],
        resolved_characters: dict[str, dict],
        resolved_environments: dict[str, dict],
        scenes_by_id: dict[str, dict[str, Any]],
        filename: str,
    ) -> dict[str, Any]:
        """
        Build a single PanelSpec dict from a resolved panel.

        The PanelSpec is fully self-contained — the Prompt Compiler needs
        no additional file I/O to process it.
        """
        page_id = page["page_id"]
        layout_id = page["layout"]
        position = panel["position"]

        panel_id = self._make_panel_id(chapter_id, page_id, layout_id, position)

        # Find the panel geometry from the layout by position
        layout_panels = layout_data["panels"]
        geometry: dict[str, Any] | None = None
        for lp in layout_panels:
            if lp["position"] == position:
                geometry = {
                    "x": lp["x"],
                    "y": lp["y"],
                    "width_px": lp["width_px"],
                    "height_px": lp["height_px"],
                }
                break

        if geometry is None:
            # This shouldn't happen after Stage 4, but guard anyway
            raise ParserError(
                f"{filename} / page {page_id} / panel {position}: "
                f"no matching position in layout {layout_id}"
            )

        # Resolve characters (now objects with character_id + optional costume)
        resolved_chars: list[dict[str, Any]] = []
        for char_entry in panel.get("characters", []):
            # Support both new object format and legacy string format
            if isinstance(char_entry, dict):
                char_id = char_entry["character_id"]
                costume = char_entry.get("costume")
            elif isinstance(char_entry, str):
                char_id = char_entry
                costume = None
            else:
                continue

            if char_id in resolved_characters:
                resolved_chars.append(
                    self._resolve_character(char_id, costume)
                )

        # Resolve environment
        env_id = panel.get("environment", "")
        resolved_env: dict[str, Any] | None = None
        if env_id and env_id in resolved_environments:
            resolved_env = self._resolve_environment(env_id)

        # Resolve scene_id and continuity narrative
        scene_id = panel.get("scene_id")
        continuity_narrative: str | None = None

        if scene_id and scene_id in scenes_by_id:
            # Look up this panel's narrative entry in the scene
            scene = scenes_by_id[scene_id]
            for entry in scene.get("panels", []):
                if entry.get("page") == page_id and entry.get("position") == position:
                    continuity_narrative = entry.get("narrative")
                    break

        return {
            "panel_id": panel_id,
            "chapter_id": chapter_id,
            "page_id": page_id,
            "layout_id": layout_id,
            "position": position,
            "panel_geometry": geometry,
            "characters": resolved_chars,
            "environment": resolved_env,
            "shot_type": panel["shot_type"],
            "mood": panel["mood"],
            "description": panel["description"],
            "continuity": page.get("continuity", {}),
            "scene_id": scene_id,
            "continuity_narrative": continuity_narrative,
            "style": style_embedded,
            "panel_seed": self._generate_panel_seed(),
            "compiler_version": self.config.compiler_version,
        }

    # -- Public API -----------------------------------------------------------

    def parse(self, chapter_file: str | Path) -> ParseResult:
        """
        Parse a Chapter Plan YAML file into PanelSpecs.

        Executes all four validation stages sequentially, halting on the
        first failure. On success, emits one PanelSpec per panel and
        persists each to disk.

        Args:
            chapter_file: Path to the chapter_N.yaml file. Can be relative
                          to the project root or absolute.

        Returns:
            ParseResult containing all emitted PanelSpecs.

        Raises:
            SchemaValidationError: Stage 1 failure.
            ReferenceResolutionError: Stage 2 failure.
            LayoutGeometryError: Stage 3 failure.
            PanelCountError: Stage 4 failure.
            ParserError: Other parsing errors.
        """
        path = Path(chapter_file)
        if not path.is_absolute():
            path = self.config.project_root / path

        if not path.exists():
            raise FileNotFoundError(f"Chapter file not found: {path}")

        filename = path.name

        # Load chapter plan YAML
        with path.open("r", encoding="utf-8") as f:
            chapter_plan = yaml.safe_load(f)

        # Stage 1: Schema validation
        self._validate_schema(chapter_plan, filename)

        chapter_id = chapter_plan["chapter_id"]

        # Stage 2: Reference resolution
        raw_chars, raw_envs, layouts = self._resolve_references(
            chapter_plan, filename
        )

        # Stage 3: Layout geometry validation
        self._validate_all_geometry(layouts)

        # Stage 4: Panel count consistency
        self._validate_panel_counts(chapter_plan, layouts, filename)

        # Embed style at parse time
        style_embedded = self._embed_style()

        # Index scenes by scene_id for quick lookup
        scenes_by_id: dict[str, dict[str, Any]] = {}
        for scene in chapter_plan.get("scenes", []):
            scenes_by_id[scene["scene_id"]] = scene

        # Build PanelSpecs
        parsed_panels: list[ParsedPanel] = []
        warnings: list[str] = []

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for page in chapter_plan["pages"]:
            layout_id = page["layout"]
            layout_data = layouts[layout_id]

            for panel in page["panels"]:
                spec = self._build_panel_spec(
                    panel=panel,
                    page=page,
                    chapter_id=chapter_id,
                    layout_data=layout_data,
                    style_embedded=style_embedded,
                    resolved_characters=raw_chars,
                    resolved_environments=raw_envs,
                    scenes_by_id=scenes_by_id,
                    filename=filename,
                )

                output_path = self.config.output_dir / f"{spec['panel_id']}.panelspec.json"

                # Persist PanelSpec to disk
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(spec, f, indent=2, ensure_ascii=False)

                parsed_panels.append(ParsedPanel(
                    panel_spec=spec,
                    output_path=output_path,
                ))

        return ParseResult(
            chapter_file=path,
            chapter_id=chapter_id,
            total_panels=len(parsed_panels),
            panels=parsed_panels,
            warnings=warnings,
        )

    def parse_chapter(self, chapter_number: int) -> ParseResult:
        """
        Convenience method: parse chapters/chapter_{N}.yaml.

        Args:
            chapter_number: The chapter number (e.g. 1).

        Returns:
            ParseResult containing all emitted PanelSpecs.
        """
        chapter_file = self.config.chapters_dir / f"chapter_{chapter_number}.yaml"
        return self.parse(chapter_file)
