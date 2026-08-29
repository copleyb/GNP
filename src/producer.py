"""
producer.py — Chapter Plan Producer.

Accepts a human-provided narrative synopsis and chapter number, calls GPT-4o
with full project context using OpenAI structured output mode, and writes a
conforming chapter_N.yaml to the chapters/ directory.

Per DESIGN.md §7: The Producer is external to the pipeline. It generates the
upstream input (Chapter Plans) that the pipeline consumes.

Per DESIGN.md §13.5: Returns structured data, no print() statements, no CLI logic.

Context management infrastructure (v1 foundation, per DESIGN.md §7):
  - context_store: placeholder for accumulated context (empty in v1)
  - curate_context(): placeholder for context selection (pass-through in v1)
  - update_context(): placeholder for post-generation context storage (no-op in v1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validate as validate_schema


class ChapterPlanProducer:
    """
    Generates Chapter Plans from narrative synopses using GPT-4o.

    The Producer is project-aware: it injects character IDs, environment IDs,
    layout IDs, and style constraints into the LLM call so the generated plan
    references only assets that exist in the project.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        config: Any,
        model: str = "gpt-4o",
        schema_path: str | Path | None = None,
    ):
        """
        Initialise the Producer.

        Args:
            config: A ProjectConfig object (from pipeline.config.load_config).
            model: The LLM model to use (default: gpt-4o).
            schema_path: Path to chapter_plan.schema.json. Defaults to
                        {config.schemas_dir}/chapter_plan.schema.json.
        """
        self.config = config
        self.model = model

        if schema_path is None:
            self.schema_path = config.schemas_dir / "chapter_plan.schema.json"
        else:
            self.schema_path = Path(schema_path)

        # Load the full validation schema
        with self.schema_path.open("r", encoding="utf-8") as f:
            self.validation_schema = json.load(f)

        # Context management infrastructure (v1 placeholders)
        self.context_store: dict[str, Any] = {}
        self.scene_context_store: dict[str, Any] = {}

    # -- Context management placeholders (v1 foundation) --------------------

    def curate_context(self) -> dict[str, Any]:
        """
        Select and format relevant context from context_store for the LLM call.

        v1: pass-through — returns the current project config context unchanged.
        Future versions may add cross-chapter continuity, narrative memory,
        or curated context windows.
        """
        return self.context_store

    def update_context(self, chapter_plan: dict[str, Any]) -> None:
        """
        Store new context worth retaining after a successful generation.

        v1: no-op. Future versions may store chapter summaries, established
        narrative facts, or recurring motifs for continuity with subsequent
        chapters.
        """
        pass

    # -- Context assembly ---------------------------------------------------

    def _load_character_context(self) -> list[dict[str, Any]]:
        """Load all character YAML files and extract context for the LLM."""
        characters = []
        char_dir = self.config.characters_dir

        for char_file in sorted(char_dir.glob("*/*.yaml")):
            with char_file.open("r", encoding="utf-8") as f:
                char_data = yaml.safe_load(f)

            characters.append({
                "character_id": char_data["character_id"],
                "display_name": char_data["display_name"],
                "physical_description": char_data["physical_description"],
                "costume_default": char_data["costumes"]["default"]["description"],
                "costume_variants": [
                    {"variant_id": v["variant_id"], "description": v["description"]}
                    for v in char_data.get("costumes", {}).get("variants", [])
                ],
                "prompt_tokens_identity": char_data["prompt_tokens"]["identity"],
                "exclusions": char_data.get("prompt_tokens", {}).get("exclusions", []),
            })

        return characters

    def _load_environment_context(self) -> list[dict[str, Any]]:
        """Load all environment YAML files and extract context for the LLM."""
        environments = []
        env_dir = self.config.environments_dir

        for env_file in sorted(env_dir.glob("*/*.yaml")):
            with env_file.open("r", encoding="utf-8") as f:
                env_data = yaml.safe_load(f)

            environments.append({
                "environment_id": env_data["environment_id"],
                "display_name": env_data["display_name"],
                "description": env_data["description"],
                "exclusions": env_data.get("prompt_tokens", {}).get("exclusions", []),
            })

        return environments

    def _load_layout_context(self) -> list[dict[str, Any]]:
        """Load all layout YAML files and extract context for the LLM."""
        layouts = []
        layout_dir = self.config.layouts_dir

        for layout_file in sorted(layout_dir.glob("*.yaml")):
            with layout_file.open("r", encoding="utf-8") as f:
                layout_data = yaml.safe_load(f)

            layouts.append({
                "layout_id": layout_data["layout_id"],
                "display_name": layout_data["display_name"],
                "panel_count": len(layout_data["panels"]),
            })

        return layouts

    def _load_style_context(self) -> dict[str, Any]:
        """Load style.yaml and extract context for the LLM."""
        return {
            "style_id": self.config.style["style_id"],
            "visual_style": self.config.style["visual_style"],
            "forbidden_elements": self.config.style.get("forbidden_elements", []),
        }

    def assemble_context(self) -> dict[str, Any]:
        """
        Assemble the full project context for the LLM call.

        Returns a dict with keys: characters, environments, layouts, style.
        This is the data injected into the user message alongside the synopsis.
        """
        return {
            "characters": self._load_character_context(),
            "environments": self._load_environment_context(),
            "layouts": self._load_layout_context(),
            "style": self._load_style_context(),
        }

    # -- Prompt construction -------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system message that defines the LLM's role and constraints."""
        return """You are a graphic novel chapter planner. Your job is to translate a narrative synopsis into a structured Chapter Plan for a graphic novel.

You will receive:
- A narrative synopsis from the author
- A roster of available characters (with IDs, descriptions, and costumes)
- A roster of available environments (with IDs, descriptions, and exclusions)
- A list of available page layouts (with IDs and panel counts)
- The project's visual style constraints

Your output must be a JSON object conforming to the Chapter Plan schema.

CRITICAL RULES:
1. Only use character_ids that exist in the provided roster. Never invent characters.
2. Characters are objects with a "character_id" field (required) and an optional "costume" field. The "costume" field references a variant_id from the character's costume variants. When a character is in their default outfit, omit the "costume" field. Use the "costume" field when a character is wearing a non-default outfit (e.g. morning_routine, sleepwear, formal).
2. Only use environment_ids that exist in the provided roster. Never invent environments.
3. Only use layout_ids that exist in the provided roster. Never invent layouts.
4. THE NUMBER OF PANELS IN EACH PAGE'S "panels" ARRAY MUST EXACTLY EQUAL THE PANEL COUNT OF THE LAYOUT ASSIGNED TO THAT PAGE. This is the most important constraint. If a layout has 3 panels, the page using that layout MUST have exactly 3 entries in its panels array — not 2, not 4. Count the panels before finalising each page.
5. Respect environment exclusions — if an environment forbids daylight, do not set time_of_day to a daytime value.
6. Respect style forbidden elements — do not describe scenes that violate the style constraints.
7. Each panel's description should be concise prose (1-3 sentences) describing what is visually depicted. This is the primary creative input to the image generation pipeline.
8. shot_type must be one of: wide, medium, close_up, extreme_close_up, overhead, low_angle, dutch_angle.
9. page_id format: {chapter_id}_{page_within_chapter} (e.g. "1_1", "1_2").
10. Panel positions are sequential integers starting at 1 within each page.
11. The mood field should be a short phrase capturing the emotional register (e.g. "tense", "contemplative", "urgent").
12. The continuity block at the page level captures time_of_day and location that apply across the page.

PANEL COUNT REFERENCE — memorise this before writing any pages:
- layout_01: 2 panels
- layout_02: 3 panels
- layout_03: 7 panels

Your output will be validated against a strict schema. If any constraint is violated, the generation will fail."""

    def _build_user_prompt(self, synopsis: str, chapter_number: int) -> str:
        """Build the user message with the synopsis and injected project context."""
        context = self.assemble_context()

        # Format characters
        char_lines = []
        for c in context["characters"]:
            char_lines.append(
                f"  - {c['character_id']} ({c['display_name']}): {c['physical_description']['build']}, "
                f"{c['physical_description']['hair']} hair, {c['physical_description']['eyes']} eyes. "
                f"Default costume: {c['costume_default']}"
            )
            if c.get("costume_variants"):
                for v in c["costume_variants"]:
                    char_lines.append(f"    Variant '{v['variant_id']}': {v['description']}")
            if c.get("exclusions"):
                char_lines.append(f"    Exclusions: {', '.join(c['exclusions'])}")

        # Format environments
        env_lines = []
        for e in context["environments"]:
            env_lines.append(
                f"  - {e['environment_id']} ({e['display_name']}): {e['description']}"
            )
            if e.get("exclusions"):
                env_lines.append(f"    Exclusions: {', '.join(e['exclusions'])}")

        # Format layouts — explicit panel count emphasis
        layout_lines = []
        for l in context["layouts"]:
            layout_lines.append(
                f"  - {l['layout_id']} ({l['display_name']}): EXACTLY {l['panel_count']} panels — "
                f"pages using this layout MUST have {l['panel_count']} entries in their panels array"
            )

        # Format style
        s = context["style"]
        style_lines = [
            f"  Style: {s['visual_style']['label']} — {s['visual_style']['description'].strip()}",
            f"  Forbidden: {', '.join(s['forbidden_elements'])}",
        ]

        return f"""Chapter {chapter_number} Synopsis:
{synopsis}

Available Characters:
{chr(10).join(char_lines)}

Available Environments:
{chr(10).join(env_lines)}

Available Layouts (each layout's panel count is a HARD constraint):
{chr(10).join(layout_lines)}

Visual Style:
{chr(10).join(style_lines)}

Generate a Chapter Plan for Chapter {chapter_number}. Use the page_id format "{chapter_number}_N" where N is the page number within this chapter. Choose appropriate layouts for the narrative pacing — don't use the same layout for every page unless the story demands it.

REMINDER: Before finalising each page, count the entries in its panels array and verify it EXACTLY matches the panel count of the layout you assigned to that page."""

    # -- Schema conversion for OpenAI structured output ---------------------

    def _to_openai_schema(self) -> dict[str, Any]:
        """
        Convert chapter_plan.schema.json to an OpenAI-compatible schema.

        OpenAI's structured output mode (strict: true) requires:
        - All properties listed in required
        - additionalProperties: false at every object level
        - Optional fields represented as nullable types

        The full validation schema is used for post-call validation.
        """
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["chapter_id", "title", "notes", "pages"],
            "properties": {
                "chapter_id": {"type": "integer"},
                "title": {"type": "string"},
                "notes": {"type": ["string", "null"]},
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["page_id", "layout", "continuity", "panels"],
                        "properties": {
                            "page_id": {"type": "string"},
                            "layout": {"type": "string"},
                            "continuity": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["time_of_day", "location"],
                                "properties": {
                                    "time_of_day": {"type": "string"},
                                    "location": {"type": "string"},
                                },
                            },
                            "panels": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "position",
                                        "characters",
                                        "environment",
                                        "shot_type",
                                        "mood",
                                        "description",
                                    ],
                                    "properties": {
                                        "position": {"type": "integer"},
                                        "characters": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "required": ["character_id"],
                                                "properties": {
                                                    "character_id": {"type": "string"},
                                                    "costume": {"type": "string"},
                                                },
                                            },
                                        },
                                        "environment": {"type": "string"},
                                        "shot_type": {
                                            "type": "string",
                                            "enum": [
                                                "wide", "medium", "close_up",
                                                "extreme_close_up", "overhead",
                                                "low_angle", "dutch_angle",
                                            ],
                                        },
                                        "mood": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    # -- LLM call (mockable) -------------------------------------------------

    def call_llm(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Call the OpenAI API with structured output mode.

        Returns the parsed JSON response as a dict.

        Raises:
            RuntimeError: If the API call fails or the response is malformed.
            ImportError: If the openai package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required. Install it with: pip install openai"
            ) from e

        client = OpenAI()

        openai_schema = self._to_openai_schema()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chapter_plan",
                    "schema": openai_schema,
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("API returned empty response content")

        return json.loads(content)

    # -- Output --------------------------------------------------------------

    def _chapter_plan_path(self, chapter_number: int) -> Path:
        """Return the path for the chapter plan file."""
        return self.config.chapters_dir / f"chapter_{chapter_number}.yaml"

    def write_chapter_plan(self, chapter_plan: dict[str, Any], chapter_number: int) -> Path:
        """
        Validate and write the chapter plan as YAML to the chapters/ directory.

        Args:
            chapter_plan: The Chapter Plan dict from the LLM.
            chapter_number: The chapter number (used for the filename).

        Returns:
            The Path to the written file.

        Raises:
            jsonschema.ValidationError: If the chapter plan doesn't conform
                                         to the full validation schema.
        """
        # Validate against the full schema (includes additionalProperties: true
        # on continuity, optional notes, etc. — stricter than the OpenAI schema)
        validate_schema(instance=chapter_plan, schema=self.validation_schema)

        # Ensure chapter_id matches the requested chapter number
        if chapter_plan["chapter_id"] != chapter_number:
            chapter_plan["chapter_id"] = chapter_number

        path = self._chapter_plan_path(chapter_number)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(chapter_plan, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return path

    # -- Main orchestration --------------------------------------------------

    def produce(self, synopsis: str, chapter_number: int) -> dict[str, Any]:
        """
        Generate a Chapter Plan from a narrative synopsis.

        Retries up to MAX_RETRIES times if the LLM produces a panel count
        mismatch (a known weakness of structured output mode, which cannot
        enforce cross-field constraints like "panels array length must match
        layout panel count" in the JSON schema itself).

        Args:
            synopsis: The narrative synopsis for this chapter (free text).
            chapter_number: The chapter number (used for the filename and chapter_id).

        Returns:
            A result dict with keys:
            - "chapter_plan": The validated Chapter Plan dict.
            - "file_path": Path to the written YAML file.
            - "model": The model used for the call.
            - "attempts": Number of LLM calls made (1 = first try succeeded).

        Raises:
            RuntimeError: If the LLM call fails after all retries.
            jsonschema.ValidationError: If the output doesn't conform to the schema.
            ValueError: If the output references unknown IDs after all retries.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(synopsis, chapter_number)

        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            # Call the LLM (mockable for testing)
            chapter_plan = self.call_llm(system_prompt, user_prompt)

            # Ensure chapter_id matches
            chapter_plan["chapter_id"] = chapter_number

            # Normalize null notes to empty string
            if chapter_plan.get("notes") is None:
                chapter_plan["notes"] = ""

            # Validate against the full schema
            validate_schema(instance=chapter_plan, schema=self.validation_schema)

            # Validate that referenced IDs exist and panel counts match
            try:
                self._validate_referenced_ids(chapter_plan)
            except ValueError as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    # Retry — LLM may produce correct panel counts on a second call
                    continue
                else:
                    raise

            # Write the file
            file_path = self.write_chapter_plan(chapter_plan, chapter_number)

            # Update context store (v1: no-op)
            self.update_context(chapter_plan)

            return {
                "chapter_plan": chapter_plan,
                "file_path": file_path,
                "model": self.model,
                "attempts": attempt,
            }

        # Should not reach here, but just in case
        raise RuntimeError(
            f"Producer failed after {self.MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def _validate_referenced_ids(self, chapter_plan: dict[str, Any]) -> None:
        """Validate that all character, environment, and layout IDs in the plan exist in the project."""
        context = self.assemble_context()

        valid_char_ids = {c["character_id"] for c in context["characters"]}
        valid_env_ids = {e["environment_id"] for e in context["environments"]}
        valid_layout_ids = {l["layout_id"] for l in context["layouts"]}

        errors = []

        for page in chapter_plan.get("pages", []):
            layout = page.get("layout", "")
            if layout not in valid_layout_ids:
                errors.append(f"Page {page.get('page_id', '?')}: unknown layout '{layout}'")

            for panel in page.get("panels", []):
                env = panel.get("environment", "")
                if env not in valid_env_ids:
                    errors.append(
                        f"Page {page.get('page_id', '?')}, panel {panel.get('position', '?')}: "
                        f"unknown environment '{env}'"
                    )

                for char_entry in panel.get("characters", []):
                    # Support both object and legacy string format
                    if isinstance(char_entry, dict):
                        char_id = char_entry["character_id"]
                        costume = char_entry.get("costume")
                    elif isinstance(char_entry, str):
                        char_id = char_entry
                        costume = None
                    else:
                        errors.append(
                            f"Page {page.get('page_id', '?')}, panel {panel.get('position', '?')}: "
                            f"invalid character entry (expected object or string)"
                        )
                        continue

                    if char_id not in valid_char_ids:
                        errors.append(
                            f"Page {page.get('page_id', '?')}, panel {panel.get('position', '?')}: "
                            f"unknown character '{char_id}'"
                        )
                    # Validate costume variant if specified
                    if costume:
                        char_ctx = next(
                            (c for c in context["characters"] if c["character_id"] == char_id),
                            None
                        )
                        if char_ctx:
                            valid_variants = {
                                v["variant_id"] for v in char_ctx.get("costume_variants", [])
                            }
                            if costume not in valid_variants:
                                errors.append(
                                    f"Page {page.get('page_id', '?')}, panel {panel.get('position', '?')}: "
                                    f"unknown costume variant '{costume}' for character '{char_id}'"
                                )

        if errors:
            raise ValueError(
                "Chapter Plan references unknown IDs:\n  " + "\n  ".join(errors)
            )

        # Validate panel count matches layout
        layout_panel_counts = {l["layout_id"]: l["panel_count"] for l in context["layouts"]}
        for page in chapter_plan.get("pages", []):
            layout = page.get("layout", "")
            expected_count = layout_panel_counts.get(layout, 0)
            actual_count = len(page.get("panels", []))
            if actual_count != expected_count:
                errors.append(
                    f"Page {page.get('page_id', '?')}: panel count {actual_count} "
                    f"does not match layout '{layout}' which expects {expected_count}"
                )

        if errors:
            raise ValueError(
                "Chapter Plan panel count mismatch:\n  " + "\n  ".join(errors)
            )
