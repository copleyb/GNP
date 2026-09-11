"""
scene_producer.py — Scene Producer (Scope Redesign, DESIGN.md §16.11).

Accepts a human-authored scene INPUT file (scene_id, chapter_tag, title,
synopsis, elements) and generates a complete scene file for the Scene Parser
via a two-stage LLM process:

  Stage 1 — Chunker (one call, whole synopsis): emits {narrative, chunks[N]}.
            The narrative is authored HERE — the only call that sees the
            whole synopsis. Chunk size should track per-element panel count.

  Stage 2 — Panel content (one call per element, SEQUENTIAL in element
            order): emits an ordered ARRAY of per-panel content. The LLM
            never sees a panel key — producer code maps array position to
            the derived positional ID. Bounded per-element retry on array
            length (the only compliance obligation). A code-assembled
            WHERE-WE-LEFT-OFF handoff carries fine state from the previous
            element; the stage-1 narrative is the shared backbone every call.

Assembly + the parse gate: the producer assembles the scene dict in memory,
validates it against scene_plan.schema.json, writes it to a temp file, and
runs ScenePlanParser.parse on it — the full Phase 1 gauntlet — BEFORE
atomically committing to scenes/. Parse's side effect emits PanelSpecs to
output/ in the same run. Any failure is loud and leaves no half-scene on disk.

Per DESIGN.md §7/§16: The Producer is external to the pipeline and
communicates solely through the filesystem.
Per DESIGN.md §13.5: Returns structured data, no print() statements, no CLI logic.

The LLM client is injectable (llm_client=...) so all tests run offline.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml
from jsonschema import validate as validate_schema

from pipeline.scene_parser import ScenePlanParser, SceneParseResult
from pipeline.templates import TemplateRegistry


# -- Exceptions ---------------------------------------------------------------

class SceneProducerError(Exception):
    """Base exception for all Scene Producer errors."""
    pass


class SceneInputError(SceneProducerError):
    """The input file is missing, malformed, or references unknown templates."""
    pass


class SceneProducerLLMError(SceneProducerError):
    """An LLM call failed or produced non-compliant output after all retries."""
    pass


# -- OpenAI structured-output schemas (strict mode) ---------------------------

_CHUNKER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["narrative", "chunks"],
    "properties": {
        "narrative": {"type": "string"},
        "chunks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_PANEL_CONTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["panels"],
    "properties": {
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "characters", "environment", "shot_type", "mood"],
                "properties": {
                    "description": {"type": "string"},
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "costume"],
                            "properties": {
                                "id": {"type": "string"},
                                "costume": {"type": ["string", "null"]},
                            },
                        },
                    },
                    "environment": {"type": "string"},
                    "shot_type": {
                        "type": "string",
                        "enum": [
                            "wide", "medium", "close_up", "extreme_close_up",
                            "overhead", "low_angle", "dutch_angle",
                        ],
                    },
                    "mood": {"type": "string"},
                },
            },
        },
    },
}


# -- Default LLM client (OpenAI structured output) ----------------------------

def _default_llm_client(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Default LLM client: OpenAI structured output mode (strict: true).

    Lazy-imports the openai package. Injectable replacement: pass any
    callable with the same signature as llm_client to SceneProducer.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from e

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise SceneProducerLLMError("API returned empty response content")
    return json.loads(content)


# -- Scene Producer -------------------------------------------------------------

class SceneProducer:
    """
    Generates scene files from human-authored input files via a two-stage
    LLM process (DESIGN.md §16.11).

    The user owns STRUCTURE (the elements list from the input file); the LLM
    owns CONTENT (narrative + per-panel content). Panel keys are never shown
    to the LLM — code maps ordered arrays onto derived positional IDs.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        config: Any,
        model: str = "gpt-4o",
        llm_client: Callable[[str, str, str, dict[str, Any]], dict[str, Any]] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialise the Scene Producer.

        Args:
            config: A ProjectConfig object with a scene block.
            model: The LLM model for both stages (default: gpt-4o).
            llm_client: Injectable callable
                (system_prompt, user_prompt, schema_name, schema) -> dict.
                Defaults to OpenAI structured output mode.
            progress_callback: Optional callable(str) for progress updates
                (CLI wires it to its elapsed-time printer; tests may capture).
                Structured data is still returned — this is notification only.
        """
        self.config = config
        self.model = model
        self._progress = progress_callback
        self._llm_client = llm_client or (
            lambda system_prompt, user_prompt, schema_name, schema:
                _default_llm_client(model, system_prompt, user_prompt, schema_name, schema)
        )
        self.registry = TemplateRegistry(config.templates_dir)
        self.parser = ScenePlanParser(config)

        with (config.schemas_dir / "scene_input.schema.json").open("r", encoding="utf-8") as f:
            self.input_schema = json.load(f)
        with (config.schemas_dir / "scene_plan.schema.json").open("r", encoding="utf-8") as f:
            self.scene_schema = json.load(f)

        # Panel-content sub-schema from the scene schema (post-call validation)
        self.panel_content_schema = self.scene_schema["definitions"]["panelContent"]

    def _notify(self, message: str) -> None:
        """Emit a progress update if a callback is wired."""
        if self._progress:
            self._progress(message)

    # -- Input loading -------------------------------------------------------

    def load_input(self, input_file: str | Path) -> dict[str, Any]:
        """
        Load and validate a scene input file.

        Validates against scene_input.schema.json AND that every element
        references a template that exists in the registry — BEFORE any LLM
        call is spent.

        Raises:
            SceneInputError: On missing file, schema failure, or unknown refs.
        """
        path = Path(input_file)
        if not path.is_absolute():
            path = self.config.project_root / path
        if not path.exists():
            raise SceneInputError(f"Scene input file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            scene_input = yaml.safe_load(f)

        try:
            validate_schema(instance=scene_input, schema=self.input_schema)
        except Exception as e:
            raise SceneInputError(
                f"{path.name}: input schema validation failed — {getattr(e, 'message', e)}"
            ) from e

        for idx, element in enumerate(scene_input["elements"], start=1):
            if "layout" in element:
                if not self.registry.has_layout(element["layout"]):
                    raise SceneInputError(
                        f"{path.name} / element {idx}: unknown layout '{element['layout']}'"
                    )
            elif "strip" in element:
                if not self.registry.has_strip(element["strip"]):
                    raise SceneInputError(
                        f"{path.name} / element {idx}: unknown strip '{element['strip']}'"
                    )
            elif "panel" in element:
                if not self.registry.has_panel(element["panel"]):
                    raise SceneInputError(
                        f"{path.name} / element {idx}: unknown panel '{element['panel']}'"
                    )

        return scene_input

    def find_input(self, scene_id: str) -> Path:
        """
        Locate an input file by scene ID: scene_inputs/{id}.input.yaml or
        scene_inputs/*_{id}.input.yaml (chapter-tagged filenames). The
        .input.yaml suffix is REQUIRED — it distinguishes author-owned
        inputs (scene_inputs/) from producer-generated scenes (scenes/),
        which never share a basename convention.
        """
        inputs_dir = self.config.scene_inputs_dir
        candidates = [
            inputs_dir / f"{scene_id}.input.yaml",
            *sorted(inputs_dir.glob(f"*_{scene_id}.input.yaml")),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise SceneInputError(
            f"No scene input file for '{scene_id}' in {inputs_dir} "
            f"(looked for {scene_id}.input.yaml or *_{scene_id}.input.yaml; "
            f"the .input.yaml suffix is required)"
        )

    # -- Context (roster injection for stage 2) --------------------------------

    def _load_character_context(self) -> list[dict[str, Any]]:
        """Load all character YAMLs and extract LLM context (chapter-producer pattern)."""
        characters = []
        for char_file in sorted(self.config.characters_dir.glob("*/*.yaml")):
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
        """Load all environment YAMLs and extract LLM context."""
        environments = []
        for env_file in sorted(self.config.environments_dir.glob("*/*.yaml")):
            with env_file.open("r", encoding="utf-8") as f:
                env_data = yaml.safe_load(f)
            environments.append({
                "environment_id": env_data["environment_id"],
                "display_name": env_data["display_name"],
                "description": env_data["description"],
                "exclusions": env_data.get("prompt_tokens", {}).get("exclusions", []),
            })
        return environments

    def _load_style_context(self) -> dict[str, Any]:
        return {
            "style_id": self.config.style["style_id"],
            "visual_style": self.config.style["visual_style"],
            "forbidden_elements": self.config.style.get("forbidden_elements", []),
        }

    # -- Stage 1: Chunker ---------------------------------------------------------

    def _build_chunker_system_prompt(self) -> str:
        return """You are a graphic novel scene chunker and continuity narrator.

You receive a scene synopsis and an ordered list of the scene's structural
elements (each with its exact panel count). You produce TWO things:

1. A narrative — a single coherent continuity storyboard for the WHOLE scene
   (3-8 sentences of prose). This narrative is the scene's shared backbone:
   it feeds every downstream panel-content call and the pipeline's continuity
   context. Write it as one author, not as a concatenation of parts.

2. Chunks — exactly one chunk of the synopsis per element. Chunk i will be
   expanded into element i's panels. Allocate story proportionally: an
   element with 14 panels needs substantially more story than one with 2.
   Chunks must be consecutive, non-overlapping beats in element order, and
   together must cover the whole synopsis.

RULES:
- The number of chunks MUST EXACTLY EQUAL the number of elements.
- Never invent story that contradicts the synopsis.
- The narrative covers the whole scene arc; the chunks divide it."""

    def _build_chunker_user_prompt(self, scene_input: dict[str, Any], element_counts: list[int]) -> str:
        element_lines = []
        for idx, (element, count) in enumerate(zip(scene_input["elements"], element_counts), start=1):
            kind, ref = next(iter(element.items()))
            element_lines.append(
                f"  - Element {idx}: {kind} '{ref}' — EXACTLY {count} panels. "
                f"Chunk {idx} will fill these {count} panels."
            )
        return f"""Scene title: {scene_input['title']}

Scene synopsis:
{scene_input['synopsis']}

Structural elements (in order; chunk count MUST equal {len(element_lines)}):
{chr(10).join(element_lines)}

Write the scene narrative and the per-element chunks."""

    def _call_chunker(self, scene_input: dict[str, Any], element_counts: list[int]) -> tuple[str, list[str]]:
        """
        Stage 1: one LLM call — emits {narrative, chunks}.

        Retries up to MAX_RETRIES on chunk-count mismatch or empty narrative
        (the known structured-output weakness for array-length constraints).
        """
        system_prompt = self._build_chunker_system_prompt()
        user_prompt = self._build_chunker_user_prompt(scene_input, element_counts)
        n = len(scene_input["elements"])
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = self._wrapped_llm_client(system_prompt, user_prompt, "scene_chunks", _CHUNKER_SCHEMA)

            narrative = result.get("narrative", "")
            chunks = result.get("chunks", [])

            if not narrative or not narrative.strip():
                last_error = SceneProducerLLMError("chunker returned empty narrative")
            elif len(chunks) != n:
                last_error = SceneProducerLLMError(
                    f"chunker returned {len(chunks)} chunks, expected {n}"
                )
            elif any(not c.strip() for c in chunks):
                last_error = SceneProducerLLMError("chunker returned an empty chunk")
            else:
                return narrative, chunks

            if attempt >= self.MAX_RETRIES:
                raise SceneProducerLLMError(
                    f"Stage 1 (chunker) failed after {self.MAX_RETRIES} attempts. "
                    f"Last error: {last_error}"
                )
            self._notify(f"Stage 1 retry ({attempt + 1}/{self.MAX_RETRIES}): {last_error}")

        raise SceneProducerLLMError("unreachable")  # pragma: no cover

    # -- Stage 2: Per-element panel content ---------------------------------------

    def _build_element_system_prompt(self) -> str:
        return """You are a graphic novel panel writer. You receive one chunk of a scene synopsis, the scene's narrative backbone, and (except for the first element) a WHERE WE LEFT OFF handoff from the previous element. You write the content for one element's panels.

CRITICAL RULES:
1. THE NUMBER OF ENTRIES IN YOUR "panels" ARRAY MUST EXACTLY EQUAL THE PANEL COUNT GIVEN IN THE USER MESSAGE. Count before finalising. This is the most important constraint.
2. Panels are in reading order (left-to-right, top-to-bottom of the element). Write them in narrative order.
3. Only use character ids from the provided roster. Never invent characters. Each character entry: {"id": <character_id>, "costume": <variant_id or null>}. Use the costume field ONLY when the character wears a non-default costume; null means default.
4. Only use environment ids from the provided roster. Never invent environments.
5. Each description must be ABSOLUTE — spatially and physically complete on its own. The image model reading it has NO memory of other panels or of the synopsis: describe position (left/center/right, foreground/background), pose, action, and what is visible, not references like "he turns" or "as before".
6. Respect environment exclusions and style forbidden elements.
7. shot_type must be one of: wide, medium, close_up, extreme_close_up, overhead, low_angle, dutch_angle.
8. mood is a short phrase (e.g. "tense", "contemplative").
9. Maintain continuity with the WHERE WE LEFT OFF handoff where given — same characters keep their costumes and carried props unless the story changes them.

Your output is validated against a strict schema. Violations fail the generation."""

    def _build_element_user_prompt(
        self,
        scene_input: dict[str, Any],
        element: dict[str, Any],
        element_index: int,
        panel_count: int,
        chunk: str,
        narrative: str,
        handoff: str | None,
    ) -> str:
        kind, ref = next(iter(element.items()))

        # Roster context (chapter-producer pattern)
        char_lines = []
        for c in self._load_character_context():
            char_lines.append(
                f"  - {c['character_id']} ({c['display_name']}): {c['physical_description']['build']}, "
                f"{c['physical_description']['hair']} hair, {c['physical_description']['eyes']} eyes. "
                f"Default costume: {c['costume_default']}"
            )
            for v in c.get("costume_variants"):
                char_lines.append(f"    Variant '{v['variant_id']}': {v['description']}")
            if c.get("exclusions"):
                char_lines.append(f"    Exclusions: {', '.join(c['exclusions'])}")

        env_lines = []
        for e in self._load_environment_context():
            env_lines.append(f"  - {e['environment_id']} ({e['display_name']}): {e['description']}")
            if e.get("exclusions"):
                env_lines.append(f"    Exclusions: {', '.join(e['exclusions'])}")

        s = self._load_style_context()
        style_lines = [
            f"  Style: {s['visual_style']['label']} — {s['visual_style']['description'].strip()}",
            f"  Forbidden: {', '.join(s['forbidden_elements'])}",
        ]

        handoff_block = (
            f"WHERE WE LEFT OFF (the previous element's final panel):\n{handoff}\n"
            if handoff
            else "This is the first element of the scene — no prior state.\n"
        )

        return f"""Scene narrative backbone (applies to the whole scene):
{narrative}

{handoff_block}Your assignment: element {element_index} ({kind} '{ref}').
Synopsis chunk for THIS element:
{chunk}

Available Characters:
{chr(10).join(char_lines)}

Available Environments:
{chr(10).join(env_lines)}

Visual Style:
{chr(10).join(style_lines)}

Write the content for EXACTLY {panel_count} panels — your "panels" array MUST have EXACTLY {panel_count} entries. Count them before finalising."""

    def _validate_element_content(
        self,
        panels: list[dict[str, Any]],
        panel_count: int,
        element_index: int,
    ) -> None:
        """
        Validate one element's returned panel array: length, per-panel schema
        conformance, and referenced IDs (characters, costumes, environments).
        Raises ValueError on any violation (retryable).
        """
        if len(panels) != panel_count:
            raise ValueError(
                f"element {element_index}: returned {len(panels)} panels, "
                f"expected {panel_count}"
            )

        valid_chars = {c["character_id"] for c in self._load_character_context()}
        char_variants = {
            c["character_id"]: {v["variant_id"] for v in c.get("costume_variants", [])}
            for c in self._load_character_context()
        }
        valid_envs = {e["environment_id"] for e in self._load_environment_context()}

        errors: list[str] = []
        for pos, panel in enumerate(panels, start=1):
            try:
                validate_schema(instance=panel, schema=self.panel_content_schema)
            except Exception as e:
                errors.append(
                    f"element {element_index} / panel {pos}: "
                    f"{getattr(e, 'message', e)}"
                )
                continue

            for char_entry in panel.get("characters", []):
                char_id = char_entry["id"]
                costume = char_entry.get("costume")
                if char_id not in valid_chars:
                    errors.append(
                        f"element {element_index} / panel {pos}: "
                        f"unknown character '{char_id}'"
                    )
                elif costume and costume not in char_variants.get(char_id, set()):
                    errors.append(
                        f"element {element_index} / panel {pos}: "
                        f"unknown costume variant '{costume}' for character '{char_id}'"
                    )

            if panel.get("environment") not in valid_envs:
                errors.append(
                    f"element {element_index} / panel {pos}: "
                    f"unknown environment '{panel.get('environment')}'"
                )

        if errors:
            raise ValueError(
                "Panel content validation failed:\n  " + "\n  ".join(errors)
            )

    @staticmethod
    def _build_handoff(last_panel: dict[str, Any]) -> str:
        """Code-assemble the WHERE-WE-LEFT-OFF block from the previous
        element's final panel (description + characters with costumes)."""
        chars = []
        for char_entry in last_panel.get("characters", []):
            costume = char_entry.get("costume")
            label = char_entry["id"] + (f" (costume: {costume})" if costume else " (default costume)")
            chars.append(label)
        char_line = ", ".join(chars) if chars else "no characters"
        return (
            f"Description: {last_panel['description']}\n"
            f"Characters present: {char_line}"
        )

    def _call_element(
        self,
        scene_input: dict[str, Any],
        element: dict[str, Any],
        element_index: int,
        panel_count: int,
        chunk: str,
        narrative: str,
        handoff: str | None,
    ) -> list[dict[str, Any]]:
        """
        Stage 2: one LLM call for one element — emits an ordered panel array.

        Retries up to MAX_RETRIES on length mismatch or invalid content
        (unknown ids, schema violations).
        """
        system_prompt = self._build_element_system_prompt()
        user_prompt = self._build_element_user_prompt(
            scene_input, element, element_index, panel_count, chunk, narrative, handoff
        )
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = self._wrapped_llm_client(system_prompt, user_prompt, "element_panels", _PANEL_CONTENT_SCHEMA)
            panels = result.get("panels", [])

            # Normalize: OpenAI strict mode emits "costume": null for the
            # default costume; the scene schema's field is an optional string.
            # null == default == omit the field.
            for panel in panels:
                for char_entry in panel.get("characters", []):
                    if char_entry.get("costume") is None:
                        char_entry.pop("costume", None)

            try:
                self._validate_element_content(panels, panel_count, element_index)
                return panels
            except ValueError as e:
                last_error = e
                if attempt >= self.MAX_RETRIES:
                    raise SceneProducerLLMError(
                        f"Stage 2 failed for element {element_index} after "
                        f"{self.MAX_RETRIES} attempts. Last error: {last_error}"
                    ) from e
                self._notify(
                    f"Stage 2 element {element_index} retry "
                    f"({attempt + 1}/{self.MAX_RETRIES}): {last_error}"
                )

        raise SceneProducerLLMError("unreachable")  # pragma: no cover

    # -- Assembly + parse gate --------------------------------------------------------

    def _scene_filename(self, scene_input: dict[str, Any]) -> str:
        scene_id = scene_input["scene_id"]
        chapter_tag = scene_input.get("chapter_tag")
        return f"{chapter_tag}_{scene_id}.yaml" if chapter_tag else f"{scene_id}.yaml"

    def _assemble_scene(
        self,
        scene_input: dict[str, Any],
        narrative: str,
        element_panels: list[list[dict[str, Any]]],
        panel_ids: list[str],
    ) -> dict[str, Any]:
        """Assemble the scene dict: header verbatim from input, panels keyed by
        derived positional IDs (code maps array position -> ID; the LLM never
        saw a key)."""
        scene: dict[str, Any] = {
            "scene_id": scene_input["scene_id"],
            "title": scene_input["title"],
            "narrative": narrative,
            "elements": scene_input["elements"],
            "panels": {},
        }
        if scene_input.get("chapter_tag"):
            scene["chapter_tag"] = scene_input["chapter_tag"]

        id_cursor = 0
        for panels in element_panels:
            for panel in panels:
                scene["panels"][panel_ids[id_cursor]] = panel
                id_cursor += 1

        # Structural sanity: coverage must be exact by construction.
        assert id_cursor == len(panel_ids), "panel id mapping desync"
        return scene

    def _commit_through_parse_gate(self, scene: dict[str, Any]) -> tuple[Path, SceneParseResult]:
        """
        Run the full Phase 1 gauntlet on the assembled scene BEFORE commit.

        Writes the scene dict to a temp file, parses it (schema, coverage,
        geometry — and emits PanelSpecs to output/ as a side effect), then
        atomically moves the file into scenes/. Any failure leaves no scene
        file on disk.
        """
        scenes_dir = self.config.scenes_dir
        scenes_dir.mkdir(parents=True, exist_ok=True)
        final_path = scenes_dir / self._scene_filename(scene)

        fd, tmp_name = tempfile.mkstemp(
            dir=str(scenes_dir), prefix=f".{final_path.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(scene, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            parse_result = self.parser.parse(tmp_path)

            # Atomic commit — only parse-clean scenes ever land in scenes/.
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return final_path, parse_result

    # -- Main orchestration -------------------------------------------------------------

    def produce_scene(self, input_file: str | Path) -> dict[str, Any]:
        """
        Produce a complete scene from an input file.

        Stage 1 (chunker) -> sequential stage 2 (per-element content) ->
        assembly -> parse gate -> atomic commit to scenes/.

        Args:
            input_file: Path to the scene input YAML (or a scene ID — resolved
                        via scene_inputs/ globbing).

        Returns:
            A result dict with keys:
            - "scene": The validated scene dict.
            - "file_path": Path to the committed scene file in scenes/.
            - "parse_result": The SceneParseResult (PanelSpecs emitted to output/).
            - "model": The model used.
            - "attempts": {"chunker": n, "elements": [n, ...]} — LLM calls per stage.
            - "total_llm_calls": int.

        Raises:
            SceneInputError: Bad input file (before any LLM call).
            SceneProducerLLMError: LLM non-compliance after all retries.
            Scene parser errors: propagate from the parse gate (scene files
                the pipeline would reject never reach scenes/).
        """
        # Resolve scene_id -> input file (convenience)
        if isinstance(input_file, str) and not (Path(input_file).exists() or "/" in input_file or "\\" in input_file):
            input_file = self.find_input(input_file)

        scene_input = self.load_input(input_file)
        scene_id = scene_input["scene_id"]

        # The panel ID space is computable BEFORE any LLM call (user owns
        # structure; the LLM never sees a key).
        panel_ids = self.parser.derive_panel_ids(scene_id, scene_input["elements"])
        if not panel_ids:
            raise SceneInputError("input has no elements")
        self._notify(
            f"Input validated: {scene_id} — {len(scene_input['elements'])} elements, "
            f"{len(panel_ids)} panels"
        )

        # Per-element panel counts (stage-1 allocation context + stage-2 targets)
        element_ids: list[list[str]] = []
        for element_index in range(1, len(scene_input["elements"]) + 1):
            prefix = f"{scene_id}_l{element_index:02d}_"
            element_ids.append([pid for pid in panel_ids if pid.startswith(prefix)])
        element_counts = [len(ids) for ids in element_ids]

        # Stage 1: narrative + chunks
        narrative, chunks = self._call_chunker(scene_input, element_counts)
        self._notify(f"Stage 1 complete: narrative written, {len(chunks)} chunks allocated")

        # Stage 2: sequential per-element calls with code-assembled handoff
        element_panels: list[list[dict[str, Any]]] = []
        attempts: list[int] = []
        handoff: str | None = None

        for element_index, (element, chunk, count) in enumerate(
            zip(scene_input["elements"], chunks, element_counts), start=1
        ):
            before = self._llm_call_count
            panels = self._call_element(
                scene_input, element, element_index, count, chunk, narrative, handoff
            )
            attempts.append(self._llm_call_count - before)
            element_panels.append(panels)
            self._notify(
                f"Stage 2: element {element_index}/{len(scene_input['elements'])} "
                f"complete ({count} panels)"
            )

            # Handoff for the next element: this element's final panel.
            handoff = self._build_handoff(panels[-1])

        # Assembly + parse gate + atomic commit
        self._notify("Parse gate: validating scene (schema, coverage, geometry)...")
        scene = self._assemble_scene(scene_input, narrative, element_panels, panel_ids)
        file_path, parse_result = self._commit_through_parse_gate(scene)
        self._notify(f"Parse gate passed — committed {file_path.name}")

        return {
            "scene": scene,
            "file_path": file_path,
            "parse_result": parse_result,
            "model": self.model,
            "attempts": {
                "chunker": self._chunker_calls,
                "elements": attempts,
            },
            "total_llm_calls": self._llm_call_count,
        }

    # -- Call accounting (for tests + result reporting) -------------------------------

    _llm_call_count = 0
    _chunker_calls = 0

    def _wrapped_llm_client(self, system_prompt: str, user_prompt: str, schema_name: str, schema: dict) -> dict:
        self._llm_call_count += 1
        if schema_name == "scene_chunks":
            self._chunker_calls += 1
        return self._llm_client(system_prompt, user_prompt, schema_name, schema)
