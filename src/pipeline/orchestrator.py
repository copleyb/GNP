"""
orchestrator.py — Generation Orchestrator.

Coordinates the full generation pipeline for a single panel or page:
  1. Compile PanelSpec → GenerationRequest (Prompt Compiler)
  2. Send to Image Backend → raw bytes (Image Generation Backend)
  3. Post-process: scale-to-fill center crop to 124dpi target (Pillow)
  4. Write output file to disk
  5. Append Generation Record to Provenance Store

Per DESIGN.md §11: The Orchestrator is the caller that writes files and
records provenance. The backend adapter is stateless and returns bytes only.
Per DESIGN.md §13.5: Returns structured data, no print() statements, no CLI logic.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .compiler import PromptCompiler, GenerationRequest
from .compiler import ScenePromptError
from .backend import ImageGenerationBackend, GenerationResult
from .provenance import ProvenanceStore

logger = logging.getLogger(__name__)

# Scale factor: 1024 / 2480 = 0.413 (300dpi → ~124dpi)
PIPELINE_SCALE_FACTOR = 1024 / 2480
PIPELINE_DPI = 124


@dataclass(frozen=True)
class PanelResult:
    """Result of generating a single panel."""
    panel_id: str
    status: str  # "success" | "failure"
    output_path: str | None
    error: str | None = None
    api_response_id: str | None = None
    post_processed: bool = False
    input_dimensions: tuple[int, int] | None = None
    output_dimensions: tuple[int, int] | None = None


@dataclass(frozen=True)
class PageResult:
    """Result of generating all panels on a page."""
    page_id: str
    panels: list[PanelResult]

    @property
    def succeeded_count(self) -> int:
        return sum(1 for p in self.panels if p.status == "success")

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.panels if p.status == "failure")


class Orchestrator:
    """
    Coordinates the generation pipeline.

    The Orchestrator is the only module that writes files and records
    provenance. The Backend Adapter and Prompt Compiler are stateless.
    """

    def __init__(self, config: Any):
        """
        Args:
            config: ProjectConfig from pipeline.config.load_config.
        """
        self.config = config
        self.compiler = PromptCompiler(config)
        self.backend = ImageGenerationBackend(project_root=config.project_root)
        self.provenance = ProvenanceStore(config.output_dir)
        self.project_root = config.project_root

    def generate_panel(
        self,
        panel_spec: dict[str, Any],
        surrounding_descriptions: list[str] | None = None,
        user_feedback: str | None = None,
        attempt_number: int | None = None,
        call_llm: Any = None,
        progress_callback: Any = None,
    ) -> PanelResult:
        """
        Generate a single panel from a PanelSpec.

        Args:
            panel_spec: The PanelSpec dict.
            surrounding_descriptions: Adjacent panel descriptions for context.
            user_feedback: Optional human override for regeneration.
            attempt_number: Override the attempt number (auto-detected if None).
            call_llm: Optional mock for the Scene Prompt Generator.

        Returns:
            PanelResult with output path and metadata.
        """
        panel_id = panel_spec["panel_id"]

        # 1. Compile the prompt
        if progress_callback:
            progress_callback("Compiling prompt...")
        try:
            gen_request = self.compiler.compile(
                panel_spec,
                surrounding_descriptions=surrounding_descriptions,
                user_feedback=user_feedback,
                call_llm=call_llm,
            )
        except ScenePromptError as e:
            logger.error("Scene prompt generation failed for %s: %s", panel_id, e)
            return PanelResult(
                panel_id=panel_id,
                status="failure",
                output_path=None,
                error=f"Scene prompt failed: {e}",
            )
        except Exception as e:
            logger.error("Unexpected error compiling prompt for %s: %s", panel_id, e)
            return PanelResult(
                panel_id=panel_id,
                status="failure",
                output_path=None,
                error=f"Compile failed: {e}",
            )

        # 2. Generate the image
        if progress_callback:
            progress_callback("Generating image (this can take 60-120s)...")
        result = self.backend.generate(gen_request)

        if not result.succeeded:
            return PanelResult(
                panel_id=panel_id,
                status="failure",
                output_path=None,
                error=result.error,
            )

        # 3. Post-process: scale-to-fill center crop
        if progress_callback:
            progress_callback("Post-processing...")
        geo = panel_spec["panel_geometry"]
        target_w = int(geo["width_px"] * PIPELINE_SCALE_FACTOR)
        target_h = int(geo["height_px"] * PIPELINE_SCALE_FACTOR)

        input_dimensions, output_dimensions = self._post_process(
            result.output_bytes,
            target_w,
            target_h,
            panel_id,
            attempt_number,
        )

        # 4. Write output file
        if progress_callback:
            progress_callback("Writing output...")
        attempt = attempt_number or self._next_attempt_number(panel_id)
        output_path = self._write_output(
            result.output_bytes if not output_dimensions else None,
            panel_id,
            attempt,
            target_w,
            target_h,
        )

        # 5. Record provenance
        self._record_provenance(
            panel_id, attempt, gen_request, result,
            input_dimensions, output_dimensions,
        )

        return PanelResult(
            panel_id=panel_id,
            status="success",
            output_path=str(output_path),
            api_response_id=result.api_response_id,
            post_processed=output_dimensions is not None,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )

    def generate_page(
        self,
        panels: list[dict[str, Any]],
        call_llm: Any = None,
        progress_callback: Any = None,
    ) -> PageResult:
        """
        Generate all panels on a page in sequence.

        Args:
            panels: List of PanelSpec dicts for a single page.
            call_llm: Optional mock for the Scene Prompt Generator.
            progress_callback: Optional callable(str) for progress updates.

        Returns:
            PageResult with results for each panel.
        """
        page_id = panels[0]["page_id"] if panels else "unknown"
        results: list[PanelResult] = []

        for i, spec in enumerate(panels):
            # Build surrounding descriptions from same-page neighbors
            surrounding: list[str] = []
            if i > 0:
                surrounding.append(f"Previous panel: {panels[i-1]['description']}")
            if i < len(panels) - 1:
                surrounding.append(f"Next panel: {panels[i+1]['description']}")

            if progress_callback:
                progress_callback(f"Panel {i+1}/{len(panels)}: {spec['panel_id']}")
            result = self.generate_panel(
                spec,
                surrounding_descriptions=surrounding,
                progress_callback=progress_callback,
            )
            results.append(result)
            logger.info("Panel %s: %s", spec["panel_id"], result.status)

        return PageResult(page_id=page_id, panels=results)

    # -- Regeneration category constants ------------------------------------

    _REGEN_CATEGORIES = ("replay", "reroll", "revise", "regenerate")

    _BACKEND_OVERRIDE_KEYS = frozenset({"seed", "quality", "thinking"})
    _SCENE_PROMPT_KEYS = frozenset({"feedback", "fresh_prompt", "scene_prompt"})
    _PANELSPEC_OVERRIDE_KEYS = frozenset({"costume", "shot_type", "mood", "description"})

    # -- Category inference -----------------------------------------------

    @staticmethod
    def _infer_category(overrides: dict[str, Any]) -> str:
        """
        Infer the regeneration category from the provided overrides.

        Priority: deepest layer of change wins.
        PanelSpec overrides > scene prompt flags > backend overrides > no flags.
        """
        if any(k in overrides for k in Orchestrator._PANELSPEC_OVERRIDE_KEYS):
            return "regenerate"
        if any(k in overrides for k in Orchestrator._SCENE_PROMPT_KEYS):
            return "revise"
        if any(k in overrides for k in Orchestrator._BACKEND_OVERRIDE_KEYS):
            return "reroll"
        return "replay"

    # -- PanelSpec patching (non-destructive) ------------------------------

    @staticmethod
    def _apply_panelspec_patches(
        panel_spec: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply PanelSpec field overrides in-memory (non-destructive).

        Returns a new dict with the patches applied. The original
        PanelSpec on disk is never modified.

        Supported fields: costume, shot_type, mood, description.
        """
        patched = dict(panel_spec)

        if "shot_type" in overrides:
            patched["shot_type"] = overrides["shot_type"]
        if "mood" in overrides:
            patched["mood"] = overrides["mood"]
        if "description" in overrides:
            patched["description"] = overrides["description"]
        if "costume" in overrides:
            # Patch costume on each character that has a matching variant
            costume = overrides["costume"]
            patched_chars = []
            for char in panel_spec.get("characters", []):
                patched_char = dict(char)
                # Check if this costume variant exists for the character
                # The PanelSpec stores refs with costume tags — we filter them
                costume_refs = [
                    ref for ref in char.get("references", [])
                    if ref.get("costume") == costume
                ]
                if costume_refs:
                    patched_char["references"] = costume_refs
                patched_chars.append(patched_char)
            patched["characters"] = patched_chars

        return patched

    # -- Structured diff computation ---------------------------------------

    @staticmethod
    def _compute_diff(
        panel_spec: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute a structured diff from the panel spec and overrides.

        For PanelSpec field overrides: {"field": {"from": old, "to": new}}
        For feedback: {"feedback": "user instruction text"}
        Composes both when present.
        """
        diff: dict[str, Any] = {}

        for key in ("shot_type", "mood", "description"):
            if key in overrides:
                diff[key] = {
                    "from": panel_spec.get(key, ""),
                    "to": overrides[key],
                }

        if "costume" in overrides:
            diff["costume"] = {
                "from": "default",
                "to": overrides["costume"],
            }

        if "feedback" in overrides:
            diff["feedback"] = overrides["feedback"]

        return diff

    # -- Scene prompt mode determination -----------------------------------

    @staticmethod
    def _determine_scene_prompt_mode(overrides: dict[str, Any]) -> str:
        """
        Determine the scene prompt mode from overrides.

        Returns: "direct" | "cold_start" | "preservation_with_feedback" |
                 "preservation" | "reused"
        """
        if "scene_prompt" in overrides:
            return "direct"
        if "fresh_prompt" in overrides:
            return "cold_start"
        if "feedback" in overrides:
            return "preservation_with_feedback"
        return "preservation"  # default for revise/regenerate without scene-prompt flags

    # -- Regeneration (main entry point) ----------------------------------

    def regenerate_panel(
        self,
        panel_spec: dict[str, Any],
        overrides: dict[str, Any] | None = None,
        surrounding_descriptions: list[str] | None = None,
        call_llm: Any = None,
        progress_callback: Any = None,
        from_attempt: int | None = None,
    ) -> PanelResult:
        """
        Regenerate a single panel using the four-category system.

        The category (replay, reroll, revise, regenerate) is inferred
        from the provided overrides. See DESIGN.md §13 for details.

        Args:
            panel_spec: The PanelSpec dict.
            overrides: Dict of regeneration parameters. Any combination of:
                Backend: seed (int), quality (str), thinking (str)
                Scene prompt: feedback (str), fresh_prompt (bool),
                              scene_prompt (str)
                PanelSpec: costume (str), shot_type (str), mood (str),
                           description (str)
            surrounding_descriptions: Adjacent panel descriptions for context.
            call_llm: Optional mock for the Scene Prompt Generator.

        Returns:
            PanelResult with output path and metadata.
        """
        overrides = overrides or {}
        category = self._infer_category(overrides)
        panel_id = panel_spec["panel_id"]

        # Get the provenance record to branch from (latest, or specific attempt)
        if from_attempt is not None:
            latest_record = self.provenance.get_record_by_attempt(
                panel_id, from_attempt
            )
            if latest_record is None:
                return PanelResult(
                    panel_id=panel_id,
                    status="failure",
                    output_path=None,
                    error=f"No provenance record found for attempt {from_attempt}",
                )
        else:
            latest_record = self.provenance.get_latest_record(panel_id)

        # Determine scene prompt mode (for later provenance recording)
        scene_prompt_mode = "reused"  # default for replay/reroll
        preservation_context = None  # set for revise/regenerate below
        if category in ("revise", "regenerate"):
            scene_prompt_mode = self._determine_scene_prompt_mode(overrides)

        # -- Build the GenerationRequest based on category --
        if progress_callback:
            stage = "Replaying prompt" if category in ("replay", "reroll") else "Generating scene prompt"
            progress_callback(stage + "...")

        if category in ("replay", "reroll"):
            # Backend-only: use stored prompt from provenance
            if latest_record is None or "generation_request" not in latest_record:
                # No prior record — fall back to full pipeline (cold start)
                logger.warning(
                    "No prior Generation Record for %s — falling back to cold start",
                    panel_id,
                )
                category = "revise"
                scene_prompt_mode = "cold_start"
                gen_request = self.compiler.compile(
                    panel_spec,
                    surrounding_descriptions=surrounding_descriptions,
                    call_llm=call_llm,
                )
            else:
                stored_gr = latest_record["generation_request"]
                gen_request = self.compiler.compile_for_replay(
                    panel_spec,
                    stored_prompt=stored_gr["prompt"],
                    seed=overrides.get("seed"),
                    quality=overrides.get("quality"),
                    thinking=overrides.get("thinking"),
                )

        else:
            # Revise or regenerate: run the compiler (possibly with preservation)
            working_spec = panel_spec
            if category == "regenerate":
                working_spec = self._apply_panelspec_patches(panel_spec, overrides)

            preservation_context = None
            user_feedback = None

            if scene_prompt_mode == "direct":
                # --scene-prompt "text" → skip LLM, use provided text
                scene_prompt_text = overrides["scene_prompt"]

                def direct_mock(model, system_prompt, user_prompt):
                    return scene_prompt_text
                call_llm_for_gen = direct_mock
            elif scene_prompt_mode == "cold_start":
                # --fresh-prompt → normal compile (no preservation)
                call_llm_for_gen = call_llm
            elif scene_prompt_mode in ("preservation", "preservation_with_feedback"):
                # Preservation mode: prior prompt + structured diff
                if latest_record and "scene_prompt" in latest_record:
                    prior_prompt = latest_record["scene_prompt"].get("output", "")
                else:
                    # Cold start exception: no prior record
                    logger.warning(
                        "No prior scene prompt for %s — using cold start",
                        panel_id,
                    )
                    scene_prompt_mode = "cold_start"
                    call_llm_for_gen = call_llm
                    prior_prompt = None

                if prior_prompt:
                    diff = self._compute_diff(panel_spec, overrides)
                    preservation_context = {
                        "prior_prompt": prior_prompt,
                        "change_summary": diff,
                    }
                    if scene_prompt_mode == "preservation_with_feedback":
                        user_feedback = overrides.get("feedback")
                        preservation_context["feedback"] = user_feedback
                    call_llm_for_gen = call_llm
                else:
                    call_llm_for_gen = call_llm
            else:
                call_llm_for_gen = call_llm

            gen_request = self.compiler.compile(
                working_spec,
                surrounding_descriptions=surrounding_descriptions,
                user_feedback=user_feedback,
                call_llm=call_llm_for_gen,
                preservation_context=preservation_context,
            )

        # -- Apply backend overrides for revise/regenerate (config defaults) --
        if category in ("revise", "regenerate"):
            patch_kwargs = {}
            if "seed" in overrides:
                patch_kwargs["seed"] = overrides["seed"]
            if "quality" in overrides:
                patch_kwargs["quality"] = overrides["quality"]
            if "thinking" in overrides:
                patch_kwargs["thinking"] = overrides["thinking"]
            if patch_kwargs:
                gen_request = replace(gen_request, **patch_kwargs)

        # -- Generate the image --
        if progress_callback:
            progress_callback("Generating image (this can take 60-120s)...")
        result = self.backend.generate(gen_request)

        if not result.succeeded:
            return PanelResult(
                panel_id=panel_id,
                status="failure",
                output_path=None,
                error=result.error,
            )

        # -- Post-process --
        if progress_callback:
            progress_callback("Post-processing...")
        geo = panel_spec["panel_geometry"]
        target_w = int(geo["width_px"] * PIPELINE_SCALE_FACTOR)
        target_h = int(geo["height_px"] * PIPELINE_SCALE_FACTOR)

        input_dimensions, output_dimensions = self._post_process(
            result.output_bytes,
            target_w,
            target_h,
            panel_id,
            None,  # attempt number determined below
        )

        # -- Write output --
        if progress_callback:
            progress_callback("Writing output...")
        attempt = self._next_attempt_number(panel_id)
        output_path = self._write_output(
            result.output_bytes if not output_dimensions else None,
            panel_id,
            attempt,
            target_w,
            target_h,
        )

        # -- Record provenance with regeneration metadata --
        self._record_provenance(
            panel_id, attempt, gen_request, result,
            input_dimensions, output_dimensions,
            regeneration_category=category,
            overrides=overrides,
            scene_prompt_mode=scene_prompt_mode,
            preservation_context=preservation_context,
        )

        return PanelResult(
            panel_id=panel_id,
            status="success",
            output_path=str(output_path),
            api_response_id=result.api_response_id,
            post_processed=output_dimensions is not None,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )

    # -- Automatic regeneration (validation-triggered) ----------------------

    def auto_regenerate_panel(
        self,
        panel_spec: dict[str, Any],
        max_attempts: int = 3,
        surrounding_descriptions: list[str] | None = None,
        call_llm: Any = None,
    ) -> PanelResult:
        """
        Run the automatic regeneration loop after validation failure.

        Per DESIGN.md §13:
        - Attempt 2 (first auto-regen): reroll with no overrides (exact replay).
          Only the image model's inherent stochasticity provides variation.
        - Attempts 3...max: revise with no flags (preservation mode).
          Fresh LLM scene prompt using prior prompt as context.

        Args:
            panel_spec: The PanelSpec dict.
            max_attempts: Maximum regeneration attempts (from project.yaml).
            surrounding_descriptions: Adjacent panel descriptions.
            call_llm: Optional mock for the Scene Prompt Generator.

        Returns:
            PanelResult from the last attempt.
        """
        panel_id = panel_spec["panel_id"]
        current_attempt = self.provenance.get_latest_attempt_number(panel_id)

        while current_attempt < max_attempts:
            next_attempt = current_attempt + 1

            if next_attempt == 2:
                # First auto-regen: reroll (exact replay, no overrides)
                logger.info("Auto-regen attempt %d for %s: reroll (exact replay)",
                           next_attempt, panel_id)
                result = self.regenerate_panel(
                    panel_spec,
                    overrides={},
                    surrounding_descriptions=surrounding_descriptions,
                    call_llm=call_llm,
                )
            else:
                # Subsequent auto-regens: revise (preservation mode)
                logger.info("Auto-regen attempt %d for %s: revise (preservation)",
                           next_attempt, panel_id)
                result = self.regenerate_panel(
                    panel_spec,
                    overrides={},  # no flags → revise with preservation
                    surrounding_descriptions=surrounding_descriptions,
                    call_llm=call_llm,
                )

            if result.status != "success":
                return result

            # TODO: Run validation here once validation is integrated.
            # For now, return the result — auto-regen loop is structurally
            # in place but requires validation to drive the loop.
            return result

        # Exhausted attempts — escalate to human
        logger.warning("Auto-regen exhausted for %s after %d attempts",
                       panel_id, max_attempts)
        return PanelResult(
            panel_id=panel_id,
            status="failure",
            output_path=None,
            error=f"Exhausted {max_attempts} auto-regeneration attempts",
        )

    def _post_process(
        self,
        image_bytes: bytes | None,
        target_w: int,
        target_h: int,
        panel_id: str,
        attempt_number: int | None,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """
        Apply scale-to-fill center crop to the generated image.

        Returns (input_dimensions, output_dimensions) or (None, None) if
        the image couldn't be processed. The cropped image replaces the
        raw bytes — we re-encode the processed result.

        Note: This modifies the image in memory. The caller should use
        the processed bytes for output, not the original.
        """
        if image_bytes is None:
            return None, None

        try:
            img = Image.open(io.BytesIO(image_bytes))
            input_dims = (img.width, img.height)

            # Scale-to-fill: scale so the shorter dimension matches target,
            # the longer overflows. Then center-crop.
            scale = max(target_w / img.width, target_h / img.height)
            scaled_w = int(img.width * scale)
            scaled_h = int(img.height * scale)
            img = img.resize((scaled_w, scaled_h), Image.LANCZOS)

            # Center crop
            left = (scaled_w - target_w) // 2
            top = (scaled_h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))

            output_dims = (img.width, img.height)

            # Store the processed image back in a way the caller can use
            # We'll write a temporary attribute that _write_output checks
            self._processed_image = img

            return input_dims, output_dims
        except Exception as e:
            logger.warning("Post-processing failed for %s: %s", panel_id, e)
            self._processed_image = None
            return None, None

    def _write_output(
        self,
        raw_bytes: bytes | None,
        panel_id: str,
        attempt: int,
        target_w: int,
        target_h: int,
    ) -> Path:
        """
        Write the generated image to disk.

        Uses the post-processed image if available, otherwise writes raw bytes.
        Output path: output/{panel_id}_attempt_{N}.png

        Prior attempts are moved to output/archive/.
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = output_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Move any existing attempts to archive
        existing = list(output_dir.glob(f"{panel_id}_attempt_*.png"))
        for f in existing:
            dest = archive_dir / f.name
            f.rename(dest)

        output_path = output_dir / f"{panel_id}_attempt_{attempt:03d}.png"

        if hasattr(self, "_processed_image") and self._processed_image is not None:
            self._processed_image.save(output_path, "PNG")
            self._processed_image = None
        elif raw_bytes is not None:
            output_path.write_bytes(raw_bytes)
        else:
            raise RuntimeError("No image data to write")

        return output_path

    def _next_attempt_number(self, panel_id: str) -> int:
        """Determine the next attempt number for a panel."""
        existing = list(self.config.output_dir.glob(f"{panel_id}_attempt_*.png"))
        archived = list((self.config.output_dir / "archive").glob(f"{panel_id}_attempt_*.png"))
        all_attempts = existing + archived
        if not all_attempts:
            return 1
        max_attempt = max(int(f.stem.split("_attempt_")[-1]) for f in all_attempts)
        return max_attempt + 1

    def _record_provenance(
        self,
        panel_id: str,
        attempt: int,
        gen_request: GenerationRequest,
        result: GenerationResult,
        input_dimensions: tuple[int, int] | None,
        output_dimensions: tuple[int, int] | None,
        regeneration_category: str | None = None,
        overrides: dict[str, Any] | None = None,
        scene_prompt_mode: str | None = None,
        preservation_context: dict[str, Any] | None = None,
    ) -> None:
        """Append a Generation Record to the Provenance Store."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        prompt_hash = hashlib.sha256(gen_request.prompt.encode()).hexdigest()

        record = {
            "record_id": f"{panel_id}_attempt_{attempt:03d}",
            "panel_id": panel_id,
            "attempt_number": attempt,
            "timestamp_utc": now,
            "compiler": {
                "version": gen_request.compiler_version,
                "prompt_hash": f"sha256:{prompt_hash[:16]}...",
            },
            "generation_request": {
                "model": gen_request.model,
                "prompt": gen_request.prompt,
                "size": gen_request.size,
                "quality": gen_request.quality,
                "thinking": getattr(gen_request, "thinking", None),
                "seed": gen_request.seed,
                "n": 1,
            },
            "outcome": {
                "status": result.status,
                "output_file": f"output/{panel_id}_attempt_{attempt:03d}.png",
                "api_response_id": result.api_response_id,
            },
            "post_processing": {
                "crop": {
                    "applied": output_dimensions is not None,
                    "strategy": "scale_to_fill_center_crop" if output_dimensions else None,
                    "input_dimensions": list(input_dimensions) if input_dimensions else None,
                    "output_dimensions": list(output_dimensions) if output_dimensions else None,
                    "pipeline_dpi": PIPELINE_DPI if output_dimensions else None,
                }
            },
        }

        # Add scene prompt sub-record if available
        if hasattr(gen_request, "_scene_prompt") and gen_request._scene_prompt:
            sp_record: dict[str, Any] = {
                "model": self.config.scene_prompt.model,
                "context_profile": gen_request._context_profile,
                "output": gen_request._scene_prompt,
            }
            # Regeneration metadata
            if scene_prompt_mode is not None:
                sp_record["mode"] = scene_prompt_mode
                sp_record["regenerated"] = scene_prompt_mode != "reused"
            if preservation_context is not None:
                sp_record["preservation_context"] = {
                    "prior_prompt": preservation_context.get("prior_prompt", ""),
                    "change_summary": preservation_context.get("change_summary", {}),
                }
            record["scene_prompt"] = sp_record

        # Add regeneration metadata
        if regeneration_category is not None:
            record["regeneration_category"] = regeneration_category
            record["overrides"] = {
                k: v for k, v in (overrides or {}).items()
                if v is not None and v is not False
            }

        # Add reference selection sub-record
        if hasattr(gen_request, "_reference_selections") and gen_request._reference_selections:
            record["reference_selection"] = {
                "budget": self.config.image_generation.reference_budget,
                "allocation_algorithm": "proportional_primary_priority_v1",
                "selected": [
                    {
                        "ref_id": s.ref_id,
                        "character_id": s.character_id,
                        "environment_id": s.environment_id,
                        "role": s.role,
                    }
                    for s in gen_request._reference_selections
                ],
            }

        self.provenance.append(record)
