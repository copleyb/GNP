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
from dataclasses import dataclass
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
        result = self.backend.generate(gen_request)

        if not result.succeeded:
            return PanelResult(
                panel_id=panel_id,
                status="failure",
                output_path=None,
                error=result.error,
            )

        # 3. Post-process: scale-to-fill center crop
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
    ) -> PageResult:
        """
        Generate all panels on a page in sequence.

        Args:
            panels: List of PanelSpec dicts for a single page.
            call_llm: Optional mock for the Scene Prompt Generator.

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

            result = self.generate_panel(spec, surrounding_descriptions=surrounding)
            results.append(result)
            logger.info("Panel %s: %s", spec["panel_id"], result.status)

        return PageResult(page_id=page_id, panels=results)

    def regenerate_panel(
        self,
        panel_spec: dict[str, Any],
        user_feedback: str | None = None,
        surrounding_descriptions: list[str] | None = None,
        full_pipeline: bool = False,
        call_llm: Any = None,
    ) -> PanelResult:
        """
        Regenerate a single panel with optional user feedback.

        Args:
            panel_spec: The PanelSpec dict.
            user_feedback: Optional human instruction for the regeneration
                           (e.g. "make the lighting warmer", "push character left").
            surrounding_descriptions: Adjacent panel descriptions for context.
            full_pipeline: If True, re-run the full pipeline (scene prompt + image).
                           If False, only re-run the image generation with the
                           existing prompt (faster, for minor adjustments).
                           Default: False (backend-only).
            call_llm: Optional mock for the Scene Prompt Generator.

        Returns:
            PanelResult with output path and metadata.

        Per DESIGN.md §11: The first regeneration attempt re-runs only the
        Backend. Subsequent attempts re-run the full pipeline. The
            full_pipeline flag allows manual override of this behavior.
        """
        if full_pipeline or user_feedback:
            # User feedback requires a new scene prompt — force full pipeline
            return self.generate_panel(
                panel_spec,
                surrounding_descriptions=surrounding_descriptions,
                user_feedback=user_feedback,
                call_llm=call_llm,
            )
        else:
            # Backend-only regeneration: recompile without LLM call,
            # then send to backend. We still need the compiled prompt.
            # For backend-only, we reuse the existing prompt by compiling
            # with a mock that returns the last scene prompt.
            panel_id = panel_spec["panel_id"]

            # Check if we have a prior scene prompt in provenance
            records = self.provenance.read_all(panel_id)
            prior_prompt = None
            if records:
                last = records[-1]
                if "scene_prompt" in last and "output" in last["scene_prompt"]:
                    prior_prompt = last["scene_prompt"]["output"]

            if prior_prompt:
                # Use a mock that returns the prior scene prompt
                def reuse_mock(model, system_prompt, user_prompt):
                    return prior_prompt
                return self.generate_panel(
                    panel_spec,
                    surrounding_descriptions=surrounding_descriptions,
                    call_llm=reuse_mock,
                )
            else:
                # No prior prompt — fall back to full pipeline
                return self.generate_panel(
                    panel_spec,
                    surrounding_descriptions=surrounding_descriptions,
                    call_llm=call_llm,
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
            record["scene_prompt"] = {
                "model": self.config.scene_prompt.model,
                "context_profile": gen_request._context_profile,
                "output": gen_request._scene_prompt,
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
