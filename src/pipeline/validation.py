"""
validation.py — Validation Pipeline.

Evaluates generated panels against project standards. Two stages:

  Stage 1: Structural validation (deterministic code, no AI)
    - Output file exists, is a valid PNG, has non-trivial file size
    - Post-processed dimensions match panel geometry target from PanelSpec

  Stage 2: Semantic validation (GPT-4o vision, per-dimension calls)
    - character_consistency: does the character match canonical reference?
    - style_adherence: does the panel match the project's visual register?

Composite score = weighted average of Stage 2 dimensions.
If composite >= threshold -> accepted. Otherwise -> escalated for human review.

Per DESIGN.md 12: Scores are heuristic LLM opinions, not measurements.
Per DESIGN.md 13.5: Returns structured data, no print() statements, no CLI logic.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

logger = logging.getLogger(__name__)

# Scale factor: 1024 / 2480 = 0.413 (300dpi -> 124dpi)
PIPELINE_SCALE_FACTOR = 1024 / 2480

# Minimum file size for a non-trivial PNG (bytes)
_MIN_FILE_SIZE = 10_000

# Supported API output sizes
_API_SIZES = {(1024, 1024), (1536, 1024), (1024, 1536)}

# Tolerance for dimension matching (pixels) -- accounts for int() truncation
_DIM_TOLERANCE = 2


# -- Exceptions --------------------------------------------------------------

class ValidationConfigError(Exception):
    """Raised when validation configuration is missing or invalid."""
    pass


# -- Result types -----------------------------------------------------------

@dataclass(frozen=True)
class StructuralResult:
    """
    Result of Stage 1 structural validation.

    Binary gate -- if this fails, Stage 2 does not run.
    """
    passed: bool
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def layout_compliance(self) -> float:
        """1.0 if passed, 0.0 if failed -- binary, not continuous."""
        return 1.0 if self.passed else 0.0


@dataclass(frozen=True)
class DimensionResult:
    """
    Result of a single Stage 2 dimension call.

    Returned by GPT-4o vision with structured output.
    """
    score: float  # 0.0-1.0
    observations: list[str] = field(default_factory=list)
    confidence: str = "medium"  # "high" | "medium" | "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "observations": list(self.observations),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ValidationResult:
    """
    Complete validation result for a single panel.

    Matches the Generation Record validation block from DESIGN.md 12.
    """
    layout_compliance: float  # 1.0 or 0.0 (binary gate)
    character_consistency: DimensionResult | None
    style_adherence: DimensionResult | None
    composite_score: float | None
    threshold: float
    weights_snapshot: dict[str, float]
    accepted_for_production: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the Provenance Store."""
        return {
            "layout_compliance": self.layout_compliance,
            "character_consistency": (
                self.character_consistency.to_dict()
                if self.character_consistency else None
            ),
            "style_adherence": (
                self.style_adherence.to_dict()
                if self.style_adherence else None
            ),
            "composite_score": self.composite_score,
            "threshold": self.threshold,
            "weights_snapshot": dict(self.weights_snapshot),
            "accepted_for_production": self.accepted_for_production,
        }


# -- Vision call type (for mockability) --------------------------------------

VisionCallFn = Callable[..., dict[str, Any]]


# -- Validation Pipeline ----------------------------------------------------

class ValidationPipeline:
    """
    Two-stage panel validation.

    Stage 1 is deterministic and fast. Stage 2 makes GPT-4o vision calls.
    Both stages return structured data; the pipeline never prints.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        weights: dict[str, float] | None = None,
        style_description: str = "",
        call_vision: VisionCallFn | None = None,
    ):
        """
        Initialise the validation pipeline.

        Args:
            threshold: Composite score threshold for acceptance.
            weights: Dimension weights (must sum to 1.0).
                Default: {"character_consistency": 0.6, "style_adherence": 0.4}.
            style_description: The project's visual style description text
                (from style.yaml). Used for the style_adherence dimension.
            call_vision: Injectable vision-call function for testing.
                If None, uses the real GPT-4o vision call via OpenAI client.
        """
        self.threshold = threshold
        self.weights = weights or {
            "character_consistency": 0.6,
            "style_adherence": 0.4,
        }
        self.style_description = style_description
        self._call_vision = call_vision

    # -- Public API ----------------------------------------------------------

    def validate_panel(
        self,
        image_path: str | Path,
        panel_geometry: dict[str, Any],
        character_refs: list[dict[str, Any]] | None = None,
        character_descriptions: list[str] | None = None,
        post_processed: bool = True,
    ) -> ValidationResult:
        """
        Validate a single generated panel.

        Args:
            image_path: Path to the output PNG file.
            panel_geometry: Panel geometry from PanelSpec
                (width_px, height_px at 300dpi).
            character_refs: List of character reference image paths
                [{"path": str, "label": str}] for character_consistency.
            character_descriptions: Human-readable descriptions of each
                character for the vision prompt.
            post_processed: Whether the image was post-processed
                (scale/crop). If True, dimensions checked against panel
                geometry target. If False, checked against API sizes.

        Returns:
            ValidationResult with all fields populated.
        """
        # Stage 1: Structural validation
        structural = self._structural_validation(
            image_path, panel_geometry, post_processed
        )

        if not structural.passed:
            logger.info(
                "Stage 1 failed -- skipping semantic validation. "
                f"Failures: {structural.failures}"
            )
            return ValidationResult(
                layout_compliance=0.0,
                character_consistency=None,
                style_adherence=None,
                composite_score=None,
                threshold=self.threshold,
                weights_snapshot=dict(self.weights),
                accepted_for_production=False,
            )

        # Stage 2: Semantic validation
        char_result = None
        style_result = None

        if character_refs:
            char_result = self._validate_character_consistency(
                image_path, character_refs, character_descriptions or []
            )

        style_result = self._validate_style_adherence(image_path)

        # Composite score
        composite = self._compute_composite(char_result, style_result)

        accepted = composite is not None and composite >= self.threshold

        return ValidationResult(
            layout_compliance=1.0,
            character_consistency=char_result,
            style_adherence=style_result,
            composite_score=composite,
            threshold=self.threshold,
            weights_snapshot=dict(self.weights),
            accepted_for_production=accepted,
        )

    # -- Stage 1: Structural validation -------------------------------------

    def _structural_validation(
        self,
        image_path: str | Path,
        panel_geometry: dict[str, Any],
        post_processed: bool,
    ) -> StructuralResult:
        """
        Deterministic structural checks. No AI.

        Checks:
        1. File exists
        2. Valid PNG (openable by Pillow)
        3. File size is non-trivial
        4. Dimensions match expected target
        """
        path = Path(image_path)
        checks: list[str] = []
        failures: list[str] = []

        # Check 1: File exists
        if not path.exists():
            failures.append(f"File does not exist: {path}")
            return StructuralResult(passed=False, checks=checks, failures=failures)
        checks.append("File exists")

        # Check 2: Valid PNG
        try:
            img = Image.open(path)
            img.verify()  # Verify without loading pixel data
            img = Image.open(path)  # Re-open after verify()
            width, height = img.size
            checks.append(f"Valid PNG ({width}x{height})")
        except Exception as e:
            failures.append(f"Invalid or corrupt PNG: {e}")
            return StructuralResult(passed=False, checks=checks, failures=failures)

        # Check 3: File size is non-trivial
        file_size = path.stat().st_size
        if file_size < _MIN_FILE_SIZE:
            failures.append(
                f"File size too small: {file_size} bytes "
                f"(minimum {_MIN_FILE_SIZE})"
            )
            return StructuralResult(passed=False, checks=checks, failures=failures)
        checks.append(f"File size adequate ({file_size:,} bytes)")

        # Check 4: Dimensions match expected target
        if post_processed:
            # Post-processed: match panel geometry at pipeline DPI
            target_w = int(panel_geometry["width_px"] * PIPELINE_SCALE_FACTOR)
            target_h = int(panel_geometry["height_px"] * PIPELINE_SCALE_FACTOR)

            if abs(width - target_w) > _DIM_TOLERANCE:
                failures.append(
                    f"Width mismatch: got {width}, expected {target_w} "
                    f"(panel geometry {panel_geometry['width_px']}px x "
                    f"{PIPELINE_SCALE_FACTOR:.3f})"
                )
            if abs(height - target_h) > _DIM_TOLERANCE:
                failures.append(
                    f"Height mismatch: got {height}, expected {target_h} "
                    f"(panel geometry {panel_geometry['height_px']}px x "
                    f"{PIPELINE_SCALE_FACTOR:.3f})"
                )
            if not failures:
                checks.append(
                    f"Dimensions match panel geometry target "
                    f"({target_w}x{target_h})"
                )
        else:
            # Raw API output: match one of the three supported sizes
            if (width, height) not in _API_SIZES:
                failures.append(
                    f"Dimensions {width}x{height} not in supported API sizes: "
                    f"{sorted(_API_SIZES)}"
                )
            else:
                checks.append(f"Dimensions match API size ({width}x{height})")

        if failures:
            return StructuralResult(passed=False, checks=checks, failures=failures)

        return StructuralResult(passed=True, checks=checks, failures=[])

    # -- Stage 2: Semantic validation --------------------------------------

    def _validate_character_consistency(
        self,
        image_path: str | Path,
        character_refs: list[dict[str, Any]],
        character_descriptions: list[str],
    ) -> DimensionResult:
        """
        GPT-4o vision call for character consistency.

        Sends the generated panel + character reference images.
        """
        content_blocks: list[dict[str, Any]] = []

        # Panel image first
        panel_b64 = self._encode_image(image_path)
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{panel_b64}",
            },
        })

        # Character reference images
        ref_labels = []
        for i, ref in enumerate(character_refs):
            ref_path = ref.get("path", "")
            if ref_path and Path(ref_path).exists():
                ref_b64 = self._encode_image(ref_path)
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{ref_b64}",
                    },
                })
                label = ref.get("label", f"Reference {i + 1}")
                ref_labels.append(f"Reference {i + 1}: {label}")

        # Build the text prompt
        char_text_parts = [
            "Evaluate character consistency in this generated panel.",
            "The first image is the generated panel.",
        ]
        if ref_labels:
            char_text_parts.append("\n".join(ref_labels))
        char_text_parts.append(
            "Score how consistently the characters match their canonical "
            "reference designs. Target: 'stylized, variation acceptable' -- "
            "minor style variation is fine, structural drift (wrong hair "
            "color, different facial structure, missing signature clothing "
            "items) is not. Return your assessment as JSON."
        )

        content_blocks.insert(0, {
            "type": "text",
            "text": "\n".join(char_text_parts),
        })

        system_msg = (
            "You are a comic art quality reviewer. You evaluate character "
            "consistency in graphic novel panels. You receive a generated panel "
            "and reference images of the canonical character designs. "
            "Score how consistently the characters in the panel match their "
            "canonical reference designs. "
            "Return a JSON object with: score (0.0-1.0), observations (list "
            "of strings), confidence (high/medium/low)."
        )

        result = self._make_vision_call(
            system_msg, content_blocks, "character_consistency"
        )

        return DimensionResult(
            score=result.get("score", 0.0),
            observations=result.get("observations", []),
            confidence=result.get("confidence", "medium"),
        )

    def _validate_style_adherence(
        self,
        image_path: str | Path,
    ) -> DimensionResult:
        """
        GPT-4o vision call for style adherence.

        Sends the generated panel + style description text.
        """
        panel_b64 = self._encode_image(image_path)

        content_blocks = [
            {
                "type": "text",
                "text": (
                    "Evaluate style adherence in this generated panel.\n\n"
                    f"PROJECT STYLE SPECIFICATION:\n{self.style_description}\n\n"
                    "Check: line weight and ink quality, contrast and shadow "
                    "approach, color palette character, lighting style, and "
                    "whether any forbidden elements (photorealism, watercolor, "
                    "anime/manga) are present. Return your assessment as JSON."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{panel_b64}",
                },
            },
        ]

        system_msg = (
            "You are a comic art quality reviewer. You evaluate style "
            "adherence in graphic novel panels. You receive a generated panel "
            "and a description of the project's canonical visual style. "
            "Score how well the panel matches the project's visual register. "
            "Return a JSON object with: score (0.0-1.0), observations (list "
            "of strings), confidence (high/medium/low)."
        )

        result = self._make_vision_call(
            system_msg, content_blocks, "style_adherence"
        )

        return DimensionResult(
            score=result.get("score", 0.0),
            observations=result.get("observations", []),
            confidence=result.get("confidence", "medium"),
        )

    # -- Composite score ----------------------------------------------------

    def _compute_composite(
        self,
        char_result: DimensionResult | None,
        style_result: DimensionResult | None,
    ) -> float | None:
        """
        Weighted average of available dimension scores.

        If a dimension is None (no character refs provided), its weight
        is redistributed to the other available dimensions proportionally.
        """
        available = {}
        if char_result is not None:
            available["character_consistency"] = (
                char_result.score * self.weights["character_consistency"]
            )
        if style_result is not None:
            available["style_adherence"] = (
                style_result.score * self.weights["style_adherence"]
            )

        if not available:
            return None

        # Sum of weights for available dimensions
        used_weight = sum(
            self.weights[dim] for dim in available
        )

        if used_weight == 0:
            return None

        total = sum(available.values())
        return round(total / used_weight, 3)

    # -- Vision call helper -------------------------------------------------

    def _make_vision_call(
        self,
        system_msg: str,
        content_blocks: list[dict[str, Any]],
        dimension_name: str,
    ) -> dict[str, Any]:
        """
        Make a GPT-4o vision call with structured output.

        Uses the injected call_vision function if available (for testing),
        otherwise makes a real OpenAI API call.
        """
        if self._call_vision is not None:
            return self._call_vision(
                system_msg=system_msg,
                content_blocks=content_blocks,
                dimension_name=dimension_name,
            )

        # Real API call
        from openai import OpenAI
        client = OpenAI()

        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "observations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["score", "observations", "confidence"],
            "additionalProperties": False,
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": content_blocks},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": dimension_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=0.2,
        )

        return json.loads(response.choices[0].message.content)

    # -- Image encoding -----------------------------------------------------

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        """Encode an image file as base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# -- Factory function -------------------------------------------------------

def create_validation_pipeline(
    config: Any,
    call_vision: VisionCallFn | None = None,
) -> ValidationPipeline:
    """
    Create a ValidationPipeline from a ProjectConfig.

    Args:
        config: ProjectConfig with validation settings.
        call_vision: Injectable vision-call function for testing.

    Returns:
        Configured ValidationPipeline instance.

    Raises:
        ValidationConfigError: If validation config is missing.
    """
    val_config = getattr(config, "validation", None)
    if val_config is None:
        raise ValidationConfigError(
            "No validation configuration found in project.yaml. "
            "Add a 'validation' block with threshold and weights."
        )

    # Build style description from style data
    style = getattr(config, "style", {})
    visual_style = style.get("visual_style", {})
    style_parts = []
    if "description" in visual_style:
        style_parts.append(visual_style["description"].strip())
    if "prompt_tokens" in visual_style:
        style_parts.append(visual_style["prompt_tokens"].strip())
    if "color_palette" in visual_style:
        style_parts.append(visual_style["color_palette"].strip())
    if "line_weight" in visual_style:
        style_parts.append(f"Line weight: {visual_style['line_weight']}")
    if "shading_approach" in visual_style:
        style_parts.append(f"Shading: {visual_style['shading_approach']}")
    if "lighting_defaults" in style:
        style_parts.append(f"Lighting: {style['lighting_defaults']}")

    forbidden = style.get("forbidden_elements", [])
    if forbidden:
        style_parts.append(
            "Forbidden: " + ", ".join(forbidden)
        )

    style_description = "\n".join(style_parts)

    return ValidationPipeline(
        threshold=val_config.threshold,
        weights=dict(val_config.weights),
        style_description=style_description,
        call_vision=call_vision,
    )
