"""
test_validation.py -- Test suite for the Validation Pipeline.

Tests:
1. Structural validation: file exists, valid PNG, adequate size, correct dimensions
2. Structural validation: failure cases (missing file, corrupt PNG, too small, wrong dims)
3. Semantic validation: character_consistency with mock vision
4. Semantic validation: style_adherence with mock vision
5. Composite score computation with weights
6. Weight redistribution when a dimension is None
7. Full validate_panel flow (Stage 1 pass -> Stage 2)
8. Full validate_panel flow (Stage 1 fail -> skip Stage 2)
9. ValidationResult.to_dict serialisation
10. create_validation_pipeline factory from ProjectConfig
11. ValidationConfigError when no validation config
12. Mock vision call receives correct parameters
"""

import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.validation import (
    ValidationPipeline,
    ValidationResult,
    DimensionResult,
    StructuralResult,
    ValidationConfigError,
    create_validation_pipeline,
    PIPELINE_SCALE_FACTOR,
)

PROJECT_ROOT = Path(__file__).parent.parent


# -- Helpers ----------------------------------------------------------------

def _make_png(path, width, height):
    """Create a PNG file with random noise (ensures non-trivial file size)."""
    random.seed(42)
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
    img.save(path, "PNG")
    return path


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture
def config():
    from pipeline.config import load_config
    return load_config(str(PROJECT_ROOT))


@pytest.fixture
def tmp_png(tmp_path):
    """Create a valid PNG file matching panel_geometry (423x470 at 124dpi)."""
    # 1024 * 0.413 = 423, 1137 * 0.413 = 469 (int truncation)
    path = tmp_path / "test_panel.png"
    return _make_png(path, 423, 469)


@pytest.fixture
def tmp_png_small(tmp_path):
    """Create a PNG that is too small (file size)."""
    path = tmp_path / "tiny.png"
    img = Image.new("RGB", (1, 1), (0, 0, 0))
    img.save(path, "PNG")
    return path


@pytest.fixture
def tmp_png_wrong_dims(tmp_path):
    """Create a valid PNG with wrong dimensions."""
    path = tmp_path / "wrong_dims.png"
    return _make_png(path, 500, 500)


@pytest.fixture
def panel_geometry():
    """Panel geometry matching tmp_png dimensions (423x469 at 124dpi)."""
    return {
        "x": 10,
        "y": 10,
        "width_px": 1024,
        "height_px": 1137,
    }


@pytest.fixture
def mock_vision():
    """A mock vision-call function that returns configurable results."""
    calls = []

    def _call(system_msg, content_blocks, dimension_name):
        calls.append({
            "system_msg": system_msg,
            "content_blocks": content_blocks,
            "dimension_name": dimension_name,
        })
        if dimension_name == "character_consistency":
            return {
                "score": 0.85,
                "observations": ["Hair matches reference.", "Jacket detail consistent."],
                "confidence": "high",
            }
        elif dimension_name == "style_adherence":
            return {
                "score": 0.92,
                "observations": ["Bold ink lines match style.", "High contrast present."],
                "confidence": "high",
            }
        return {"score": 0.5, "observations": [], "confidence": "low"}

    _call.calls = calls
    return _call


@pytest.fixture
def pipeline(mock_vision):
    """A ValidationPipeline with a mock vision function."""
    return ValidationPipeline(
        threshold=0.80,
        weights={"character_consistency": 0.6, "style_adherence": 0.4},
        style_description="High contrast, bold ink lines, graphic novel style.",
        call_vision=mock_vision,
    )


@pytest.fixture
def char_ref_images(tmp_path):
    """Create dummy character reference images."""
    refs = []
    for name in ["alyssa_ref", "hood_ref"]:
        path = tmp_path / f"{name}.png"
        _make_png(path, 512, 512)
        refs.append({"path": str(path), "label": name})
    return refs


# -- Stage 1: Structural validation tests ---------------------------------

class TestStructuralValidation:
    """Tests for Stage 1 deterministic structural validation."""

    def test_passes_all_checks(self, pipeline, tmp_png, panel_geometry):
        """All structural checks pass for a valid PNG with correct dims."""
        result = pipeline._structural_validation(tmp_png, panel_geometry, post_processed=True)
        assert result.passed is True
        assert len(result.failures) == 0
        assert "File exists" in result.checks
        assert "Valid PNG" in " ".join(result.checks)
        assert result.layout_compliance == 1.0

    def test_fails_missing_file(self, pipeline, panel_geometry):
        """Fails when file does not exist."""
        result = pipeline._structural_validation(
            "/nonexistent/path.png", panel_geometry, post_processed=True
        )
        assert result.passed is False
        assert "does not exist" in result.failures[0]
        assert result.layout_compliance == 0.0

    def test_fails_corrupt_png(self, pipeline, panel_geometry, tmp_path):
        """Fails when file is not a valid PNG."""
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"not a png file at all")
        result = pipeline._structural_validation(path, panel_geometry, post_processed=True)
        assert result.passed is False
        assert "Invalid or corrupt PNG" in result.failures[0]

    def test_fails_too_small(self, pipeline, tmp_png_small, panel_geometry):
        """Fails when file size is below minimum."""
        result = pipeline._structural_validation(
            tmp_png_small, panel_geometry, post_processed=True
        )
        assert result.passed is False
        assert "File size too small" in " ".join(result.failures)

    def test_fails_wrong_dimensions_post_processed(self, pipeline, tmp_png_wrong_dims, panel_geometry):
        """Fails when post-processed dims do not match panel geometry target."""
        result = pipeline._structural_validation(
            tmp_png_wrong_dims, panel_geometry, post_processed=True
        )
        assert result.passed is False
        assert "Width mismatch" in " ".join(result.failures)
        assert "Height mismatch" in " ".join(result.failures)

    def test_passes_raw_api_size(self, pipeline, tmp_path):
        """Passes when raw (non-post-processed) image matches an API size."""
        path = tmp_path / "api_output.png"
        _make_png(path, 1024, 1024)
        result = pipeline._structural_validation(
            path, {"width_px": 1024, "height_px": 1024}, post_processed=False
        )
        assert result.passed is True
        assert "Dimensions match API size" in " ".join(result.checks)

    def test_fails_raw_wrong_size(self, pipeline, tmp_png_wrong_dims):
        """Fails when raw image does not match any API size."""
        result = pipeline._structural_validation(
            tmp_png_wrong_dims, {"width_px": 500, "height_px": 500}, post_processed=False
        )
        assert result.passed is False
        assert "not in supported API sizes" in " ".join(result.failures)


# -- Stage 2: Semantic validation tests -----------------------------------

class TestSemanticValidation:
    """Tests for Stage 2 GPT-4o vision calls (mocked)."""

    def test_character_consistency_calls_vision(
        self, pipeline, tmp_png, char_ref_images, mock_vision
    ):
        """Character consistency makes a vision call and returns DimensionResult."""
        result = pipeline._validate_character_consistency(
            tmp_png, char_ref_images, ["Alyssa: silver braid", "Hood: hooded figure"]
        )
        assert isinstance(result, DimensionResult)
        assert result.score == 0.85
        assert result.confidence == "high"
        assert len(result.observations) == 2

    def test_character_consistency_passes_refs(
        self, pipeline, tmp_png, char_ref_images, mock_vision
    ):
        """Vision call receives both panel image and reference images."""
        pipeline._validate_character_consistency(
            tmp_png, char_ref_images, ["Alyssa", "Hood"]
        )
        call = mock_vision.calls[-1]
        # Content blocks: text + panel image + 2 ref images = 4
        assert len(call["content_blocks"]) == 4
        assert call["dimension_name"] == "character_consistency"

    def test_style_adherence_calls_vision(self, pipeline, tmp_png, mock_vision):
        """Style adherence makes a vision call and returns DimensionResult."""
        result = pipeline._validate_style_adherence(tmp_png)
        assert isinstance(result, DimensionResult)
        assert result.score == 0.92
        assert result.confidence == "high"
        assert len(result.observations) == 2

    def test_style_adherence_includes_style_description(
        self, pipeline, tmp_png, mock_vision
    ):
        """Style adherence prompt includes the style description text."""
        pipeline._validate_style_adherence(tmp_png)
        call = mock_vision.calls[-1]
        text_block = call["content_blocks"][0]
        assert "High contrast" in text_block["text"]
        assert text_block["type"] == "text"

    def test_handles_no_character_refs(self, pipeline, tmp_png):
        """When no character refs provided, character_consistency is skipped."""
        result = pipeline.validate_panel(
            image_path=tmp_png,
            panel_geometry={"width_px": 1024, "height_px": 1137},
            character_refs=None,
        )
        assert result.character_consistency is None
        assert result.style_adherence is not None
        # Composite only from style
        assert result.composite_score is not None


# -- Composite score tests --------------------------------------------------

class TestCompositeScore:
    """Tests for composite score computation."""

    def test_weighted_average(self, pipeline):
        """Composite = 0.6 * char + 0.4 * style."""
        char = DimensionResult(score=0.80, observations=[], confidence="high")
        style = DimensionResult(score=0.90, observations=[], confidence="high")
        composite = pipeline._compute_composite(char, style)
        expected = round(0.6 * 0.80 + 0.4 * 0.90, 3)
        assert composite == expected  # 0.84

    def test_only_style_when_char_none(self, pipeline):
        """When char is None, weight redistributes to style only."""
        style = DimensionResult(score=0.90, observations=[], confidence="high")
        composite = pipeline._compute_composite(None, style)
        assert composite == 0.90

    def test_only_char_when_style_none(self, pipeline):
        """When style is None, weight redistributes to char only."""
        char = DimensionResult(score=0.70, observations=[], confidence="high")
        composite = pipeline._compute_composite(char, None)
        assert composite == 0.70

    def test_both_none_returns_none(self, pipeline):
        """When both dimensions are None, composite is None."""
        composite = pipeline._compute_composite(None, None)
        assert composite is None

    def test_accepted_above_threshold(self):
        """Pipeline threshold correctly configured."""
        p = ValidationPipeline(
            threshold=0.80,
            call_vision=lambda **kw: {"score": 0.95, "observations": [], "confidence": "high"},
        )
        assert p.threshold == 0.80

    def test_rejected_below_threshold(self):
        """Pipeline threshold correctly configured."""
        p = ValidationPipeline(
            threshold=0.80,
            call_vision=lambda **kw: {"score": 0.50, "observations": [], "confidence": "low"},
        )
        assert p.threshold == 0.80


# -- Full validate_panel tests ---------------------------------------------

class TestValidatePanel:
    """Tests for the full validate_panel flow."""

    def test_full_flow_stage1_pass_stage2_runs(
        self, pipeline, tmp_png, panel_geometry, char_ref_images, mock_vision
    ):
        """Stage 1 passes, Stage 2 runs, composite computed."""
        result = pipeline.validate_panel(
            image_path=tmp_png,
            panel_geometry=panel_geometry,
            character_refs=char_ref_images,
            character_descriptions=["Alyssa: silver braid", "Hood: hooded"],
        )
        assert result.layout_compliance == 1.0
        assert result.character_consistency is not None
        assert result.style_adherence is not None
        assert result.composite_score is not None
        # 0.6 * 0.85 + 0.4 * 0.92 = 0.878
        assert result.composite_score == round(0.6 * 0.85 + 0.4 * 0.92, 3)
        assert result.accepted_for_production is True

    def test_full_flow_stage1_fail_skips_stage2(
        self, pipeline, panel_geometry, mock_vision
    ):
        """Stage 1 fails, Stage 2 skipped, all semantic fields None."""
        result = pipeline.validate_panel(
            image_path="/nonexistent/file.png",
            panel_geometry=panel_geometry,
        )
        assert result.layout_compliance == 0.0
        assert result.character_consistency is None
        assert result.style_adherence is None
        assert result.composite_score is None
        assert result.accepted_for_production is False
        # No vision calls made
        assert len(mock_vision.calls) == 0

    def test_accepted_when_above_threshold(
        self, pipeline, tmp_png, panel_geometry, char_ref_images
    ):
        """Panel accepted when composite >= 0.80 threshold."""
        result = pipeline.validate_panel(
            image_path=tmp_png,
            panel_geometry=panel_geometry,
            character_refs=char_ref_images,
        )
        assert result.accepted_for_production is True

    def test_rejected_when_below_threshold(
        self, tmp_png, panel_geometry, char_ref_images
    ):
        """Panel rejected when composite < threshold."""
        low_mock = lambda **kw: {"score": 0.50, "observations": ["Drift detected."], "confidence": "low"}
        p = ValidationPipeline(
            threshold=0.80,
            style_description="test style",
            call_vision=low_mock,
        )
        result = p.validate_panel(
            image_path=tmp_png,
            panel_geometry=panel_geometry,
            character_refs=char_ref_images,
        )
        assert result.accepted_for_production is False
        assert result.composite_score < 0.80


# -- Serialisation tests ----------------------------------------------------

class TestSerialisation:
    """Tests for to_dict serialisation."""

    def test_validation_result_to_dict_pass(
        self, pipeline, tmp_png, panel_geometry, char_ref_images
    ):
        """to_dict produces the DESIGN.md validation block structure."""
        result = pipeline.validate_panel(
            image_path=tmp_png,
            panel_geometry=panel_geometry,
            character_refs=char_ref_images,
        )
        d = result.to_dict()
        assert d["layout_compliance"] == 1.0
        assert d["character_consistency"]["score"] == 0.85
        assert d["style_adherence"]["score"] == 0.92
        assert d["composite_score"] == round(0.6 * 0.85 + 0.4 * 0.92, 3)
        assert d["threshold"] == 0.80
        assert d["weights_snapshot"] == {"character_consistency": 0.6, "style_adherence": 0.4}
        assert d["accepted_for_production"] is True

    def test_validation_result_to_dict_stage1_fail(self, pipeline, panel_geometry):
        """to_dict for Stage 1 failure has None for semantic fields."""
        result = pipeline.validate_panel(
            image_path="/nonexistent.png",
            panel_geometry=panel_geometry,
        )
        d = result.to_dict()
        assert d["layout_compliance"] == 0.0
        assert d["character_consistency"] is None
        assert d["style_adherence"] is None
        assert d["composite_score"] is None
        assert d["accepted_for_production"] is False

    def test_dimension_result_to_dict(self):
        """DimensionResult.to_dict has correct structure."""
        dr = DimensionResult(
            score=0.81,
            observations=["Hair matches.", "Coat differs slightly."],
            confidence="medium",
        )
        d = dr.to_dict()
        assert d == {
            "score": 0.81,
            "observations": ["Hair matches.", "Coat differs slightly."],
            "confidence": "medium",
        }


# -- Factory function tests -------------------------------------------------

class TestCreateValidationPipeline:
    """Tests for the create_validation_pipeline factory."""

    def test_creates_from_config(self, config):
        """Factory creates pipeline with correct settings from ProjectConfig."""
        pipeline = create_validation_pipeline(config)
        assert pipeline.threshold == 0.80
        assert pipeline.weights == {"character_consistency": 0.6, "style_adherence": 0.4}
        assert "graphic novel" in pipeline.style_description.lower()
        assert "bold ink lines" in pipeline.style_description

    def test_style_description_includes_forbidden(self, config):
        """Style description includes forbidden elements."""
        pipeline = create_validation_pipeline(config)
        assert "Forbidden" in pipeline.style_description
        assert "photorealistic" in pipeline.style_description

    def test_raises_on_missing_validation_config(self):
        """Raises ValidationConfigError when validation config is None."""
        mock_config = MagicMock()
        mock_config.validation = None
        mock_config.style = {}
        with pytest.raises(ValidationConfigError, match="No validation configuration"):
            create_validation_pipeline(mock_config)

    def test_injects_call_vision(self, config, mock_vision):
        """Factory accepts and passes call_vision."""
        pipeline = create_validation_pipeline(config, call_vision=mock_vision)
        assert pipeline._call_vision is mock_vision


# -- Mock vision call parameter tests --------------------------------------

class TestVisionCallParameters:
    """Tests that the mock vision function receives correct parameters."""

    def test_character_call_has_correct_dimension_name(
        self, pipeline, tmp_png, char_ref_images, mock_vision
    ):
        """Character consistency call has correct dimension_name."""
        pipeline._validate_character_consistency(tmp_png, char_ref_images, [])
        assert mock_vision.calls[-1]["dimension_name"] == "character_consistency"

    def test_style_call_has_correct_dimension_name(
        self, pipeline, tmp_png, mock_vision
    ):
        """Style adherence call has correct dimension_name."""
        pipeline._validate_style_adherence(tmp_png)
        assert mock_vision.calls[-1]["dimension_name"] == "style_adherence"

    def test_two_calls_made_for_full_validation(
        self, pipeline, tmp_png, panel_geometry, char_ref_images, mock_vision
    ):
        """Full validate_panel makes exactly 2 vision calls (char + style)."""
        pipeline.validate_panel(
            image_path=tmp_png,
            panel_geometry=panel_geometry,
            character_refs=char_ref_images,
        )
        assert len(mock_vision.calls) == 2
        names = [c["dimension_name"] for c in mock_vision.calls]
        assert "character_consistency" in names
        assert "style_adherence" in names
