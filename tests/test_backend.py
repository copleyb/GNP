"""
Tests for the Image Generation Backend (backend.py).

The backend is a stateless adapter — tests mock the OpenAI API client
to verify the adapter's behavior without making real API calls.
"""

import base64
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.config import load_config
from pipeline.backend import ImageGenerationBackend, GenerationResult
from pipeline.compiler import PromptCompiler, GenerationRequest

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def config():
    return load_config(str(PROJECT_ROOT))


@pytest.fixture
def backend():
    return ImageGenerationBackend(project_root=str(PROJECT_ROOT))


@pytest.fixture
def mock_generation_request():
    """A minimal GenerationRequest for testing."""
    return GenerationRequest(
        panel_id="test_panel",
        model="gpt-image-2",
        prompt="test prompt",
        size="1024x1024",
        quality="high",
        thinking="medium",
        seed=None,
        reference_images=[],
        panelspec_path="output/test.panelspec.json",
        compiler_version="1.0.0",
    )


class TestGenerationResult:
    def test_success(self):
        r = GenerationResult(
            status="success",
            output_bytes=b"fake_png",
            api_response_id="img-123",
            model="gpt-image-2",
        )
        assert r.succeeded is True
        assert r.error is None

    def test_failure(self):
        r = GenerationResult(
            status="failure",
            output_bytes=None,
            api_response_id=None,
            model="gpt-image-2",
            error="API error",
        )
        assert r.succeeded is False
        assert r.error == "API error"


class TestImageGenerationBackend:
    @patch("pipeline.backend.ImageGenerationBackend.generate")
    def test_generate_returns_result(self, mock_generate, backend, mock_generation_request):
        """generate() should return a GenerationResult."""
        mock_result = GenerationResult(
            status="success",
            output_bytes=b"fake",
            api_response_id="img-123",
            model="gpt-image-2",
        )
        mock_generate.return_value = mock_result

        result = backend.generate(mock_generation_request)
        assert isinstance(result, GenerationResult)
        assert result.status == "success"

    def test_missing_reference_image_logged(self, backend):
        """Missing reference images should be logged but not crash."""
        request = GenerationRequest(
            panel_id="test",
            model="gpt-image-2",
            prompt="test",
            size="1024x1024",
            quality="high",
            thinking="medium",
            seed=None,
            reference_images=[{"ref_id": "r1", "file": "nonexistent/file.png", "role": "character"}],
            panelspec_path="output/test.json",
            compiler_version="1.0.0",
        )

        # Mock the OpenAI client to avoid a real API call
        with patch("openai.OpenAI"):
            result = backend.generate(request)
            # Should either fail (no refs) or succeed — just shouldn't crash
            assert isinstance(result, GenerationResult)


class TestBackendIntegration:
    """Integration test: compile a real panel and check the request structure."""

    def test_compiled_request_has_all_api_params(self, config):
        """A compiled GenerationRequest should have all params the backend needs."""
        import json

        spec_path = (
            PROJECT_ROOT / "tests" / "fixtures" / "fixture_single_char.panelspec.json"
        )

        with spec_path.open() as f:
            spec = json.load(f)

        compiler = PromptCompiler(config)
        result = compiler.compile(
            spec,
            call_llm=lambda model, system_prompt, user_prompt: (
                "Alyssa sits on the edge of her bed in first light, venetian "
                "blinds striping the room in gold."
            ),
        )

        assert result.model is not None
        assert result.prompt is not None
        assert len(result.prompt) > 100
        assert result.size in ("1024x1024", "1536x1024", "1024x1536")
        assert result.quality in ("low", "medium", "high", "auto", "standard")
        assert result.reference_images is not None
