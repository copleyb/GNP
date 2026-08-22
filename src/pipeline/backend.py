"""
backend.py — Image Generation Backend Adapter.

Thin, stateless wrapper around the OpenAI gpt-image-2 API.
Receives a GenerationRequest, returns a GenerationResult with raw bytes.
Does NOT write files, does NOT do post-processing, does NOT know about PanelSpecs.

Per DESIGN.md §10: The adapter's only job is API communication.
Per DESIGN.md §13.5: Returns structured data, no print() statements, no CLI logic.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    """
    Result of an image generation API call.

    The adapter returns raw bytes — the caller (Orchestrator) is responsible
    for writing files and recording provenance.
    """
    status: str  # "success" | "failure" | "content_filtered"
    output_bytes: bytes | None
    api_response_id: str | None
    model: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


class ImageGenerationBackend:
    """
    Stateless adapter for the OpenAI gpt-image-2 images.edit endpoint.

    Receives a GenerationRequest from the Prompt Compiler and returns
    a GenerationResult containing raw PNG bytes and API metadata.

    This class does NOT:
    - Write files (caller's responsibility)
    - Do aspect ratio selection (done by the Prompt Compiler)
    - Do scale/crop post-processing (done by the Orchestrator)
    - Know about PanelSpecs or provenance
    """

    def __init__(self, project_root: str | Path = "."):
        """
        Args:
            project_root: Root directory of the project. Reference image
                          file paths in the GenerationRequest are relative
                          to this directory.
        """
        self.project_root = Path(project_root)

    def generate(self, request: Any) -> GenerationResult:
        """
        Send a GenerationRequest to the gpt-image-2 API and return the result.

        Args:
            request: A GenerationRequest (from the Prompt Compiler).

        Returns:
            GenerationResult with raw PNG bytes on success, or error info on failure.
        """
        from openai import OpenAI

        client = OpenAI()

        # Load reference image files
        image_files = []
        for ref in request.reference_images:
            path = self.project_root / ref["file"]
            if not path.exists():
                logger.warning("Reference image not found: %s", path)
                continue
            image_files.append(open(path, "rb"))

        if not image_files:
            logger.warning("No reference images loaded for panel %s", request.panel_id)

        try:
            # Build API kwargs
            api_kwargs: dict[str, Any] = {
                "model": request.model,
                "prompt": request.prompt,
                "image": image_files,
                "size": request.size,
                "quality": request.quality,
            }

            # Add seed if configured (null = random, don't send)
            if request.seed is not None:
                api_kwargs["seed"] = request.seed

            # Note: The 'thinking' parameter is specified in DESIGN.md but
            # the current gpt-image-2 API (SDK v2.48.0) does not support it
            # on the images.edit endpoint. We attempt it via extra_body and
            # fall back gracefully if rejected.
            if hasattr(request, "thinking") and request.thinking:
                api_kwargs["extra_body"] = {"thinking": request.thinking}

            response = client.images.edit(**api_kwargs)

            # Decode base64 PNG
            image_data = base64.b64decode(response.data[0].b64_json)

            return GenerationResult(
                status="success",
                output_bytes=image_data,
                api_response_id=getattr(response, "id", None),
                model=request.model,
            )

        except Exception as e:
            error_str = str(e)

            # If the error is about 'thinking', retry without it
            if "thinking" in error_str and "extra_body" in api_kwargs:
                logger.info("API rejected 'thinking' parameter, retrying without it")
                del api_kwargs["extra_body"]

                try:
                    response = client.images.edit(**api_kwargs)
                    image_data = base64.b64decode(response.data[0].b64_json)
                    return GenerationResult(
                        status="success",
                        output_bytes=image_data,
                        api_response_id=getattr(response, "id", None),
                        model=request.model,
                        error="thinking_parameter_rejected",
                    )
                except Exception as e2:
                    return GenerationResult(
                        status="failure",
                        output_bytes=None,
                        api_response_id=None,
                        model=request.model,
                        error=str(e2),
                    )

            return GenerationResult(
                status="failure",
                output_bytes=None,
                api_response_id=None,
                model=request.model,
                error=error_str,
            )

        finally:
            for f in image_files:
                f.close()
