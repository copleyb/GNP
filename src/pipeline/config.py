"""
config.py — Project configuration loader and validator.

Loads project.yaml, validates it against project.schema.json, resolves all
directory paths relative to the project root, and loads style.yaml.

This is the foundation module — every other module imports from here to access
project configuration. No other module reads project.yaml directly.

Per DESIGN.md §13.5: returns structured data, no print() statements, no CLI logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validate as validate_schema


# -- Schema paths -----------------------------------------------------------

_SCHEMAS_DIR = "schemas"
_PROJECT_SCHEMA = "project.schema.json"
_STYLE_SCHEMA = "style.schema.json"


# -- Data classes ------------------------------------------------------------

@dataclass(frozen=True)
class ImageGenerationConfig:
    """Image generation backend configuration from project.yaml."""
    backend: str
    model: str
    quality: str
    reference_budget: int
    thinking: str | None = None    # off | low | medium | high. Default: medium.
    seed: int | None = None       # int32 for loose reproducibility, or None for random.


@dataclass(frozen=True)
class ScenePromptConfig:
    """Scene Prompt Generator configuration from project.yaml."""
    model: str
    context_profile: str


@dataclass(frozen=True)
class ValidationConfig:
    """Validation pipeline configuration from project.yaml."""
    threshold: float
    weights: dict[str, float]
    max_regeneration_attempts: int


@dataclass(frozen=True)
class ProjectConfig:
    """
    Fully resolved project configuration.

    All paths are absolute Path objects resolved relative to the project root.
    This is the single object other modules import and use.
    """
    project_id: str
    title: str
    version: str
    compiler_version: str

    # Resolved absolute paths
    project_root: Path
    style_path: Path
    characters_dir: Path
    environments_dir: Path
    layouts_dir: Path
    chapters_dir: Path
    output_dir: Path
    output_archive_dir: Path
    schemas_dir: Path

    # Loaded and validated style data
    style: dict[str, Any]

    # Optional config blocks (may be None if not specified)
    image_generation: ImageGenerationConfig | None = None
    scene_prompt: ScenePromptConfig | None = None
    validation: ValidationConfig | None = None

    notes: str | None = None


# -- Loader ------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_schema(schemas_dir: Path, filename: str) -> dict[str, Any]:
    """Load a JSON schema file from the schemas directory."""
    path = schemas_dir / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_weights_sum(weights: dict[str, float]) -> None:
    """Ensure validation weights sum to 1.0 (within floating point tolerance)."""
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Validation weights must sum to 1.0, got {total:.3f}. "
            f"Weights: {weights}"
        )


def load_config(project_root: str | Path) -> ProjectConfig:
    """
    Load and validate the full project configuration.

    Args:
        project_root: Path to the project directory containing project.yaml.

    Returns:
        ProjectConfig with all paths resolved and all config blocks parsed.

    Raises:
        FileNotFoundError: If project.yaml or style.yaml is missing.
        jsonschema.ValidationError: If project.yaml or style.yaml fails schema validation.
        ValueError: If validation weights don't sum to 1.0 or paths don't resolve.
    """
    root = Path(project_root).resolve()

    # Locate schemas directory
    schemas_dir = root / _SCHEMAS_DIR
    if not schemas_dir.is_dir():
        raise FileNotFoundError(
            f"Schemas directory not found at {schemas_dir}"
        )

    # Load and validate project.yaml
    project_path = root / "project.yaml"
    if not project_path.exists():
        raise FileNotFoundError(f"project.yaml not found at {project_path}")

    project_data = _load_yaml(project_path)
    project_schema = _load_schema(schemas_dir, _PROJECT_SCHEMA)
    validate_schema(instance=project_data, schema=project_schema)

    # Load and validate style.yaml
    style_path = root / project_data["style"]
    if not style_path.exists():
        raise FileNotFoundError(f"Style file not found at {style_path}")

    style_data = _load_yaml(style_path)
    style_schema = _load_schema(schemas_dir, _STYLE_SCHEMA)
    validate_schema(instance=style_data, schema=style_schema)

    # Resolve directory paths
    characters_dir = root / project_data["characters_dir"]
    environments_dir = root / project_data["environments_dir"]
    layouts_dir = root / project_data["layouts_dir"]
    chapters_dir = root / project_data["chapters_dir"]
    output_dir = root / project_data["output_dir"]
    output_archive_dir = output_dir / "archive"

    # Parse optional config blocks
    image_gen_config = None
    if "image_generation" in project_data:
        ig = project_data["image_generation"]
        image_gen_config = ImageGenerationConfig(
            backend=ig["backend"],
            model=ig["model"],
            quality=ig["quality"],
            reference_budget=ig["reference_budget"],
            thinking=ig.get("thinking", "medium"),
            seed=ig.get("seed"),
        )

    scene_prompt_config = None
    if "scene_prompt" in project_data:
        sp = project_data["scene_prompt"]
        scene_prompt_config = ScenePromptConfig(
            model=sp["model"],
            context_profile=sp["context_profile"],
        )

    validation_config = None
    if "validation" in project_data:
        val = project_data["validation"]
        _validate_weights_sum(val["weights"])
        validation_config = ValidationConfig(
            threshold=val["threshold"],
            weights=val["weights"],
            max_regeneration_attempts=val["max_regeneration_attempts"],
        )

    return ProjectConfig(
        project_id=project_data["project_id"],
        title=project_data["title"],
        version=project_data["version"],
        compiler_version=project_data["compiler_version"],
        project_root=root,
        style_path=style_path,
        characters_dir=characters_dir,
        environments_dir=environments_dir,
        layouts_dir=layouts_dir,
        chapters_dir=chapters_dir,
        output_dir=output_dir,
        output_archive_dir=output_archive_dir,
        schemas_dir=schemas_dir,
        style=style_data,
        image_generation=image_gen_config,
        scene_prompt=scene_prompt_config,
        validation=validation_config,
        notes=project_data.get("notes"),
    )
