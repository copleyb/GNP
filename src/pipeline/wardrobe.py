"""
Wardrobe — character costume resolution module.

Handles the resolution of raw character YAML into PanelSpec-ready dicts:
costume variant selection, identity string composition, and reference
image filtering by costume variant.

Used by the Parser (at parse time) and the Orchestrator (at regeneration
time when --costume overrides are applied).
"""

from __future__ import annotations

import warnings
import yaml
from pathlib import Path
from typing import Any


class Wardrobe:
    """
    Resolves character data from raw YAML into PanelSpec-ready dicts.

    Encapsulates costume variant selection, identity string composition,
    and reference image filtering. Uses internal caching for character YAML.

    Args:
        characters_dir: Path to the characters/ directory in the project.
    """

    def __init__(self, characters_dir: Path) -> None:
        self.characters_dir = characters_dir
        self._character_cache: dict[str, dict[str, Any]] = {}

    # -- Asset loading (cached) ----------------------------------------------

    def load_character(self, character_id: str) -> dict[str, Any]:
        """Load and cache a character YAML by ID."""
        if character_id not in self._character_cache:
            char_file = self.characters_dir / character_id / f"{character_id}.yaml"
            if not char_file.exists():
                raise FileNotFoundError(f"Character file not found: {char_file}")
            with char_file.open("r", encoding="utf-8") as f:
                self._character_cache[character_id] = yaml.safe_load(f)
        return self._character_cache[character_id]

    # -- Character resolution ------------------------------------------------

    def resolve_character(
        self, character_id: str, costume_variant: str | None = None
    ) -> dict[str, Any]:
        """
        Resolve a character ID to its full data for PanelSpec embedding.

        Composes the identity string (costume-agnostic identity + costume
        description), filters references by costume variant, and embeds
        costume_variant for provenance.

        Args:
            character_id: The character to resolve.
            costume_variant: Optional variant_id from the panel's costume field.
                             If None, uses the character's default costume.

        Returns:
            A dict with character_id, display_name, prompt_tokens (with
            composed identity), costume_variant, and filtered references.
        """
        char_data = self.load_character(character_id)

        # Resolve costume
        costume_desc, resolved_variant = self._resolve_costume(
            char_data, costume_variant
        )

        # Compose identity string: costume-agnostic identity + costume description
        base_identity = char_data["prompt_tokens"]["identity"]
        composed_identity = f"{base_identity}. Currently wearing: {costume_desc}"

        # Filter references by costume variant
        filtered_refs = self._filter_refs_by_costume(
            char_data.get("references", []), resolved_variant
        )

        return {
            "character_id": char_data["character_id"],
            "display_name": char_data["display_name"],
            "prompt_tokens": {
                **char_data["prompt_tokens"],
                "identity": composed_identity,
            },
            "costume_variant": resolved_variant,
            "references": filtered_refs,
        }

    def resolve_character_in_panel(
        self,
        character_id: str,
        costume_variant: str,
        panel_references: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Re-resolve a character for a different costume variant at regeneration
        time, using the character's source YAML for identity and reference data.

        Unlike resolve_character (which starts from a character ID), this
        method is designed to be called from the Orchestrator when a PanelSpec
        already exists but needs a costume swap. It loads the raw character
        YAML, resolves the new costume, and returns a fresh character dict
        suitable for substituting into a PanelSpec.

        Args:
            character_id: The character to re-resolve.
            costume_variant: The new costume variant to apply.
            panel_references: The character's current references from the
                              PanelSpec (used as fallback if source YAML refs
                              for the variant are not found).

        Returns:
            A dict with character_id, display_name, prompt_tokens (with
            composed identity for the new costume), costume_variant, and
            filtered references for the new variant.
        """
        char_data = self.load_character(character_id)

        # Resolve the new costume
        costume_desc, resolved_variant = self._resolve_costume(
            char_data, costume_variant
        )

        # Compose new identity string
        base_identity = char_data["prompt_tokens"]["identity"]
        composed_identity = f"{base_identity}. Currently wearing: {costume_desc}"

        # Filter references from the SOURCE YAML by the new costume variant
        source_refs = char_data.get("references", [])
        filtered_refs = self._filter_refs_by_costume(source_refs, resolved_variant)

        # If no refs matched the variant (already warned inside _filter_refs_by_costume),
        # fall back to the panel's existing references
        if not filtered_refs or filtered_refs == source_refs:
            filtered_refs = panel_references

        return {
            "character_id": char_data["character_id"],
            "display_name": char_data["display_name"],
            "prompt_tokens": {
                **char_data["prompt_tokens"],
                "identity": composed_identity,
            },
            "costume_variant": resolved_variant,
            "references": filtered_refs,
        }

    # -- Costume resolution (internal) --------------------------------------

    def _resolve_costume(
        self, char_data: dict[str, Any], variant_id: str | None
    ) -> tuple[str, str]:
        """
        Resolve a costume variant to its description.

        Returns (description, resolved_variant_id).
        If variant_id is None, uses the default costume.
        If variant_id doesn't match any variant, falls back to default with a warning.
        """
        costumes = char_data.get("costumes", {})
        default_desc = costumes.get("default", {}).get("description", "")

        if variant_id is None:
            return default_desc, "default"

        # Look for matching variant
        for variant in costumes.get("variants", []):
            if variant["variant_id"] == variant_id:
                return variant["description"], variant_id

        # Fallback: variant not found
        warnings.warn(
            f"Character '{char_data['character_id']}': costume variant "
            f"'{variant_id}' not found. Falling back to default costume."
        )
        return default_desc, "default"

    def _filter_refs_by_costume(
        self, references: list[dict[str, Any]], costume_variant: str
    ) -> list[dict[str, Any]]:
        """
        Filter reference images by costume variant.

        Each ref may have an optional 'costume' field (defaults to 'default').
        Only refs matching the selected costume variant are returned.
        If no refs match the variant, falls back to default-costume refs with a warning.
        """
        matching = [
            ref for ref in references
            if ref.get("costume", "default") == costume_variant
        ]

        if matching:
            return matching

        # Fallback: no refs match the variant, use default refs
        if costume_variant != "default":
            default_refs = [
                ref for ref in references
                if ref.get("costume", "default") == "default"
            ]
            if default_refs:
                warnings.warn(
                    f"No reference images found for costume variant "
                    f"'{costume_variant}'. Falling back to default costume references."
                )
                return default_refs

        # Last resort: return all refs (no filtering possible)
        return references
