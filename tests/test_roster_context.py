"""
Tests for the shared roster context builder (src/roster_context.py) —
the exclusion-cue convention used by both producers (DESIGN.md §7).

All tests use the real project assets (alyssa, hood, enforcer, brute).
"""

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roster_context import CONCEALMENT_MARKER, build_character_context_lines

PROJECT_ROOT = Path(__file__).parent.parent


def load_char(character_id: str) -> dict:
    """Load a real character YAML and shape it as both producers do."""
    with (PROJECT_ROOT / "characters" / character_id / f"{character_id}.yaml").open("r") as f:
        d = yaml.safe_load(f)
    return {
        "character_id": d["character_id"],
        "display_name": d["display_name"],
        "physical_description": d["physical_description"],
        "costume_default": d["costumes"]["default"]["description"],
        "costume_variants": [
            {"variant_id": v["variant_id"], "description": v["description"]}
            for v in d.get("costumes", {}).get("variants", [])
        ],
        "exclusions": d.get("prompt_tokens", {}).get("exclusions", []),
    }


class TestConcealmentSuppression:

    def test_hood_hidden_features_not_injected(self):
        """Hood's 'rarely seen' hair/eyes must NOT appear as writing cues."""
        lines = build_character_context_lines(load_char("hood"))
        text = "\n".join(lines)
        assert CONCEALMENT_MARKER not in text
        assert "dark brown" not in text          # eye color is a forbidden cue
        # Build silhouette survives — it describes shape, not a hidden feature.
        assert "tall, lean, athletic" in lines[0]

    def test_enforcer_visible_eyes_survive(self):
        """No marker on the Enforcer's signature eyes — cue passes through."""
        lines = build_character_context_lines(load_char("enforcer"))
        assert "glowing blue 'X' shaped eyes" in lines[0]

    def test_alyssa_hair_survives(self):
        """Alyssa's canonical blonde hair carries no marker — cue passes through."""
        lines = build_character_context_lines(load_char("alyssa"))
        assert "blonde" in lines[0]


class TestIdentityRules:

    def test_exclusion_promoted_to_identity_rule(self):
        lines = build_character_context_lines(load_char("hood"))
        rules = [l for l in lines if l.lstrip().startswith("IDENTITY RULE")]
        assert len(rules) == 1
        # Exclusion text spliced verbatim + fixed amplification template.
        assert rules[0].lstrip().startswith("IDENTITY RULE — never show eyes or face.")
        assert "absolute in every panel" in rules[0]
        # Old trailing footnote format is gone.
        assert not any(l.lstrip().startswith("Exclusions:") for l in lines)

    def test_rule_precedes_costume_lines(self):
        lines = build_character_context_lines(load_char("hood"))
        rule_idx = next(i for i, l in enumerate(lines) if "IDENTITY RULE" in l)
        costume_idx = next(i for i, l in enumerate(lines) if "Default costume" in l)
        assert rule_idx < costume_idx

    def test_costume_variants_still_listed(self):
        lines = build_character_context_lines(load_char("hood"))
        assert any("Variant 'roadwork'" in l for l in lines)


class TestDeterminism:

    def test_same_input_byte_identical(self):
        c = load_char("hood")
        assert build_character_context_lines(c) == build_character_context_lines(c)
