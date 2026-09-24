"""
roster_context.py — shared character roster line builder for the Chapter
Plan Producer and the Scene Producer.

Implements the DESIGN.md exclusion-cue convention for Producer prompt
context (both producers, identical output):

- Concealment marker: a physical_description field whose value contains
  "rarely seen" describes a hidden feature. It is NOT injected as a
  positive writing cue — the model never sees an eye/hair color it is
  forbidden to show. Visible signature features (e.g. Enforcer's glowing
  eyes) carry no marker and pass through unchanged.
- Identity rules: prompt_tokens.exclusions are promoted from a trailing
  footnote into per-rule IDENTITY RULE lines on the character's main
  roster entry, with a fixed amplification template. The exclusion text
  is spliced in verbatim from the YAML.

Everything here is deterministic: the same character dict always yields
byte-identical lines. No LLM calls. physical_description itself remains
unchanged in the asset YAML — the vision Validation stage still reads
the full description (it needs to know what to check); only the
Producer's writing context is redacted.
"""

from typing import Any

# Marker string in physical_description values marking a concealed feature.
CONCEALMENT_MARKER = "rarely seen"


def build_character_context_lines(character: dict[str, Any]) -> list[str]:
    """
    Build the deterministic roster lines for one character.

    Expected dict shape (produced identically by the chapter producer's
    assemble_context and the scene producer's _load_character_context):
    character_id, display_name, physical_description
    (build/hair/eyes), costume_default, costume_variants, exclusions.

    Returns a list of prompt lines:
      - header line (build + visible hair/eyes cues only),
      - one IDENTITY RULE line per exclusion (if any),
      - default costume line,
      - one line per costume variant.
    """
    pd = character["physical_description"]

    cues = [pd["build"]]
    if CONCEALMENT_MARKER not in pd["hair"].lower():
        cues.append(f"{pd['hair']} hair")
    if CONCEALMENT_MARKER not in pd["eyes"].lower():
        cues.append(f"{pd['eyes']} eyes")

    lines = [
        f"  - {character['character_id']} ({character['display_name']}): "
        f"{', '.join(cues)}."
    ]
    for exclusion in character.get("exclusions") or []:
        lines.append(
            f"    IDENTITY RULE — {exclusion}. This rule is absolute in "
            f"every panel: never describe or depict anything it forbids."
        )
    lines.append(f"    Default costume: {character['costume_default']}")
    for v in character.get("costume_variants") or []:
        lines.append(f"    Variant '{v['variant_id']}': {v['description']}")

    return lines
