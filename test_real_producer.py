#!/usr/bin/env python3
"""
Real API test for the Chapter Plan Producer.
Calls GPT-4o with structured output mode using the actual project context
(Alyssa + Hood, real environments, three layouts).
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline.config import load_config
from producer import ChapterPlanProducer

SYNOPSIS = """Chapter 1: First Light.

Alyssa wakes before dawn in her small apartment, the city still dark outside.
She pulls on her technical jacket and heads out into the cold morning streets,
moving through the waking city with purpose. She has a meeting to get to.

Hood spots her from a rooftop above — he's been tracking her movements for
days, trying to figure out who she is and why she's in New Bridgeton. He
follows at a distance, staying to the shadows and elevated walkways.

Alyssa arrives at a quiet bar that's just opening up. She meets a contact
inside who gives her a small, sealed package. As she steps back out into the
street, she senses she's being watched. She pauses at a corner, scanning
the rooftops, but Hood has already melted back into the city's geometry.

The chapter ends with Alyssa walking away, clutching the package, and Hood
watching from three blocks out — both aware something has begun."""

def main():
    print("=== Loading project config ===")
    config = load_config(".")
    print(f"  Project: {config.project_id}")
    print(f"  Characters dir: {config.characters_dir}")
    print(f"  Environments dir: {config.environments_dir}")
    print(f"  Layouts dir: {config.layouts_dir}")

    print("\n=== Creating Producer ===")
    producer = ChapterPlanProducer(config, model="gpt-4o")
    print(f"  Model: {producer.model}")
    print(f"  Schema: {producer.schema_path}")

    # Show what context will be injected
    print("\n=== Assembling project context ===")
    context = producer.assemble_context()
    print(f"  Characters: {[c['character_id'] for c in context['characters']]}")
    print(f"  Environments: {[e['environment_id'] for e in context['environments']]}")
    print(f"  Layouts: {[(l['layout_id'], l['panel_count']) for l in context['layouts']]}")
    print(f"  Style: {context['style']['style_id']}")

    print("\n=== Calling GPT-4o (real API call) ===")
    print("  This may take 10-30 seconds...")
    try:
        result = producer.produce(SYNOPSIS, chapter_number=1)
    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    attempts = result.get("attempts", 1)
    print(f"\n✓ Generation succeeded (attempt {attempts} of {producer.MAX_RETRIES})")
    print(f"  Model: {result['model']}")
    print(f"  File: {result['file_path']}")

    chapter_plan = result["chapter_plan"]
    print(f"\n=== Chapter Plan Summary ===")
    print(f"  Chapter ID: {chapter_plan['chapter_id']}")
    print(f"  Title: {chapter_plan['title']}")
    print(f"  Notes: {chapter_plan.get('notes', '(none)')}")
    print(f"  Pages: {len(chapter_plan['pages'])}")

    for page in chapter_plan["pages"]:
        print(f"\n  Page {page['page_id']} (layout: {page['layout']}, {len(page['panels'])} panels)")
        print(f"    Continuity: {page['continuity']}")
        for panel in page["panels"]:
            chars = ", ".join(panel["characters"]) if panel["characters"] else "(none)"
            print(f"    Panel {panel['position']}: [{chars}] env={panel['environment']} "
                  f"shot={panel['shot_type']} mood={panel['mood']}")
            desc = panel["description"]
            if len(desc) > 120:
                desc = desc[:117] + "..."
            print(f"      {desc}")

    print(f"\n=== Validating panel counts match layouts ===")
    layout_counts = {l["layout_id"]: l["panel_count"] for l in context["layouts"]}
    all_good = True
    for page in chapter_plan["pages"]:
        layout_id = page["layout"]
        expected = layout_counts.get(layout_id, "?")
        actual = len(page["panels"])
        status = "✓" if expected == actual else "✗"
        if expected != actual:
            all_good = False
        print(f"  {status} Page {page['page_id']}: layout={layout_id} expected={expected} actual={actual}")

    if all_good:
        print("\n✓ All panel counts match their layouts.")
    else:
        print("\n✗ Panel count mismatches detected!")

    print(f"\n=== Full YAML written to: {result['file_path']} ===")
    print("Done.")


if __name__ == "__main__":
    main()
