"""
cli.py — Command-line interface for the Graphic Novel Pipeline.

Provides subcommands for the full production workflow:
  produce   — Generate a Chapter Plan from a synopsis (GPT-4o)
  parse     — Parse a Chapter Plan into PanelSpecs (4-stage validation)
  generate  — Generate panel images (single panel, page, or full chapter)
  regenerate— Regenerate a panel with optional textual feedback
  list      — List panels, provenance records, or project status
  status    — Show generation status for a chapter
  validate  — Validate generated panels against project standards (GPT-4o vision)

Usage:
  python -m pipeline.cli <command> [options]

Per DESIGN.md §13.5: This is the ONLY module with print() statements and
CLI logic. All other modules return structured data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# -- Pipeline imports (lazy where they trigger API calls) -------------------

from pipeline.config import load_config, ProjectConfig
from pipeline.parser import ChapterPlanParser, ParseResult, ParserError
from pipeline.orchestrator import Orchestrator, PanelResult, PageResult
from pipeline.provenance import ProvenanceStore


# -- Formatting helpers ------------------------------------------------------

def _fmt_status(status: str) -> str:
    """Colourise status strings for terminal output."""
    if status == "success":
        return f"\033[32m{status}\033[0m"  # green
    elif status == "failure":
        return f"\033[31m{status}\033[0m"  # red
    return status


def _fmt_panel_id(panel_id: str) -> str:
    """Pretty-print a panel ID."""
    return f"\033[36m{panel_id}\033[0m"  # cyan


def _print_panel_result(result: PanelResult, indent: int = 0) -> None:
    """Print a single PanelResult."""
    pad = " " * indent
    status_str = _fmt_status(result.status)
    pid = _fmt_panel_id(result.panel_id)

    if result.status == "success":
        dims = ""
        if result.input_dimensions and result.output_dimensions:
            iw, ih = result.input_dimensions
            ow, oh = result.output_dimensions
            dims = f" [{iw}x{ih} → {ow}x{oh}]"
        pp = " (post-processed)" if result.post_processed else ""
        print(f"{pad}{pid}: {status_str}{pp}{dims}")
        if result.output_path:
            print(f"{pad}  → {result.output_path}")
    else:
        print(f"{pad}{pid}: {status_str}")
        if result.error:
            print(f"{pad}  error: {result.error}")


def _print_page_result(result: PageResult) -> None:
    """Print a PageResult and its constituent PanelResults."""
    print(f"\nPage {result.page_id} ({result.succeeded_count}/{len(result.panels)} succeeded):")
    for pr in result.panels:
        _print_panel_result(pr, indent=2)
    if result.failed_count > 0:
        print(f"\n  \033[33m{result.failed_count} panel(s) failed\033[0m")


# -- Command: produce --------------------------------------------------------

def cmd_produce(args: argparse.Namespace) -> int:
    """Generate a Chapter Plan from a synopsis using GPT-4o."""
    from producer import ChapterPlanProducer

    config = load_config(args.project)

    # Read synopsis from --synopsis or --synopsis-file
    if args.synopsis_file:
        with Path(args.synopsis_file).open("r") as f:
            synopsis = f.read().strip()
    elif args.synopsis:
        synopsis = args.synopsis
    else:
        print("Error: provide --synopsis or --synopsis-file")
        return 1

    if not synopsis:
        print("Error: synopsis is empty")
        return 1

    print(f"Producing Chapter {args.chapter}...")
    print(f"  Model: {args.model or 'gpt-4o'}")
    print(f"  Synopsis: {synopsis[:100]}{'...' if len(synopsis) > 100 else ''}")
    print()

    producer = ChapterPlanProducer(
        config,
        model=args.model or "gpt-4o",
    )

    try:
        result = producer.produce(synopsis, args.chapter)
    except Exception as e:
        print(f"\033[31mFailed: {e}\033[0m")
        return 1

    print(f"\033[32mSuccess!\033[0m")
    print(f"  Chapter plan: {result['file_path']}")
    print(f"  Model: {result['model']}")
    print(f"  Attempts: {result['attempts']}")
    return 0


# -- Command: parse ----------------------------------------------------------

def cmd_parse(args: argparse.Namespace) -> int:
    """Parse a Chapter Plan into PanelSpecs."""
    config = load_config(args.project)
    parser = ChapterPlanParser(config)

    print(f"Parsing Chapter {args.chapter}...")
    try:
        result = parser.parse_chapter(args.chapter)
    except ParserError as e:
        print(f"\033[31mParse error: {e}\033[0m")
        return 1
    except FileNotFoundError as e:
        print(f"\033[31mFile not found: {e}\033[0m")
        return 1

    print(f"\033[32mParsed {result.total_panels} panels\033[0m")
    print(f"  Chapter file: {result.chapter_file}")
    print()

    # Group by page
    pages: dict[str, list] = {}
    for p in result.panels:
        pid = p.panel_spec["page_id"]
        pages.setdefault(pid, []).append(p)

    for page_id in sorted(pages):
        panels = pages[page_id]
        print(f"  Page {page_id} ({len(panels)} panels):")
        for p in panels:
            spec = p.panel_spec
            print(f"    {_fmt_panel_id(spec['panel_id'])}  "
                  f"[{spec['shot_type']}/{spec['mood']}]  "
                  f"{spec['description'][:60]}...")
            print(f"      → {p.output_path}")

    if result.warnings:
        print(f"\n  \033[33mWarnings ({len(result.warnings)}):\033[0m")
        for w in result.warnings:
            print(f"    - {w}")

    return 0


# -- Command: generate -------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    """Generate panel images."""
    config = load_config(args.project)

    # Parse the chapter to get PanelSpecs
    parser = ChapterPlanParser(config)
    try:
        parse_result = parser.parse_chapter(args.chapter)
    except (ParserError, FileNotFoundError) as e:
        print(f"\033[31mParse error: {e}\033[0m")
        return 1

    # Filter to requested scope
    if args.panel:
        # Single panel
        matching = [p for p in parse_result.panels if p.panel_spec["panel_id"] == args.panel]
        if not matching:
            print(f"Error: panel '{args.panel}' not found in chapter {args.chapter}")
            return 1
        return _generate_single(config, matching[0].panel_spec, parse_result)
    elif args.page:
        # Single page — accept either a full page_id (e.g. "2_1") or a plain
        # page number (e.g. "1") which is matched against the page portion
        # of each panel's page_id (the segment after the last underscore).
        page_arg = args.page
        matching = [
            p for p in parse_result.panels
            if p.panel_spec["page_id"] == page_arg
            or p.panel_spec["page_id"].rsplit("_", 1)[-1] == page_arg
        ]
        if not matching:
            print(f"Error: page '{page_arg}' not found in chapter {args.chapter}")
            return 1
        return _generate_page(config, [p.panel_spec for p in matching])
    else:
        # Full chapter
        pages: dict[str, list] = {}
        for p in parse_result.panels:
            pages.setdefault(p.panel_spec["page_id"], []).append(p.panel_spec)
        return _generate_chapter(config, pages)


def _generate_single(
    config: ProjectConfig,
    panel_spec: dict[str, Any],
    parse_result: ParseResult,
) -> int:
    """Generate a single panel."""
    orch = Orchestrator(config)

    # Build surrounding descriptions from same-page neighbors
    page_id = panel_spec["page_id"]
    page_panels = [p for p in parse_result.panels if p.panel_spec["page_id"] == page_id]
    idx = next(i for i, p in enumerate(page_panels) if p.panel_spec["panel_id"] == panel_spec["panel_id"])

    surrounding: list[str] = []
    if idx > 0:
        surrounding.append(f"Previous panel: {page_panels[idx-1].panel_spec['description']}")
    if idx < len(page_panels) - 1:
        surrounding.append(f"Next panel: {page_panels[idx+1].panel_spec['description']}")

    print(f"Generating {_fmt_panel_id(panel_spec['panel_id'])}...")
    print(f"  Shot: {panel_spec['shot_type']}, Mood: {panel_spec['mood']}")
    print(f"  Size: {panel_spec['panel_geometry']['width_px']}x{panel_spec['panel_geometry']['height_px']}px")
    print()

    t0 = time.time()
    result = orch.generate_panel(panel_spec, surrounding_descriptions=surrounding)
    elapsed = time.time() - t0

    _print_panel_result(result)
    print(f"\n  Elapsed: {elapsed:.1f}s")
    return 0 if result.status == "success" else 1


def _generate_page(
    config: ProjectConfig,
    panels: list[dict[str, Any]],
) -> int:
    """Generate all panels on a single page."""
    orch = Orchestrator(config)
    page_id = panels[0]["page_id"]

    print(f"Generating page {page_id} ({len(panels)} panels)...")

    t0 = time.time()
    result = orch.generate_page(panels)
    elapsed = time.time() - t0

    _print_page_result(result)
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Succeeded: {result.succeeded_count}/{len(result.panels)}")

    return 0 if result.failed_count == 0 else 1


def _generate_chapter(
    config: ProjectConfig,
    pages: dict[str, list[dict[str, Any]]],
) -> int:
    """Generate all pages in a chapter."""
    orch = Orchestrator(config)

    total_panels = sum(len(p) for p in pages.values())
    print(f"Generating chapter ({len(pages)} pages, {total_panels} panels)...")
    print()

    total_succeeded = 0
    total_failed = 0
    t0 = time.time()

    for page_id in sorted(pages):
        panels = pages[page_id]
        print(f"--- Page {page_id} ({len(panels)} panels) ---")
        result = orch.generate_page(panels)
        _print_page_result(result)
        total_succeeded += result.succeeded_count
        total_failed += result.failed_count
        print()

    elapsed = time.time() - t0

    print(f"{'='*50}")
    print(f"Chapter complete: {total_succeeded}/{total_panels} panels succeeded")
    print(f"  Failed: {total_failed}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return 0 if total_failed == 0 else 1


# -- Command: regenerate -----------------------------------------------------

def cmd_regenerate(args: argparse.Namespace) -> int:
    """Regenerate a panel with optional user feedback."""
    config = load_config(args.project)

    # Find the panel — try parsing the chapter first
    parser = ChapterPlanParser(config)
    try:
        parse_result = parser.parse_chapter(args.chapter)
    except (ParserError, FileNotFoundError) as e:
        print(f"\033[31mParse error: {e}\033[0m")
        return 1

    matching = [p for p in parse_result.panels if p.panel_spec["panel_id"] == args.panel]
    if not matching:
        print(f"Error: panel '{args.panel}' not found in chapter {args.chapter}")
        return 1

    panel_spec = matching[0].panel_spec

    # Build surrounding descriptions
    page_id = panel_spec["page_id"]
    page_panels = [p for p in parse_result.panels if p.panel_spec["page_id"] == page_id]
    idx = next(i for i, p in enumerate(page_panels) if p.panel_spec["panel_id"] == args.panel)
    surrounding: list[str] = []
    if idx > 0:
        surrounding.append(f"Previous panel: {page_panels[idx-1].panel_spec['description']}")
    if idx < len(page_panels) - 1:
        surrounding.append(f"Next panel: {page_panels[idx+1].panel_spec['description']}")

    orch = Orchestrator(config)

    # Determine mode
    if args.feedback:
        mode = "full (with director's note)"
    elif args.full_pipeline:
        mode = "full (forced)"
    else:
        mode = "backend-only (reuse last scene prompt)"

    print(f"Regenerating {_fmt_panel_id(args.panel)}...")
    print(f"  Mode: {mode}")
    if args.feedback:
        print(f"  Feedback: \"{args.feedback}\"")
    print()

    t0 = time.time()
    result = orch.regenerate_panel(
        panel_spec,
        user_feedback=args.feedback,
        surrounding_descriptions=surrounding,
        full_pipeline=args.full_pipeline,
    )
    elapsed = time.time() - t0

    _print_panel_result(result)
    print(f"\n  Elapsed: {elapsed:.1f}s")
    return 0 if result.status == "success" else 1


# -- Command: list -----------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    """List panels or provenance records."""
    config = load_config(args.project)

    if args.what == "panels":
        return _list_panels(config, args)
    elif args.what == "provenance":
        return _list_provenance(config, args)
    else:
        print(f"Unknown list target: {args.what}")
        return 1


def _list_panels(config: ProjectConfig, args: argparse.Namespace) -> int:
    """List all panels in a chapter."""
    parser = ChapterPlanParser(config)
    try:
        result = parser.parse_chapter(args.chapter)
    except (ParserError, FileNotFoundError) as e:
        print(f"\033[31mParse error: {e}\033[0m")
        return 1

    prov = ProvenanceStore(config.output_dir)

    print(f"Chapter {args.chapter}: {result.total_panels} panels\n")

    pages: dict[str, list] = {}
    for p in result.panels:
        pages.setdefault(p.panel_spec["page_id"], []).append(p)

    for page_id in sorted(pages):
        panels = pages[page_id]
        print(f"Page {page_id} ({len(panels)} panels):")
        for p in panels:
            spec = p.panel_spec
            pid = spec["panel_id"]

            # Check provenance
            has_records = prov.has_records(pid)
            if has_records:
                latest = prov.get_latest_record(pid)
                attempt = latest.get("attempt_number", "?")
                status = latest.get("outcome", {}).get("status", "?")
                status_str = _fmt_status(status)
                print(f"  {_fmt_panel_id(pid)}  [{spec['shot_type']}/{spec['mood']}]  "
                      f"{status_str} (attempt {attempt})")
            else:
                print(f"  {_fmt_panel_id(pid)}  [{spec['shot_type']}/{spec['mood']}]  "
                      f"\033[90mnot generated\033[0m")
        print()

    return 0


def _list_provenance(config: ProjectConfig, args: argparse.Namespace) -> int:
    """List provenance records for a panel."""
    prov = ProvenanceStore(config.output_dir)

    if args.panel:
        records = prov.read_all(args.panel)
        if not records:
            print(f"No provenance records for {args.panel}")
            return 0
        print(f"Provenance for {_fmt_panel_id(args.panel)} ({len(records)} records):\n")
        for r in records:
            status = r.get("outcome", {}).get("status", "?")
            attempt = r.get("attempt_number", "?")
            timestamp = r.get("timestamp_utc", "?")
            print(f"  Attempt {attempt}: {_fmt_status(status)}  ({timestamp})")
            if status == "failure" and r.get("outcome", {}).get("error"):
                print(f"    error: {r['outcome']['error']}")
            pp = r.get("post_processing", {})
            if pp.get("crop", {}).get("applied"):
                dims = pp["crop"]
                print(f"    crop: {dims['input_dimensions']} → {dims['output_dimensions']} "
                      f"({dims['strategy']})")
            scene = r.get("scene_prompt", {})
            if scene.get("output"):
                print(f"    scene: {scene['output'][:80]}...")
            print()
    else:
        panels = prov.list_all_panels()
        if not panels:
            print("No provenance records found.")
            return 0
        print(f"Panels with provenance ({len(panels)}):\n")
        for pid in sorted(panels):
            latest = prov.get_latest_record(pid)
            status = latest.get("outcome", {}).get("status", "?")
            attempt = latest.get("attempt_number", "?")
            print(f"  {_fmt_panel_id(pid)}  {_fmt_status(status)}  (attempt {attempt})")

    return 0


# -- Command: status ---------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """Show generation status for a chapter."""
    config = load_config(args.project)
    parser = ChapterPlanParser(config)
    prov = ProvenanceStore(config.output_dir)

    try:
        result = parser.parse_chapter(args.chapter)
    except (ParserError, FileNotFoundError) as e:
        print(f"\033[31mParse error: {e}\033[0m")
        return 1

    total = result.total_panels
    generated = 0
    succeeded = 0
    failed = 0
    not_generated = 0

    for p in result.panels:
        pid = p.panel_spec["panel_id"]
        if not prov.has_records(pid):
            not_generated += 1
            continue
        generated += 1
        latest = prov.get_latest_record(pid)
        status = latest.get("outcome", {}).get("status", "?")
        if status == "success":
            succeeded += 1
        else:
            failed += 1

    print(f"Chapter {args.chapter} Status")
    print(f"{'='*40}")
    print(f"  Total panels:    {total}")
    print(f"  \033[32mSucceeded:       {succeeded}\033[0m")
    print(f"  \033[31mFailed:          {failed}\033[0m")
    print(f"  \033[90mNot generated:   {not_generated}\033[0m")
    print(f"  Generated total: {generated}")
    print()

    if failed > 0:
        print("Failed panels:")
        for p in result.panels:
            pid = p.panel_spec["panel_id"]
            if prov.has_records(pid):
                latest = prov.get_latest_record(pid)
                if latest.get("outcome", {}).get("status") == "failure":
                    print(f"  {_fmt_panel_id(pid)}: {latest['outcome'].get('error', 'unknown')}")

    if not_generated > 0:
        print("\nNot yet generated:")
        for p in result.panels:
            pid = p.panel_spec["panel_id"]
            if not prov.has_records(pid):
                print(f"  {_fmt_panel_id(pid)}")

    return 0



# -- validate command -------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """Validate generated panel(s) against project standards."""
    from pipeline.validation import create_validation_pipeline

    config = load_config(args.project)

    # Parse chapter to get PanelSpecs
    parser_obj = ChapterPlanParser(config)
    try:
        result = parser_obj.parse_chapter(args.chapter)
    except (ParserError, FileNotFoundError) as e:
        print(f"\033[31mParse error: {e}\033[0m")
        return 1

    # Filter to requested scope
    if args.panel:
        panels = [p for p in result.panels if p.panel_spec["panel_id"] == args.panel]
        if not panels:
            print(f"\033[31mPanel '{args.panel}' not found in chapter {args.chapter}\033[0m")
            return 1
    elif args.page:
        page_arg = args.page
        panels = [
            p for p in result.panels
            if p.panel_spec["page_id"] == page_arg
            or p.panel_spec["page_id"].rsplit("_", 1)[-1] == page_arg
        ]
        if not panels:
            print(f"\033[31mPage '{page_arg}' not found in chapter {args.chapter}\033[0m")
            return 1
    else:
        panels = result.panels

    # Create validation pipeline from config
    try:
        vp = create_validation_pipeline(config)
    except Exception as e:
        print(f"\033[31mValidation config error: {e}\033[0m")
        return 1

    output_dir = config.output_dir
    validated = 0
    skipped = 0
    failed = 0

    for p in panels:
        spec = p.panel_spec
        pid = spec["panel_id"]

        # Find the output file (latest attempt)
        candidates = sorted(output_dir.glob(f"{pid}_attempt_*.png"), reverse=True)
        if not candidates:
            print(f"  \033[90m{_fmt_panel_id(pid)}: no output file, skipped\033[0m")
            skipped += 1
            continue

        image_path = candidates[0]
        geometry = spec["panel_geometry"]

        # Gather character reference images from PanelSpec
        char_refs = []
        for char in spec.get("characters", []):
            char_id = char.get("character_id", "")
            display_name = char.get("display_name", char_id)
            for ref in char.get("references", []):
                ref_path = ref.get("file", "")
                # Resolve relative to project root
                full_path = config.project_root / ref_path
                if full_path.exists():
                    char_refs.append({
                        "path": str(full_path),
                        "label": f"{display_name} ({ref.get('purpose', 'reference')})",
                    })

        print(f"  Validating {_fmt_panel_id(pid)}...")

        try:
            vr = vp.validate_panel(
                image_path=image_path,
                panel_geometry=geometry,
                character_refs=char_refs if char_refs else None,
            )
        except Exception as e:
            print(f"  \033[31m{_fmt_panel_id(pid)}: ERROR — {e}\033[0m")
            failed += 1
            continue

        validated += 1

        # Print result
        if vr.layout_compliance == 0.0:
            print(f"    \033[31mStage 1 FAILED (layout_compliance=0.0)\033[0m")
            continue

        char_s = "—"
        if vr.character_consistency:
            char_s = f"{vr.character_consistency.score:.2f} ({vr.character_consistency.confidence})"

        style_s = "—"
        if vr.style_adherence:
            style_s = f"{vr.style_adherence.score:.2f} ({vr.style_adherence.confidence})"

        comp_s = "—"
        if vr.composite_score is not None:
            comp_s = f"{vr.composite_score:.3f}"

        accepted = vr.accepted_for_production
        verdict = "\033[32mACCEPTED\033[0m" if accepted else "\033[33mESCALATED\033[0m"

        print(f"    layout_compliance: 1.0")
        print(f"    character_consistency: {char_s}")
        print(f"    style_adherence:       {style_s}")
        print(f"    composite_score:       {comp_s}")
        print(f"    threshold:             {vr.threshold}")
        print(f"    verdict:               {verdict}")

        # Show observations if escalated
        if not accepted and vr.character_consistency:
            for obs in vr.character_consistency.observations:
                print(f"      char: {obs}")
        if not accepted and vr.style_adherence:
            for obs in vr.style_adherence.observations:
                print(f"      style: {obs}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Validation Summary: {validated} validated, {skipped} skipped, {failed} errors")
    if validated > 0:
        print(f"  (2 GPT-4o vision calls per validated panel = {validated * 2} total calls)")
    print(f"{'='*50}")

    return 0 if failed == 0 else 1

# -- Argument parser ---------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="pipeline.cli",
        description="Graphic Novel Pipeline — generate comic panels from chapter plans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Generate a chapter plan from a synopsis
  python -m pipeline.cli produce --chapter 1 --synopsis "Alyssa wakes up..."

  # Parse a chapter plan into PanelSpecs
  python -m pipeline.cli parse --chapter 1

  # Generate all panels for a chapter
  python -m pipeline.cli generate --chapter 1

  # Generate a single page
  python -m pipeline.cli generate --chapter 1 --page 1

  # Generate a single panel
  python -m pipeline.cli generate --chapter 1 --panel c01_pg1_l02_pn01

  # Regenerate a panel with feedback
  python -m pipeline.cli regenerate --chapter 1 --panel c01_pg1_l02_pn03 \\
      --feedback "Make the lighting warmer"

  # Regenerate backend-only (reuse last scene prompt, faster)
  python -m pipeline.cli regenerate --chapter 1 --panel c01_pg1_l02_pn01

  # List panels and their generation status
  python -m pipeline.cli list panels --chapter 1

  # Show provenance for a panel
  python -m pipeline.cli list provenance --panel c01_pg1_l02_pn03

  # Show chapter status summary
  python -m pipeline.cli status --chapter 1

  # Validate generated panels (GPT-4o vision scoring)
  python -m pipeline.cli validate --chapter 1 --page 1
  python -m pipeline.cli validate --chapter 1 --panel c01_pg1_l02_pn03
        """,
    )

    parser.add_argument(
        "--project",
        default=".",
        help="Path to the project root (default: current directory)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (pipeline internals)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- produce --
    p_produce = subparsers.add_parser(
        "produce",
        help="Generate a Chapter Plan from a synopsis (GPT-4o)",
    )
    p_produce.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_produce.add_argument("--synopsis", help="Chapter synopsis text")
    p_produce.add_argument("--synopsis-file", help="Path to a file containing the synopsis")
    p_produce.add_argument("--model", help="Override the LLM model (default: gpt-4o)")
    p_produce.set_defaults(func=cmd_produce)

    # -- parse --
    p_parse = subparsers.add_parser(
        "parse",
        help="Parse a Chapter Plan into PanelSpecs",
    )
    p_parse.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_parse.set_defaults(func=cmd_parse)

    # -- generate --
    p_gen = subparsers.add_parser(
        "generate",
        help="Generate panel images",
    )
    p_gen.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_gen.add_argument("--page", type=str, help="Generate only this page (e.g. 1 or 2_1)")
    p_gen.add_argument("--panel", help="Generate only this panel ID")
    p_gen.set_defaults(func=cmd_generate)

    # -- regenerate --
    p_regen = subparsers.add_parser(
        "regenerate",
        help="Regenerate a panel with optional feedback",
    )
    p_regen.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_regen.add_argument("--panel", required=True, help="Panel ID to regenerate")
    p_regen.add_argument("--feedback", help="Textual feedback for the regeneration (Director's Note)")
    p_regen.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Force full pipeline re-run (new scene prompt + image). "
             "Default: backend-only if no feedback, full if feedback provided.",
    )
    p_regen.set_defaults(func=cmd_regenerate)

    # -- list --
    p_list = subparsers.add_parser(
        "list",
        help="List panels or provenance records",
    )
    p_list.add_argument("what", choices=["panels", "provenance"], help="What to list")
    p_list.add_argument("--chapter", type=int, help="Chapter number (for listing panels)")
    p_list.add_argument("--panel", help="Panel ID (for listing provenance)")
    p_list.set_defaults(func=cmd_list)

    # -- status --
    p_status = subparsers.add_parser(
        "status",
        help="Show generation status for a chapter",
    )
    p_status.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_status.set_defaults(func=cmd_status)

    # -- validate --
    p_val = subparsers.add_parser(
        "validate",
        help="Validate generated panels against project standards (GPT-4o vision)",
    )
    p_val.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_val.add_argument("--page", type=str, help="Validate only this page (e.g. 1 or 2_1)")
    p_val.add_argument("--panel", help="Validate only this panel ID")
    p_val.set_defaults(func=cmd_validate)

    return parser


# -- Entry point -------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy library logs
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\033[33mInterrupted\033[0m")
        return 130
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        print(f"\033[31mError: {e}\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
