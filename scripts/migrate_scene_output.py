#!/usr/bin/env python3
"""
One-time migration: add the chapter_tag display prefix to scene panel outputs.

Renames output PNGs  s{scene}_l..._attempt_NNN.png  ->  {tag}_s{scene}_l..._attempt_NNN.png
(includes output/archive/), and patches every matching path stored inside
{scene}_*.provenance.jsonl records (outcome.output_file and any nested
references), so provenance keeps pointing at real files.

Does NOT touch:
- {panel_id}.panelspec.json       (tag-free identity, by design)
- {panel_id}.provenance.jsonl     (store filename stays tag-free, by design)
- record_id / panel_id fields     (identity is tag-free)

Usage (Windows):
    set PYTHONPATH=src
    python scripts\\migrate_scene_output.py --scene s702 --tag c01          (dry-run, default)
    python scripts\\migrate_scene_output.py --scene s702 --tag c01 --apply  (execute)

Run the dry-run first; it prints every planned change and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _patch_record(record: dict, tag: str, scene: str) -> int:
    """Walk a provenance record and patch path-like strings in place. Returns patch count."""
    patches = 0

    def walk(node):
        nonlocal patches
        if isinstance(node, dict):
            for key, val in node.items():
                if isinstance(val, str):
                    for prefix in ("output/", "output/archive/"):
                        if val.startswith(prefix):
                            rest = val[len(prefix):]
                            if rest.startswith(f"{scene}_") and not rest.startswith(f"{tag}_{scene}_"):
                                node[key] = f"{prefix}{tag}_{rest}"
                                patches += 1
                                break
                else:
                    walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(record)
    return patches


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", required=True, help="Scene ID, e.g. s702")
    ap.add_argument("--tag", required=True, help="Chapter tag to add, e.g. c01")
    ap.add_argument("--project", default=".", help="Path to project root (default: current dir)")
    ap.add_argument("--apply", action="store_true", help="Execute changes (default is dry-run)")
    args = ap.parse_args()

    project = Path(args.project)
    output_dir = project / "output"
    if not output_dir.is_dir():
        print(f"Error: {output_dir} does not exist.")
        return 1

    # --- 1. PNG renames (output/ and output/archive/) ---
    pngs: list[tuple[Path, Path]] = []
    for sub in ("", "archive"):
        d = output_dir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob(f"{args.scene}_*_attempt_*.png")):
            pngs.append((f, f.with_name(f"{args.tag}_{f.name}")))

    # --- 2. Provenance patches ---
    prov_files = sorted(output_dir.rglob(f"{args.scene}_*.provenance.jsonl"))

    mode = "APPLY" if args.apply else "DRY-RUN - no changes will be made"
    print(f"Migration plan ({mode}):")
    print(f"  Scene: {args.scene}   Tag to add: {args.tag}")
    print()
    print(f"  PNG renames ({len(pngs)}):")
    for src, dst in pngs:
        print(f"    {src.relative_to(project)}  ->  {dst.relative_to(project)}")

    total_patches = 0
    print(f"  Provenance files to patch ({len(prov_files)}):")
    for pf in prov_files:
        patches = 0
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                patches += _patch_record(json.loads(line), args.tag, args.scene)
        total_patches += patches
        print(f"    {pf.relative_to(project)}: {patches} path patch(es)")

    print()
    print(f"  Total: {len(pngs)} renames, {total_patches} provenance path patches.")

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to execute.")
        return 0

    # --- Apply ---
    for src, dst in pngs:
        if dst.exists():
            print(f"  SKIP (target exists): {dst}")
            continue
        src.rename(dst)

    for pf in prov_files:
        patched_lines = []
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                _patch_record(rec, args.tag, args.scene)
                patched_lines.append(json.dumps(rec))
            else:
                patched_lines.append(line)
        tmp = pf.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")
        tmp.replace(pf)

    print("\nApplied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
