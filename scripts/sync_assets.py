#!/usr/bin/env python3
"""
sync_assets.py — Reference image sync, verify, and manifest generation.

Three modes:

  generate   Scans characters/ and environments/ for PNG files, reads the
             entity YAMLs to map ref_ids and purposes, computes SHA-256
             hashes, and writes assets_manifest.yaml.

  sync       Reads assets_manifest.yaml, checks each file against the local
             copy. Missing or hash-mismatched files are copied from the
             source directory (configured in project.yaml as assets_source,
             or via --source flag).

  check      Reads assets_manifest.yaml, verifies every listed file exists
             locally and matches its recorded hash. Reports missing,
             mismatched, and unexpected files. Exits non-zero if anything
             is wrong — use as a pre-flight check before generation runs.

Usage:
  python scripts/sync_assets.py generate
  python scripts/sync_assets.py sync --source ../graphic-novel-assets
  python scripts/sync_assets.py sync          # uses assets_source from project.yaml
  python scripts/sync_assets.py check

The script is standalone — it only needs PyYAML and the standard library.
It does NOT import pipeline modules, so it works from a fresh clone before
PYTHONPATH or dependencies are set up.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml")
    sys.exit(1)


# -- Constants ---------------------------------------------------------------

MANIFEST_FILENAME = "assets_manifest.yaml"
CHUNK_SIZE = 65536  # 64KB for file hashing


# -- Helpers -----------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_project_config(project_root: Path) -> dict:
    """Load project.yaml and return as a dict."""
    config_path = project_root / "project.yaml"
    if not config_path.exists():
        print(f"Error: project.yaml not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_entity(path: Path) -> dict | None:
    """Load a character or environment YAML file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_entity_yaml(directory: Path) -> Path | None:
    """
    Find the YAML config file in an entity directory.
    Convention: one .yaml file per directory, named after the entity_id.
    """
    yaml_files = list(directory.glob("*.yaml"))
    if len(yaml_files) == 1:
        return yaml_files[0]
    # If multiple, prefer the one matching the directory name
    for yf in yaml_files:
        if yf.stem == directory.name:
            return yf
    return yaml_files[0] if yaml_files else None


def build_ref_map(entity_yaml: dict) -> dict[str, dict]:
    """
    Build a map of file_basename -> {ref_id, purpose} from an entity YAML.
    Keys are filenames like 'ref_front.png' (the basename of the file field).
    """
    ref_map = {}
    for ref in entity_yaml.get("references", []):
        file_field = ref.get("file", "")
        basename = os.path.basename(file_field)
        ref_map[basename] = {
            "ref_id": ref.get("ref_id", ""),
            "purpose": ref.get("purpose", ""),
        }
    return ref_map


# -- Generate mode -----------------------------------------------------------

def cmd_generate(project_root: Path) -> int:
    """Scan the project directory and generate assets_manifest.yaml."""
    config = load_project_config(project_root)
    characters_dir = project_root / config.get("characters_dir", "characters")
    environments_dir = project_root / config.get("environments_dir", "environments")

    assets = []

    # Scan characters
    if characters_dir.exists():
        for entity_dir in sorted(characters_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            entity_id = entity_dir.name
            yaml_path = find_entity_yaml(entity_dir)
            ref_map = {}
            if yaml_path:
                entity_data = load_yaml_entity(yaml_path)
                if entity_data:
                    ref_map = build_ref_map(entity_data)

            for png in sorted(entity_dir.glob("*.png")):
                rel_path = png.relative_to(project_root)
                rel_path_str = str(rel_path).replace("\\", "/")
                sha = sha256_file(png)
                size = png.stat().st_size
                basename = png.name
                ref_info = ref_map.get(basename, {})

                assets.append({
                    "path": rel_path_str,
                    "sha256": sha,
                    "size_bytes": size,
                    "asset_type": "character",
                    "entity_id": entity_id,
                    "ref_id": ref_info.get("ref_id", ""),
                    "purpose": ref_info.get("purpose", ""),
                })
                print(f"  {rel_path_str} ({size:,} bytes)")

    # Scan environments
    if environments_dir.exists():
        for entity_dir in sorted(environments_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            entity_id = entity_dir.name
            yaml_path = find_entity_yaml(entity_dir)
            ref_map = {}
            if yaml_path:
                entity_data = load_yaml_entity(yaml_path)
                if entity_data:
                    ref_map = build_ref_map(entity_data)

            for png in sorted(entity_dir.glob("*.png")):
                rel_path = png.relative_to(project_root)
                rel_path_str = str(rel_path).replace("\\", "/")
                sha = sha256_file(png)
                size = png.stat().st_size
                basename = png.name
                ref_info = ref_map.get(basename, {})

                assets.append({
                    "path": rel_path_str,
                    "sha256": sha,
                    "size_bytes": size,
                    "asset_type": "environment",
                    "entity_id": entity_id,
                    "ref_id": ref_info.get("ref_id", ""),
                    "purpose": ref_info.get("purpose", ""),
                })
                print(f"  {rel_path_str} ({size:,} bytes)")

    # Write manifest
    manifest = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_id": config.get("project_id", ""),
        "assets": assets,
    }

    manifest_path = project_root / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"\nManifest written: {manifest_path}")
    print(f"  {len(assets)} assets tracked")
    total_size = sum(a["size_bytes"] for a in assets)
    print(f"  {total_size:,} bytes total ({total_size / 1024 / 1024:.1f} MB)")

    return 0


# -- Sync mode ---------------------------------------------------------------

def cmd_sync(project_root: Path, source_dir: Path | None) -> int:
    """Sync images from external source to the project directory."""
    config = load_project_config(project_root)

    # Determine source directory
    if source_dir is None:
        source_path = config.get("assets_source", "")
        if not source_path:
            print("Error: No source directory specified.")
            print("  Either set 'assets_source' in project.yaml or use --source.")
            return 1
        source_dir = (project_root / source_path).resolve()
        if not source_dir.exists():
            source_dir = Path(source_path).resolve()
    else:
        source_dir = source_dir.resolve()

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1

    # Load manifest
    manifest_path = project_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        print("  Run 'python scripts/sync_assets.py generate' first.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    assets = manifest.get("assets", [])
    if not assets:
        print("Manifest is empty — nothing to sync.")
        return 0

    print(f"Source: {source_dir}")
    print(f"Manifest: {len(assets)} assets\n")

    synced = 0
    already_current = 0
    missing_from_source = 0
    errors = 0

    for asset in assets:
        rel_path = asset["path"]
        expected_sha = asset["sha256"]
        dest_path = project_root / rel_path
        source_path = source_dir / rel_path

        # Check if local file already matches
        if dest_path.exists():
            local_sha = sha256_file(dest_path)
            if local_sha == expected_sha:
                already_current += 1
                continue

        # Need to sync — check source
        if not source_path.exists():
            print(f"  MISSING: {rel_path} (not in source)")
            missing_from_source += 1
            continue

        # Verify source hash matches manifest
        source_sha = sha256_file(source_path)
        if source_sha != expected_sha:
            print(f"  WARN: {rel_path}")
            print(f"    manifest hash: {expected_sha}")
            print(f"    source hash:   {source_sha}")
            print(f"    Source file may have been updated. Syncing anyway.")
            # Update the destination with the source version
            # (The manifest may need regeneration)

        # Copy from source to destination
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.copy2(source_path, dest_path)
            print(f"  SYNCED: {rel_path}")
            synced += 1
        except Exception as e:
            print(f"  ERROR: {rel_path} — {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Sync complete: {synced} synced, {already_current} already current, "
          f"{missing_from_source} missing from source, {errors} errors")
    print(f"{'='*50}")

    return 1 if errors > 0 or missing_from_source > 0 else 0


# -- Check mode --------------------------------------------------------------

def cmd_check(project_root: Path) -> int:
    """Verify local files against the manifest. Pre-flight check."""
    manifest_path = project_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        print("  Run 'python scripts/sync_assets.py generate' first.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    assets = manifest.get("assets", [])
    if not assets:
        print("Manifest is empty — nothing to check.")
        return 0

    print(f"Manifest: {len(assets)} assets\n")

    ok = 0
    missing = 0
    hash_mismatch = 0
    mismatches = []

    for asset in assets:
        rel_path = asset["path"]
        expected_sha = asset["sha256"]
        local_path = project_root / rel_path

        if not local_path.exists():
            print(f"  MISSING: {rel_path}")
            missing += 1
            mismatches.append(rel_path)
            continue

        local_sha = sha256_file(local_path)
        if local_sha != expected_sha:
            print(f"  HASH MISMATCH: {rel_path}")
            print(f"    expected: {expected_sha}")
            print(f"    got:      {local_sha}")
            hash_mismatch += 1
            mismatches.append(rel_path)
            continue

        ok += 1

    # Check for unexpected PNGs (in the directory but not in the manifest)
    config = load_project_config(project_root)
    characters_dir = project_root / config.get("characters_dir", "characters")
    environments_dir = project_root / config.get("environments_dir", "environments")

    manifest_paths = {a["path"] for a in assets}
    unexpected = []

    for base_dir in [characters_dir, environments_dir]:
        if not base_dir.exists():
            continue
        for png in base_dir.rglob("*.png"):
            rel_path = str(png.relative_to(project_root)).replace("\\", "/")
            if rel_path not in manifest_paths:
                unexpected.append(rel_path)

    if unexpected:
        print(f"\n  UNEXPECTED files ({len(unexpected)}):")
        for p in unexpected:
            print(f"    {p}")

    print(f"\n{'='*50}")
    print(f"Check: {ok} OK, {missing} missing, {hash_mismatch} hash mismatches, "
          f"{len(unexpected)} unexpected")
    print(f"{'='*50}")

    if missing > 0:
        print(f"\n  {missing} file(s) missing — run 'sync' to fetch from external storage.")
    if hash_mismatch > 0:
        print(f"\n  {hash_mismatch} file(s) have been modified — regenerate the manifest")
        print(f"  or re-sync from source.")

    return 1 if missing > 0 or hash_mismatch > 0 else 0


# -- Main --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sync_assets",
        description="Sync, verify, and generate the reference image manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Generate manifest from current local files
  python scripts/sync_assets.py generate

  # Sync from external storage (path from project.yaml)
  python scripts/sync_assets.py sync

  # Sync from a specific source directory
  python scripts/sync_assets.py sync --source ../graphic-novel-assets

  # Verify all files are present and match (pre-flight check)
  python scripts/sync_assets.py check
        """,
    )

    parser.add_argument(
        "--project",
        default=".",
        help="Path to project root (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate", help="Scan local files and write assets_manifest.yaml")
    p_gen.set_defaults(func=lambda a: cmd_generate(Path(a.project).resolve()))

    p_sync = subparsers.add_parser("sync", help="Copy images from external source to project")
    p_sync.add_argument("--source", help="Source directory (overrides assets_source in project.yaml)")
    p_sync.set_defaults(func=lambda a: cmd_sync(Path(a.project).resolve(),
                                                Path(a.source).resolve() if a.source else None))

    p_check = subparsers.add_parser("check", help="Verify local files against manifest")
    p_check.set_defaults(func=lambda a: cmd_check(Path(a.project).resolve()))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
