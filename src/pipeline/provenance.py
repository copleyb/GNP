"""
provenance.py — Write-once audit log for panel generation attempts.

Storage: JSONL files in output/, one per panel.
  File naming: {panel_id}.provenance.jsonl
  One line per attempt, appended on each generation run.

Per DESIGN.md §11: Never mutates records — only appends.
Per DESIGN.md §13.5: Returns structured data, no print() statements, no CLI logic.

The Store is a dumb storage layer. It receives complete Generation Records
(assembled by the Orchestrator) and stores them. It does not compute hashes,
assemble records, or perform domain validation — that is the caller's job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# -- Required top-level fields in a Generation Record -----------------------

_REQUIRED_FIELDS = frozenset({
    "record_id",
    "panel_id",
    "attempt_number",
    "timestamp_utc",
    "compiler",
    "generation_request",
    "outcome",
})


class ProvenanceStore:
    """
    Write-once audit log of every generation attempt at the panel level.

    Encapsulates the JSONL storage backend. Migrating to SQLite later requires
    replacing this class's implementation only — no changes to the Prompt
    Compiler, Backend Adapter, or Validation Pipeline.
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialise the store, creating the output and archive directories if needed.

        Args:
            output_dir: Path to the output directory (from ProjectConfig).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.output_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # -- Path helpers -------------------------------------------------------

    def _provenance_path(self, panel_id: str) -> Path:
        """Return the path to the panel's provenance file."""
        return self.output_dir / f"{panel_id}.provenance.jsonl"

    def _all_panel_ids(self) -> list[str]:
        """List all panel_ids that have provenance files."""
        return sorted(
            f.name.replace(".provenance.jsonl", "")
            for f in self.output_dir.glob("*.provenance.jsonl")
            if f.is_file() and f.stat().st_size > 0
        )

    # -- Record validation --------------------------------------------------

    def _validate_record(self, record: dict[str, Any]) -> None:
        """Validate that a record has all required top-level fields."""
        if not isinstance(record, dict):
            raise TypeError("Generation Record must be a dict")

        missing = _REQUIRED_FIELDS - set(record.keys())
        if missing:
            raise ValueError(f"Generation Record missing required fields: {sorted(missing)}")

        if not isinstance(record["panel_id"], str) or not record["panel_id"]:
            raise ValueError("panel_id must be a non-empty string")

        if not isinstance(record["attempt_number"], int) or record["attempt_number"] < 1:
            raise ValueError("attempt_number must be a positive integer")

        if not isinstance(record["record_id"], str) or not record["record_id"]:
            raise ValueError("record_id must be a non-empty string")

    # -- Core operations (DESIGN.md §11) ------------------------------------

    def append(self, record: dict[str, Any]) -> None:
        """
        Append a Generation Record to the panel's provenance file.

        The record is written as a single JSON line. The file is created if
        it doesn't exist. Records are never modified or deleted — this is
        a write-once, append-only log.

        Args:
            record: A complete Generation Record dict. Must contain all
                    required top-level fields (see _REQUIRED_FIELDS).

        Raises:
            TypeError: If record is not a dict.
            ValueError: If record is missing required fields or has invalid values.
        """
        self._validate_record(record)
        path = self._provenance_path(record["panel_id"])
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self, panel_id: str) -> list[dict[str, Any]]:
        """
        Read all records for a panel, ordered by attempt_number.

        Args:
            panel_id: The panel identifier (e.g. "c02_pg03_l02_pn01").

        Returns:
            List of Generation Record dicts, sorted by attempt_number ascending.
            Empty list if no records exist.

        Raises:
            ValueError: If a line in the provenance file is not valid JSON.
        """
        path = self._provenance_path(panel_id)
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Malformed JSON at line {line_num} in {path.name}: {e}"
                    ) from e

        records.sort(key=lambda r: r.get("attempt_number", 0))
        return records

    # -- Attempt numbering (DESIGN.md §13) ----------------------------------

    def get_latest_attempt_number(self, panel_id: str) -> int:
        """
        Get the highest attempt_number from existing records.

        Returns 0 if no records exist (i.e. next attempt would be #1).
        """
        records = self.read_all(panel_id)
        if not records:
            return 0
        return max(r.get("attempt_number", 0) for r in records)

    def get_next_attempt_number(self, panel_id: str) -> int:
        """Get the next attempt number for a panel. Returns 1 if no records exist."""
        return self.get_latest_attempt_number(panel_id) + 1

    def get_latest_record(self, panel_id: str) -> dict[str, Any] | None:
        """
        Get the most recent record for a panel.

        Returns None if no records exist. Returns the record with the highest
        attempt_number.
        """
        records = self.read_all(panel_id)
        if not records:
            return None
        return records[-1]

    def has_records(self, panel_id: str) -> bool:
        """Check if any records exist for a panel."""
        path = self._provenance_path(panel_id)
        return path.exists() and path.stat().st_size > 0

    # -- Query methods (DESIGN.md §13.5 requirement 3) ---------------------

    def query_by_page(self, chapter: int, page: int) -> dict[str, list[dict[str, Any]]]:
        """
        Get all records for all panels on a specific page.

        Panel IDs follow the format c{chapter:02d}_pg{page:02d}_l{layout}_pn{position}.
        This method matches the chapter and page prefix.

        Args:
            chapter: Chapter number (e.g. 2).
            page: Page within chapter (e.g. 3).

        Returns:
            Dict mapping panel_id → list of records (sorted by attempt_number).
            Empty dict if no panels match.
        """
        prefix = f"c{chapter:02d}_pg{page:02d}_"
        result: dict[str, list[dict[str, Any]]] = {}
        for panel_id in self._all_panel_ids():
            if panel_id.startswith(prefix):
                result[panel_id] = self.read_all(panel_id)
        return result

    def query_by_chapter(self, chapter: int) -> dict[str, list[dict[str, Any]]]:
        """
        Get all records for all panels in a chapter.

        Args:
            chapter: Chapter number (e.g. 2).

        Returns:
            Dict mapping panel_id → list of records (sorted by attempt_number).
            Empty dict if no panels match.
        """
        prefix = f"c{chapter:02d}_pg"
        result: dict[str, list[dict[str, Any]]] = {}
        for panel_id in self._all_panel_ids():
            if panel_id.startswith(prefix):
                result[panel_id] = self.read_all(panel_id)
        return result

    def query_failed(self) -> list[dict[str, Any]]:
        """
        Get latest records for panels that are not accepted for production.

        A panel is considered "failed" if its latest record either:
        - Has outcome.status != "success", OR
        - Has validation.accepted_for_production == false (or missing validation)

        Returns:
            List of latest Generation Record dicts for failed panels.
        """
        failed: list[dict[str, Any]] = []
        for panel_id in self._all_panel_ids():
            latest = self.get_latest_record(panel_id)
            if latest is None:
                continue

            status = latest.get("outcome", {}).get("status", "")
            if status != "success":
                failed.append(latest)
                continue

            validation = latest.get("validation", {})
            if not validation.get("accepted_for_production", False):
                failed.append(latest)

        return failed

    def query_latest_per_panel(
        self, panel_ids: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        """
        Get the latest record for each specified panel.

        Args:
            panel_ids: List of panel IDs to query.

        Returns:
            Dict mapping panel_id → latest record (or None if no records).
        """
        return {pid: self.get_latest_record(pid) for pid in panel_ids}

    def list_all_panels(self) -> list[str]:
        """
        List all panel IDs that have at least one provenance record.

        Returns:
            Sorted list of panel_id strings.
        """
        return self._all_panel_ids()
