"""
templates.py — Template Registry (Scope Redesign, DESIGN.md §16.2).

Loads and validates the three template menu files and serves as the single
source of panel/strip/layout definitions for the Scene Parser.

The registry model (per DESIGN.md §16.2):
  - templates/panels.yaml   — named size presets: width + height, nothing more.
  - templates/strips.yaml   — ordered panel refs. No strip-level geometry:
                              strip height is DERIVED (max of its panels).
  - templates/layouts.yaml   — ordered strip/panel refs; one full page.
  - Menu files may cross-reference each other (strips/layouts reference panels).

Validation is eager and loud: the registry refuses to load a malformed or
internally-inconsistent menu. This is the silent-breakage guard — a scene
referencing a missing template must fail at parse time, and the best way to
guarantee that is to guarantee the menu itself is complete and coherent.

Per DESIGN.md §13.5: returns structured data, no print() statements, no CLI logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -- Exceptions ---------------------------------------------------------------

class TemplateError(Exception):
    """Base exception for all Template Registry errors."""
    pass


class TemplateLoadError(TemplateError):
    """A template menu file is missing, unreadable, or malformed."""
    pass


class TemplateValidationError(TemplateError):
    """A template menu entry fails structural or cross-reference validation."""
    pass


# -- Data types ---------------------------------------------------------------

@dataclass(frozen=True)
class PanelPreset:
    """A named panel size preset. Width and height, nothing more."""
    panel_id: str
    width_px: int
    height_px: int


@dataclass(frozen=True)
class StripMember:
    """One panel entry within a strip: a preset ref, optionally floating.

    Floating members carry explicit coords relative to the strip's top-left
    corner. Floats do not participate in horizontal flow and do not affect
    the strip's derived height (they are overlays).
    """
    panel_id: str
    position: int
    floating: tuple[int, int] | None = None


@dataclass(frozen=True)
class LayoutMember:
    """One element entry within a layout: a strip or panel ref, optionally
    floating.

    Floating members carry explicit coords relative to the page's top-left
    margin (usable-area origin). Floats do not participate in vertical flow.
    """
    ref_id: str
    kind: str                     # "strip" | "panel"
    strip_index: int              # st slot within the layout (floats included)
    floating: tuple[int, int] | None = None


# -- Internal validation helpers ----------------------------------------------

def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TemplateLoadError(f"Template menu file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise TemplateLoadError(f"{path.name}: invalid YAML — {e}") from e
    if not isinstance(data, dict):
        raise TemplateLoadError(f"{path.name}: top level must be a mapping")
    return data


# -- Template Registry --------------------------------------------------------

class TemplateRegistry:
    """
    Loads and validates the template menu files, then serves lookups.

    All three files load and validate at construction time. Any malformed
    entry, duplicate, or dangling cross-reference raises immediately.

    Args:
        templates_dir: Path to the templates/ directory containing
            panels.yaml, strips.yaml, and layouts.yaml.
    """

    def __init__(self, templates_dir: Path):
        self.templates_dir = Path(templates_dir)
        self._panels: dict[str, PanelPreset] = {}
        self._strips: dict[str, list[StripMember]] = {}
        self._layouts: dict[str, list[LayoutMember]] = {}
        self._load_panels()
        self._load_strips()
        self._load_layouts()

    # -- Loading -------------------------------------------------------------

    def _load_panels(self) -> None:
        data = _load_yaml_file(self.templates_dir / "panels.yaml")
        for panel_id, entry in data.items():
            if not isinstance(panel_id, str) or not panel_id:
                raise TemplateValidationError(
                    "panels.yaml: preset names must be non-empty strings"
                )
            if not isinstance(entry, dict):
                raise TemplateValidationError(
                    f"panels.yaml / {panel_id}: must be a mapping of "
                    f"width_px and height_px"
                )
            extra_keys = set(entry) - {"width_px", "height_px"}
            if extra_keys:
                raise TemplateValidationError(
                    f"panels.yaml / {panel_id}: unexpected fields "
                    f"{sorted(extra_keys)} — size presets are width + "
                    f"height, nothing more"
                )
            if not _is_positive_int(entry.get("width_px")) or not _is_positive_int(entry.get("height_px")):
                raise TemplateValidationError(
                    f"panels.yaml / {panel_id}: width_px and height_px "
                    f"must be positive integers"
                )
            self._panels[panel_id] = PanelPreset(
                panel_id=panel_id,
                width_px=entry["width_px"],
                height_px=entry["height_px"],
            )

    def _load_strips(self) -> None:
        data = _load_yaml_file(self.templates_dir / "strips.yaml")
        strips = data.get("strips")
        if not isinstance(strips, dict) or not strips:
            raise TemplateValidationError(
                "strips.yaml: must contain a non-empty 'strips' mapping"
            )
        for strip_id, strip_def in strips.items():
            members = self._validate_member_list(
                container=f"strips.yaml / {strip_id}",
                raw=strip_def,
                list_key="panels",
                valid_refs=set(self._panels),
                ref_key="panel",
            )
            self._strips[strip_id] = members

    def _load_layouts(self) -> None:
        data = _load_yaml_file(self.templates_dir / "layouts.yaml")
        layouts = data.get("layouts")
        if not isinstance(layouts, dict) or not layouts:
            raise TemplateValidationError(
                "layouts.yaml: must contain a non-empty 'layouts' mapping"
            )
        for layout_id, layout_def in layouts.items():
            elements = layout_def.get("elements") if isinstance(layout_def, dict) else None
            if not isinstance(elements, list) or not elements:
                raise TemplateValidationError(
                    f"layouts.yaml / {layout_id}: must contain a non-empty "
                    f"'elements' list"
                )
            members: list[LayoutMember] = []
            for idx, element in enumerate(elements, start=1):
                if not isinstance(element, dict):
                    raise TemplateValidationError(
                        f"layouts.yaml / {layout_id} / element {idx}: must "
                        f"be a mapping"
                    )
                ref_key = next((k for k in ("strip", "panel") if k in element), None)
                if ref_key is None:
                    raise TemplateValidationError(
                        f"layouts.yaml / {layout_id} / element {idx}: must "
                        f"reference a 'strip' or a 'panel'"
                    )
                ref_id = element[ref_key]
                valid_refs = set(self._strips) if ref_key == "strip" else set(self._panels)
                if ref_id not in valid_refs:
                    raise TemplateValidationError(
                        f"layouts.yaml / {layout_id} / element {idx}: unknown "
                        f"{ref_key} '{ref_id}'"
                    )
                floating = self._extract_floating(
                    container=f"layouts.yaml / {layout_id} / element {idx}",
                    element=element,
                )
                members.append(LayoutMember(
                    ref_id=ref_id,
                    kind=ref_key,
                    strip_index=idx,
                    floating=floating,
                ))
            self._layouts[layout_id] = members

    def _validate_member_list(
        self,
        container: str,
        raw: Any,
        list_key: str,
        valid_refs: set[str],
        ref_key: str,
    ) -> list[StripMember]:
        """Validate a strip's panel list. Returns StripMembers in order."""
        if not isinstance(raw, dict):
            raise TemplateValidationError(
                f"{container}: must be a mapping with a '{list_key}' list"
            )
        items = raw.get(list_key)
        if not isinstance(items, list) or not items:
            raise TemplateValidationError(
                f"{container}: must contain a non-empty '{list_key}' list"
            )
        members: list[StripMember] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict) or ref_key not in item:
                raise TemplateValidationError(
                    f"{container} / {list_key} item {idx}: must be a "
                    f"mapping with a '{ref_key}' ref"
                )
            ref_id = item[ref_key]
            if ref_id not in valid_refs:
                raise TemplateValidationError(
                    f"{container} / {list_key} item {idx}: unknown "
                    f"{ref_key} '{ref_id}'"
                )
            floating = self._extract_floating(
                container=f"{container} / item {idx}",
                element=item,
            )
            members.append(StripMember(
                panel_id=ref_id,
                position=idx,
                floating=floating,
            ))
        return members

    def _extract_floating(self, container: str, element: dict[str, Any]) -> tuple[int, int] | None:
        """Extract and validate an optional floating {x, y} coordinate."""
        if "floating" not in element:
            return None
        fl = element["floating"]
        if not isinstance(fl, dict) or set(fl) != {"x", "y"}:
            raise TemplateValidationError(
                f"{container}: floating must be a mapping of exactly "
                f"'x' and 'y'"
            )
        for axis in ("x", "y"):
            v = fl[axis]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise TemplateValidationError(
                    f"{container}: floating {axis} must be a non-negative "
                    f"integer"
                )
        return (int(fl["x"]), int(fl["y"]))

    # -- Public API ----------------------------------------------------------

    def panel_count(self) -> int:
        return len(self._panels)

    def strip_count(self) -> int:
        return len(self._strips)

    def layout_count(self) -> int:
        return len(self._layouts)

    def get_panel(self, panel_id: str) -> PanelPreset:
        """Return a panel size preset by ID. Raises if unknown."""
        try:
            return self._panels[panel_id]
        except KeyError:
            raise TemplateValidationError(
                f"Unknown panel preset '{panel_id}'"
            ) from None

    def has_panel(self, panel_id: str) -> bool:
        return panel_id in self._panels

    def has_strip(self, strip_id: str) -> bool:
        return strip_id in self._strips

    def has_layout(self, layout_id: str) -> bool:
        return layout_id in self._layouts

    def get_strip(self, strip_id: str) -> list[StripMember]:
        """Return a strip's member list in order. Raises if unknown."""
        try:
            return self._strips[strip_id]
        except KeyError:
            raise TemplateValidationError(
                f"Unknown strip '{strip_id}'"
            ) from None

    def get_layout(self, layout_id: str) -> list[LayoutMember]:
        """Return a layout's member list in order. Raises if unknown."""
        try:
            return self._layouts[layout_id]
        except KeyError:
            raise TemplateValidationError(
                f"Unknown layout '{layout_id}'"
            ) from None
