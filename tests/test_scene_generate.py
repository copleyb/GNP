"""Tests for scene-scoped generation context (element-grouped surrounding panels).

Covers pipeline.cli._scene_surrounding — the context-window builder for
`generate --scene` (Scope Redesign, pre-cutover addendum, DESIGN.md §16.12).
"""

from types import SimpleNamespace

from pipeline.cli import _element_of, _scene_surrounding


def _panel(panel_id: str, description: str | None = None) -> SimpleNamespace:
    """Fake parse-result panel: only panel_id and description matter here."""
    return SimpleNamespace(panel_spec={
        "panel_id": panel_id,
        "description": description or f"desc:{panel_id}",
    })


# Reading-order fixture: element 1 = 2-strip layout (2+1 panels),
# element 2 = 1-strip layout (2 panels). Mirrors s702's shape.
PANELS = [
    _panel("s702_l01_st01_pn01"),
    _panel("s702_l01_st01_pn02"),
    _panel("s702_l01_st02_pn01"),
    _panel("s702_l02_st01_pn01"),
    _panel("s702_l02_st01_pn02"),
]


class TestElementOf:
    def test_element_key(self):
        assert _element_of("s702_l01_st01_pn01") == "s702_l01"
        assert _element_of("s999_l12_st03_pn02") == "s999_l12"


class TestSceneSurrounding:
    def test_middle_panel_gets_same_element_prev_and_next(self):
        out = _scene_surrounding(PANELS, "s702_l01_st01_pn02")
        assert out == [
            "Previous panel: desc:s702_l01_st01_pn01",
            "Next panel: desc:s702_l01_st02_pn01",
        ]

    def test_first_panel_of_scene_has_no_prev(self):
        out = _scene_surrounding(PANELS, "s702_l01_st01_pn01")
        assert out == ["Next panel: desc:s702_l01_st01_pn02"]

    def test_element_start_gets_cross_element_prev_handoff(self):
        # First panel of element 2: previous = LAST panel of element 1
        # (cross-element handoff) + same-element next as usual
        out = _scene_surrounding(PANELS, "s702_l02_st01_pn01")
        assert out == [
            "Previous panel: desc:s702_l01_st02_pn01",
            "Next panel: desc:s702_l02_st01_pn02",
        ]

    def test_last_panel_of_scene_has_no_next(self):
        out = _scene_surrounding(PANELS, "s702_l02_st01_pn02")
        assert out == ["Previous panel: desc:s702_l02_st01_pn01"]

    def test_strip_boundary_within_element_is_seamless(self):
        # Last panel of strip 1, single-panel strip 2: Prev crosses strips
        # within the element (same element); NO cross-element Next by design.
        out = _scene_surrounding(PANELS, "s702_l01_st02_pn01")
        assert out == ["Previous panel: desc:s702_l01_st01_pn02"]

    def test_last_panel_of_element_has_no_cross_element_next(self):
        # Element 1's final panel: no Next — cross-element context is
        # Previous-side only (mirrors the producer's WHERE-WE-LEFT-OFF).
        out = _scene_surrounding(PANELS, "s702_l01_st02_pn01")
        assert all(not s.startswith("Next") for s in out)

    def test_single_panel_element_gets_only_cross_element_prev(self):
        panels = [
            _panel("s100_l01_st01_pn01"),
            _panel("s100_l02_st01_pn01"),  # single-panel element
            _panel("s100_l03_st01_pn01"),  # single-panel element
        ]
        out = _scene_surrounding(panels, "s100_l03_st01_pn01")
        assert out == ["Previous panel: desc:s100_l02_st01_pn01"]

    def test_single_panel_scene_has_empty_context(self):
        out = _scene_surrounding([_panel("s100_l01_st01_pn01")], "s100_l01_st01_pn01")
        assert out == []

    def test_panels_list_may_be_unsorted_helper_uses_given_order(self):
        # The caller sorts; the helper itself just walks the list it gets.
        shuffled = list(reversed(PANELS))
        out = _scene_surrounding(shuffled, "s702_l01_st01_pn01")
        assert out == ["Previous panel: desc:s702_l01_st01_pn02"]
