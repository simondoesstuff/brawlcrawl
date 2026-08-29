"""Tests for sixth_pick_annotations and render_brawler_table two-grid split."""

from dataclasses import dataclass, field
from io import StringIO

import pytest
from rich.console import Console

from pick.constants import ANN_BRAIN, ANN_X, ANNOTATION_SLOT, PICKRATE_HIGH_Z, PICKRATE_LOW_Z, WIN_PROB_THRESHOLD
from pick.display import render_brawler_table
from pick.main import sixth_pick_annotations
from pick.score import BrawlerInfo, DraftContext, EventInfo


# ── minimal stubs ─────────────────────────────────────────────────────────────

def _brawler(bid: int, name: str = "COLT") -> BrawlerInfo:
    return BrawlerInfo(id=bid, name=name, brawler_class="Damage Dealer", char_idx=bid, meta=(0, 0, 0))


def _event(eid: int = 15000005) -> EventInfo:
    return EventInfo(id=eid, mode="bounty", mode_id=3, map_name="Shooting Star", event_idx=0, mode_idx=0)


def _ctx(pickrates: dict[int, dict[int, float]]) -> DraftContext:
    return DraftContext(  # type: ignore[call-arg]
        model=None,  # type: ignore[arg-type]
        brawlers=[],
        events=[],
        brawler_names=[],
        map_names=[],
        pickrates=pickrates,
        _char_idxs=None,  # type: ignore[arg-type]
        _char_metas=None,  # type: ignore[arg-type]
        _brawler_by_char_idx={},
        _brawler_by_name={},
    )


# ── sixth_pick_annotations ────────────────────────────────────────────────────

def test_brain_icon_when_high_winprob_low_pickrate():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_LOW_Z - 0.1}})
    anns = sixth_pick_annotations(ctx, e, [(b, WIN_PROB_THRESHOLD)])
    assert anns[1] == ANN_BRAIN


def test_no_brain_icon_when_pickrate_at_threshold():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_LOW_Z}})
    anns = sixth_pick_annotations(ctx, e, [(b, WIN_PROB_THRESHOLD)])
    assert 1 not in anns


def test_x_icon_when_low_winprob_high_pickrate():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_HIGH_Z + 0.1}})
    anns = sixth_pick_annotations(ctx, e, [(b, WIN_PROB_THRESHOLD - 0.01)])
    assert anns[1] == ANN_X


def test_no_x_icon_when_pickrate_below_high_threshold():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_HIGH_Z - 0.1}})
    anns = sixth_pick_annotations(ctx, e, [(b, WIN_PROB_THRESHOLD - 0.01)])
    assert 1 not in anns


def test_no_annotation_when_pickrate_missing():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({})  # no pickrate data
    anns = sixth_pick_annotations(ctx, e, [(b, WIN_PROB_THRESHOLD)])
    assert anns == {}


def test_no_annotation_when_event_missing():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {99999: 2.0}})  # different event
    anns = sixth_pick_annotations(ctx, e, [(b, WIN_PROB_THRESHOLD)])
    assert anns == {}


def test_mixed_annotations():
    b1 = _brawler(1, "COLT")    # high win, low pickrate → brain
    b2 = _brawler(2, "SHELLY")  # low win, high pickrate → X
    b3 = _brawler(3, "BULL")    # high win, high pickrate → no annotation
    b4 = _brawler(4, "BO")      # no pickrate data → no annotation
    e = _event(15000005)
    ctx = _ctx({
        1: {15000005: -0.5},   # low pickrate
        2: {15000005: 1.5},    # high pickrate
        3: {15000005: 1.5},    # high pickrate
    })
    scores = [(b1, 0.60), (b2, 0.40), (b3, 0.60), (b4, 0.60)]
    anns = sixth_pick_annotations(ctx, e, scores)
    assert anns.get(1) == ANN_BRAIN
    assert anns.get(2) == ANN_X
    assert 3 not in anns   # high win AND high pickrate → no annotation
    assert 4 not in anns   # missing pickrate


# ── render_brawler_table two-grid split ───────────────────────────────────────

def _console(width: int = 120) -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, width=width, highlight=False, markup=False, emoji=False)
    return c, buf


def _make_entries(n: int, above_50: int) -> list[tuple[BrawlerInfo, float]]:
    result: list[tuple[BrawlerInfo, float]] = []
    for i in range(above_50):
        result.append((_brawler(i, f"BRW{i:02d}"), 0.55 - i * 0.001))
    for i in range(n - above_50):
        j = above_50 + i
        result.append((_brawler(j, f"BRW{j:02d}"), 0.45 - i * 0.001))
    return result


def test_divider_splits_output_into_two_grids():
    entries = _make_entries(10, above_50=4)
    c, buf = _console()
    render_brawler_table(c, entries, str, divider=WIN_PROB_THRESHOLD)
    output = buf.getvalue()
    # Rich rule produces a line of '─' characters
    assert "─" in output


def test_no_divider_when_all_above_threshold():
    entries = _make_entries(5, above_50=5)
    c, buf = _console()
    render_brawler_table(c, entries, str, divider=WIN_PROB_THRESHOLD)
    output = buf.getvalue()
    assert "─" not in output  # no rule rendered


def test_no_divider_when_none_above_threshold():
    entries = _make_entries(5, above_50=0)
    c, buf = _console()
    render_brawler_table(c, entries, str, divider=WIN_PROB_THRESHOLD)
    output = buf.getvalue()
    assert "─" not in output


def test_rank_continuity_across_grids():
    entries = _make_entries(6, above_50=3)
    c, buf = _console(width=200)
    render_brawler_table(c, entries, str, divider=WIN_PROB_THRESHOLD)
    output = buf.getvalue()
    # ranks 1, 2, 3 in first grid; ranks 4, 5, 6 in second
    assert "  1." in output
    assert "  4." in output


def test_annotations_appear_in_output():
    b = _brawler(0, "BRW00")
    entries = [(b, 0.60)]
    c, buf = _console()
    render_brawler_table(c, entries, str, annotations={0: " XX"}, annotation_slot=3)
    output = buf.getvalue()
    assert "XX" in output


def test_empty_entries_renders_nothing():
    c, buf = _console()
    render_brawler_table(c, [], str)
    assert buf.getvalue() == ""
