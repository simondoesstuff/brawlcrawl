"""Tests for pick_annotations, overview_annotations, render_brawler_table,
_z_score_q, _ban_excluded, _build_obs, and _pick6_team_split."""

from io import StringIO

import numpy as np
import pytest
from rich.console import Console

from pick.constants import (
    ANN_BRAIN, ANN_SECRET, ANN_X, ANNOTATION_SLOT,
    PICKRATE_HIGH_Z, PICKRATE_LOW_Z, Q_THRESHOLD,
)
from pick.display import render_brawler_table
from pick.main import _ban_excluded, _phase_default_filter, _z_score_q, overview_annotations, pick_annotations
from pick.score import BrawlerInfo, DraftContext, EventInfo, _build_obs, _pick6_team_split
from geneus.draft.env import AVAILABLE, GLOBALLY_BANNED, LOCALLY_BANNED, PICKED_A, PICKED_B


# ── minimal stubs ─────────────────────────────────────────────────────────────

def _brawler(bid: int, name: str = "COLT") -> BrawlerInfo:
    return BrawlerInfo(id=bid, name=name, brawler_class="Damage Dealer", rarity="Rare", char_idx=bid)


def _event(eid: int = 15000005) -> EventInfo:
    return EventInfo(id=eid, mode="bounty", mode_id=3, map_name="Shooting Star", event_idx=0, mode_idx=0)


def _ctx(pickrates: dict[int, dict[int, float]]) -> DraftContext:
    return DraftContext(  # type: ignore[call-arg]
        q_net=None,  # type: ignore[arg-type]
        terminal_model=None,  # type: ignore[arg-type]
        char_meta_table=None,  # type: ignore[arg-type]
        char_encs_all=None,  # type: ignore[arg-type]
        event_enc_row={},
        brawlers=[],
        events=[],
        brawler_names=[],
        map_names=[],
        n_chars=0,
        winrates={},
        pickrates=pickrates,
        _brawler_by_char_idx={},
        _brawler_by_name={},
    )


# ── pick_annotations ──────────────────────────────────────────────────────────

def test_brain_icon_when_positive_q_low_pickrate():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_LOW_Z - 0.1}})
    anns = pick_annotations(ctx, e, [(b, Q_THRESHOLD + 0.1)])
    assert anns[1] == ANN_BRAIN


def test_no_brain_icon_when_q_at_threshold():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_LOW_Z - 0.1}})
    anns = pick_annotations(ctx, e, [(b, Q_THRESHOLD)])
    assert 1 not in anns


def test_x_icon_when_negative_q_high_pickrate():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_HIGH_Z + 0.1}})
    anns = pick_annotations(ctx, e, [(b, Q_THRESHOLD - 0.01)])
    assert anns[1] == ANN_X


def test_no_x_icon_when_pickrate_below_high_threshold():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_HIGH_Z - 0.1}})
    anns = pick_annotations(ctx, e, [(b, Q_THRESHOLD - 0.01)])
    assert 1 not in anns


def test_no_annotation_when_pickrate_missing():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({})
    anns = pick_annotations(ctx, e, [(b, Q_THRESHOLD + 0.5)])
    assert anns == {}


def test_no_annotation_when_event_missing():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {99999: 2.0}})
    anns = pick_annotations(ctx, e, [(b, Q_THRESHOLD + 0.5)])
    assert anns == {}


def test_mixed_pick_annotations():
    b1 = _brawler(1, "COLT")    # Q > 0, low pickrate → brain
    b2 = _brawler(2, "SHELLY")  # Q ≤ 0, high pickrate → X
    b3 = _brawler(3, "BULL")    # Q > 0, high pickrate → no annotation
    b4 = _brawler(4, "BO")      # no pickrate data → no annotation
    e = _event(15000005)
    ctx = _ctx({
        1: {15000005: PICKRATE_LOW_Z - 0.1},
        2: {15000005: PICKRATE_HIGH_Z + 0.1},
        3: {15000005: PICKRATE_HIGH_Z + 0.1},
    })
    scores = [(b1, 0.5), (b2, -0.1), (b3, 0.5), (b4, 0.5)]
    anns = pick_annotations(ctx, e, scores)
    assert anns.get(1) == ANN_BRAIN
    assert anns.get(2) == ANN_X
    assert 3 not in anns   # Q > 0 but high pickrate → no annotation
    assert 4 not in anns   # missing pickrate


# ── overview_annotations ──────────────────────────────────────────────────────

def test_secret_icon_when_positive_z_low_pickrate():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_LOW_Z - 0.1}})
    anns = overview_annotations(ctx, e, [(b, 0.5)])
    assert anns[1] == ANN_SECRET


def test_x_icon_overview_when_negative_z_high_pickrate():
    b = _brawler(1, "COLT")
    e = _event(15000005)
    ctx = _ctx({1: {15000005: PICKRATE_HIGH_Z + 0.1}})
    anns = overview_annotations(ctx, e, [(b, -0.5)])
    assert anns[1] == ANN_X


# ── render_brawler_table two-grid split ───────────────────────────────────────

def _console(width: int = 120) -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, width=width, highlight=False, markup=False, emoji=False)
    return c, buf


def _make_entries(n: int, above_threshold: int) -> list[tuple[BrawlerInfo, float]]:
    result: list[tuple[BrawlerInfo, float]] = []
    for i in range(above_threshold):
        result.append((_brawler(i, f"BRW{i:02d}"), 1.0 - i * 0.1))
    for i in range(n - above_threshold):
        j = above_threshold + i
        result.append((_brawler(j, f"BRW{j:02d}"), -0.1 - i * 0.1))
    return result


def test_divider_splits_output_into_two_grids():
    entries = _make_entries(10, above_threshold=4)
    c, buf = _console()
    render_brawler_table(c, entries, str, divider=Q_THRESHOLD)
    output = buf.getvalue()
    assert "─" in output


def test_no_divider_when_all_above_threshold():
    entries = _make_entries(5, above_threshold=5)
    c, buf = _console()
    render_brawler_table(c, entries, str, divider=Q_THRESHOLD)
    output = buf.getvalue()
    assert "─" not in output


def test_no_divider_when_none_above_threshold():
    entries = _make_entries(5, above_threshold=0)
    c, buf = _console()
    render_brawler_table(c, entries, str, divider=Q_THRESHOLD)
    output = buf.getvalue()
    assert "─" not in output


def test_rank_continuity_across_grids():
    entries = _make_entries(6, above_threshold=3)
    c, buf = _console(width=200)
    render_brawler_table(c, entries, str, divider=Q_THRESHOLD)
    output = buf.getvalue()
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


# ── _z_score_q ────────────────────────────────────────────────────────────────

def test_z_score_q_normalizes_correctly():
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = _z_score_q(q, [0, 1, 2, 3, 4])
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std()) - 1.0) < 1e-6


def test_z_score_q_empty_available_returns_unchanged():
    q = np.array([1.0, 2.0, 3.0])
    z = _z_score_q(q, [])
    np.testing.assert_array_equal(z, q)


def test_z_score_q_constant_returns_zeros():
    q = np.array([3.0, 3.0, 3.0])
    z = _z_score_q(q, [0, 1, 2])
    np.testing.assert_array_equal(z, np.zeros(3))


def test_z_score_q_subset_of_indices():
    q = np.array([10.0, 1.0, 2.0, 3.0])
    # Only indices 1-3 are available; mean=2, std=sqrt(2/3)
    vals = np.array([1.0, 2.0, 3.0])
    mean, std = float(vals.mean()), float(vals.std())
    z = _z_score_q(q, [1, 2, 3])
    assert abs(z[1] - (1.0 - mean) / std) < 1e-6
    assert abs(z[2] - (2.0 - mean) / std) < 1e-6
    assert abs(z[3] - (3.0 - mean) / std) < 1e-6


# ── _ban_excluded ────────────────────────────────────────────────────────────

def _b(bid: int) -> BrawlerInfo:
    return BrawlerInfo(id=bid, name=f"B{bid}", brawler_class="Damage Dealer", rarity="Rare", char_idx=bid)


def test_ban_excluded_ally_phase_excludes_ally_bans_only():
    ally = [_b(1), _b(2)]
    enemy: list[BrawlerInfo] = []
    # During ally ban phase, only ally bans excluded — enemy bans not yet present anyway
    for phase in range(3):
        excl = _ban_excluded(phase, ally, enemy, [])
        assert excl == {1, 2}


def test_ban_excluded_enemy_phase_excludes_enemy_bans_only():
    ally = [_b(1), _b(2), _b(3)]
    enemy = [_b(4)]
    for phase in range(3, 6):
        excl = _ban_excluded(phase, ally, enemy, [])
        # Ally bans (1,2,3) NOT excluded — enemies can re-ban them
        assert excl == {4}
        assert 1 not in excl
        assert 2 not in excl


def test_ban_excluded_enemy_can_ban_ally_banned():
    ally = [_b(10)]
    enemy: list[BrawlerInfo] = []
    excl = _ban_excluded(4, ally, enemy, [])
    assert 10 not in excl  # ally-banned brawler still available to enemy


def test_ban_excluded_pick_phase_excludes_all():
    ally = [_b(1), _b(2), _b(3)]
    enemy = [_b(4), _b(5), _b(6)]
    picks = [(True, _b(7)), (False, _b(8))]
    for phase in range(6, 12):
        excl = _ban_excluded(phase, ally, enemy, picks)
        assert excl == {1, 2, 3, 4, 5, 6, 7, 8}


def test_ban_excluded_inter_team_duplicate_allowed():
    # Both teams ban brawler 99 — this is allowed (inter-team sharing)
    ally = [_b(99)]
    enemy = [_b(99)]
    picks: list[tuple[bool, BrawlerInfo]] = []
    # During pick phase, 99 is excluded from picks (banned by both, but set deduplicates)
    excl = _ban_excluded(6, ally, enemy, picks)
    assert 99 in excl


# ── _build_obs ────────────────────────────────────────────────────────────────

def _make_brawler(bid: int, char_idx: int) -> BrawlerInfo:
    return BrawlerInfo(id=bid, name=f"B{bid}", brawler_class="Damage Dealer", rarity="Rare", char_idx=char_idx)


def test_build_obs_ban_phase_ally():
    # Phase 0–2: only ally bans → GLOBALLY_BANNED; everything else AVAILABLE
    b1 = _make_brawler(1, 0)
    b2 = _make_brawler(2, 1)
    b3 = _make_brawler(3, 2)
    obs, turn_token = _build_obs(5, [b1, b2], [], [], phase=0, ally_first=True, brawlers=[b1, b2, b3])
    assert obs[0] == GLOBALLY_BANNED
    assert obs[1] == GLOBALLY_BANNED
    assert obs[2] == AVAILABLE


def test_build_obs_locally_banned_non_pool():
    # Non-pool brawlers (not in local_pool and AVAILABLE) become LOCALLY_BANNED
    b1 = _make_brawler(1, 0)
    b2 = _make_brawler(2, 1)
    b3 = _make_brawler(3, 2)
    obs, _ = _build_obs(5, [], [], [], phase=0, ally_first=True,
                        local_pool={1}, brawlers=[b1, b2, b3])
    assert obs[0] == AVAILABLE       # b1 is in pool
    assert obs[1] == LOCALLY_BANNED  # b2 not in pool
    assert obs[2] == LOCALLY_BANNED  # b3 not in pool


def test_build_obs_globally_banned_not_overridden_by_local():
    # GLOBALLY_BANNED takes precedence over LOCALLY_BANNED when neither is in pool
    b1 = _make_brawler(1, 0)  # ally banned → GLOBALLY_BANNED
    b2 = _make_brawler(2, 1)  # not banned, not in pool → LOCALLY_BANNED
    obs, _ = _build_obs(5, [b1], [], [], phase=0, ally_first=True,
                        local_pool=set(), brawlers=[b1, b2])
    assert obs[0] == GLOBALLY_BANNED  # ban takes precedence
    assert obs[1] == LOCALLY_BANNED


def test_build_obs_pick_phase_sets_picked_states():
    b1 = _make_brawler(1, 0)  # team A pick (ally, ally_first=True)
    b2 = _make_brawler(2, 1)  # team B pick (enemy, ally_first=True)
    picks = [(True, b1), (False, b2)]
    obs, _ = _build_obs(5, [], [], picks, phase=8, ally_first=True, brawlers=[b1, b2])
    assert obs[0] == PICKED_A
    assert obs[1] == PICKED_B


# ── _pick6_team_split ─────────────────────────────────────────────────────────

def test_pick6_team_split_assigns_by_turn_schedule():
    # TURN_SCHEDULE[6:11] teams: A, B, B, A, A
    # So pick indices 0,3,4 → team A; indices 1,2 → team B
    brawlers = [_make_brawler(i, i) for i in range(5)]
    picks = [(True, b) for b in brawlers]  # is_ally ignored for split
    team_a, team_b = _pick6_team_split(picks)
    team_a_idxs = {b.char_idx for b in team_a}
    team_b_idxs = {b.char_idx for b in team_b}
    assert team_a_idxs == {0, 3, 4}  # pick positions 0,3,4 are team A
    assert team_b_idxs == {1, 2}     # pick positions 1,2 are team B


def test_pick6_team_split_team_b_has_two_entries():
    brawlers = [_make_brawler(i, i) for i in range(5)]
    picks = [(True, b) for b in brawlers]
    _, team_b = _pick6_team_split(picks)
    assert len(team_b) == 2  # 6th pick (B seat 2) is the missing slot


# ── _phase_default_filter ─────────────────────────────────────────────────────

def test_phase_default_filter_ban_phases_always_off():
    # Phases 0-5 are ban phases — filter always off regardless of ally_first
    for phase in range(6):
        assert _phase_default_filter(phase, True) is False
        assert _phase_default_filter(phase, False) is False


def test_phase_default_filter_ally_first_true():
    # TURN_SCHEDULE[6..10] teams: A,B,B,A,A → ally picks at 6,9,10; enemy at 7,8
    # ally_first=True: ally=team A
    # filter ON for ally's picks (6,9,10), OFF for enemy's (7,8)
    assert _phase_default_filter(6, True) is True   # pick 1: team A (ally)
    assert _phase_default_filter(7, True) is False  # pick 2: team B (enemy)
    assert _phase_default_filter(8, True) is False  # pick 3: team B (enemy)
    assert _phase_default_filter(9, True) is True   # pick 4: team A (ally)
    assert _phase_default_filter(10, True) is True  # pick 5: team A (ally)
    assert _phase_default_filter(11, True) is False # pick 6: team B (enemy)


def test_phase_default_filter_ally_first_false():
    # ally_first=False: ally=team B
    # filter ON for ally's picks (7,8,11), OFF for enemy's (6,9,10)
    assert _phase_default_filter(6, False) is False  # pick 1: team A (enemy)
    assert _phase_default_filter(7, False) is True   # pick 2: team B (ally)
    assert _phase_default_filter(8, False) is True   # pick 3: team B (ally)
    assert _phase_default_filter(9, False) is False  # pick 4: team A (enemy)
    assert _phase_default_filter(10, False) is False # pick 5: team A (enemy)
    assert _phase_default_filter(11, False) is True  # pick 6: team B (ally)
