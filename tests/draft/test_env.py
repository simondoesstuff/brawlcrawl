"""Tests for the draft MDP environment."""

import numpy as np
import pytest

from geneus.draft.env import (
    AVAILABLE,
    BELLMAN_SIGNS,
    BAN_PHASE,
    GLOBALLY_BANNED,
    LOCALLY_BANNED,
    PICKED_A,
    PICKED_B,
    PICK_1,
    PICK_2,
    PICK_3,
    PICK_4,
    PICK_5,
    PICK_6,
    SAME_TEAM_NEXT,
    TERMINAL_SIGN,
    TURN_SCHEDULE,
    TURN_TEAMS,
    DraftConfig,
    DraftState,
    PlayerConfig,
    get_player_observed_states,
    get_valid_action_mask,
    player_idx,
    step,
)

N = 20  # small char count for tests


def _make_state(n: int = N, pool: list | None = None) -> DraftState:
    all_pool = np.ones(n, dtype=bool)
    if pool is None:
        pool = [all_pool.copy() for _ in range(6)]
    configs = [
        PlayerConfig(local_pool=p, temperature=1.0, team="A" if i < 3 else "B")
        for i, p in enumerate(pool)
    ]
    return DraftState(
        team_a_bans=np.zeros(n, dtype=bool),
        team_b_bans=np.zeros(n, dtype=bool),
        picks_a=np.zeros(n, dtype=bool),
        picks_b=np.zeros(n, dtype=bool),
        char_encs=np.zeros((n, 4), dtype=np.float32),
        event_idx=0,
        mode_idx=0,
        player_configs=configs,
    )


# ---------------------------------------------------------------------------
# Schedule correctness
# ---------------------------------------------------------------------------

def test_schedule_length():
    assert len(TURN_SCHEDULE) == 12


def test_ban_phase_tokens():
    for turn_idx in range(6):
        token, _, _ = TURN_SCHEDULE[turn_idx]
        assert token == BAN_PHASE, f"Turn {turn_idx} should have BAN_PHASE token"


def test_pick_tokens():
    expected = [PICK_1, PICK_2, PICK_3, PICK_4, PICK_5, PICK_6]
    for i, (turn_idx, exp) in enumerate(zip(range(6, 12), expected)):
        token, _, _ = TURN_SCHEDULE[turn_idx]
        assert token == exp, f"Turn {turn_idx} should have token {exp}, got {token}"


def test_pick_order_teams():
    pick_teams = [TURN_SCHEDULE[t][1] for t in range(6, 12)]
    assert pick_teams == ["A", "B", "B", "A", "A", "B"], f"Got {pick_teams}"


def test_ban_order_teams():
    ban_teams = [TURN_SCHEDULE[t][1] for t in range(6)]
    assert ban_teams == ["A", "A", "A", "B", "B", "B"]


def test_turn_teams_shape():
    assert TURN_TEAMS.shape == (12,)
    assert set(TURN_TEAMS.tolist()) == {0, 1}


def test_same_team_next_shape():
    assert SAME_TEAM_NEXT.shape == (11,)


def test_bellman_signs_shape():
    assert BELLMAN_SIGNS.shape == (11,)
    assert set(BELLMAN_SIGNS.tolist()) == {1.0, -1.0}


def test_terminal_sign():
    # Last turn (11) is team B, so Q_local = -logit_A
    last_team = TURN_SCHEDULE[11][1]
    assert last_team == "B"
    assert TERMINAL_SIGN == -1.0


def test_bellman_signs_values():
    for t in range(11):
        expected_sign = 1.0 if TURN_TEAMS[t] == TURN_TEAMS[t + 1] else -1.0
        assert BELLMAN_SIGNS[t] == expected_sign, f"Wrong sign at t={t}"


def test_player_idx():
    assert player_idx("A", 0) == 0
    assert player_idx("A", 1) == 1
    assert player_idx("A", 2) == 2
    assert player_idx("B", 0) == 3
    assert player_idx("B", 1) == 4
    assert player_idx("B", 2) == 5


# ---------------------------------------------------------------------------
# Valid action masks
# ---------------------------------------------------------------------------

def test_ban_phase_all_valid_initially():
    state = _make_state()
    for turn_idx in range(6):
        mask = get_valid_action_mask(state, turn_idx)
        assert mask.all(), f"All chars should be valid at turn {turn_idx} with no bans"


def test_ban_phase_no_local_pool_restriction():
    # Restrict local pool for all players: only char 0 is allowed
    tiny_pool = np.zeros(N, dtype=bool)
    tiny_pool[0] = True
    state = _make_state(pool=[tiny_pool.copy() for _ in range(6)])
    # Ban phase should still allow all chars (no pool restriction)
    for turn_idx in range(6):
        mask = get_valid_action_mask(state, turn_idx)
        assert mask.all(), f"Ban phase should not restrict by local pool at turn {turn_idx}"


def test_pick_phase_restricted_by_local_pool():
    pool = np.zeros(N, dtype=bool)
    pool[0] = pool[1] = pool[2] = True  # only 3 chars allowed
    state = _make_state(pool=[pool.copy() for _ in range(6)])
    for turn_idx in range(6, 12):
        mask = get_valid_action_mask(state, turn_idx)
        assert mask.sum() == 3, f"Only pooled chars valid at pick turn {turn_idx}"
        assert mask[0] and mask[1] and mask[2]


def test_ban_removes_from_own_team_valid_actions():
    state = _make_state()
    state = step(state, action=5, turn_idx=0)  # A0 bans char 5
    # A1 cannot ban char 5 again
    mask = get_valid_action_mask(state, turn_idx=1)
    assert not mask[5]
    assert mask.sum() == N - 1


def test_ban_does_not_restrict_other_team():
    state = _make_state()
    state = step(state, action=5, turn_idx=0)  # A0 bans char 5
    # B0 at turn 3 can still ban char 5 (cross-team overlap is allowed)
    mask = get_valid_action_mask(state, turn_idx=3)
    assert mask[5], "Team B should be able to ban the same char as team A"
    assert mask.sum() == N


def test_pick_phase_excludes_all_bans():
    state = _make_state()
    # A0 bans 0, A1 bans 1, A2 bans 2
    state = step(state, action=0, turn_idx=0)
    state = step(state, action=1, turn_idx=1)
    state = step(state, action=2, turn_idx=2)
    # B0 bans 3, B1 bans 4, B2 bans 5
    state = step(state, action=3, turn_idx=3)
    state = step(state, action=4, turn_idx=4)
    state = step(state, action=5, turn_idx=5)
    # At pick 1, chars 0-5 should be banned
    mask = get_valid_action_mask(state, turn_idx=6)
    assert not any(mask[:6])
    assert mask[6:].all()


def test_cross_team_ban_overlap_at_pick():
    state = _make_state()
    # A0 bans char 7
    state = step(state, action=7, turn_idx=0)
    state = step(state, action=0, turn_idx=1)
    state = step(state, action=1, turn_idx=2)
    # B0 also bans char 7 (cross-team overlap)
    state = step(state, action=7, turn_idx=3)
    state = step(state, action=2, turn_idx=4)
    state = step(state, action=3, turn_idx=5)
    # At pick 1: char 7 banned by both teams (still globally banned)
    mask = get_valid_action_mask(state, turn_idx=6)
    assert not mask[7], "Char 7 double-banned — still globally banned"
    # Only unique bans count: chars 0, 1, 2, 3, 7 → 5 banned
    assert mask.sum() == N - 5


def test_pick_excludes_already_picked():
    state = _make_state()
    for t in range(6):  # no bans (simulate with dummy actions on separate chars)
        state = step(state, action=t, turn_idx=t)  # bans 0-5
    state = step(state, action=6, turn_idx=6)  # A0 picks char 6
    mask = get_valid_action_mask(state, turn_idx=7)  # B0's turn
    assert not mask[6], "Picked char should be invalid"


# ---------------------------------------------------------------------------
# Observed states
# ---------------------------------------------------------------------------

def test_ban_phase_hides_other_team_bans():
    state = _make_state()
    # A0 bans char 0 (turn 0)
    state = step(state, action=0, turn_idx=0)
    # Now at B0's turn (turn 3): team B should NOT see team A's ban
    obs = get_player_observed_states(state, turn_idx=3)
    assert obs[0] == AVAILABLE, "Team B should not see team A's ban during ban phase"


def test_ban_phase_shows_own_team_bans():
    state = _make_state()
    state = step(state, action=0, turn_idx=0)  # A0 bans char 0
    # At A1's turn (turn 1): team A sees their own ban
    obs = get_player_observed_states(state, turn_idx=1)
    assert obs[0] == GLOBALLY_BANNED, "Team A should see their own ban"


def test_pick_phase_reveals_all_bans():
    state = _make_state()
    state = step(state, action=0, turn_idx=0)  # A bans 0
    state = step(state, action=1, turn_idx=1)
    state = step(state, action=2, turn_idx=2)
    state = step(state, action=3, turn_idx=3)  # B bans 3
    state = step(state, action=4, turn_idx=4)
    state = step(state, action=5, turn_idx=5)
    # At PICK_1 (turn 6), team A sees all 6 bans
    obs = get_player_observed_states(state, turn_idx=6)
    for i in range(6):
        assert obs[i] == GLOBALLY_BANNED, f"Char {i} should be globally banned at pick phase"


def test_locally_banned_state():
    pool = np.ones(N, dtype=bool)
    pool[0] = False  # char 0 not in pool
    state = _make_state(pool=[pool.copy() for _ in range(6)])
    obs = get_player_observed_states(state, turn_idx=6)  # pick phase
    assert obs[0] == LOCALLY_BANNED


def test_picked_states_visible():
    state = _make_state()
    for t in range(6):
        state = step(state, action=t, turn_idx=t)  # bans 0-5
    state = step(state, action=6, turn_idx=6)  # A0 picks char 6 → PICKED_A
    obs = get_player_observed_states(state, turn_idx=7)
    assert obs[6] == PICKED_A


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_step_ban_sets_team_a_bans():
    state = _make_state()
    new_state = step(state, action=3, turn_idx=0)
    assert new_state.team_a_bans[3]
    assert not new_state.team_b_bans[3]


def test_step_ban_sets_team_b_bans():
    state = _make_state()
    new_state = step(state, action=3, turn_idx=3)  # B0's ban turn
    assert new_state.team_b_bans[3]
    assert not new_state.team_a_bans[3]


def test_step_pick_sets_picks_a():
    state = _make_state()
    for t in range(6):
        state = step(state, action=t, turn_idx=t)
    new_state = step(state, action=6, turn_idx=6)  # PICK_1 = team A
    assert new_state.picks_a[6]
    assert not new_state.picks_b[6]


def test_step_pick_sets_picks_b():
    state = _make_state()
    for t in range(6):
        state = step(state, action=t, turn_idx=t)
    state = step(state, action=6, turn_idx=6)
    new_state = step(state, action=7, turn_idx=7)  # PICK_2 = team B
    assert new_state.picks_b[7]
    assert not new_state.picks_a[7]


def test_step_does_not_mutate_original():
    state = _make_state()
    _ = step(state, action=0, turn_idx=0)
    assert not state.team_a_bans[0], "Original state should be immutable"


def test_full_draft_yields_3v3():
    state = _make_state()
    for turn_idx in range(12):
        mask = get_valid_action_mask(state, turn_idx)
        action = int(np.where(mask)[0][0])  # pick first valid
        state = step(state, action, turn_idx)
    assert state.picks_a.sum() == 3
    assert state.picks_b.sum() == 3
    assert (state.picks_a & state.picks_b).sum() == 0  # no overlap
