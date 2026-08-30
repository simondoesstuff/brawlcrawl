"""Draft MDP: state, transitions, and action masking.

Ban phase is modelled as two independent sequential sub-MDPs — team A bans first
(A0→A1→A2), then team B (B0→B1→B2). Each team sees only its own bans during this
phase; the opposing team's bans are hidden until pick 1, matching the simultaneous
ban mechanic. Cross-team ban overlap is allowed: both teams can select the same
character, resulting in one globally banned character at pick 1.

Team A is defined as the first-picking team (no coin-flip in the model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Character observation states
# ---------------------------------------------------------------------------
AVAILABLE = 0
GLOBALLY_BANNED = 1  # banned by own team during ban phase; by either team during picks
LOCALLY_BANNED = 2   # available globally but not in this player's local pool
PICKED_A = 3
PICKED_B = 4

N_CHAR_STATES = 5

# ---------------------------------------------------------------------------
# Draft turn tokens  (embedding indices for DraftQNetwork)
# ---------------------------------------------------------------------------
BAN_PHASE = 0   # shared token for all 6 ban turns
PICK_1 = 1      # Team A, seat 0
PICK_2 = 2      # Team B, seat 0
PICK_3 = 3      # Team B, seat 1
PICK_4 = 4      # Team A, seat 1
PICK_5 = 5      # Team A, seat 2
PICK_6 = 6      # Team B, seat 2 (last pick)

N_DRAFT_TOKENS = 7

# ---------------------------------------------------------------------------
# Full 12-turn schedule: (turn_token, team, seat_within_team)
#
# Bans:  A0, A1, A2, B0, B1, B2   — sequential approximation of simultaneous
# Picks: A BB AA B  (team A = first-picking team by convention)
# ---------------------------------------------------------------------------
TURN_SCHEDULE: list[tuple[int, str, int]] = [
    (BAN_PHASE, "A", 0),
    (BAN_PHASE, "A", 1),
    (BAN_PHASE, "A", 2),
    (BAN_PHASE, "B", 0),
    (BAN_PHASE, "B", 1),
    (BAN_PHASE, "B", 2),
    (PICK_1, "A", 0),
    (PICK_2, "B", 0),
    (PICK_3, "B", 1),
    (PICK_4, "A", 1),
    (PICK_5, "A", 2),
    (PICK_6, "B", 2),
]

# Team index per turn: 0 = A, 1 = B  (shape [12])
TURN_TEAMS = np.array([0 if t[1] == "A" else 1 for t in TURN_SCHEDULE], dtype=np.int32)

# Whether the team at turn t equals the team at turn t+1  (shape [11])
SAME_TEAM_NEXT: np.ndarray = (TURN_TEAMS[:-1] == TURN_TEAMS[1:])

# Bellman sign per turn t (applied to V_{t+1} when computing target_t)
BELLMAN_SIGNS: np.ndarray = np.where(SAME_TEAM_NEXT, 1.0, -1.0).astype(np.float32)

# The last pick is always team B in this schedule → Q_local = -logit_A at terminal
TERMINAL_SIGN: float = -1.0


def player_idx(team: str, seat: int) -> int:
    """Player list index 0-5: A0=0, A1=1, A2=2, B0=3, B1=4, B2=5."""
    return seat if team == "A" else 3 + seat


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerConfig:
    local_pool: np.ndarray  # bool[n_chars] — chars this player can pick
    temperature: float
    team: str               # "A" or "B"


@dataclass
class DraftState:
    """Full (true) draft state.  Per-player observations are derived from this."""
    team_a_bans: np.ndarray  # bool[n_chars] — visible only to team A during ban phase
    team_b_bans: np.ndarray  # bool[n_chars] — visible only to team B during ban phase
    picks_a: np.ndarray      # bool[n_chars]
    picks_b: np.ndarray      # bool[n_chars]
    char_encs: np.ndarray    # float[n_chars, h] — frozen per-event char encodings
    event_idx: int
    mode_idx: int
    player_configs: list[PlayerConfig]  # 6 entries: A0, A1, A2, B0, B1, B2


@dataclass
class DraftConfig:
    pool_min: int = 10    # minimum chars in a player's local pool
    pool_max: int = 40    # maximum chars in a player's local pool
    temp_min: float = 0.05
    temp_max: float = 2.0


# ---------------------------------------------------------------------------
# Core MDP functions
# ---------------------------------------------------------------------------

def get_player_observed_states(state: DraftState, turn_idx: int) -> np.ndarray:
    """Character states from the acting player's partial observation.

    Ban phase: own team's bans appear as GLOBALLY_BANNED; opposing bans are hidden
    (appear as AVAILABLE or LOCALLY_BANNED depending on the player's pool).
    Pick phase: all bans and picks are revealed.
    """
    _, team, seat = TURN_SCHEDULE[turn_idx]
    p = state.player_configs[player_idx(team, seat)]
    obs = np.zeros(len(state.team_a_bans), dtype=np.int32)

    if turn_idx < 6:
        own_bans = state.team_a_bans if team == "A" else state.team_b_bans
        obs = np.where(own_bans, GLOBALLY_BANNED, obs)
    else:
        all_bans = state.team_a_bans | state.team_b_bans
        obs = np.where(all_bans, GLOBALLY_BANNED, obs)
        obs = np.where(state.picks_a, PICKED_A, obs)
        obs = np.where(state.picks_b, PICKED_B, obs)

    obs = np.where((obs == AVAILABLE) & ~p.local_pool, LOCALLY_BANNED, obs)
    return obs


def get_valid_action_mask(state: DraftState, turn_idx: int) -> np.ndarray:
    """Boolean mask of valid actions for the acting player at turn_idx.

    True = valid.  Invalid actions must never be sampled.

    Ban phase:  any char not yet banned by own team (cross-team overlap is allowed).
    Pick phase: chars that are available globally (post-ban-reveal) and in local pool.
    """
    _, team, seat = TURN_SCHEDULE[turn_idx]

    if turn_idx < 6:
        own_bans = state.team_a_bans if team == "A" else state.team_b_bans
        return ~own_bans

    all_bans = state.team_a_bans | state.team_b_bans
    all_picks = state.picks_a | state.picks_b
    p = state.player_configs[player_idx(team, seat)]
    return ~all_bans & ~all_picks & p.local_pool


def step(state: DraftState, action: int, turn_idx: int) -> DraftState:
    """Apply action (ban or pick) and return the updated state.

    Every action shrinks the remaining action space for all future agents:
    bans add to team_*_bans, picks add to picks_a/picks_b.  Both are
    checked by get_valid_action_mask so no char can be selected twice.
    """
    _, team, _ = TURN_SCHEDULE[turn_idx]

    team_a_bans = state.team_a_bans.copy()
    team_b_bans = state.team_b_bans.copy()
    picks_a = state.picks_a.copy()
    picks_b = state.picks_b.copy()

    if turn_idx < 6:
        if team == "A":
            team_a_bans[action] = True
        else:
            team_b_bans[action] = True
    else:
        if team == "A":
            picks_a[action] = True
        else:
            picks_b[action] = True

    return DraftState(
        team_a_bans=team_a_bans,
        team_b_bans=team_b_bans,
        picks_a=picks_a,
        picks_b=picks_b,
        char_encs=state.char_encs,
        event_idx=state.event_idx,
        mode_idx=state.mode_idx,
        player_configs=state.player_configs,
    )
