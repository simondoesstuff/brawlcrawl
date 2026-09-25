"""Model loading and inference for draft scoring."""

import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from geneus.data import load_vocabs
from geneus.draft.env import (
    AVAILABLE,
    BAN_PHASE_FIRST_PICK,
    BAN_PHASE_SIXTH_PICK,
    GLOBALLY_BANNED,
    LOCALLY_BANNED,
    PICKED_A,
    PICKED_B,
    TURN_SCHEDULE,
)
from geneus.draft.model import DraftQNetwork
from geneus.draft.train import precompute_char_encs
from geneus.model import BrawlModel


@dataclass(frozen=True)
class BrawlerInfo:
    id: int
    name: str
    brawler_class: str
    rarity: str
    char_idx: int  # vocabulary index


@dataclass(frozen=True)
class EventInfo:
    id: int
    mode: str
    mode_id: int
    map_name: str
    event_idx: int  # vocabulary index
    mode_idx: int   # vocabulary index


@dataclass
class DraftContext:
    q_net: DraftQNetwork
    terminal_model: BrawlModel
    char_meta_table: np.ndarray            # [n_chars, 3] int32: (class_idx, range_idx, destruct_idx)
    char_encs_all: np.ndarray              # [n_events_sorted, n_chars, h]
    event_enc_row: dict[int, int]          # event vocab_idx → row in char_encs_all
    brawlers: list[BrawlerInfo]
    events: list[EventInfo]
    brawler_names: list[str]
    map_names: list[str]
    n_chars: int                           # total vocabulary chars
    winrates: dict[int, dict[int, float]]  # char_id → event_id → z_score
    pickrates: dict[int, dict[int, float]] # char_id → event_id → z_score
    _brawler_by_char_idx: dict[int, BrawlerInfo]
    _brawler_by_name: dict[str, BrawlerInfo]


@eqx.filter_jit
def _q_apply(
    q_net: DraftQNetwork,
    char_encs: jax.Array,
    player_states: jax.Array,
    turn_token: jax.Array,
) -> jax.Array:
    return q_net(char_encs, player_states, turn_token)


@eqx.filter_jit
def _terminal_score_pick6_batch(
    terminal_model: BrawlModel,
    event_idx: jax.Array,              # []
    mode_idx: jax.Array,               # []
    team_a_chars: jax.Array,           # [3]
    team_a_meta: jax.Array,            # [3, 3]
    team_b_partial_chars: jax.Array,   # [2]
    team_b_partial_meta: jax.Array,    # [2, 3]
    cand_chars: jax.Array,             # [n]
    cand_meta: jax.Array,              # [n, 3]
) -> jax.Array:  # [n] logits, team-A-wins perspective
    def score_one(c, m):
        b_chars = jnp.concatenate([team_b_partial_chars, c[None]])
        b_meta = jnp.concatenate([team_b_partial_meta, m[None]])
        return terminal_model(event_idx, mode_idx, team_a_chars, team_a_meta, b_chars, b_meta)
    return jax.vmap(score_one)(cand_chars, cand_meta)


@eqx.filter_jit
def q_values_batch(
    q_net: DraftQNetwork,
    char_encs: jax.Array,      # [n_chars, h] -- shared across the batch (single event)
    player_states: jax.Array,  # [B, n_chars]
    turn_token: jax.Array,     # [] -- shared across the batch (single turn)
) -> jax.Array:  # [B, n_chars]
    """Q-values for a batch of observations that share one event and one turn.

    Used by full-draft Monte Carlo rollouts, where many independent drafts are
    simulated in lockstep and each step queries the same acting-team turn.
    """
    return jax.vmap(lambda obs: q_net(char_encs, obs, turn_token))(player_states)


@eqx.filter_jit
def terminal_logits_grid(
    terminal_model: BrawlModel,
    event_idx: jax.Array,             # []
    mode_idx: jax.Array,              # []
    team_a_chars: jax.Array,          # [B, 3]
    team_a_meta: jax.Array,           # [B, 3, 3]
    team_b_partial_chars: jax.Array,  # [B, 2]
    team_b_partial_meta: jax.Array,   # [B, 2, 3]
    cand_chars: jax.Array,            # [n] -- same candidate grid for every row
    cand_meta: jax.Array,             # [n, 3]
) -> jax.Array:  # [B, n] logits, team-A-wins perspective
    """Score every candidate as team B's last pick, for a batch of independent
    partial drafts that each already have their own team A and team B roster.

    Used to pick team B's exact optimal final pick across many simultaneously
    simulated drafts: the last pick has no further draft to look ahead
    through, so the terminal model itself is the correct read (matches
    `get_terminal_pick6_scores`, batched over rows instead of one draft).
    """
    def score_row(ta_c, ta_m, tb_c, tb_m):
        def score_one(c, m):
            b_chars = jnp.concatenate([tb_c, c[None]])
            b_meta = jnp.concatenate([tb_m, m[None]])
            return terminal_model(event_idx, mode_idx, ta_c, ta_m, b_chars, b_meta)
        return jax.vmap(score_one)(cand_chars, cand_meta)
    return jax.vmap(score_row)(team_a_chars, team_a_meta, team_b_partial_chars, team_b_partial_meta)


def _build_obs(
    n_chars: int,
    ally_bans: list[BrawlerInfo],
    enemy_bans: list[BrawlerInfo],
    picks: list[tuple[bool, BrawlerInfo]],
    phase: int,
    ally_first: bool,
    local_pool: set[int] | None = None,
    brawlers: list[BrawlerInfo] | None = None,
) -> tuple[np.ndarray, int]:
    """Build the character observation vector and turn token for the acting player.

    local_pool: set of brawler IDs the acting player can select.  Brawlers not in
    the pool that are still AVAILABLE are marked LOCALLY_BANNED.

    The ban-phase turn token depends on which model team (A or B) is acting:
    the model's team A always gets the first pick, team B the sixth (last).
    ally_first says whether the ally maps to model team A or B, so it also
    determines which team is acting during each ban sub-phase.
    """
    obs = np.zeros(n_chars, dtype=np.int32)

    if phase < 3:
        for b in ally_bans:
            obs[b.char_idx] = GLOBALLY_BANNED
        acting_is_team_a = ally_first
        turn_token = BAN_PHASE_FIRST_PICK if acting_is_team_a else BAN_PHASE_SIXTH_PICK
    elif phase < 6:
        for b in enemy_bans:
            obs[b.char_idx] = GLOBALLY_BANNED
        acting_is_team_a = not ally_first
        turn_token = BAN_PHASE_FIRST_PICK if acting_is_team_a else BAN_PHASE_SIXTH_PICK
    else:
        for b in ally_bans + enemy_bans:
            obs[b.char_idx] = GLOBALLY_BANNED
        for is_ally, b in picks:
            obs[b.char_idx] = PICKED_A if (is_ally == ally_first) else PICKED_B
        pick_idx = phase - 6
        turn_token = TURN_SCHEDULE[6 + pick_idx][0]

    if local_pool is not None and brawlers is not None:
        for b in brawlers:
            if obs[b.char_idx] == AVAILABLE and b.id not in local_pool:
                obs[b.char_idx] = LOCALLY_BANNED

    return obs, turn_token


def _pick6_team_split(
    picks: list[tuple[bool, BrawlerInfo]],
) -> tuple[list[BrawlerInfo], list[BrawlerInfo]]:
    """Split 5 picks into model team A (first-picking) and team B by TURN_SCHEDULE.

    Returns (team_a_picks, team_b_partial) where team_b_partial has 2 entries;
    the 6th pick (TURN_SCHEDULE[11] = team B seat 2) is always the missing slot.
    """
    team_a: list[BrawlerInfo] = []
    team_b: list[BrawlerInfo] = []
    for pick_idx, (_, brawler) in enumerate(picks):
        _, team, _ = TURN_SCHEDULE[6 + pick_idx]
        (team_a if team == "A" else team_b).append(brawler)
    return team_a, team_b


def load_context(
    data_dir: Path,
    terminal_ckpt: Path,
    draft_ckpt: Path,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
) -> DraftContext:
    vocabs = load_vocabs(data_dir)

    terminal_model = BrawlModel(
        n_events=vocabs.n_events,
        n_modes=vocabs.n_modes,
        n_chars=vocabs.n_chars,
        n_classes=vocabs.n_classes,
        n_ranges=vocabs.n_ranges,
        n_destructs=vocabs.n_destructs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout_p=0.3,
        key=jax.random.PRNGKey(0),
    )
    terminal_model = eqx.tree_deserialise_leaves(terminal_ckpt, terminal_model)

    char_encs_all, event_idxs_arr, _ = precompute_char_encs(terminal_model, vocabs, data_dir)
    h_terminal = char_encs_all.shape[-1]
    event_enc_row = {int(event_idxs_arr[i]): i for i in range(len(event_idxs_arr))}

    q_net = DraftQNetwork(
        h_terminal=h_terminal,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        key=jax.random.PRNGKey(0),
    )
    q_net = eqx.tree_deserialise_leaves(draft_ckpt, q_net)

    bclass: list[dict] = json.loads((data_dir / "brawler_class.json").read_text())
    brawlers: list[BrawlerInfo] = []
    for b in bclass:
        bid = b["id"]
        if bid in vocabs.char_to_idx:
            brawlers.append(BrawlerInfo(
                id=bid,
                name=b["name"],
                brawler_class=b["class"],
                rarity=b["rarity"],
                char_idx=vocabs.char_to_idx[bid],
            ))
    brawlers.sort(key=lambda b: b.char_idx)

    # char_meta_table[char_idx] = (class_idx, range_idx, destruct_idx)
    brawler_range_data: list[dict] = json.loads((data_dir / "brawler_effective_range.json").read_text())
    brawler_destruct_data: list[dict] = json.loads((data_dir / "brawler_destruction.json").read_text())
    class_map = {x["id"]: vocabs.class_to_idx[x["class"]] for x in bclass}
    range_map = {x["id"]: vocabs.range_to_idx[x["range"]] for x in brawler_range_data}
    destruct_map = {x["id"]: vocabs.destruct_to_idx[x["destruction"]] for x in brawler_destruct_data}
    char_meta_table = np.zeros((vocabs.n_chars, 3), dtype=np.int32)
    for char_id, cidx in vocabs.char_to_idx.items():
        char_meta_table[cidx] = [class_map[char_id], range_map[char_id], destruct_map[char_id]]

    events_raw: list[dict] = json.loads((data_dir / "events.json").read_text())
    events: list[EventInfo] = []
    for e in events_raw:
        if e["id"] in vocabs.event_to_idx:
            events.append(EventInfo(
                id=e["id"],
                mode=e["mode"],
                mode_id=e["modeId"],
                map_name=e["map"],
                event_idx=vocabs.event_to_idx[e["id"]],
                mode_idx=vocabs.mode_to_idx[e["modeId"]],
            ))

    winrates_raw: list[dict] = json.loads((data_dir / "winrates_leg1_20260827.json").read_text())
    winrates: dict[int, dict[int, float]] = {}
    for entry in winrates_raw:
        winrates.setdefault(entry["char_id"], {})[entry["event_id"]] = entry["z_score"]

    pickrates_raw: list[dict] = json.loads((data_dir / "pickrates_leg1_20260827.json").read_text())
    pickrates: dict[int, dict[int, float]] = {}
    for entry in pickrates_raw:
        pickrates.setdefault(entry["char_id"], {})[entry["event_id"]] = entry["z_score"]

    return DraftContext(
        q_net=q_net,
        terminal_model=terminal_model,
        char_meta_table=char_meta_table,
        char_encs_all=char_encs_all,
        event_enc_row=event_enc_row,
        brawlers=brawlers,
        events=events,
        brawler_names=sorted(b.name for b in brawlers),
        map_names=[e.map_name for e in events],
        n_chars=vocabs.n_chars,
        winrates=winrates,
        pickrates=pickrates,
        _brawler_by_char_idx={b.char_idx: b for b in brawlers},
        _brawler_by_name={b.name: b for b in brawlers},
    )


def score_map(ctx: DraftContext, event: EventInfo) -> list[tuple[BrawlerInfo, float]]:
    """Rank brawlers by winrate z-score for this event."""
    results = [
        (b, ctx.winrates.get(b.id, {}).get(event.id, 0.0))
        for b in ctx.brawlers
    ]
    return sorted(results, key=lambda x: x[1], reverse=True)


def get_q_values(
    ctx: DraftContext,
    event: EventInfo,
    ally_bans: list[BrawlerInfo],
    enemy_bans: list[BrawlerInfo],
    picks: list[tuple[bool, BrawlerInfo]],
    phase: int,
    ally_first: bool,
    local_pool: set[int] | None = None,
) -> np.ndarray:
    """Compute Q-values [n_chars] for the current draft phase.

    Phase 0-2: ally bans; phase 3-5: enemy bans; phase 6-11: picks 0-5.
    local_pool: brawler IDs the acting player can select; others are LOCALLY_BANNED.
    """
    obs, turn_token = _build_obs(
        ctx.n_chars, ally_bans, enemy_bans, picks, phase, ally_first,
        local_pool=local_pool, brawlers=ctx.brawlers,
    )
    enc_row = ctx.event_enc_row[event.event_idx]
    return np.array(_q_apply(
        ctx.q_net,
        jnp.array(ctx.char_encs_all[enc_row], dtype=jnp.float32),
        jnp.array(obs, dtype=jnp.int32),
        jnp.array(turn_token, dtype=jnp.int32),
    ))


def get_terminal_pick6_scores(
    ctx: DraftContext,
    event: EventInfo,
    picks: list[tuple[bool, BrawlerInfo]],
    excluded: set[int],
    ally_first: bool = False,
) -> list[tuple[BrawlerInfo, float]]:
    """Score pick-6 candidates with the terminal BrawlModel.

    Returns (brawler, prob) sorted descending by P(ally wins), where prob is
    always from the ally's perspective regardless of which team picks last.
    """
    team_a_picks, team_b_partial = _pick6_team_split(picks)

    team_a_chars = np.array([b.char_idx for b in team_a_picks], dtype=np.int32)    # [3]
    team_a_meta = ctx.char_meta_table[team_a_chars]                                  # [3, 3]
    team_b_partial_chars = np.array([b.char_idx for b in team_b_partial], dtype=np.int32)  # [2]
    team_b_partial_meta = ctx.char_meta_table[team_b_partial_chars]                  # [2, 3]

    candidates = [b for b in ctx.brawlers if b.id not in excluded]
    if not candidates:
        return []

    cand_chars = np.array([b.char_idx for b in candidates], dtype=np.int32)
    cand_meta = ctx.char_meta_table[cand_chars]  # [n, 3]

    logits = np.array(_terminal_score_pick6_batch(
        ctx.terminal_model,
        jnp.array(event.event_idx),
        jnp.array(event.mode_idx),
        jnp.array(team_a_chars),
        jnp.array(team_a_meta),
        jnp.array(team_b_partial_chars),
        jnp.array(team_b_partial_meta),
        jnp.array(cand_chars),
        jnp.array(cand_meta),
    ))

    # sigmoid(-logit) = P(team B wins); team B always picks last
    probs = 1.0 / (1.0 + np.exp(logits))
    if ally_first:
        # ally is team A, so team B is the enemy; flip to get P(ally wins)
        probs = 1.0 - probs
    results = [(b, float(p)) for b, p in zip(candidates, probs)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results
