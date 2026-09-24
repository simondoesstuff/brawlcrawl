"""Soft Q-learning training for the draft-phase agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from tqdm import tqdm

from geneus.data import Vocabs, load_vocabs
from geneus.draft.env import (
    BELLMAN_SIGNS,
    TERMINAL_SIGN,
    TURN_SCHEDULE,
    DraftConfig,
    DraftState,
    PlayerConfig,
    get_player_observed_states,
    get_valid_action_mask,
    player_idx,
    step,
)
from geneus.draft.model import DraftQNetwork
from geneus.model import BrawlModel

app = typer.Typer(add_completion=False)

_DATA_DIR = Path("data")
_OUT_DIR = Path("data/draft_model")
_MODEL_FILENAME = "draft_q.eqx"
_BEST_MODEL_FILENAME = "draft_q_best.eqx"
_CHECKPOINTS_DIRNAME = "checkpoints"

# Precomputed JAX constants for the Bellman backup
_BELLMAN_SIGNS_JAX = jnp.array(BELLMAN_SIGNS)  # [11]
_TERMINAL_SIGN_JAX = jnp.array(TERMINAL_SIGN)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _build_char_meta_table(vocabs: Vocabs, data_dir: Path) -> np.ndarray:
    """Build [n_chars, 3] int32 array: (class_idx, range_idx, destruct_idx) per char_idx."""
    brawler_class: list[dict[str, Any]] = json.loads(
        (data_dir / "brawler_class.json").read_text()
    )
    brawler_range: list[dict[str, Any]] = json.loads(
        (data_dir / "brawler_effective_range.json").read_text()
    )
    brawler_destruct: list[dict[str, Any]] = json.loads(
        (data_dir / "brawler_destruction.json").read_text()
    )

    class_map = {x["id"]: vocabs.class_to_idx[x["class"]] for x in brawler_class}
    range_map = {x["id"]: vocabs.range_to_idx[x["range"]] for x in brawler_range}
    destruct_map = {
        x["id"]: vocabs.destruct_to_idx[x["destruction"]] for x in brawler_destruct
    }

    table = np.zeros((vocabs.n_chars, 3), dtype=np.int32)
    for char_id, cidx in vocabs.char_to_idx.items():
        table[cidx] = [class_map[char_id], range_map[char_id], destruct_map[char_id]]
    return table


def precompute_char_encs(
    terminal_model: BrawlModel,
    vocabs: Vocabs,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frozen character encodings for every (event, char) pair.

    Returns:
        char_encs_all: float32 [n_events, n_chars, h]
        event_idxs:    int32   [n_events]
        mode_idxs:     int32   [n_events]
    """
    events: list[dict[str, Any]] = json.loads((data_dir / "events.json").read_text())
    events_sorted = sorted(events, key=lambda x: x["id"])

    event_idxs = np.array(
        [vocabs.event_to_idx[e["id"]] for e in events_sorted], dtype=np.int32
    )
    mode_idxs = np.array(
        [vocabs.mode_to_idx[e["modeId"]] for e in events_sorted], dtype=np.int32
    )

    char_meta_table = _build_char_meta_table(vocabs, data_dir)
    all_char_idxs = jnp.arange(vocabs.n_chars, dtype=jnp.int32)
    all_char_meta = jnp.array(char_meta_table)  # [n_chars, 3]

    # vmap _encode_char over all chars for a single event
    def _encode_all_chars(ev_idx: jax.Array, mo_idx: jax.Array) -> jax.Array:
        map_emb = terminal_model.embed_map_id(ev_idx) + terminal_model.embed_map_mode(
            mo_idx
        )
        return jax.vmap(lambda c, m: terminal_model._encode_char(c, m, map_emb))(
            all_char_idxs, all_char_meta
        )

    _encode_all_chars_jit = eqx.filter_jit(_encode_all_chars)

    n_events = len(event_idxs)
    # Determine h_terminal from a single call
    sample_enc = _encode_all_chars_jit(
        jnp.array(event_idxs[0]), jnp.array(mode_idxs[0])
    )
    h_terminal = sample_enc.shape[-1]

    char_encs_all = np.zeros((n_events, vocabs.n_chars, h_terminal), dtype=np.float32)
    char_encs_all[0] = np.array(sample_enc)
    for i in range(1, n_events):
        encs = _encode_all_chars_jit(jnp.array(event_idxs[i]), jnp.array(mode_idxs[i]))
        char_encs_all[i] = np.array(encs)

    return char_encs_all, event_idxs, mode_idxs


# ---------------------------------------------------------------------------
# Episode simulation
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _q_apply(
    q_net: DraftQNetwork,
    char_encs: jax.Array,
    player_states: jax.Array,
    turn_token: jax.Array,
) -> jax.Array:
    return q_net(char_encs, player_states, turn_token)


@eqx.filter_jit
def _terminal_logit(
    terminal_model: BrawlModel,
    event_idx: jax.Array,
    mode_idx: jax.Array,
    picks_a: jax.Array,
    picks_a_meta: jax.Array,
    picks_b: jax.Array,
    picks_b_meta: jax.Array,
) -> jax.Array:
    return terminal_model(
        event_idx, mode_idx, picks_a, picks_a_meta, picks_b, picks_b_meta
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()


def simulate_episode(
    q_net: DraftQNetwork,
    char_encs_all: np.ndarray,  # [n_events, n_chars, h]
    event_idxs: np.ndarray,
    mode_idxs: np.ndarray,
    char_meta_table: np.ndarray,  # [n_chars, 3]
    terminal_model: BrawlModel,
    config: DraftConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Simulate one draft episode and return trajectory data.

    With probability config.skip_ban_prob, the ban phase is skipped entirely:
    no char is banned, and turns 0-5 are left inactive (excluded from the loss
    by `active`) rather than simulated. Pick 1 then observes every char as
    AVAILABLE or LOCALLY_BANNED — a state the ordinary ban sub-MDP can never
    produce, since each of its 6 turns must select a char to ban.

    Returns a dict with keys:
        char_encs:    [n_chars, h]   — same for all 12 turns
        player_states:[12, n_chars]
        turn_tokens:  [12]
        valid_masks:  [12, n_chars]
        actions:      [12]
        temperatures: [12]
        active:       [12]          — False for skipped (no-ban) turns
        terminal_logit: scalar
    """
    n_events = len(event_idxs)
    n_chars = char_encs_all.shape[1]
    event_i = int(rng.integers(0, n_events))
    char_encs = char_encs_all[event_i]  # [n_chars, h]

    # Initialise player configs
    player_configs: list[PlayerConfig] = []
    for i in range(6):
        team = "A" if i < 3 else "B"
        pool_size = int(rng.integers(config.pool_min, config.pool_max + 1))
        perm = rng.permutation(n_chars)
        local_pool = np.zeros(n_chars, dtype=bool)
        local_pool[perm[:pool_size]] = True
        log_t = rng.uniform(np.log(config.temp_min), np.log(config.temp_max))
        player_configs.append(
            PlayerConfig(
                local_pool=local_pool, temperature=float(np.exp(log_t)), team=team
            )
        )

    state = DraftState(
        team_a_bans=np.zeros(n_chars, dtype=bool),
        team_b_bans=np.zeros(n_chars, dtype=bool),
        picks_a=np.zeros(n_chars, dtype=bool),
        picks_b=np.zeros(n_chars, dtype=bool),
        char_encs=char_encs,
        event_idx=int(event_idxs[event_i]),
        mode_idx=int(mode_idxs[event_i]),
        player_configs=player_configs,
    )

    all_player_states = np.zeros((12, n_chars), dtype=np.int32)
    all_turn_tokens = np.zeros(12, dtype=np.int32)
    all_valid_masks = np.zeros((12, n_chars), dtype=bool)
    all_actions = np.zeros(12, dtype=np.int32)
    all_temperatures = np.zeros(12, dtype=np.float32)
    all_active = np.zeros(12, dtype=bool)

    skip_bans = rng.random() < config.skip_ban_prob
    start_turn = 6 if skip_bans else 0

    for turn_idx in range(start_turn, 12):
        token, team, seat = TURN_SCHEDULE[turn_idx]
        p = player_configs[player_idx(team, seat)]

        obs = get_player_observed_states(state, turn_idx)
        mask = get_valid_action_mask(state, turn_idx)

        assert mask.any(), f"No valid actions at turn {turn_idx} — pool_min too small?"

        q_vals = np.array(
            _q_apply(
                q_net,
                jnp.array(char_encs, dtype=jnp.float32),
                jnp.array(obs, dtype=jnp.int32),
                jnp.array(token, dtype=jnp.int32),
            )
        )

        q_masked = np.where(mask, q_vals, -np.inf)
        temp = max(p.temperature, 1e-6)
        probs = _softmax(q_masked / temp)
        action = int(rng.choice(n_chars, p=probs))

        all_player_states[turn_idx] = obs
        all_turn_tokens[turn_idx] = token
        all_valid_masks[turn_idx] = mask
        all_actions[turn_idx] = action
        all_temperatures[turn_idx] = temp
        all_active[turn_idx] = True

        state = step(state, action, turn_idx)

    # Terminal reward: win logit for team A
    picks_a_idxs = np.where(state.picks_a)[0]
    picks_b_idxs = np.where(state.picks_b)[0]
    logit = float(
        _terminal_logit(
            terminal_model,
            jnp.array(state.event_idx, dtype=jnp.int32),
            jnp.array(state.mode_idx, dtype=jnp.int32),
            jnp.array(picks_a_idxs, dtype=jnp.int32),
            jnp.array(char_meta_table[picks_a_idxs], dtype=jnp.int32),
            jnp.array(picks_b_idxs, dtype=jnp.int32),
            jnp.array(char_meta_table[picks_b_idxs], dtype=jnp.int32),
        )
    )

    return {
        "char_encs": char_encs,  # [n_chars, h]
        "player_states": all_player_states,  # [12, n_chars]
        "turn_tokens": all_turn_tokens,  # [12]
        "valid_masks": all_valid_masks,  # [12, n_chars]
        "actions": all_actions,  # [12]
        "temperatures": all_temperatures,  # [12]
        "active": all_active,  # [12]
        "terminal_logit": np.float32(logit),
    }


def simulate_batch(
    q_net: DraftQNetwork,
    char_encs_all: np.ndarray,
    event_idxs: np.ndarray,
    mode_idxs: np.ndarray,
    char_meta_table: np.ndarray,
    terminal_model: BrawlModel,
    config: DraftConfig,
    rng: np.random.Generator,
    n_episodes: int,
) -> dict[str, Any]:
    """Collect n_episodes and stack into JAX arrays for the loss computation."""
    episodes = [
        simulate_episode(
            q_net,
            char_encs_all,
            event_idxs,
            mode_idxs,
            char_meta_table,
            terminal_model,
            config,
            rng,
        )
        for _ in range(n_episodes)
    ]

    return {
        "char_encs": jnp.array(
            np.stack([e["char_encs"] for e in episodes])
        ),  # [N, n_chars, h]
        "player_states": jnp.array(
            np.stack([e["player_states"] for e in episodes])
        ),  # [N, 12, n_chars]
        "turn_tokens": jnp.array(
            np.stack([e["turn_tokens"] for e in episodes])
        ),  # [N, 12]
        "valid_masks": jnp.array(
            np.stack([e["valid_masks"] for e in episodes])
        ),  # [N, 12, n_chars]
        "actions": jnp.array(np.stack([e["actions"] for e in episodes])),  # [N, 12]
        "temperatures": jnp.array(
            np.stack([e["temperatures"] for e in episodes])
        ),  # [N, 12]
        "active": jnp.array(np.stack([e["active"] for e in episodes])),  # [N, 12]
        "terminal_logits": jnp.array(
            np.array([e["terminal_logit"] for e in episodes])
        ),  # [N]
    }


# ---------------------------------------------------------------------------
# Loss and training step
# ---------------------------------------------------------------------------


def _compute_loss(
    q_net: DraftQNetwork, batch: dict[str, Any], bellman_temp: float
) -> jax.Array:
    """Soft Q MSE loss with Bellman backup.

    Q_local is in logit space from the acting team's perspective.
    Bellman targets flip sign between consecutive turns that switch teams
    (zero-sum minimax with soft entropy regularisation).

    `bellman_temp` is a single fixed entropy temperature used to build every
    V — deliberately NOT the per-player, per-episode random `temperatures`
    used only for action sampling during rollout. V[t] becomes the
    regression target for a *different* player's turn (t-1); using each
    player's own random exploration temperature there would make identical
    (obs, turn_token) pairs regress toward different targets depending on an
    unobserved random draw belonging to someone else's turn. See
    DraftConfig.bellman_temp.

    `active` excludes turns skipped by config.skip_ban_prob (the ban phase
    wasn't simulated for that episode) from the loss. Those positions still
    get a forward pass and a well-defined (if meaningless) target — masking
    happens only in the final average — so no NaN/inf guarding is needed:
    zero-filled valid_masks there make V finite (logsumexp of an all -1e9
    row).
    """
    char_encs = batch["char_encs"]  # [N, n_chars, h]
    player_states = batch["player_states"]  # [N, 12, n_chars]
    turn_tokens = batch["turn_tokens"]  # [N, 12]
    valid_masks = batch["valid_masks"]  # [N, 12, n_chars]
    actions = batch["actions"]  # [N, 12]
    active = batch["active"]  # [N, 12]
    terminal_logits = batch["terminal_logits"]  # [N]

    # Q values at all (episode, turn) pairs  →  [N, 12, n_chars]
    def q_for_episode(ce, ps_seq, tt_seq):
        return jax.vmap(lambda ps, tt: q_net(ce, ps, tt))(ps_seq, tt_seq)

    q_all = jax.vmap(q_for_episode)(char_encs, player_states, turn_tokens)

    # Bellman targets (stopgradient so we don't bootstrap through gradients)
    q_sg = jax.lax.stop_gradient(q_all)

    # Soft values: V[n, t] = bellman_temp * logsumexp(q_sg[n,t,valid] / bellman_temp)
    q_masked = jnp.where(valid_masks, q_sg, -1e9)
    V = bellman_temp * jax.scipy.special.logsumexp(
        q_masked / bellman_temp, axis=-1
    )  # [N, 12]

    # targets[:, 0..10] = V[:, 1..11] * bellman_sign
    # targets[:, 11]    = terminal_logit * TERMINAL_SIGN
    targets_early = V[:, 1:] * _BELLMAN_SIGNS_JAX[None, :]  # [N, 11]
    targets_last = terminal_logits * _TERMINAL_SIGN_JAX  # [N]
    targets = jnp.concatenate([targets_early, targets_last[:, None]], axis=1)  # [N, 12]

    # Q predictions at the taken actions
    N = q_all.shape[0]
    ep = jnp.arange(N)[:, None]
    tr = jnp.arange(12)[None, :]
    q_pred = q_all[ep, tr, actions]  # [N, 12]

    active_f = active.astype(q_pred.dtype)
    sq_err = (q_pred - targets) ** 2 * active_f
    return jnp.sum(sq_err) / jnp.clip(jnp.sum(active_f), 1.0, None)


def make_step_fn(optimizer: optax.GradientTransformation, bellman_temp: float):
    """Return a JIT-compiled (q_net, opt_state, batch) → (q_net, opt_state, loss) step."""

    @eqx.filter_jit
    def step(
        q_net: DraftQNetwork,
        opt_state: optax.OptState,
        batch: dict[str, Any],
    ) -> tuple[DraftQNetwork, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(_compute_loss)(
            q_net, batch, bellman_temp
        )
        updates, new_state = optimizer.update(
            grads, opt_state, eqx.filter(q_net, eqx.is_array)
        )
        return eqx.apply_updates(q_net, updates), new_state, loss

    return step


# ---------------------------------------------------------------------------
# Checkpointing (full training state for resumption)
# ---------------------------------------------------------------------------


def save_checkpoint(
    ckpt_dir: Path,
    q_net: DraftQNetwork,
    opt_state: optax.OptState,
    rng: np.random.Generator,
    iteration: int,
    loss_history: list[float],
    best_win_rate: float,
    eval_history: list[dict[str, float]],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt_dir / "model.eqx", q_net)
    eqx.tree_serialise_leaves(ckpt_dir / "opt_state.eqx", opt_state)
    np.save(ckpt_dir / "rng.npy", rng.bit_generator.state, allow_pickle=True)  # type: ignore[arg-type]
    meta = {
        "iteration": iteration,
        "loss_history": loss_history,
        "best_win_rate": best_win_rate,
        "eval_history": eval_history,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))


def load_checkpoint(
    ckpt_dir: Path,
    q_net: DraftQNetwork,
    opt_state: optax.OptState,
    rng: np.random.Generator,
) -> tuple[
    DraftQNetwork,
    optax.OptState,
    np.random.Generator,
    int,
    list[float],
    float,
    list[dict[str, float]],
]:
    q_net = eqx.tree_deserialise_leaves(ckpt_dir / "model.eqx", q_net)
    opt_state = eqx.tree_deserialise_leaves(ckpt_dir / "opt_state.eqx", opt_state)
    rng.bit_generator.state = np.load(ckpt_dir / "rng.npy", allow_pickle=True).item()
    meta: dict[str, Any] = json.loads((ckpt_dir / "meta.json").read_text())
    return (
        q_net,
        opt_state,
        rng,
        meta["iteration"],
        meta["loss_history"],
        # Older checkpoints (pre win-rate tracking) won't have these keys.
        meta.get("best_win_rate", -1.0),
        meta.get("eval_history", []),
    )


# ---------------------------------------------------------------------------
# Win-rate evaluation vs random opponent
# ---------------------------------------------------------------------------


def _eval_episode(
    q_net: DraftQNetwork,
    char_encs_all: np.ndarray,
    event_idxs: np.ndarray,
    mode_idxs: np.ndarray,
    char_meta_table: np.ndarray,
    terminal_model: BrawlModel,
    config: DraftConfig,
    rng: np.random.Generator,
    eval_team: str,
    eval_temp: float,
) -> bool:
    """One episode: eval_team uses Q-net at eval_temp, opponent plays uniformly at random.
    Returns True if eval_team wins (terminal logit favours them)."""
    n_events = len(event_idxs)
    n_chars = char_encs_all.shape[1]
    event_i = int(rng.integers(0, n_events))
    char_encs = char_encs_all[event_i]

    player_configs: list[PlayerConfig] = []
    for i in range(6):
        team = "A" if i < 3 else "B"
        pool_size = int(rng.integers(config.pool_min, config.pool_max + 1))
        perm = rng.permutation(n_chars)
        local_pool = np.zeros(n_chars, dtype=bool)
        local_pool[perm[:pool_size]] = True
        player_configs.append(
            PlayerConfig(local_pool=local_pool, temperature=eval_temp, team=team)
        )

    state = DraftState(
        team_a_bans=np.zeros(n_chars, dtype=bool),
        team_b_bans=np.zeros(n_chars, dtype=bool),
        picks_a=np.zeros(n_chars, dtype=bool),
        picks_b=np.zeros(n_chars, dtype=bool),
        char_encs=char_encs,
        event_idx=int(event_idxs[event_i]),
        mode_idx=int(mode_idxs[event_i]),
        player_configs=player_configs,
    )

    for turn_idx in range(12):
        token, team, seat = TURN_SCHEDULE[turn_idx]
        obs = get_player_observed_states(state, turn_idx)
        mask = get_valid_action_mask(state, turn_idx)

        if team == eval_team:
            q_vals = np.array(
                _q_apply(
                    q_net,
                    jnp.array(char_encs, dtype=jnp.float32),
                    jnp.array(obs, dtype=jnp.int32),
                    jnp.array(token, dtype=jnp.int32),
                )
            )
            q_masked = np.where(mask, q_vals, -np.inf)
            action = int(
                rng.choice(n_chars, p=_softmax(q_masked / max(eval_temp, 1e-6)))
            )
        else:
            valid_idxs = np.where(mask)[0]
            action = int(rng.choice(valid_idxs))

        state = step(state, action, turn_idx)

    picks_a_idxs = np.where(state.picks_a)[0]
    picks_b_idxs = np.where(state.picks_b)[0]
    logit = float(
        _terminal_logit(
            terminal_model,
            jnp.array(state.event_idx, dtype=jnp.int32),
            jnp.array(state.mode_idx, dtype=jnp.int32),
            jnp.array(picks_a_idxs, dtype=jnp.int32),
            jnp.array(char_meta_table[picks_a_idxs], dtype=jnp.int32),
            jnp.array(picks_b_idxs, dtype=jnp.int32),
            jnp.array(char_meta_table[picks_b_idxs], dtype=jnp.int32),
        )
    )
    return (logit > 0) if eval_team == "A" else (logit < 0)


def evaluate_vs_random(
    q_net: DraftQNetwork,
    char_encs_all: np.ndarray,
    event_idxs: np.ndarray,
    mode_idxs: np.ndarray,
    char_meta_table: np.ndarray,
    terminal_model: BrawlModel,
    config: DraftConfig,
    rng: np.random.Generator,
    n_episodes: int = 400,
    eval_temp: float = 0.1,
) -> dict[str, float]:
    """Evaluate Q-net win rate vs a uniformly-random opponent over n_episodes.
    Plays n_episodes//2 as team A and n_episodes//2 as team B to remove first-pick bias.
    Returns win rates as A, as B, and combined."""
    half = n_episodes // 2
    wins_a = sum(
        _eval_episode(
            q_net,
            char_encs_all,
            event_idxs,
            mode_idxs,
            char_meta_table,
            terminal_model,
            config,
            rng,
            "A",
            eval_temp,
        )
        for _ in range(half)
    )
    wins_b = sum(
        _eval_episode(
            q_net,
            char_encs_all,
            event_idxs,
            mode_idxs,
            char_meta_table,
            terminal_model,
            config,
            rng,
            "B",
            eval_temp,
        )
        for _ in range(half)
    )
    return {
        "win_rate_as_a": wins_a / half,
        "win_rate_as_b": wins_b / half,
        "win_rate_combined": (wins_a + wins_b) / (2 * half),
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def train(
    terminal_ckpt: Annotated[
        Path, typer.Option(help="Frozen BrawlModel checkpoint (.eqx)")
    ] = Path("data/model/model.eqx"),
    out: Annotated[
        Path,
        typer.Option(
            help=(
                "Output directory for training artifacts "
                f"({_MODEL_FILENAME}, {_BEST_MODEL_FILENAME}, {_CHECKPOINTS_DIRNAME}/)"
            )
        ),
    ] = _OUT_DIR,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = _DATA_DIR,
    n_iters: Annotated[int, typer.Option(help="Training iterations")] = 7_000,
    batch_episodes: Annotated[
        int, typer.Option(help="Episodes per gradient step")
    ] = 64,
    lr: Annotated[float, typer.Option(help="Adam learning rate")] = 1e-3,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay")] = 1e-4,
    d_model: Annotated[int, typer.Option(help="Transformer hidden dimension")] = 64,
    n_heads: Annotated[int, typer.Option(help="Attention heads")] = 4,
    n_layers: Annotated[int, typer.Option(help="Transformer layers")] = 2,
    terminal_embed_dim: Annotated[
        int, typer.Option(help="embed_dim of the frozen BrawlModel")
    ] = 32,
    terminal_hidden_dim: Annotated[
        int, typer.Option(help="hidden_dim of the frozen BrawlModel")
    ] = 64,
    pool_min: Annotated[
        int, typer.Option(help="Minimum local pool size per player")
    ] = 12,
    pool_max: Annotated[
        int | None,
        typer.Option(help="Maximum local pool size per player (default: unbounded)"),
    ] = None,
    temp_min: Annotated[float, typer.Option(help="Minimum player temperature")] = 0.05,
    temp_max: Annotated[float, typer.Option(help="Maximum player temperature")] = 2.0,
    skip_ban_prob: Annotated[
        float,
        typer.Option(
            help="Probability an episode skips the ban phase entirely (no chars banned)"
        ),
    ] = 0.05,
    bellman_temp: Annotated[
        float,
        typer.Option(
            help=(
                "Fixed entropy temperature for the soft-Bellman target V "
                "(decoupled from the per-player random exploration temperature)"
            )
        ),
    ] = 0.1,
    checkpoint_every: Annotated[
        int, typer.Option(help="Save checkpoint every N iters (0=off)")
    ] = 5000,
    log_every: Annotated[int, typer.Option(help="Log interval in iterations")] = 100,
    eval_every: Annotated[
        int, typer.Option(help="Win-rate-vs-random eval interval in iterations (0=off)")
    ] = 100,
    eval_episodes: Annotated[
        int, typer.Option(help="Episodes per periodic eval (split evenly A/B)")
    ] = 200,
    eval_temp: Annotated[
        float, typer.Option(help="Q-net temperature during periodic eval")
    ] = 0.1,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    resume_from: Annotated[
        Path | None, typer.Option(help="Checkpoint dir to resume from")
    ] = None,
) -> None:
    typer.echo("Loading vocabs and terminal model...")
    vocabs = load_vocabs(data_dir)
    pool_max = pool_max if pool_max is not None else vocabs.n_chars

    terminal_model = BrawlModel(
        n_events=vocabs.n_events,
        n_modes=vocabs.n_modes,
        n_chars=vocabs.n_chars,
        n_classes=vocabs.n_classes,
        n_ranges=vocabs.n_ranges,
        n_destructs=vocabs.n_destructs,
        embed_dim=terminal_embed_dim,
        hidden_dim=terminal_hidden_dim,
        key=jax.random.PRNGKey(0),
    )
    terminal_model = eqx.tree_deserialise_leaves(terminal_ckpt, terminal_model)
    typer.echo(f"  Loaded terminal model from {terminal_ckpt}")

    typer.echo("Precomputing frozen character encodings...")
    char_encs_all, event_idxs, mode_idxs = precompute_char_encs(
        terminal_model, vocabs, data_dir
    )
    h_terminal = char_encs_all.shape[-1]
    char_meta_table = _build_char_meta_table(vocabs, data_dir)
    typer.echo(
        f"  {char_encs_all.shape[0]} events × {char_encs_all.shape[1]} chars × {h_terminal}d"
    )

    key = jax.random.PRNGKey(seed)
    q_net = DraftQNetwork(
        h_terminal=h_terminal,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        key=key,
    )
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(q_net, eqx.is_array)))
    typer.echo(f"  DraftQNetwork: {n_params:,} parameters")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=max(1, n_iters // 100),
        decay_steps=n_iters,
        end_value=lr * 0.001,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(q_net, eqx.is_array))
    step_fn = make_step_fn(optimizer, bellman_temp)

    config = DraftConfig(
        pool_min=pool_min,
        pool_max=pool_max,
        temp_min=temp_min,
        temp_max=temp_max,
        skip_ban_prob=skip_ban_prob,
        bellman_temp=bellman_temp,
    )
    rng = np.random.default_rng(seed)

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / _MODEL_FILENAME  # latest weights, overwritten periodically
    best_model_path = out / _BEST_MODEL_FILENAME  # best win-rate-vs-random so far
    ckpt_dir = out / _CHECKPOINTS_DIRNAME
    if checkpoint_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_history: list[float] = []
    best_win_rate = -1.0
    eval_history: list[dict[str, float]] = []
    start_iter = 1

    if resume_from is not None:
        (
            q_net,
            opt_state,
            rng,
            resumed_iter,
            loss_history,
            best_win_rate,
            eval_history,
        ) = load_checkpoint(resume_from, q_net, opt_state, rng)
        start_iter = resumed_iter + 1
        typer.echo(
            f"  Resumed from iter {resumed_iter}  (best_win_rate={best_win_rate:.3f})"
        )

    typer.echo(
        f"\nTraining iters {start_iter}–{n_iters} ({batch_episodes} episodes/step)..."
    )
    pbar = tqdm(range(start_iter, n_iters + 1), desc="Draft RL", unit="iter")
    for i in pbar:
        batch = simulate_batch(
            q_net,
            char_encs_all,
            event_idxs,
            mode_idxs,
            char_meta_table,
            terminal_model,
            config,
            rng,
            batch_episodes,
        )
        q_net, opt_state, loss = step_fn(q_net, opt_state, batch)
        loss_val = float(loss)
        loss_history.append(loss_val)

        # Self-play loss is not a fitness signal: a converged minimax
        # equilibrium can sit at a nonzero, noisy loss indefinitely, and
        # nothing about "lower loss so far" implies "stronger policy" when
        # both sides are moving targets for each other. So `model_path`
        # tracks the *latest* weights (overwritten periodically below, plus
        # unconditionally once training ends), and `best_model_path` tracks
        # the checkpoint with the best measured win-rate-vs-random — the one
        # number here that's actually evaluated against a fixed opponent.
        if i % log_every == 0:
            eqx.tree_serialise_leaves(model_path, q_net)
            mean_loss = float(np.mean(loss_history[-log_every:]))
            pbar.set_postfix(loss=f"{mean_loss:.4f}", best_wr=f"{best_win_rate:.3f}")

        if checkpoint_every > 0 and i % checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"ckpt_{i:07d}",
                q_net,
                opt_state,
                rng,
                i,
                loss_history,
                best_win_rate,
                eval_history,
            )

        if eval_every > 0 and i % eval_every == 0:
            # A fresh RNG keeps this independent of (and reproducible across)
            # the training rng, so it doesn't perturb resume determinism, and
            # gives a fixed, low-variance eval scenario set to compare
            # iterations against.
            eval_results = evaluate_vs_random(
                q_net,
                char_encs_all,
                event_idxs,
                mode_idxs,
                char_meta_table,
                terminal_model,
                config,
                np.random.default_rng(seed),
                n_episodes=eval_episodes,
                eval_temp=eval_temp,
            )
            eval_history.append({"iteration": i, **eval_results})
            pbar.write(
                f"  [eval @ iter {i}] "
                f"win_A={eval_results['win_rate_as_a']:.3f}  "
                f"win_B={eval_results['win_rate_as_b']:.3f}  "
                f"combined={eval_results['win_rate_combined']:.3f}"
            )
            if eval_results["win_rate_combined"] > best_win_rate:
                best_win_rate = eval_results["win_rate_combined"]
                eqx.tree_serialise_leaves(best_model_path, q_net)

    eqx.tree_serialise_leaves(model_path, q_net)
    typer.echo(f"\nLatest weights  →  {model_path}")
    if best_win_rate >= 0:
        typer.echo(
            f"Best win-rate-vs-random ({best_win_rate:.3f})  →  {best_model_path}"
        )
    else:
        typer.echo(
            "No periodic eval ran (--eval-every 0) — no best-win-rate checkpoint saved"
        )


def _load_terminal_model_and_encs(
    terminal_ckpt: Path,
    data_dir: Path,
    terminal_embed_dim: int,
    terminal_hidden_dim: int,
) -> tuple[BrawlModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, "Vocabs"]:
    vocabs = load_vocabs(data_dir)
    terminal_model = BrawlModel(
        n_events=vocabs.n_events,
        n_modes=vocabs.n_modes,
        n_chars=vocabs.n_chars,
        n_classes=vocabs.n_classes,
        n_ranges=vocabs.n_ranges,
        n_destructs=vocabs.n_destructs,
        embed_dim=terminal_embed_dim,
        hidden_dim=terminal_hidden_dim,
        key=jax.random.PRNGKey(0),
    )
    terminal_model = eqx.tree_deserialise_leaves(terminal_ckpt, terminal_model)
    char_encs_all, event_idxs, mode_idxs = precompute_char_encs(
        terminal_model, vocabs, data_dir
    )
    char_meta_table = _build_char_meta_table(vocabs, data_dir)
    return terminal_model, char_encs_all, event_idxs, mode_idxs, char_meta_table, vocabs


@app.command()
def eval(
    checkpoint: Annotated[
        Path, typer.Argument(help="Checkpoint dir (contains model.eqx + meta.json)")
    ],
    terminal_ckpt: Annotated[Path, typer.Option()] = Path("data/model/model.eqx"),
    data_dir: Annotated[Path, typer.Option()] = _DATA_DIR,
    terminal_embed_dim: Annotated[int, typer.Option()] = 32,
    terminal_hidden_dim: Annotated[int, typer.Option()] = 64,
    n_episodes: Annotated[
        int, typer.Option(help="Eval episodes (split evenly A/B)")
    ] = 400,
    eval_temp: Annotated[
        float, typer.Option(help="Q-net temperature during eval")
    ] = 0.1,
    d_model: Annotated[int, typer.Option()] = 64,
    n_heads: Annotated[int, typer.Option()] = 4,
    n_layers: Annotated[int, typer.Option()] = 2,
    pool_min: Annotated[int, typer.Option()] = 12,
    pool_max: Annotated[
        int | None,
        typer.Option(help="Maximum local pool size per player (default: unbounded)"),
    ] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Evaluate a checkpoint's Q-net win rate vs a uniformly-random opponent."""
    terminal_model, char_encs_all, event_idxs, mode_idxs, char_meta_table, vocabs = (
        _load_terminal_model_and_encs(
            terminal_ckpt, data_dir, terminal_embed_dim, terminal_hidden_dim
        )
    )
    pool_max = pool_max if pool_max is not None else vocabs.n_chars
    h_terminal = char_encs_all.shape[-1]

    q_net = DraftQNetwork(
        h_terminal=h_terminal,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        key=jax.random.PRNGKey(0),
    )
    q_net = eqx.tree_deserialise_leaves(checkpoint / "model.eqx", q_net)

    meta: dict[str, Any] = json.loads((checkpoint / "meta.json").read_text())
    iteration = meta["iteration"]

    config = DraftConfig(
        pool_min=pool_min, pool_max=pool_max, temp_min=0.1, temp_max=0.1
    )
    rng = np.random.default_rng(seed)

    typer.echo(
        f"Evaluating ckpt @ iter {iteration} over {n_episodes} episodes (temp={eval_temp})..."
    )
    results = evaluate_vs_random(
        q_net,
        char_encs_all,
        event_idxs,
        mode_idxs,
        char_meta_table,
        terminal_model,
        config,
        rng,
        n_episodes=n_episodes,
        eval_temp=eval_temp,
    )
    typer.echo(
        f"  iter={iteration:7d}  "
        f"win_A={results['win_rate_as_a']:.3f}  "
        f"win_B={results['win_rate_as_b']:.3f}  "
        f"combined={results['win_rate_combined']:.3f}"
    )


@app.command()
def eval_sweep(
    checkpoints_dir: Annotated[
        Path, typer.Argument(help="Dir containing ckpt_NNNNNNN subdirs")
    ] = _OUT_DIR / _CHECKPOINTS_DIRNAME,
    terminal_ckpt: Annotated[Path, typer.Option()] = Path("data/model/model.eqx"),
    data_dir: Annotated[Path, typer.Option()] = _DATA_DIR,
    terminal_embed_dim: Annotated[int, typer.Option()] = 32,
    terminal_hidden_dim: Annotated[int, typer.Option()] = 64,
    n_episodes: Annotated[int, typer.Option(help="Eval episodes per checkpoint")] = 400,
    eval_temp: Annotated[float, typer.Option()] = 0.1,
    d_model: Annotated[int, typer.Option()] = 64,
    n_heads: Annotated[int, typer.Option()] = 4,
    n_layers: Annotated[int, typer.Option()] = 2,
    pool_min: Annotated[int, typer.Option()] = 12,
    pool_max: Annotated[
        int | None,
        typer.Option(help="Maximum local pool size per player (default: unbounded)"),
    ] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Evaluate all checkpoints in a directory and print a win-rate progression table."""
    ckpt_dirs = sorted(
        p
        for p in checkpoints_dir.iterdir()
        if p.is_dir() and (p / "model.eqx").exists()
    )
    if not ckpt_dirs:
        typer.echo(f"No checkpoints found in {checkpoints_dir}")
        raise typer.Exit(1)

    terminal_model, char_encs_all, event_idxs, mode_idxs, char_meta_table, vocabs = (
        _load_terminal_model_and_encs(
            terminal_ckpt, data_dir, terminal_embed_dim, terminal_hidden_dim
        )
    )
    pool_max = pool_max if pool_max is not None else vocabs.n_chars
    h_terminal = char_encs_all.shape[-1]
    config = DraftConfig(
        pool_min=pool_min, pool_max=pool_max, temp_min=0.1, temp_max=0.1
    )

    typer.echo(
        f"{'iter':>8}  {'win_A':>6}  {'win_B':>6}  {'combined':>8}  {'recent_loss':>11}"
    )
    typer.echo("-" * 50)

    for ckpt_dir in tqdm(ckpt_dirs, desc="sweep", unit="ckpt"):
        meta: dict[str, Any] = json.loads((ckpt_dir / "meta.json").read_text())
        iteration = meta["iteration"]
        # Informational only — self-play loss isn't a fitness signal, the
        # win-rate columns to the left are. Mean of the last 100 logged
        # losses, just to eyeball optimizer behavior alongside win rate.
        loss_history = meta["loss_history"]
        recent_loss = (
            float(np.mean(loss_history[-100:])) if loss_history else float("nan")
        )

        q_net = DraftQNetwork(
            h_terminal=h_terminal,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            key=jax.random.PRNGKey(0),
        )
        q_net = eqx.tree_deserialise_leaves(ckpt_dir / "model.eqx", q_net)

        rng = np.random.default_rng(seed)
        results = evaluate_vs_random(
            q_net,
            char_encs_all,
            event_idxs,
            mode_idxs,
            char_meta_table,
            terminal_model,
            config,
            rng,
            n_episodes=n_episodes,
            eval_temp=eval_temp,
        )
        typer.echo(
            f"{iteration:>8d}  "
            f"{results['win_rate_as_a']:>6.3f}  "
            f"{results['win_rate_as_b']:>6.3f}  "
            f"{results['win_rate_combined']:>8.3f}  "
            f"{recent_loss:>11.4f}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
