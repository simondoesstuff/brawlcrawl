"""Monte Carlo first-pick tier lists, scored by the terminal BrawlModel.

The raw winrates.json is biased by real pick order (a brawler that's always
taken last has inflated win variance from being reactively countered/paired).
Instead we ask a cleaner question per map: "if I pick this brawler first, what's
my win probability if the rest of the draft plays out under some policy?"

Two policies for the other five picks:

  - "random" (`run_map_trials`): every remaining seat, on both teams, picks
    uniformly among the legal brawlers. Team A is the first-picking team (see
    `geneus.draft.env`). One Monte Carlo trial draws a random permutation of
    all valid brawlers: the first two fill out team A's other two seats, the
    next three fill team B, and *every remaining brawler* is scored as team
    A's first pick against that same fixed continuation. So a single trial
    yields one terminal-model data point for almost every brawler at once,
    rather than one trial per brawler.

  - "optimal" (`run_map_trials_optimal`): every remaining seat plays the
    deployed DraftQNetwork policy instead — softmax(Q / eval_temp) sampling,
    same as `geneus.draft.train.evaluate_vs_random`'s eval rollout, except
    the last (sixth) pick uses the terminal model's own exact argmin, since
    that pick has no further draft to look ahead through. This trick doesn't
    batch across candidates the way the random policy does (the optimal
    continuation depends on who was picked first), so each candidate needs
    its own simulated rollout; the win rate reported is always from the
    candidate's (team A's) perspective.

Both policies keep running trials (per map) until the win-rate ranking
stabilizes: every `--check-every` trials we Spearman-correlate the current
ranking against the previous checkpoint, and stop once that correlation
clears `--threshold` for `--patience` consecutive checks (or `--max-trials`
is hit) — see `geneus.mc.estimate_until_converged`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import typer
from rich.console import Console

from geneus.draft.env import AVAILABLE, LOCALLY_BANNED, PICKED_A, PICKED_B, TURN_SCHEDULE
from geneus.mc import estimate_until_converged
from pick.score import DraftContext, EventInfo, load_context, q_values_batch, terminal_logits_grid

typer_app = typer.Typer(add_completion=False, help="Monte Carlo per-map first-pick tier lists.")
console = Console()


@eqx.filter_jit
def _score_batch(
    terminal_model,
    event_idx: jax.Array,      # [N]
    mode_idx: jax.Array,       # [N]
    team_a_chars: jax.Array,   # [N, 3]
    team_a_meta: jax.Array,    # [N, 3, 3]
    team_b_chars: jax.Array,   # [N, 3]
    team_b_meta: jax.Array,    # [N, 3, 3]
) -> jax.Array:  # [N] logits, P(team A wins) = sigmoid(logit)
    return jax.vmap(terminal_model)(
        event_idx, mode_idx, team_a_chars, team_a_meta, team_b_chars, team_b_meta
    )


def _sample_permutations(rng: np.random.Generator, n_trials: int, n: int) -> np.ndarray:
    """[n_trials, n] int array; each row a uniform random permutation of 0..n-1."""
    return np.argsort(rng.random((n_trials, n)), axis=1)


def _simulate_batch(
    ctx: DraftContext,
    event: EventInfo,
    perms: np.ndarray,  # [T, n] brawler-array indices (into ctx.brawlers)
) -> tuple[np.ndarray, np.ndarray]:
    """Run one batch of random-policy trials for a single map.

    Returns (candidate_idx, prob): flat arrays of length T * (n - 5), indexing
    into `ctx.brawlers` and giving P(candidate's team wins) when `candidate` is
    that team's first pick and the rest of that trial's draft is fixed.
    """
    n = len(ctx.brawlers)
    t = perms.shape[0]
    n_cand = n - 5

    char_idxs = np.array([b.char_idx for b in ctx.brawlers], dtype=np.int32)  # [n]
    meta = ctx.char_meta_table[char_idxs]  # [n, 3]

    partner_idx = perms[:, 0:2]  # [T, 2] -> ally seats alongside the candidate
    enemy_idx = perms[:, 2:5]  # [T, 3]
    cand_idx = perms[:, 5:]  # [T, n_cand]

    partner_chars = char_idxs[partner_idx]  # [T, 2]
    partner_meta = meta[partner_idx]  # [T, 2, 3]
    enemy_chars = np.broadcast_to(char_idxs[enemy_idx][:, None, :], (t, n_cand, 3))  # [T, n_cand, 3]
    enemy_meta = np.broadcast_to(meta[enemy_idx][:, None, :, :], (t, n_cand, 3, 3))  # [T, n_cand, 3, 3]

    cand_chars = char_idxs[cand_idx]  # [T, n_cand]
    cand_meta = meta[cand_idx]  # [T, n_cand, 3]
    team_a_chars = np.concatenate(
        [cand_chars[:, :, None], np.broadcast_to(partner_chars[:, None, :], (t, n_cand, 2))], axis=-1
    )  # [T, n_cand, 3]
    team_a_meta = np.concatenate(
        [cand_meta[:, :, None, :], np.broadcast_to(partner_meta[:, None, :, :], (t, n_cand, 2, 3))], axis=2
    )  # [T, n_cand, 3, 3]

    flat = t * n_cand
    logits = np.array(_score_batch(
        ctx.terminal_model,
        jnp.full((flat,), event.event_idx, dtype=jnp.int32),
        jnp.full((flat,), event.mode_idx, dtype=jnp.int32),
        jnp.array(team_a_chars.reshape(flat, 3)),
        jnp.array(team_a_meta.reshape(flat, 3, 3)),
        jnp.array(enemy_chars.reshape(flat, 3)),
        jnp.array(enemy_meta.reshape(flat, 3, 3)),
    ))
    probs = 1.0 / (1.0 + np.exp(-logits))
    return cand_idx.reshape(flat), probs


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sample_rows(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """One categorical sample per row of `probs` (each row sums to 1)."""
    cum = np.cumsum(probs, axis=-1)
    u = rng.random((probs.shape[0], 1))
    return (u < cum).argmax(axis=-1)


def _simulate_optimal_batch(
    ctx: DraftContext,
    event: EventInfo,
    rng: np.random.Generator,
    n_trials: int,
    eval_temp: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one batch of optimal-policy trials for a single map.

    Every brawler is, in turn, fixed as the candidate first pick (team A,
    seat 0); no bans, and both teams draft from the full brawler pool. The
    remaining five picks are drafted by the deployed DraftQNetwork policy —
    softmax(Q / eval_temp) sampling, matching
    `geneus.draft.train.evaluate_vs_random`'s eval rollout — except the last
    (sixth) pick, team B's, which has no further draft to look ahead through
    so the terminal model's own exact argmin is the correct "optimal" read
    (matches `pick.score.get_terminal_pick6_scores`).

    Unlike `_simulate_batch`, one trial can't share a single continuation
    across every candidate (the optimal continuation depends on who was
    picked first), so this scores exactly one draft per candidate per trial.

    Returns (candidate_idx, prob): flat arrays of length T * n, indexing into
    `ctx.brawlers` and giving P(team A / the candidate's team wins).
    """
    n = len(ctx.brawlers)
    char_idxs = np.array([b.char_idx for b in ctx.brawlers], dtype=np.int32)  # [n]
    t = n_trials
    b = t * n
    rows = np.arange(b)

    obs = np.full((b, ctx.n_chars), LOCALLY_BANNED, dtype=np.int32)
    valid = np.zeros(ctx.n_chars, dtype=bool)
    valid[char_idxs] = True
    obs[:, valid] = AVAILABLE

    cand_char = np.tile(char_idxs, t)  # [B]
    obs[rows, cand_char] = PICKED_A

    team_a_roster = np.empty((b, 3), dtype=np.int32)
    team_b_roster = np.empty((b, 3), dtype=np.int32)
    team_a_roster[:, 0] = cand_char
    a_fill, b_fill = 1, 0

    enc_row = ctx.event_enc_row[event.event_idx]
    char_encs = jnp.array(ctx.char_encs_all[enc_row], dtype=jnp.float32)  # [n_chars, h]
    all_chars = jnp.arange(ctx.n_chars, dtype=jnp.int32)
    all_meta = jnp.array(ctx.char_meta_table, dtype=jnp.int32)

    for pick_idx in range(1, 6):
        turn_token, team, _seat = TURN_SCHEDULE[6 + pick_idx]
        available = obs == AVAILABLE

        if pick_idx < 5:
            q_vals = np.array(q_values_batch(
                ctx.q_net, char_encs, jnp.array(obs), jnp.array(turn_token),
            ))
            q_masked = np.where(available, q_vals, -np.inf)
            action = _sample_rows(rng, _softmax_rows(q_masked / max(eval_temp, 1e-6)))
        else:
            team_a_meta = ctx.char_meta_table[team_a_roster]
            team_b_partial_meta = ctx.char_meta_table[team_b_roster[:, :b_fill]]
            logits = np.array(terminal_logits_grid(
                ctx.terminal_model,
                jnp.array(event.event_idx), jnp.array(event.mode_idx),
                jnp.array(team_a_roster), jnp.array(team_a_meta),
                jnp.array(team_b_roster[:, :b_fill]), jnp.array(team_b_partial_meta),
                all_chars, all_meta,
            ))
            # team B picks last and acts for itself: minimize team A's win logit
            action = np.where(available, logits, np.inf).argmin(axis=-1)

        if team == "A":
            obs[rows, action] = PICKED_A
            team_a_roster[:, a_fill] = action
            a_fill += 1
        else:
            obs[rows, action] = PICKED_B
            team_b_roster[:, b_fill] = action
            b_fill += 1

    team_a_meta = ctx.char_meta_table[team_a_roster]
    team_b_meta = ctx.char_meta_table[team_b_roster]
    logits = np.array(_score_batch(
        ctx.terminal_model,
        jnp.full((b,), event.event_idx, dtype=jnp.int32),
        jnp.full((b,), event.mode_idx, dtype=jnp.int32),
        jnp.array(team_a_roster), jnp.array(team_a_meta),
        jnp.array(team_b_roster), jnp.array(team_b_meta),
    ))
    probs = 1.0 / (1.0 + np.exp(-logits))
    cand_out = np.tile(np.arange(n), t)
    return cand_out, probs


@dataclass
class MapResult:
    event: EventInfo
    trials: int
    converged: bool
    win_rate: np.ndarray  # [n_brawlers]
    samples: np.ndarray  # [n_brawlers] int


def run_map_trials(
    ctx: DraftContext,
    event: EventInfo,
    rng: np.random.Generator,
    batch_size: int = 128,
    check_every: int = 3,
    patience: int = 5,
    threshold: float = 0.999,
    min_trials: int = 30,
    max_trials: int = 20_000,
) -> MapResult:
    """Random-policy first-pick win rates for one map; runs until ranks stabilize."""
    n = len(ctx.brawlers)

    def sample_batch(rng: np.random.Generator, bs: int) -> tuple[np.ndarray, np.ndarray]:
        return _simulate_batch(ctx, event, _sample_permutations(rng, bs, n))

    result = estimate_until_converged(
        n, sample_batch, rng,
        batch_size=batch_size, check_every=check_every, patience=patience,
        threshold=threshold, min_trials=min_trials, max_trials=max_trials,
    )
    return MapResult(event, result.trials, result.converged, result.mean, result.samples)


def run_map_trials_optimal(
    ctx: DraftContext,
    event: EventInfo,
    rng: np.random.Generator,
    eval_temp: float = 0.1,
    batch_size: int = 16,
    check_every: int = 3,
    patience: int = 5,
    threshold: float = 0.999,
    min_trials: int = 30,
    max_trials: int = 20_000,
) -> MapResult:
    """Optimal-policy first-pick win rates for one map; runs until ranks stabilize.

    Same convergence procedure as `run_map_trials`, but the other five picks
    are drafted by the deployed DraftQNetwork policy instead of uniformly at
    random. Each trial simulates a full draft per candidate (no shared-trial
    trick), so the default `batch_size` is much smaller than the random
    policy's.
    """
    n = len(ctx.brawlers)

    def sample_batch(rng: np.random.Generator, bs: int) -> tuple[np.ndarray, np.ndarray]:
        return _simulate_optimal_batch(ctx, event, rng, bs, eval_temp=eval_temp)

    result = estimate_until_converged(
        n, sample_batch, rng,
        batch_size=batch_size, check_every=check_every, patience=patience,
        threshold=threshold, min_trials=min_trials, max_trials=max_trials,
    )
    return MapResult(event, result.trials, result.converged, result.mean, result.samples)


def _map_result_to_json(ctx: DraftContext, result: MapResult) -> dict[str, object]:
    order = np.argsort(-result.win_rate)
    return {
        "event_id": result.event.id,
        "map_name": result.event.map_name,
        "mode": result.event.mode,
        "trials": result.trials,
        "converged": result.converged,
        "tier_list": [
            {
                "brawler_id": ctx.brawlers[i].id,
                "name": ctx.brawlers[i].name,
                "win_rate": float(result.win_rate[i]),
                "samples": int(result.samples[i]),
            }
            for i in order
        ],
    }


@typer_app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    terminal_ckpt: Annotated[Path, typer.Option(help="Frozen BrawlModel checkpoint (.eqx)")] = Path("data/model/model.eqx"),
    draft_ckpt: Annotated[Path, typer.Option(help="Draft Q-network weights (.eqx)")] = Path("data/draft_model/draft_q.eqx"),
    output: Annotated[Path, typer.Option(help="Output tier-list JSON path")] = Path("data/tier_lists.json"),
    policy: Annotated[str, typer.Option(help="Continuation policy for the other five picks: 'random' or 'optimal' (DraftQNetwork)")] = "random",
    eval_temp: Annotated[float, typer.Option(help="Softmax temperature for the optimal policy's Q-network sampling (--policy optimal only)")] = 0.1,
    event_id: Annotated[int | None, typer.Option(help="Only generate for this event id (default: all maps)")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Trials per Monte Carlo batch (default: 128 for random, 16 for optimal)")] = None,
    check_every: Annotated[int, typer.Option(help="Batches between convergence checks")] = 3,
    patience: Annotated[int, typer.Option(help="Consecutive stable checks required to stop")] = 5,
    threshold: Annotated[float, typer.Option(help="Spearman correlation required to count as stable")] = 0.999,
    min_trials: Annotated[int, typer.Option(help="Trials before convergence can be declared")] = 30,
    max_trials: Annotated[int, typer.Option(help="Safety cap on trials per map")] = 20_000,
    seed: Annotated[int, typer.Option(help="RNG seed")] = 0,
) -> None:
    if policy not in ("random", "optimal"):
        raise typer.BadParameter(f"--policy must be 'random' or 'optimal', got {policy!r}")

    console.print(f"Loading terminal model [dim]{terminal_ckpt}[/dim]...")
    ctx = load_context(data_dir, terminal_ckpt, draft_ckpt)

    events = [e for e in ctx.events if event_id is None or e.id == event_id]
    if not events:
        raise typer.BadParameter(f"No event with id {event_id}")

    bs = batch_size if batch_size is not None else (128 if policy == "random" else 16)

    rng = np.random.default_rng(seed)
    results = []
    for event in events:
        if policy == "random":
            result = run_map_trials(
                ctx, event, rng,
                batch_size=bs, check_every=check_every, patience=patience,
                threshold=threshold, min_trials=min_trials, max_trials=max_trials,
            )
        else:
            result = run_map_trials_optimal(
                ctx, event, rng, eval_temp=eval_temp,
                batch_size=bs, check_every=check_every, patience=patience,
                threshold=threshold, min_trials=min_trials, max_trials=max_trials,
            )
        status = "[green]converged[/green]" if result.converged else "[yellow]hit max-trials[/yellow]"
        console.print(f"{event.map_name} ({event.mode}): {status} after {result.trials} trials")
        results.append(_map_result_to_json(ctx, result))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"policy": policy, "maps": results}, indent=2))
    console.print(f"[bold green]Wrote {output}[/bold green]")


def entrypoint() -> None:
    typer_app()
