"""Monte Carlo first-pick tier lists, scored by the terminal BrawlModel.

The raw winrates.json is biased by real pick order (a brawler that's always
taken last has inflated win variance from being reactively countered/paired).
Instead we ask a cleaner question per map: "if I pick this brawler first, what's
my win probability if the rest of the draft is uniformly random?"

Team A is the first-picking team (see `geneus.draft.env`). One Monte Carlo trial
draws a random permutation of all valid brawlers: the first two fill out team
A's other two seats, the next three fill team B, and *every remaining brawler*
is scored as team A's first pick against that same fixed continuation. So a
single trial yields one terminal-model data point for almost every brawler at
once, rather than one trial per brawler.

Trials keep running (per map) until the win-rate ranking stabilizes: every
`--check-every` trials we Spearman-correlate the current ranking against the
previous checkpoint, and stop once that correlation clears `--threshold` for
`--patience` consecutive checks (or `--max-trials` is hit).
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
from scipy.stats import spearmanr

from pick.score import DraftContext, EventInfo, load_context

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
    """Run one batch of trials for a single map.

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


def _rank_correlation(
    prev: np.ndarray | None,
    prev_counted: np.ndarray | None,
    curr: np.ndarray,
    counted: np.ndarray,
) -> float:
    """Spearman correlation between two win-rate estimates, over brawlers seen in both.

    Returns 0.0 (never converged) until `prev` exists and at least two brawlers
    have samples in both checkpoints, so the very first checkpoint can never
    look "stable".
    """
    if prev is None or prev_counted is None:
        return 0.0
    both = counted & prev_counted
    if both.sum() < 2:
        return 0.0
    corr, _ = spearmanr(prev[both], curr[both])
    return float(corr) if np.isfinite(corr) else 0.0


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
    """Monte Carlo first-pick win rates for one map; runs until ranks stabilize."""
    n = len(ctx.brawlers)
    sum_prob = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    prev_means: np.ndarray | None = None
    prev_counted: np.ndarray | None = None
    stable_streak = 0
    trials_done = 0
    batches_since_check = 0

    while trials_done < max_trials:
        perms = _sample_permutations(rng, batch_size, n)
        cand_idx, probs = _simulate_batch(ctx, event, perms)
        np.add.at(sum_prob, cand_idx, probs)
        np.add.at(count, cand_idx, 1)
        trials_done += batch_size
        batches_since_check += 1

        if batches_since_check < check_every:
            continue
        batches_since_check = 0

        counted = count > 0
        curr_means = np.divide(sum_prob, count, out=np.zeros(n), where=counted)
        if trials_done >= min_trials:
            corr = _rank_correlation(prev_means, prev_counted, curr_means, counted)
            stable_streak = stable_streak + 1 if corr >= threshold else 0
        prev_means, prev_counted = curr_means, counted
        if stable_streak >= patience:
            return MapResult(event, trials_done, True, curr_means, count)

    counted = count > 0
    final_means = np.divide(sum_prob, count, out=np.zeros(n), where=counted)
    return MapResult(event, trials_done, False, final_means, count)


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
    event_id: Annotated[int | None, typer.Option(help="Only generate for this event id (default: all maps)")] = None,
    batch_size: Annotated[int, typer.Option(help="Trials per Monte Carlo batch")] = 128,
    check_every: Annotated[int, typer.Option(help="Batches between convergence checks")] = 3,
    patience: Annotated[int, typer.Option(help="Consecutive stable checks required to stop")] = 5,
    threshold: Annotated[float, typer.Option(help="Spearman correlation required to count as stable")] = 0.999,
    min_trials: Annotated[int, typer.Option(help="Trials before convergence can be declared")] = 30,
    max_trials: Annotated[int, typer.Option(help="Safety cap on trials per map")] = 20_000,
    seed: Annotated[int, typer.Option(help="RNG seed")] = 0,
) -> None:
    console.print(f"Loading terminal model [dim]{terminal_ckpt}[/dim]...")
    ctx = load_context(data_dir, terminal_ckpt, draft_ckpt)

    events = [e for e in ctx.events if event_id is None or e.id == event_id]
    if not events:
        raise typer.BadParameter(f"No event with id {event_id}")

    rng = np.random.default_rng(seed)
    results = []
    for event in events:
        result = run_map_trials(
            ctx, event, rng,
            batch_size=batch_size, check_every=check_every, patience=patience,
            threshold=threshold, min_trials=min_trials, max_trials=max_trials,
        )
        status = "[green]converged[/green]" if result.converged else "[yellow]hit max-trials[/yellow]"
        console.print(f"{event.map_name} ({event.mode}): {status} after {result.trials} trials")
        results.append(_map_result_to_json(ctx, result))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"maps": results}, indent=2))
    console.print(f"[bold green]Wrote {output}[/bold green]")


def entrypoint() -> None:
    typer_app()
