"""Tests for pick.tier_list — Monte Carlo first-pick tier lists, on a freshly
initialized small model (no dependency on trained checkpoints in data/)."""

import jax
import numpy as np
import pytest

from geneus.draft.model import DraftQNetwork
from geneus.model import BrawlModel
from pick.score import BrawlerInfo, DraftContext, EventInfo
from pick.tier_list import (
    _rank_correlation,
    _sample_permutations,
    _simulate_batch,
    run_map_trials,
)

N_CHARS = 12
N_CLASSES = 3
N_RANGES = 2
N_DESTRUCTS = 2


@pytest.fixture
def ctx() -> DraftContext:
    terminal_model = BrawlModel(
        n_events=2, n_modes=1, n_chars=N_CHARS,
        n_classes=N_CLASSES, n_ranges=N_RANGES, n_destructs=N_DESTRUCTS,
        embed_dim=8, hidden_dim=16, dropout_p=0.0,
        key=jax.random.PRNGKey(0),
    )
    q_net = DraftQNetwork(h_terminal=16, d_model=12, n_heads=2, n_layers=1, key=jax.random.PRNGKey(1))
    rng = np.random.default_rng(0)
    char_meta_table = np.stack([
        rng.integers(0, N_CLASSES, N_CHARS),
        rng.integers(0, N_RANGES, N_CHARS),
        rng.integers(0, N_DESTRUCTS, N_CHARS),
    ], axis=1).astype(np.int32)

    events = [EventInfo(id=0, mode="bounty", mode_id=0, map_name="map0", event_idx=0, mode_idx=0)]
    brawlers = [
        BrawlerInfo(id=1000 + i, name=f"Brawler{i}", brawler_class="Damage Dealer", rarity="Rare", char_idx=i)
        for i in range(N_CHARS)
    ]

    return DraftContext(
        q_net=q_net,
        terminal_model=terminal_model,
        char_meta_table=char_meta_table,
        char_encs_all=np.zeros((1, N_CHARS, 16), dtype=np.float32),
        event_enc_row={0: 0},
        brawlers=brawlers,
        events=events,
        brawler_names=sorted(b.name for b in brawlers),
        map_names=[e.map_name for e in events],
        n_chars=N_CHARS,
        winrates={},
        pickrates={},
        _brawler_by_char_idx={b.char_idx: b for b in brawlers},
        _brawler_by_name={b.name: b for b in brawlers},
    )


def test_sample_permutations_are_valid_permutations() -> None:
    rng = np.random.default_rng(0)
    perms = _sample_permutations(rng, n_trials=50, n=N_CHARS)
    assert perms.shape == (50, N_CHARS)
    for row in perms:
        assert sorted(row.tolist()) == list(range(N_CHARS))


def test_simulate_batch_covers_almost_every_brawler(ctx: DraftContext) -> None:
    rng = np.random.default_rng(0)
    perms = _sample_permutations(rng, n_trials=64, n=N_CHARS)
    cand_idx, probs = _simulate_batch(ctx, ctx.events[0], perms)

    n_cand = N_CHARS - 5
    assert cand_idx.shape == (64 * n_cand,)
    assert probs.shape == cand_idx.shape
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # every brawler is overwhelmingly likely to appear as a candidate at least
    # once across 64 trials (excluded only when drawn into the fixed 5)
    assert set(cand_idx.tolist()) == set(range(N_CHARS))


def test_rank_correlation_requires_two_checkpoints() -> None:
    curr = np.array([0.1, 0.2, 0.3])
    counted = np.array([True, True, True])
    assert _rank_correlation(None, None, curr, counted) == 0.0


def test_rank_correlation_perfect_when_identical() -> None:
    vals = np.array([0.1, 0.5, 0.3, 0.9])
    counted = np.array([True, True, True, True])
    assert _rank_correlation(vals, counted, vals, counted) == pytest.approx(1.0)


def test_run_map_trials_converges_and_ranks_all_brawlers(ctx: DraftContext) -> None:
    rng = np.random.default_rng(0)
    result = run_map_trials(
        ctx, ctx.events[0], rng,
        batch_size=64, check_every=1, patience=2, threshold=0.9,
        min_trials=1, max_trials=5_000,
    )
    assert result.converged
    assert np.all(result.samples > 0)
    assert np.all((result.win_rate >= 0.0) & (result.win_rate <= 1.0))
