"""Tests for geneus.draft.train — episode simulation, batching, and loss,
on freshly-initialized small models (no dependency on trained checkpoints)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geneus.draft.env import AVAILABLE, DraftConfig, GLOBALLY_BANNED, LOCALLY_BANNED
from geneus.draft.model import DraftQNetwork
from geneus.draft.train import _compute_loss, simulate_batch, simulate_episode
from geneus.model import BrawlModel

N_CHARS = 20
N_EVENTS = 3
H = 16
D_MODEL = 12
N_HEADS = 2
N_LAYERS = 1

N_CLASSES = 3
N_RANGES = 2
N_DESTRUCTS = 2


@pytest.fixture
def terminal_model() -> BrawlModel:
    return BrawlModel(
        n_events=N_EVENTS, n_modes=2, n_chars=N_CHARS,
        n_classes=N_CLASSES, n_ranges=N_RANGES, n_destructs=N_DESTRUCTS,
        embed_dim=8, hidden_dim=H, key=jax.random.PRNGKey(0),
    )


@pytest.fixture
def q_net() -> DraftQNetwork:
    return DraftQNetwork(
        h_terminal=H, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        key=jax.random.PRNGKey(1),
    )


@pytest.fixture
def char_encs_all() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(N_EVENTS, N_CHARS, H)).astype(np.float32)


@pytest.fixture
def event_idxs() -> np.ndarray:
    return np.arange(N_EVENTS, dtype=np.int32)


@pytest.fixture
def mode_idxs() -> np.ndarray:
    return np.zeros(N_EVENTS, dtype=np.int32)


@pytest.fixture
def char_meta_table() -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.stack([
        rng.integers(0, N_CLASSES, N_CHARS),
        rng.integers(0, N_RANGES, N_CHARS),
        rng.integers(0, N_DESTRUCTS, N_CHARS),
    ], axis=1).astype(np.int32)


def _simulate(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model,
    skip_ban_prob: float, seed: int = 0,
):
    config = DraftConfig(pool_min=N_CHARS, pool_max=N_CHARS, skip_ban_prob=skip_ban_prob)
    rng = np.random.default_rng(seed)
    return simulate_episode(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng,
    )


# ---------------------------------------------------------------------------
# Ban-phase skip
# ---------------------------------------------------------------------------


def test_skip_ban_prob_zero_never_skips(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    for seed in range(5):
        ep = _simulate(
            q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
            terminal_model, skip_ban_prob=0.0, seed=seed,
        )
        assert ep["active"].all()


def test_skip_ban_prob_one_always_skips_and_leaves_no_bans(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    ep = _simulate(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, skip_ban_prob=1.0,
    )
    active = ep["active"]
    assert not active[:6].any(), "Ban turns should be inactive when skipped"
    assert active[6:].all(), "Pick turns should still be simulated"

    # Turns 0-5 were never simulated: no actions taken, so PICK_1's
    # observation (turn 6) has zero GLOBALLY_BANNED chars — every char is
    # either AVAILABLE or LOCALLY_BANNED (pool is full here, so AVAILABLE).
    pick1_obs = ep["player_states"][6]
    assert not (pick1_obs == GLOBALLY_BANNED).any()
    assert set(np.unique(pick1_obs).tolist()) <= {AVAILABLE, LOCALLY_BANNED}


def test_skip_ban_local_pool_still_applies(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    # With a restricted pool, a skipped-ban episode's pick-1 obs should still
    # show LOCALLY_BANNED for out-of-pool chars (only global bans are absent).
    config = DraftConfig(pool_min=3, pool_max=3, skip_ban_prob=1.0)
    rng = np.random.default_rng(0)
    ep = simulate_episode(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng,
    )
    pick1_obs = ep["player_states"][6]
    assert (pick1_obs == LOCALLY_BANNED).sum() == N_CHARS - 3
    assert not (pick1_obs == GLOBALLY_BANNED).any()


# ---------------------------------------------------------------------------
# Loss masking
# ---------------------------------------------------------------------------


def test_loss_finite_with_skipped_episodes(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    config = DraftConfig(pool_min=N_CHARS, pool_max=N_CHARS, skip_ban_prob=1.0)
    rng = np.random.default_rng(0)
    batch = simulate_batch(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng, n_episodes=4,
    )
    assert not batch["active"][:, :6].any()
    loss = _compute_loss(q_net, batch)
    assert jnp.isfinite(loss)


def test_loss_well_defined_when_all_active(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    config = DraftConfig(pool_min=N_CHARS, pool_max=N_CHARS, skip_ban_prob=0.0)
    rng = np.random.default_rng(0)
    batch = simulate_batch(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng, n_episodes=4,
    )
    assert batch["active"].all()
    loss = _compute_loss(q_net, batch)
    assert jnp.isfinite(loss)
    assert loss >= 0
