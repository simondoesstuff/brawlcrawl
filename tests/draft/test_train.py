"""Tests for geneus.draft.train — episode simulation, batching, and loss,
on freshly-initialized small models (no dependency on trained checkpoints)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geneus.draft.env import AVAILABLE, BELLMAN_SIGNS, DraftConfig, GLOBALLY_BANNED, LOCALLY_BANNED
from geneus.draft.model import DraftQNetwork
from geneus.draft.train import (
    _compute_loss,
    _soft_value,
    _terminal_logits_pick6,
    simulate_batch,
    simulate_episode,
)
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
        d_model=H, n_heads=2, n_blocks=1, key=jax.random.PRNGKey(0),
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
    loss = _compute_loss(q_net, batch, 0.1)
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
    loss = _compute_loss(q_net, batch, 0.1)
    assert jnp.isfinite(loss)
    assert loss >= 0


def test_bellman_targets_ignore_batch_temperatures(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    """The whole point of the fix: the Bellman-target V must not depend on
    the per-player, per-episode exploration `temperatures` recorded in the
    batch — only on the fixed `bellman_temp` passed to _compute_loss. Two
    batches differing only in `temperatures` must give the same loss."""
    config = DraftConfig(pool_min=N_CHARS, pool_max=N_CHARS, skip_ban_prob=0.0)
    rng = np.random.default_rng(0)
    batch = simulate_batch(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng, n_episodes=4,
    )

    loss_low_temp = _compute_loss(q_net, {**batch, "temperatures": jnp.full_like(batch["temperatures"], 0.05)}, 0.1)
    loss_high_temp = _compute_loss(q_net, {**batch, "temperatures": jnp.full_like(batch["temperatures"], 2.0)}, 0.1)

    assert jnp.allclose(loss_low_temp, loss_high_temp), (
        "Loss must be invariant to batch['temperatures'] — the Bellman "
        "target should only depend on bellman_temp"
    )


# ---------------------------------------------------------------------------
# 6th (last) pick — exact terminal-model brute force, not the Q-network
# ---------------------------------------------------------------------------


def test_simulate_episode_covers_only_the_eleven_tokened_turns(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    """The 6th pick has no draft-turn token (see env.py), so it must never
    appear in the Q-network's training arrays — only turns 0-10 do."""
    ep = _simulate(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, skip_ban_prob=0.0,
    )
    assert ep["player_states"].shape == (11, N_CHARS)
    assert ep["turn_tokens"].shape == (11,)
    assert ep["valid_masks"].shape == (11, N_CHARS)
    assert ep["actions"].shape == (11,)
    assert ep["temperatures"].shape == (11,)
    assert ep["active"].shape == (11,)
    assert "pick6_bootstrap" in ep
    assert "terminal_logit" not in ep
    assert np.isfinite(ep["pick6_bootstrap"])


def test_simulate_batch_stacks_pick6_bootstrap_not_terminal_logits(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    config = DraftConfig(pool_min=N_CHARS, pool_max=N_CHARS, skip_ban_prob=0.0)
    rng = np.random.default_rng(0)
    batch = simulate_batch(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng, n_episodes=3,
    )
    assert batch["player_states"].shape == (3, 11, N_CHARS)
    assert batch["pick6_bootstrap"].shape == (3,)
    assert "terminal_logits" not in batch


def test_terminal_logits_pick6_scores_every_candidate(terminal_model, char_meta_table):
    picks_a_idx = jnp.array([0, 1, 2], dtype=jnp.int32)
    picks_b_partial_idx = jnp.array([3, 4], dtype=jnp.int32)
    cand_idx = jnp.arange(N_CHARS, dtype=jnp.int32)
    logits = _terminal_logits_pick6(
        terminal_model,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        picks_a_idx,
        jnp.array(char_meta_table[np.array(picks_a_idx)]),
        picks_b_partial_idx,
        jnp.array(char_meta_table[np.array(picks_b_partial_idx)]),
        cand_idx,
        jnp.array(char_meta_table),
    )
    assert logits.shape == (N_CHARS,)
    assert jnp.isfinite(logits).all()
    # A freshly-initialized terminal model is not degenerate w.r.t. its last
    # brawler slot — different candidates should score differently.
    assert not jnp.allclose(logits, logits[0])


def test_soft_value_reduces_to_the_only_valid_logit():
    # A single valid action (rest masked to -inf): the soft value must equal
    # that logit exactly, regardless of temperature.
    logits = np.full(6, -np.inf)
    logits[2] = 3.5
    assert np.isclose(_soft_value(logits, temp=0.1), 3.5)
    assert np.isclose(_soft_value(logits, temp=5.0), 3.5)


def test_soft_value_matches_hand_computed_logsumexp():
    logits = np.array([1.0, -1.0, -np.inf, -np.inf])
    temp = 1.0
    expected = temp * np.log(np.exp(1.0 / temp) + np.exp(-1.0 / temp))
    assert np.isclose(_soft_value(logits, temp), expected)


def test_pick5_bellman_target_sign_matches_bootstrap_sign():
    """PICK_5 (the last network-trained turn) is team A; the untokened turn
    11 is team B — so BELLMAN_SIGNS[10] must be -1.0, and _compute_loss's
    target for PICK_5 (pick6_bootstrap * BELLMAN_SIGNS[10]) must therefore
    negate pick6_bootstrap. Getting this backwards makes team A's PICK_5 Q
    positive exactly when team A is losing."""
    assert BELLMAN_SIGNS[10] == -1.0


def test_bellman_temp_changes_the_loss(
    q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table, terminal_model
):
    # Sanity check the fixture actually exercises bellman_temp: a materially
    # different bellman_temp should (generically) change the loss value.
    config = DraftConfig(pool_min=N_CHARS, pool_max=N_CHARS, skip_ban_prob=0.0)
    rng = np.random.default_rng(0)
    batch = simulate_batch(
        q_net, char_encs_all, event_idxs, mode_idxs, char_meta_table,
        terminal_model, config, rng, n_episodes=4,
    )
    loss_a = _compute_loss(q_net, batch, 0.1)
    loss_b = _compute_loss(q_net, batch, 1.0)
    assert not jnp.allclose(loss_a, loss_b)
