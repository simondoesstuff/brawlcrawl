"""Tests for geneus.mc — generic Monte Carlo estimation with rank-stability
convergence, independent of any particular sampling domain."""

import numpy as np
import pytest

from geneus.mc import estimate_until_converged, rank_stability


def test_rank_stability_requires_two_checkpoints() -> None:
    curr = np.array([0.1, 0.2, 0.3])
    counted = np.array([True, True, True])
    assert rank_stability(None, None, curr, counted) == 0.0


def test_rank_stability_perfect_when_identical() -> None:
    vals = np.array([0.1, 0.5, 0.3, 0.9])
    counted = np.array([True, True, True, True])
    assert rank_stability(vals, counted, vals, counted) == pytest.approx(1.0)


def test_estimate_until_converged_on_deterministic_batch() -> None:
    """A batch fn returning identical values every call should converge fast."""
    n = 5
    fixed_values = np.array([0.9, 0.1, 0.5, 0.7, 0.3])

    def sample_batch(rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.tile(np.arange(n), batch_size)
        vals = np.tile(fixed_values, batch_size)
        return idx, vals

    rng = np.random.default_rng(0)
    result = estimate_until_converged(
        n, sample_batch, rng,
        batch_size=4, check_every=1, patience=2, threshold=0.9,
        min_trials=1, max_trials=1000,
    )
    assert result.converged
    assert np.allclose(result.mean, fixed_values)
    assert np.all(result.samples > 0)


def test_estimate_until_converged_hits_max_trials_when_unstable() -> None:
    n = 4

    def sample_batch(rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(n)
        vals = rng.random(n)  # fresh random ranking every batch: never stabilizes
        return idx, vals

    rng = np.random.default_rng(1)
    result = estimate_until_converged(
        n, sample_batch, rng,
        batch_size=1, check_every=1, patience=100, threshold=0.999,
        min_trials=1, max_trials=20,
    )
    assert not result.converged
    assert result.trials == 20
