"""Generic Monte Carlo estimation with rank-stability convergence.

Estimates the eventual value of each of several candidate pre-conditions (e.g.
"brawler X is the map's first pick") by repeatedly drawing batches of sampled
outcomes and averaging, stopping once the per-candidate ranking of running
means stops moving. Not specific to tier lists — any "keep sampling until this
set of running averages stabilizes" problem can reuse `estimate_until_converged`.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

# (rng, batch_size) -> (candidate_idx, value): flat arrays of equal length,
# candidate_idx in [0, n). A single call may yield any number of (candidate,
# value) pairs per underlying trial — e.g. one trial can score many
# candidates against a shared continuation, or exactly one candidate per
# trial — the accumulator doesn't care.
SampleBatchFn = Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]


@dataclass
class ConvergenceResult:
    trials: int
    converged: bool
    mean: np.ndarray     # [n] running mean estimate per candidate
    samples: np.ndarray  # [n] int, number of samples backing each mean


def rank_stability(
    prev: np.ndarray | None,
    prev_counted: np.ndarray | None,
    curr: np.ndarray,
    counted: np.ndarray,
) -> float:
    """Spearman correlation between two running estimates, over candidates seen in both.

    Returns 0.0 (never converged) until `prev` exists and at least two candidates
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


def estimate_until_converged(
    n: int,
    sample_batch: SampleBatchFn,
    rng: np.random.Generator,
    batch_size: int = 128,
    check_every: int = 3,
    patience: int = 5,
    threshold: float = 0.999,
    min_trials: int = 30,
    max_trials: int = 20_000,
) -> ConvergenceResult:
    """Run `sample_batch` until the per-candidate ranking of running means stabilizes.

    Every `check_every` batches, Spearman-correlates the current per-candidate
    mean against the previous checkpoint; stops once that correlation clears
    `threshold` for `patience` consecutive checks (or `max_trials` is hit).
    """
    sum_val = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    prev_means: np.ndarray | None = None
    prev_counted: np.ndarray | None = None
    stable_streak = 0
    trials_done = 0
    batches_since_check = 0

    while trials_done < max_trials:
        cand_idx, values = sample_batch(rng, batch_size)
        np.add.at(sum_val, cand_idx, values)
        np.add.at(count, cand_idx, 1)
        trials_done += batch_size
        batches_since_check += 1

        if batches_since_check < check_every:
            continue
        batches_since_check = 0

        counted = count > 0
        curr_means = np.divide(sum_val, count, out=np.zeros(n), where=counted)
        if trials_done >= min_trials:
            corr = rank_stability(prev_means, prev_counted, curr_means, counted)
            stable_streak = stable_streak + 1 if corr >= threshold else 0
        prev_means, prev_counted = curr_means, counted
        if stable_streak >= patience:
            return ConvergenceResult(trials_done, True, curr_means, count)

    counted = count > 0
    final_means = np.divide(sum_val, count, out=np.zeros(n), where=counted)
    return ConvergenceResult(trials_done, False, final_means, count)
