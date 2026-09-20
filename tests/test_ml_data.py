"""Tests for geneus.data — vocab building, battle loading, filtering."""

from pathlib import Path

import numpy as np
import pytest

from geneus.data import BattleArrays, Vocabs, WinrateArrays, load_battles, load_vocabs, load_winrates, train_val_split

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def vocabs() -> Vocabs:
    return load_vocabs(DATA_DIR)


@pytest.fixture(scope="module")
def battles(vocabs: Vocabs) -> BattleArrays:
    return load_battles(DATA_DIR, vocabs=vocabs)


class TestVocabs:
    def test_event_indices_contiguous(self, vocabs: Vocabs) -> None:
        assert set(vocabs.event_to_idx.values()) == set(range(vocabs.n_events))

    def test_mode_indices_contiguous(self, vocabs: Vocabs) -> None:
        assert set(vocabs.mode_to_idx.values()) == set(range(vocabs.n_modes))

    def test_char_indices_contiguous(self, vocabs: Vocabs) -> None:
        assert set(vocabs.char_to_idx.values()) == set(range(vocabs.n_chars))

    def test_deterministic_ordering(self) -> None:
        v1 = load_vocabs(DATA_DIR)
        v2 = load_vocabs(DATA_DIR)
        assert v1.event_to_idx == v2.event_to_idx
        assert v1.char_to_idx == v2.char_to_idx
        assert v1.class_to_idx == v2.class_to_idx

    def test_known_vocab_sizes(self, vocabs: Vocabs) -> None:
        assert vocabs.n_events == 31
        assert vocabs.n_modes == 6
        assert vocabs.n_chars == 108
        assert vocabs.n_classes == 7
        assert vocabs.n_ranges == 4
        assert vocabs.n_destructs == 4


class TestBattleLoading:
    def test_battles_loaded(self, battles: BattleArrays) -> None:
        assert len(battles) > 0

    def test_unknown_events_dropped(self, battles: BattleArrays, vocabs: Vocabs) -> None:
        # All event indices must be in vocab range
        assert battles.event_idx.min() >= 0
        assert battles.event_idx.max() < vocabs.n_events

    def test_char_indices_in_range(self, battles: BattleArrays, vocabs: Vocabs) -> None:
        assert battles.team_a_chars.min() >= 0
        assert battles.team_a_chars.max() < vocabs.n_chars
        assert battles.team_b_chars.min() >= 0
        assert battles.team_b_chars.max() < vocabs.n_chars

    def test_meta_indices_in_range(self, battles: BattleArrays, vocabs: Vocabs) -> None:
        assert battles.team_a_meta[:, :, 0].max() < vocabs.n_classes
        assert battles.team_a_meta[:, :, 1].max() < vocabs.n_ranges
        assert battles.team_a_meta[:, :, 2].max() < vocabs.n_destructs

    def test_teams_have_three_brawlers(self, battles: BattleArrays) -> None:
        assert battles.team_a_chars.shape[1] == 3
        assert battles.team_b_chars.shape[1] == 3

    def test_a_wins_le_total(self, battles: BattleArrays) -> None:
        assert (battles.a_wins <= battles.totals).all()

    def test_totals_positive(self, battles: BattleArrays) -> None:
        assert (battles.totals > 0).all()

    def test_array_dtypes(self, battles: BattleArrays) -> None:
        for arr in [
            battles.event_idx, battles.mode_idx,
            battles.team_a_chars, battles.team_a_meta,
            battles.team_b_chars, battles.team_b_meta,
            battles.a_wins, battles.totals,
        ]:
            assert arr.dtype == np.int32


@pytest.fixture(scope="module")
def winrates(vocabs: Vocabs) -> WinrateArrays:
    return load_winrates(DATA_DIR, vocabs=vocabs)


class TestWinrateLoading:
    def test_winrates_loaded(self, winrates: WinrateArrays) -> None:
        assert len(winrates) > 0

    def test_event_indices_in_range(self, winrates: WinrateArrays, vocabs: Vocabs) -> None:
        assert winrates.event_idx.min() >= 0
        assert winrates.event_idx.max() < vocabs.n_events

    def test_char_indices_in_range(self, winrates: WinrateArrays, vocabs: Vocabs) -> None:
        assert winrates.char_idx.min() >= 0
        assert winrates.char_idx.max() < vocabs.n_chars

    def test_meta_indices_in_range(self, winrates: WinrateArrays, vocabs: Vocabs) -> None:
        assert winrates.char_meta[:, 0].max() < vocabs.n_classes
        assert winrates.char_meta[:, 1].max() < vocabs.n_ranges
        assert winrates.char_meta[:, 2].max() < vocabs.n_destructs

    def test_z_scores_finite(self, winrates: WinrateArrays) -> None:
        assert np.isfinite(winrates.z_scores).all()

    def test_z_scores_near_zero_mean_per_event(self, winrates: WinrateArrays) -> None:
        for event_val in np.unique(winrates.event_idx):
            mask = winrates.event_idx == event_val
            assert abs(winrates.z_scores[mask].mean()) < 1e-4

    def test_index_dtypes(self, winrates: WinrateArrays) -> None:
        for arr in [winrates.event_idx, winrates.mode_idx, winrates.char_idx, winrates.char_meta]:
            assert arr.dtype == np.int32

    def test_z_score_dtype(self, winrates: WinrateArrays) -> None:
        assert winrates.z_scores.dtype == np.float32

    def test_char_meta_shape(self, winrates: WinrateArrays) -> None:
        assert winrates.char_meta.shape == (len(winrates), 3)


class TestTrainValSplit:
    def test_sizes_sum_to_total(self, battles: BattleArrays) -> None:
        train, val = train_val_split(battles)
        assert len(train) + len(val) == len(battles)

    def test_val_fraction(self, battles: BattleArrays) -> None:
        _, val = train_val_split(battles, val_frac=0.2)
        assert abs(len(val) / len(battles) - 0.2) < 0.01

    def test_deterministic(self, battles: BattleArrays) -> None:
        t1, v1 = train_val_split(battles, seed=0)
        t2, v2 = train_val_split(battles, seed=0)
        np.testing.assert_array_equal(t1.event_idx, t2.event_idx)
        np.testing.assert_array_equal(v1.event_idx, v2.event_idx)

    def test_different_seeds_differ(self, battles: BattleArrays) -> None:
        _, v1 = train_val_split(battles, seed=0)
        _, v2 = train_val_split(battles, seed=1)
        assert not np.array_equal(v1.event_idx, v2.event_idx)
