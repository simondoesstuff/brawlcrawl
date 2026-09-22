import json
from pathlib import Path

import pytest

from crawl.dataset import Composition, Dataset, WinLoss, merge_stats


class TestComposition:
    def test_valid_canonical_order(self):
        c = Composition(event_id=1, team_a_ids=frozenset({1, 2}), team_b_ids=frozenset({3, 4}))
        assert c.team_a_ids == frozenset({1, 2})

    def test_equal_teams_allowed(self):
        c = Composition(event_id=1, team_a_ids=frozenset({1, 2}), team_b_ids=frozenset({1, 2}))
        assert c.team_a_ids == c.team_b_ids

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            Composition(event_id=1, team_a_ids=frozenset({5, 6}), team_b_ids=frozenset({1, 2}))

    def test_hashable(self):
        c = Composition(event_id=1, team_a_ids=frozenset({1}), team_b_ids=frozenset({2}))
        assert {c: 42}[c] == 42

    def test_from_sides_canonical_order(self):
        comp, a_won = Composition.from_sides(
            event_id=1,
            winning=frozenset({10, 11, 12}),
            losing=frozenset({1, 2, 3}),
        )
        assert comp.team_a_ids == frozenset({1, 2, 3})
        assert comp.team_b_ids == frozenset({10, 11, 12})
        assert not a_won  # team_a (losing side) did not win

    def test_from_sides_already_canonical(self):
        comp, a_won = Composition.from_sides(
            event_id=1,
            winning=frozenset({1, 2, 3}),
            losing=frozenset({10, 11, 12}),
        )
        assert comp.team_a_ids == frozenset({1, 2, 3})
        assert a_won  # team_a (winning side) won


class TestDataset:
    def test_from_seed(self):
        db = Dataset.from_seed({"#A", "#B"})
        assert db.seen_tags == {"#A", "#B"}
        assert db.frontier == {"#A", "#B"}
        assert db.seen_battle_ids == set()
        assert db.bad_tags == set()
        assert db.stats == {}

    def test_save_load_roundtrip(self, tmp_path: Path):
        db = Dataset.from_seed({"#A"})
        comp = Composition(1, frozenset({1, 2, 3}), frozenset({4, 5, 6}))
        db.stats[comp] = WinLoss(a_wins=10, total=17)
        db.seen_battle_ids.add("abc123")
        db.frontier.add("#B")
        db.seen_tags.add("#B")

        path = tmp_path / "dataset.json"
        db.save(path)
        db2 = Dataset.load(path)

        assert db2.seen_tags == db.seen_tags
        assert db2.frontier == db.frontier
        assert db2.seen_battle_ids == db.seen_battle_ids
        assert len(db2.stats) == 1
        wl = db2.stats[comp]
        assert wl.a_wins == 10
        assert wl.total == 17

    def test_save_produces_valid_json(self, tmp_path: Path):
        db = Dataset.from_seed({"#A"})
        path = tmp_path / "dataset.json"
        db.save(path)
        data = json.loads(path.read_text())
        assert "seen_tags" in data
        assert "frontier" in data
        assert "seen_battle_ids" in data
        assert "bad_tags" in data
        assert "stats" in data

    def test_bad_tags_roundtrip(self, tmp_path: Path):
        db = Dataset.from_seed({"#A"})
        db.bad_tags.add("#DEAD")
        path = tmp_path / "dataset.json"
        db.save(path)
        db2 = Dataset.load(path)
        assert db2.bad_tags == {"#DEAD"}

    def test_stats_win_loss_preserved(self, tmp_path: Path):
        db = Dataset.from_seed(set())
        comp = Composition(99, frozenset({7}), frozenset({8}))
        db.stats[comp] = WinLoss(a_wins=3, total=5)
        path = tmp_path / "dataset.json"
        db.save(path)
        db2 = Dataset.load(path)
        wl = db2.stats[comp]
        assert wl.a_wins == 3
        assert wl.total == 5

    def test_load_empty_dataset(self, tmp_path: Path):
        db = Dataset.from_seed(set())
        path = tmp_path / "dataset.json"
        db.save(path)
        db2 = Dataset.load(path)
        assert db2.stats == {}
        assert db2.seen_tags == set()


class TestMergeStats:
    def test_alpha_zero_is_plain_pool(self):
        old = WinLoss(a_wins=3, total=10)
        new = WinLoss(a_wins=8, total=10)
        merged = merge_stats(old, new, alpha=0.0)
        assert merged.a_wins == old.a_wins + new.a_wins
        assert merged.total == old.total + new.total

    def test_alpha_one_fully_trusts_new(self):
        old = WinLoss(a_wins=3, total=10)
        new = WinLoss(a_wins=8, total=10)
        merged = merge_stats(old, new, alpha=1.0)
        assert merged.a_wins == new.a_wins
        assert merged.total == new.total

    def test_intermediate_alpha_discounts_old_without_inflating_total(self):
        old = WinLoss(a_wins=0, total=100)
        new = WinLoss(a_wins=5, total=10)
        merged = merge_stats(old, new, alpha=0.5)
        assert new.total <= merged.total <= old.total + new.total
        assert merged.a_wins <= merged.total

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            merge_stats(WinLoss(1, 2), WinLoss(1, 2), alpha=1.5)
        with pytest.raises(ValueError):
            merge_stats(WinLoss(1, 2), WinLoss(1, 2), alpha=-0.1)


class TestDatasetMerge:
    def test_union_of_metadata(self):
        old = Dataset.from_seed({"#A"})
        old.bad_tags.add("#DEAD")
        old.seen_battle_ids.add("battle1")
        new = Dataset.from_seed({"#B"})
        new.seen_battle_ids.add("battle2")

        merged = Dataset.merge(old, new, alpha=0.5)

        assert merged.seen_tags == {"#A", "#B"}
        assert merged.frontier == {"#A", "#B"}
        assert merged.seen_battle_ids == {"battle1", "battle2"}
        assert merged.bad_tags == {"#DEAD"}

    def test_comp_only_in_one_dataset_is_unchanged(self):
        comp = Composition(1, frozenset({1, 2, 3}), frozenset({4, 5, 6}))
        old = Dataset.from_seed(set())
        old.stats[comp] = WinLoss(a_wins=4, total=9)
        new = Dataset.from_seed(set())

        merged = Dataset.merge(old, new, alpha=0.9)
        assert merged.stats[comp] == WinLoss(a_wins=4, total=9)

    def test_comp_in_both_datasets_is_combined(self):
        comp = Composition(1, frozenset({1, 2, 3}), frozenset({4, 5, 6}))
        old = Dataset.from_seed(set())
        old.stats[comp] = WinLoss(a_wins=3, total=10)
        new = Dataset.from_seed(set())
        new.stats[comp] = WinLoss(a_wins=8, total=10)

        merged = Dataset.merge(old, new, alpha=0.0)
        assert merged.stats[comp] == WinLoss(a_wins=11, total=20)
