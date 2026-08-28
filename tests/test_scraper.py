from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crawl.api import Battle, BrawlApiError, Brawler, Event, Player, TagInaccessibleError
from crawl.dataset import Composition, Dataset, WinLoss
from crawl.scraper import Scraper, consume

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)


def make_player(tag: str, brawler_id: int, trophies: int = 1000) -> Player:
    return Player(tag=tag, name=tag, brawler=Brawler(id=brawler_id, name="X", power=11, trophies=trophies))


def make_battle(
    teams: list[list[Player]],
    result: str,
    event_id: int = 1,
    battle_time: datetime = _T0,
) -> Battle:
    return Battle(
        battle_time=battle_time,
        event=Event(id=event_id, mode="gemGrab", map="Test"),
        mode="gemGrab",
        type="soloRanked",
        result=result,
        teams=teams,
        players=None,
    )


def _ab_teams(offset: int = 0) -> tuple[list[Player], list[Player]]:
    """Two 3-player teams with deterministic brawler IDs. a_ids < b_ids by canonical order."""
    a = [make_player(f"#A{i + offset}", brawler_id=i + offset) for i in range(3)]
    b = [make_player(f"#B{i + offset}", brawler_id=i + offset + 100) for i in range(3)]
    return a, b


class TestConsume:
    def test_new_battle_added_to_ids(self):
        a, b = _ab_teams()
        result = consume("#A0", [make_battle([a, b], "victory")], seen_battle_ids=set())
        assert len(result.new_battle_ids) == 1

    def test_already_seen_battle_skipped(self):
        a, b = _ab_teams()
        battle = make_battle([a, b], "victory")
        r1 = consume("#A0", [battle], seen_battle_ids=set())
        r2 = consume("#A0", [battle], seen_battle_ids=r1.new_battle_ids)
        assert r2.new_battle_ids == set()
        assert r2.stat_updates == {}

    def test_victory_increments_a_wins(self):
        # a brawler_ids {0,1,2}, b brawler_ids {100,101,102}: a is canonical team_a
        a, b = _ab_teams()
        result = consume("#A0", [make_battle([a, b], "victory")], seen_battle_ids=set())
        comp, wl = next(iter(result.stat_updates.items()))
        assert comp.team_a_ids == frozenset({0, 1, 2})
        assert wl.total == 1
        assert wl.a_wins == 1

    def test_defeat_does_not_increment_a_wins(self):
        # a loses: b wins, but b_ids > a_ids so team_a = a_ids, a_won = False
        a, b = _ab_teams()
        result = consume("#A0", [make_battle([a, b], "defeat")], seen_battle_ids=set())
        _, wl = next(iter(result.stat_updates.items()))
        assert wl.total == 1
        assert wl.a_wins == 0

    def test_cross_player_dedup(self):
        """Same battle appearing in two players' logs is counted once."""
        a, b = _ab_teams()
        battle = make_battle([a, b], "victory")
        r1 = consume("#A0", [battle], seen_battle_ids=set())
        r2 = consume("#A0", [battle], seen_battle_ids=r1.new_battle_ids)
        assert r2.new_battle_ids == set()

    def test_discovers_player_tags(self):
        a, b = _ab_teams()
        result = consume("#A0", [make_battle([a, b], "victory")], seen_battle_ids=set())
        assert result.discovered_tags == {p.tag for p in a + b}

    def test_showdown_battles_produce_no_stats(self):
        players = [make_player(f"#P{i}", brawler_id=i) for i in range(4)]
        showdown = Battle(
            battle_time=_T0,
            event=Event(id=1, mode="soloShowdown", map="Test"),
            mode="soloShowdown",
            type="soloShowdown",
            result=None,
            teams=None,
            players=players,
        )
        result = consume("#P0", [showdown], seen_battle_ids=set())
        assert result.stat_updates == {}

    def test_multiple_battles_accumulate(self):
        a, b = _ab_teams()
        # Second battle: same owner (#A0) with different teammates
        a2 = [make_player("#A0", brawler_id=0)] + [make_player(f"#C{i}", brawler_id=i + 200) for i in range(2)]
        b2 = [make_player(f"#D{i}", brawler_id=i + 300) for i in range(3)]
        battles = [
            make_battle([a, b], "victory", battle_time=_T0),
            make_battle([a2, b2], "defeat", battle_time=_T1),
        ]
        result = consume("#A0", battles, seen_battle_ids=set())
        assert len(result.new_battle_ids) == 2

    def test_non_solo_ranked_excluded(self):
        a, b = _ab_teams()
        ranked = Battle(
            battle_time=_T0,
            event=Event(id=1, mode="gemGrab", map="Test"),
            mode="gemGrab",
            type="ranked",
            result="victory",
            teams=[a, b],
            players=None,
        )
        result = consume("#A0", [ranked], seen_battle_ids=set())
        assert result.stat_updates == {}
        assert result.new_battle_ids == set()

    def test_since_filter_excludes_old_battles(self):
        a, b = _ab_teams()
        old = make_battle([a, b], "victory", battle_time=_T0)
        result = consume("#A0", [old], seen_battle_ids=set(), since=_T1)
        assert result.stat_updates == {}

    def test_min_trophies_filters_discovered_tags(self):
        # 5 high-trophy players pass the min_players=5 threshold; low-trophy player is excluded from discovery
        high = [make_player(f"#H{i}", brawler_id=i, trophies=2000) for i in range(5)]
        low = make_player("#LOW", brawler_id=99, trophies=500)
        a = high[:3]
        b = high[3:] + [low]
        result = consume(a[0].tag, [make_battle([a, b], "victory")], seen_battle_ids=set(), min_trophies=1000)
        assert result.discovered_tags == {p.tag for p in high}

    def test_trophy_filter_excludes_battle_with_too_few_qualifiers(self):
        # Only 3 of 6 players meet the threshold: battle dropped entirely
        a = [make_player(f"#A{i}", brawler_id=i, trophies=500) for i in range(3)]
        b = [make_player(f"#B{i}", brawler_id=i + 100, trophies=2000) for i in range(3)]
        result = consume("#A0", [make_battle([a, b], "victory")], seen_battle_ids=set(), min_trophies=1000)
        assert result.stat_updates == {}
        assert result.discovered_tags == set()


class TestScraper:
    def _make_db(self, seed: set[str] | None = None) -> Dataset:
        return Dataset.from_seed(seed or {"#SEED"})

    def test_step_returns_new_battle_count(self):
        a, b = _ab_teams()
        client = MagicMock()
        client.get_battlelog.return_value = [make_battle([a, b], "victory")]
        db = Dataset.from_seed({a[0].tag})  # seed tag must be in the battle
        assert Scraper(client, db).step() == 1

    def test_step_updates_frontier_with_discovered_tags(self):
        a, b = _ab_teams()
        client = MagicMock()
        client.get_battlelog.return_value = [make_battle([a, b], "victory")]
        db = Dataset.from_seed({a[0].tag})  # seed tag is "#A0"
        Scraper(client, db).step()
        all_tags = {p.tag for p in a + b}
        assert db.frontier == all_tags - {a[0].tag}

    def test_step_deduplicates_same_battle_across_frontier_tags(self):
        a, b = _ab_teams()
        battle = make_battle([a, b], "victory")
        client = MagicMock()
        client.get_battlelog.return_value = [battle]
        db = Dataset.from_seed({"#A0", "#A1"})
        n = Scraper(client, db).step()
        assert n == 1  # same battle, counted once

    def test_step_accumulates_stats(self):
        a, b = _ab_teams()
        client = MagicMock()
        client.get_battlelog.return_value = [make_battle([a, b], "victory")]
        db = Dataset.from_seed({a[0].tag})
        Scraper(client, db).step()
        assert len(db.stats) == 1
        wl = next(iter(db.stats.values()))
        assert wl.total == 1

    def test_frontier_cap_limits_api_calls(self):
        client = MagicMock()
        client.get_battlelog.return_value = []
        db = Dataset.from_seed({"#A", "#B", "#C"})
        Scraper(client, db, frontier_cap=2).step()
        assert client.get_battlelog.call_count == 2

    def test_frontier_cap_retains_unprocessed_tags(self):
        # 3 tags in frontier, cap=2: the unprocessed tag must survive into next step
        client = MagicMock()
        client.get_battlelog.return_value = []
        db = Dataset.from_seed({"#A", "#B", "#C"})
        Scraper(client, db, frontier_cap=2).step()
        assert len(db.frontier) == 1  # the uncapped tag is still queued

    def test_seen_tags_grows_after_step(self):
        a, b = _ab_teams()
        client = MagicMock()
        client.get_battlelog.return_value = [make_battle([a, b], "victory")]
        db = self._make_db()
        Scraper(client, db).step()
        all_tags = {p.tag for p in a + b} | {"#SEED"}
        assert db.seen_tags == all_tags

    def test_run_calls_step_n_times(self):
        client = MagicMock()
        client.get_battlelog.return_value = []
        db = self._make_db()
        scraper = Scraper(client, db)
        scraper.step = MagicMock(return_value=0)  # type: ignore[method-assign]
        scraper.run(3)
        assert scraper.step.call_count == 3

    def test_run_saves_after_each_step(self, tmp_path):
        client = MagicMock()
        client.get_battlelog.return_value = []
        db = self._make_db()
        path = tmp_path / "dataset.json"
        Scraper(client, db).run(2, save_path=path)
        assert path.exists()

    def test_request_interval_sleeps_between_calls(self):
        client = MagicMock()
        client.get_battlelog.return_value = []
        db = Dataset.from_seed({"#A", "#B", "#C"})
        with patch("crawl.scraper.time") as mock_time:
            Scraper(client, db, frontier_cap=3, request_interval=0.5).step()
        assert mock_time.sleep.call_count == 3
        mock_time.sleep.assert_called_with(0.5)

    def test_no_sleep_when_interval_is_zero(self):
        client = MagicMock()
        client.get_battlelog.return_value = []
        db = Dataset.from_seed({"#A"})
        with patch("crawl.scraper.time") as mock_time:
            Scraper(client, db, request_interval=0.0).step()
        mock_time.sleep.assert_not_called()

    def test_api_error_propagates_from_step(self):
        client = MagicMock()
        client.get_battlelog.side_effect = BrawlApiError("malformed response")
        db = Dataset.from_seed({"#A"})
        with pytest.raises(BrawlApiError):
            Scraper(client, db).step()

    def test_inaccessible_tag_added_to_bad_tags(self):
        client = MagicMock()
        client.get_battlelog.side_effect = TagInaccessibleError("404 for #DEAD")
        db = Dataset.from_seed({"#DEAD"})
        Scraper(client, db).step()
        assert "#DEAD" in db.bad_tags

    def test_inaccessible_tag_added_to_seen_tags(self):
        client = MagicMock()
        client.get_battlelog.side_effect = TagInaccessibleError("404 for #DEAD")
        db = Dataset.from_seed({"#DEAD"})
        Scraper(client, db).step()
        assert "#DEAD" in db.seen_tags

    def test_inaccessible_tag_not_requeued(self):
        """A 404 tag must not reappear in the frontier after the step."""
        client = MagicMock()
        client.get_battlelog.side_effect = TagInaccessibleError("404 for #DEAD")
        db = Dataset.from_seed({"#DEAD"})
        Scraper(client, db).step()
        assert "#DEAD" not in db.frontier

    def test_too_many_bad_tags_raises(self):
        client = MagicMock()
        client.get_battlelog.side_effect = TagInaccessibleError("404")
        db = Dataset.from_seed({f"#{i}" for i in range(12)})
        with pytest.raises(RuntimeError, match="possible API block"):
            Scraper(client, db, max_bad_per_step=10).step()

    def test_bad_tags_at_threshold_does_not_raise(self):
        client = MagicMock()
        client.get_battlelog.side_effect = TagInaccessibleError("404")
        db = Dataset.from_seed({f"#{i}" for i in range(10)})
        Scraper(client, db, max_bad_per_step=10).step()  # exactly 10: no raise

    def test_inaccessible_tag_does_not_crash_step(self):
        """Step continues past a 404 tag and processes subsequent tags."""
        a, b = _ab_teams()

        def side_effect(tag: str) -> list[Battle]:
            if tag == "#DEAD":
                raise TagInaccessibleError("404")
            return [make_battle([a, b], "victory")]

        client = MagicMock()
        client.get_battlelog.side_effect = side_effect
        db = Dataset.from_seed({"#DEAD", a[0].tag})
        n = Scraper(client, db).step()
        assert n == 1
        assert db.bad_tags == {"#DEAD"}

    def test_crash_mid_step_leaves_frontier_intact(self):
        """Processed tags stay in frontier on crash; re-querying them is safe via seen_battle_ids dedup."""
        a, b = _ab_teams()
        battle = make_battle([a, b], "victory")
        call_count = 0

        def side_effect(tag: str) -> list[Battle]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise BrawlApiError("second tag failed")
            return [battle]

        client = MagicMock()
        client.get_battlelog.side_effect = side_effect
        db = Dataset.from_seed({"#A0", "#A1"})

        with pytest.raises(BrawlApiError):
            Scraper(client, db).step()

        # Frontier not drained; both tags survive
        assert len(db.frontier) == 2
        # The first tag's battles were recorded; re-running dedupes them
        assert len(db.seen_battle_ids) == 1
