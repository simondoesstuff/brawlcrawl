import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from crawl.api import Battle, Brawler, Event, Player, _parse_battle
from crawl.battles import (
    BattleRecord,
    filter_by_min_trophies,
    filter_since,
    filter_solo_ranked,
    to_battle_records,
    unique_player_tags,
)

DATA = Path(__file__).parent.parent / "data"
OWNER_TAG = "#PJPV2VRLP"


def load_battles(filename: str) -> list[Battle]:
    raw = json.loads((DATA / filename).read_text())
    return [_parse_battle(item) for item in raw["items"]]


@pytest.fixture
def pjp_battles() -> list[Battle]:
    return load_battles("battles_pjpv2vrlp.json")


def make_player(tag: str, trophies: int, brawler_id: int = 1) -> Player:
    return Player(tag=tag, name=tag, brawler=Brawler(id=brawler_id, name="X", power=11, trophies=trophies))


_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_battle(
    teams: list[list[Player]],
    result: str,
    event_id: int = 1,
    battle_type: str = "soloRanked",
    battle_time: datetime = _T0,
) -> Battle:
    return Battle(
        battle_time=battle_time,
        event=Event(id=event_id, mode="gemGrab", map="Test Map"),
        mode="gemGrab",
        type=battle_type,
        result=result,
        teams=teams,
        players=None,
    )


class TestUniquePlayerTags:
    def test_returns_tags_above_threshold(self, pjp_battles: list[Battle]):
        tags = unique_player_tags(pjp_battles, min_trophies=1000)
        assert OWNER_TAG in tags

    def test_excludes_tags_below_threshold(self, pjp_battles: list[Battle]):
        tags = unique_player_tags(pjp_battles, min_trophies=99999)
        assert len(tags) == 0

    def test_all_tags_at_zero_threshold(self, pjp_battles: list[Battle]):
        tags_all = unique_player_tags(pjp_battles, min_trophies=0)
        tags_none = unique_player_tags(pjp_battles, min_trophies=99999)
        assert len(tags_all) > len(tags_none)

    def test_includes_showdown_players(self, pjp_battles: list[Battle]):
        showdown = [b for b in pjp_battles if b.teams is None and b.players is not None]
        assert showdown, "fixture must contain showdown battles"
        tags = unique_player_tags(showdown, min_trophies=0)
        assert len(tags) > 0

    def test_fixture_counts(self):
        p1 = make_player("#A", trophies=1000)
        p2 = make_player("#B", trophies=500)
        battle = make_battle(teams=[[p1], [p2]], result="victory")
        assert unique_player_tags([battle], min_trophies=750) == {"#A"}
        assert unique_player_tags([battle], min_trophies=400) == {"#A", "#B"}


class TestFilterSince:
    def test_keeps_battles_at_or_after_cutoff(self):
        early = make_battle([], "victory", battle_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
        late  = make_battle([], "victory", battle_time=datetime(2026, 6, 1, tzinfo=timezone.utc))
        cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
        assert filter_since([early, late], cutoff) == [late]

    def test_inclusive_on_exact_cutoff(self):
        t = datetime(2026, 3, 1, tzinfo=timezone.utc)
        battle = make_battle([], "victory", battle_time=t)
        assert filter_since([battle], t) == [battle]

    def test_real_data_parses_battle_time(self, pjp_battles: list[Battle]):
        assert all(b.battle_time.tzinfo is not None for b in pjp_battles)

    def test_real_data_cutoff(self, pjp_battles: list[Battle]):
        newest = max(b.battle_time for b in pjp_battles)
        assert filter_since(pjp_battles, newest) == [next(b for b in pjp_battles if b.battle_time == newest)]


class TestFilterSoloRanked:
    def test_keeps_solo_ranked(self, pjp_battles: list[Battle]):
        result = filter_solo_ranked(pjp_battles)
        assert all(b.type == "soloRanked" for b in result)

    def test_removes_other_types(self, pjp_battles: list[Battle]):
        non_solo = [b for b in pjp_battles if b.type != "soloRanked"]
        assert filter_solo_ranked(non_solo) == []

    def test_real_data_has_solo_ranked(self, pjp_battles: list[Battle]):
        assert len(filter_solo_ranked(pjp_battles)) > 0

    def test_fixture(self):
        solo = make_battle(teams=[], result="victory", battle_type="soloRanked")
        ranked = make_battle(teams=[], result="victory", battle_type="ranked")
        assert filter_solo_ranked([solo, ranked]) == [solo]


class TestFilterByMinTrophies:
    def test_keeps_battles_with_enough_qualifiers(self):
        players = [make_player(f"#{i}", trophies=1000, brawler_id=i) for i in range(6)]
        battle = make_battle(teams=[players[:3], players[3:]], result="victory")
        assert filter_by_min_trophies([battle], min_trophies=999) == [battle]

    def test_drops_battles_below_threshold(self):
        high = make_player("#H", trophies=1000)
        low = [make_player(f"#{i}", trophies=100, brawler_id=i) for i in range(5)]
        battle = make_battle(teams=[[high, low[0], low[1]], [low[2], low[3], low[4]]], result="victory")
        assert filter_by_min_trophies([battle], min_trophies=999) == []

    def test_min_players_parameter(self):
        players = [make_player(f"#{i}", trophies=1000, brawler_id=i) for i in range(3)]
        low = [make_player(f"#L{i}", trophies=100, brawler_id=i + 10) for i in range(3)]
        battle = make_battle(teams=[players, low], result="victory")
        assert filter_by_min_trophies([battle], min_trophies=999, min_players=3) == [battle]
        assert filter_by_min_trophies([battle], min_trophies=999, min_players=4) == []

    def test_real_data(self, pjp_battles: list[Battle]):
        result = filter_by_min_trophies(pjp_battles, min_trophies=1000)
        assert len(result) > 0
        assert len(result) <= len(pjp_battles)


class TestToBattleRecords:
    def test_victory_assigns_winner_correctly(self):
        owner = make_player("#OWNER", trophies=1000, brawler_id=10)
        ally = make_player("#ALLY",  trophies=1000, brawler_id=20)
        ally2 = make_player("#ALLY2", trophies=1000, brawler_id=30)
        opp1 = make_player("#OPP1", trophies=1000, brawler_id=40)
        opp2 = make_player("#OPP2", trophies=1000, brawler_id=50)
        opp3 = make_player("#OPP3", trophies=1000, brawler_id=60)
        battle = make_battle(teams=[[owner, ally, ally2], [opp1, opp2, opp3]], result="victory")
        records = to_battle_records([battle], "#OWNER")
        assert set(records[0].winning_brawler_ids) == {10, 20, 30}
        assert set(records[0].losing_brawler_ids) == {40, 50, 60}

    def test_defeat_assigns_winner_correctly(self):
        owner = make_player("#OWNER", trophies=1000, brawler_id=10)
        ally = make_player("#ALLY",  trophies=1000, brawler_id=20)
        ally2 = make_player("#ALLY2", trophies=1000, brawler_id=30)
        opp1 = make_player("#OPP1", trophies=1000, brawler_id=40)
        opp2 = make_player("#OPP2", trophies=1000, brawler_id=50)
        opp3 = make_player("#OPP3", trophies=1000, brawler_id=60)
        battle = make_battle(teams=[[owner, ally, ally2], [opp1, opp2, opp3]], result="defeat")
        records = to_battle_records([battle], "#OWNER")
        assert set(records[0].winning_brawler_ids) == {40, 50, 60}
        assert set(records[0].losing_brawler_ids) == {10, 20, 30}

    def test_owner_in_second_team(self):
        p1 = make_player("#P1", trophies=1000, brawler_id=1)
        p2 = make_player("#P2", trophies=1000, brawler_id=2)
        p3 = make_player("#P3", trophies=1000, brawler_id=3)
        owner = make_player("#OWNER", trophies=1000, brawler_id=99)
        p5 = make_player("#P5", trophies=1000, brawler_id=5)
        p6 = make_player("#P6", trophies=1000, brawler_id=6)
        battle = make_battle(teams=[[p1, p2, p3], [owner, p5, p6]], result="defeat")
        records = to_battle_records([battle], "#OWNER")
        assert set(records[0].winning_brawler_ids) == {1, 2, 3}
        assert 99 in records[0].losing_brawler_ids

    def test_skips_showdown_battles(self, pjp_battles: list[Battle]):
        showdown = [b for b in pjp_battles if b.teams is None]
        assert to_battle_records(showdown, OWNER_TAG) == []

    def test_event_id_preserved(self):
        players = [make_player(f"#{i}", trophies=1000, brawler_id=i) for i in range(6)]
        battle = make_battle(teams=[players[:3], players[3:]], result="victory", event_id=42)
        records = to_battle_records([battle], players[0].tag)
        assert records[0].event_id == 42

    def test_real_data(self, pjp_battles: list[Battle]):
        records = to_battle_records(pjp_battles, OWNER_TAG)
        assert len(records) > 0
        for r in records:
            assert isinstance(r, BattleRecord)
            assert len(r.winning_brawler_ids) > 0
            assert len(r.losing_brawler_ids) > 0


class TestBattleId:
    def _six_player_battle(self, result: str = "victory", t: datetime = _T0) -> Battle:
        players = [make_player(f"#P{i}", trophies=1000, brawler_id=i) for i in range(6)]
        return make_battle(teams=[players[:3], players[3:]], result=result, battle_time=t)

    def test_id_is_deterministic(self):
        battle = self._six_player_battle()
        r1 = to_battle_records([battle], "#P0")[0]
        r2 = to_battle_records([battle], "#P0")[0]
        assert r1.battle_id == r2.battle_id

    def test_same_battle_same_id_regardless_of_owner(self):
        battle = self._six_player_battle()
        id_from_p0 = to_battle_records([battle], "#P0")[0].battle_id
        id_from_p3 = to_battle_records([battle], "#P3")[0].battle_id
        assert id_from_p0 == id_from_p3

    def test_different_time_different_id(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
        id1 = to_battle_records([self._six_player_battle(t=t1)], "#P0")[0].battle_id
        id2 = to_battle_records([self._six_player_battle(t=t2)], "#P0")[0].battle_id
        assert id1 != id2

    def test_different_players_different_id(self):
        p_a = [make_player(f"#A{i}", trophies=1000, brawler_id=i) for i in range(6)]
        p_b = [make_player(f"#B{i}", trophies=1000, brawler_id=i) for i in range(6)]
        battle_a = make_battle(teams=[p_a[:3], p_a[3:]], result="victory")
        battle_b = make_battle(teams=[p_b[:3], p_b[3:]], result="victory")
        id_a = to_battle_records([battle_a], "#A0")[0].battle_id
        id_b = to_battle_records([battle_b], "#B0")[0].battle_id
        assert id_a != id_b

    def test_real_data_ids_are_unique(self, pjp_battles: list[Battle]):
        records = to_battle_records(pjp_battles, OWNER_TAG)
        ids = [r.battle_id for r in records]
        assert len(ids) == len(set(ids))


class TestComposition:
    def test_solo_ranked_then_trophy_filter_then_records(self, pjp_battles: list[Battle]):
        records = to_battle_records(
            filter_by_min_trophies(filter_solo_ranked(pjp_battles), min_trophies=1000),
            OWNER_TAG,
        )
        assert isinstance(records, list)
        assert all(isinstance(r, BattleRecord) for r in records)
