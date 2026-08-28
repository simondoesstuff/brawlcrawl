import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crawl.api import Battle, Brawler, Event, Player, TagInaccessibleError
from crawl.events import crawl_events, load_events, save_events

_EVENT_A = Event(id=15000001, mode="gemGrab", map="Crystal Arcade", mode_id=3)
_EVENT_B = Event(id=15000002, mode="heist", map="Kaboom Canyon", mode_id=2)


def _make_battle(event: Event) -> Battle:
    player = Player(tag="#X", name="X", brawler=Brawler(id=1, name="X", power=11, trophies=1000))
    return Battle(
        battle_time=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        event=event,
        mode=event.mode,
        type=None,
        result=None,
        teams=None,
        players=[player],
    )


class TestCrawlEvents:
    def test_yields_event_when_found(self):
        client = MagicMock()
        client.get_battlelog.return_value = [_make_battle(_EVENT_A)]
        results = list(crawl_events(client, ["#TAG1"], {_EVENT_A.id}))
        assert results == [_EVENT_A]

    def test_stops_early_when_all_found(self):
        client = MagicMock()
        client.get_battlelog.return_value = [_make_battle(_EVENT_A)]
        list(crawl_events(client, ["#TAG1", "#TAG2", "#TAG3"], {_EVENT_A.id}))
        assert client.get_battlelog.call_count == 1

    def test_skips_inaccessible_tags(self):
        client = MagicMock()
        client.get_battlelog.side_effect = [
            TagInaccessibleError("404"),
            [_make_battle(_EVENT_A)],
        ]
        results = list(crawl_events(client, ["#DEAD", "#GOOD"], {_EVENT_A.id}))
        assert results == [_EVENT_A]

    def test_finds_multiple_events_across_tags(self):
        client = MagicMock()
        client.get_battlelog.side_effect = [
            [_make_battle(_EVENT_A)],
            [_make_battle(_EVENT_B)],
        ]
        results = list(crawl_events(client, ["#T1", "#T2"], {_EVENT_A.id, _EVENT_B.id}))
        assert {e.id for e in results} == {_EVENT_A.id, _EVENT_B.id}

    def test_finds_multiple_events_from_one_tag(self):
        client = MagicMock()
        client.get_battlelog.return_value = [_make_battle(_EVENT_A), _make_battle(_EVENT_B)]
        results = list(crawl_events(client, ["#T1"], {_EVENT_A.id, _EVENT_B.id}))
        assert {e.id for e in results} == {_EVENT_A.id, _EVENT_B.id}
        assert client.get_battlelog.call_count == 1

    def test_returns_empty_when_no_tags(self):
        client = MagicMock()
        results = list(crawl_events(client, [], {_EVENT_A.id}))
        assert results == []

    def test_returns_empty_when_needed_ids_empty(self):
        client = MagicMock()
        client.get_battlelog.return_value = [_make_battle(_EVENT_A)]
        results = list(crawl_events(client, ["#T1"], set()))
        assert results == []
        client.get_battlelog.assert_not_called()

    def test_ignores_unneeded_event_ids(self):
        client = MagicMock()
        client.get_battlelog.side_effect = [
            [_make_battle(_EVENT_A)],  # _EVENT_A not in needed
            [_make_battle(_EVENT_B)],
        ]
        results = list(crawl_events(client, ["#T1", "#T2"], {_EVENT_B.id}))
        assert results == [_EVENT_B]

    def test_each_event_id_yielded_once(self):
        client = MagicMock()
        client.get_battlelog.side_effect = [
            [_make_battle(_EVENT_A), _make_battle(_EVENT_A)],
        ]
        results = list(crawl_events(client, ["#T1"], {_EVENT_A.id}))
        assert len(results) == 1


class TestSaveLoadEvents:
    def test_round_trip(self, tmp_path: Path):
        events = {_EVENT_A.id: _EVENT_A, _EVENT_B.id: _EVENT_B}
        path = tmp_path / "events.json"
        save_events(events, path)
        loaded = load_events(path)
        assert loaded == events

    def test_saved_json_structure(self, tmp_path: Path):
        path = tmp_path / "events.json"
        save_events({_EVENT_A.id: _EVENT_A}, path)
        data = json.loads(path.read_text())
        assert data == [{"id": 15000001, "mode": "gemGrab", "modeId": 3, "map": "Crystal Arcade"}]

    def test_saved_sorted_by_id(self, tmp_path: Path):
        events = {_EVENT_B.id: _EVENT_B, _EVENT_A.id: _EVENT_A}
        path = tmp_path / "events.json"
        save_events(events, path)
        data = json.loads(path.read_text())
        assert data[0]["id"] < data[1]["id"]

    def test_load_missing_mode_id_defaults_to_zero(self, tmp_path: Path):
        path = tmp_path / "events.json"
        path.write_text(json.dumps([{"id": 1, "mode": "gemGrab", "map": "Arena"}]))
        loaded = load_events(path)
        assert loaded[1].mode_id == 0
