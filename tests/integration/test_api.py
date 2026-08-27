"""
API client tests — uses pytest-httpx to mock HTTP; no real network calls.
Run separately from the main suite: pytest tests/integration
"""
import pytest
from pytest_httpx import HTTPXMock

from brawl.api import Battle, BrawlStarsClient

_BATTLELOG_RESPONSE = {
    "items": [
        {
            "battleTime": "20260815T054351.000Z",
            "event": {"id": 15000007, "mode": "gemGrab", "modeId": 0, "map": "Hard Rock Mine"},
            "battle": {
                "mode": "gemGrab",
                "type": "ranked",
                "result": "defeat",
                "duration": 138,
                "teams": [
                    [
                        {"tag": "#AAA", "name": "Alice", "brawler": {"id": 1, "name": "SHELLY", "power": 11, "trophies": 900}},
                        {"tag": "#BBB", "name": "Bob",   "brawler": {"id": 2, "name": "SPIKE",  "power": 11, "trophies": 850}},
                        {"tag": "#CCC", "name": "Carol", "brawler": {"id": 3, "name": "EVE",    "power": 11, "trophies": 800}},
                    ],
                    [
                        {"tag": "#OWNER", "name": "Owner", "brawler": {"id": 4, "name": "BUZZ",  "power": 11, "trophies": 950}},
                        {"tag": "#EEE",   "name": "Eve",   "brawler": {"id": 5, "name": "TRUNK", "power": 11, "trophies": 920}},
                        {"tag": "#FFF",   "name": "Frank", "brawler": {"id": 6, "name": "DRACO", "power": 11, "trophies": 880}},
                    ],
                ],
            },
        },
        {
            "battleTime": "20260815T050000.000Z",
            "event": {"id": 15000001, "mode": "soloShowdown", "modeId": 0, "map": "Skull Creek"},
            "battle": {
                "mode": "soloShowdown",
                "type": "ranked",
                "result": None,
                "players": [
                    {"tag": "#OWNER", "name": "Owner", "brawler": {"id": 7, "name": "CLANCY", "power": 11, "trophies": 1000}},
                    {"tag": "#GGG",   "name": "Gary",  "brawler": {"id": 8, "name": "BUSTER", "power": 11, "trophies": 980}},
                ],
            },
        },
    ]
}

_BRAWLERS_RESPONSE = {
    "items": [
        {"id": 16000000, "name": "SHELLY"},
        {"id": 16000001, "name": "COLT"},
    ]
}


@pytest.fixture
def client(httpx_mock: HTTPXMock) -> BrawlStarsClient:
    return BrawlStarsClient(token="test-token")


class TestGetBattlelog:
    def test_returns_battle_list(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json=_BATTLELOG_RESPONSE)
        battles = client.get_battlelog("#OWNER")
        assert len(battles) == 2

    def test_url_encodes_hash(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json=_BATTLELOG_RESPONSE,
            url="https://api.brawlstars.com/v1/players/%23OWNER/battlelog",
        )
        battles = client.get_battlelog("#OWNER")
        assert len(battles) == 2

    def test_team_battle_parsed(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json=_BATTLELOG_RESPONSE)
        battle = client.get_battlelog("#OWNER")[0]
        assert isinstance(battle, Battle)
        assert battle.event.id == 15000007
        assert battle.event.mode == "gemGrab"
        assert battle.event.map == "Hard Rock Mine"
        assert battle.mode == "gemGrab"
        assert battle.type == "ranked"
        assert battle.result == "defeat"
        assert battle.teams is not None
        assert battle.players is None
        assert len(battle.teams) == 2
        assert len(battle.teams[0]) == 3

    def test_team_player_fields(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json=_BATTLELOG_RESPONSE)
        player = client.get_battlelog("#OWNER")[0].teams[0][0]  # type: ignore[index]
        assert player.tag == "#AAA"
        assert player.name == "Alice"
        assert player.brawler.id == 1
        assert player.brawler.trophies == 900

    def test_showdown_battle_parsed(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json=_BATTLELOG_RESPONSE)
        battle = client.get_battlelog("#OWNER")[1]
        assert battle.teams is None
        assert battle.players is not None
        assert len(battle.players) == 2
        assert battle.players[0].tag == "#OWNER"

    def test_auth_header_sent(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            json=_BATTLELOG_RESPONSE,
            match_headers={"Authorization": "Bearer test-token"},
        )
        battles = client.get_battlelog("OWNER")
        assert len(battles) == 2

    def test_raises_on_http_error(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=403)
        with pytest.raises(Exception):
            client.get_battlelog("#OWNER")


class TestGetBrawlers:
    def test_returns_list(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json=_BRAWLERS_RESPONSE)
        brawlers = client.get_brawlers()
        assert len(brawlers) == 2
        assert brawlers[0]["id"] == 16000000
        assert brawlers[0]["name"] == "SHELLY"

    def test_raises_on_http_error(self, client: BrawlStarsClient, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=500)
        with pytest.raises(Exception):
            client.get_brawlers()


class TestContextManager:
    def test_context_manager(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(json=_BATTLELOG_RESPONSE)
        with BrawlStarsClient(token="test-token") as client:
            battles = client.get_battlelog("#OWNER")
        assert len(battles) == 2
