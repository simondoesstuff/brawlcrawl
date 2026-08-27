import os
from dataclasses import dataclass
from types import TracebackType
from typing import TypedDict, cast

import httpx

_BASE_URL = "https://api.brawlstars.com/v1"


# --- Raw API shapes (private) ---

class _RawBrawler(TypedDict):
    id: int
    name: str
    power: int
    trophies: int


class _RawPlayer(TypedDict):
    tag: str
    name: str
    brawler: _RawBrawler


class _RawEvent(TypedDict, total=False):
    id: int
    mode: str
    map: str


class _RawBattle(TypedDict, total=False):
    mode: str
    type: str | None
    result: str | None
    teams: list[list[_RawPlayer]]
    players: list[_RawPlayer]


class _RawBattleItem(TypedDict):
    event: _RawEvent
    battle: _RawBattle


class _RawBattlelogResponse(TypedDict):
    items: list[_RawBattleItem]


class _RawBrawlersResponse(TypedDict):
    items: list[_RawBrawler]


# --- Public dataclasses ---

@dataclass(frozen=True)
class Brawler:
    id: int
    name: str
    power: int
    trophies: int


@dataclass(frozen=True)
class Player:
    tag: str
    name: str
    brawler: Brawler


@dataclass(frozen=True)
class Event:
    id: int
    mode: str
    map: str


@dataclass(frozen=True)
class Battle:
    event: Event
    mode: str
    type: str | None
    result: str | None
    teams: list[list[Player]] | None
    players: list[Player] | None


# --- Parsers ---

def _parse_brawler(raw: _RawBrawler) -> Brawler:
    return Brawler(
        id=raw["id"],
        name=raw["name"],
        power=raw["power"],
        trophies=raw["trophies"],
    )


def _parse_player(raw: _RawPlayer) -> Player:
    return Player(
        tag=raw["tag"],
        name=raw["name"],
        brawler=_parse_brawler(raw["brawler"]),
    )


def _parse_battle(raw: _RawBattleItem) -> Battle:
    event_raw = raw.get("event") or {}
    event = Event(
        id=event_raw.get("id") or 0,
        mode=event_raw.get("mode") or "",
        map=event_raw.get("map") or "",
    )
    battle = raw["battle"]
    teams_raw = battle.get("teams")
    players_raw = battle.get("players")
    return Battle(
        event=event,
        mode=battle.get("mode") or "",
        type=battle.get("type"),
        result=battle.get("result"),
        teams=[[_parse_player(p) for p in team] for team in teams_raw] if teams_raw else None,
        players=[_parse_player(p) for p in players_raw] if players_raw else None,
    )


# --- Client ---

class BrawlStarsClient:
    _client: httpx.Client

    def __init__(self, token: str | None = None):
        token = token or os.environ["BSTOK"]
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
        )

    def get_battlelog(self, tag: str) -> list[Battle]:
        encoded = tag.lstrip("#")
        response = self._client.get(f"/players/%23{encoded}/battlelog")
        _ = response.raise_for_status()
        data = cast(_RawBattlelogResponse, response.json())
        return [_parse_battle(item) for item in data["items"]]

    def get_brawlers(self) -> list[_RawBrawler]:
        response = self._client.get("/brawlers")
        _ = response.raise_for_status()
        data = cast(_RawBrawlersResponse, response.json())
        return list(data["items"])

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BrawlStarsClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
