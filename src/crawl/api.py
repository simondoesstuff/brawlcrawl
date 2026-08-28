import os
from dataclasses import dataclass
from datetime import datetime, timezone
from types import TracebackType
from typing import TypedDict, cast

import httpx


class BrawlApiError(Exception):
    """Raised when the Brawl Stars API returns an unparseable response."""


class TagInaccessibleError(BrawlApiError):
    """Raised on 404/403 — tag exists in battle records but has no accessible battlelog."""

_BASE_URL = "https://api.brawlstars.com/v1"


# --- Raw API shapes (private) ---

class _RawBrawler(TypedDict):
    id: int
    name: str
    power: int
    trophies: int


class _RawPlayerRequired(TypedDict):
    tag: str
    name: str

class _RawPlayer(_RawPlayerRequired, total=False):
    brawler: _RawBrawler


class _RawEvent(TypedDict, total=False):
    id: int
    mode: str
    modeId: int
    map: str


class _RawBattle(TypedDict, total=False):
    mode: str
    type: str | None
    result: str | None
    teams: list[list[_RawPlayer]]
    players: list[_RawPlayer]


class _RawBattleItem(TypedDict):
    battleTime: str
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
    mode_id: int = 0


@dataclass(frozen=True)
class Battle:
    battle_time: datetime
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


def _parse_player(raw: _RawPlayer) -> Player | None:
    brawler_raw = raw.get("brawler")
    if brawler_raw is None:
        return None
    return Player(
        tag=raw["tag"],
        name=raw["name"],
        brawler=_parse_brawler(brawler_raw),
    )


def _parse_players(raws: list[_RawPlayer]) -> list[Player]:
    return [p for raw in raws if (p := _parse_player(raw)) is not None]


_BATTLE_TIME_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def _parse_battle(raw: _RawBattleItem) -> Battle:
    event_raw = raw.get("event") or {}
    event = Event(
        id=event_raw.get("id") or 0,
        mode=event_raw.get("mode") or "",
        map=event_raw.get("map") or "",
        mode_id=event_raw.get("modeId") or 0,
    )
    battle = raw["battle"]
    teams_raw = battle.get("teams")
    players_raw = battle.get("players")
    return Battle(
        battle_time=datetime.strptime(raw["battleTime"], _BATTLE_TIME_FORMAT).replace(tzinfo=timezone.utc),
        event=event,
        mode=battle.get("mode") or "",
        type=battle.get("type"),
        result=battle.get("result"),
        teams=[_parse_players(team) for team in teams_raw] if teams_raw else None,
        players=_parse_players(players_raw) if players_raw else None,
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
        if response.status_code in (403, 404):
            raise TagInaccessibleError(f"{response.status_code} for tag {tag}")
        _ = response.raise_for_status()
        try:
            data = cast(_RawBattlelogResponse, response.json())
            return [_parse_battle(item) for item in data["items"]]
        except Exception as exc:
            raise BrawlApiError(
                f"failed to parse battlelog for {tag}: {response.text[:500]}"
            ) from exc

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
