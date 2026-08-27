import hashlib
from dataclasses import dataclass
from datetime import datetime

from brawl.api import Battle, Player


def _all_players(battle: Battle) -> list[Player]:
    if battle.teams is not None:
        return [p for team in battle.teams for p in team]
    if battle.players is not None:
        return battle.players
    return []


def unique_player_tags(battles: list[Battle], min_trophies: int) -> set[str]:
    """Unique player tags where that player's brawler had >= min_trophies."""
    tags: set[str] = set()
    for battle in battles:
        for player in _all_players(battle):
            if player.brawler.trophies >= min_trophies:
                tags.add(player.tag)
    return tags


def filter_since(battles: list[Battle], since: datetime) -> list[Battle]:
    """Keep battles that occurred at or after `since`."""
    return [b for b in battles if b.battle_time >= since]


def filter_solo_ranked(battles: list[Battle]) -> list[Battle]:
    """Keep only soloRanked battles."""
    return [b for b in battles if b.type == "soloRanked"]


def filter_by_min_trophies(battles: list[Battle], min_trophies: int, min_players: int = 5) -> list[Battle]:
    """Keep battles where at least min_players players have brawler trophies >= min_trophies."""
    def qualifies(battle: Battle) -> bool:
        count = sum(1 for p in _all_players(battle) if p.brawler.trophies >= min_trophies)
        return count >= min_players
    return [b for b in battles if qualifies(b)]


@dataclass(frozen=True)
class BattleRecord:
    battle_id: str
    event_id: int
    winning_brawler_ids: list[int]
    losing_brawler_ids: list[int]


def _battle_id(battle: Battle) -> str:
    tags = sorted(p.tag for team in (battle.teams or []) for p in team)
    payload = battle.battle_time.isoformat() + "|" + ",".join(tags)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def to_battle_records(battles: list[Battle], owner_tag: str) -> list[BattleRecord]:
    """
    Condense team battles into BattleRecords with winner indicated.

    Skips non-team battles and any battle where the owner's team cannot be found.
    owner_tag identifies whose battlelog these battles came from, used to
    interpret the result field (which is always from that player's perspective).
    """
    records: list[BattleRecord] = []
    for battle in battles:
        if battle.teams is None or battle.result not in ("victory", "defeat"):
            continue

        owner_team = next(
            (i for i, team in enumerate(battle.teams) if any(p.tag == owner_tag for p in team)),
            None,
        )
        if owner_team is None:
            continue

        winner_idx = owner_team if battle.result == "victory" else 1 - owner_team
        loser_idx = 1 - winner_idx

        records.append(BattleRecord(
            battle_id=_battle_id(battle),
            event_id=battle.event.id,
            winning_brawler_ids=[p.brawler.id for p in battle.teams[winner_idx]],
            losing_brawler_ids=[p.brawler.id for p in battle.teams[loser_idx]],
        ))

    return records
