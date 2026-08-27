import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast


class _StatEntry(TypedDict):
    event_id: int
    team_a: list[int]
    team_b: list[int]
    a_wins: int
    total: int


class _DatasetJson(TypedDict):
    seen_tags: list[str]
    frontier: list[str]
    seen_battle_ids: list[str]
    bad_tags: list[str]
    stats: list[_StatEntry]


@dataclass(frozen=True)
class Composition:
    event_id: int
    team_a_ids: frozenset[int]
    team_b_ids: frozenset[int]

    def __post_init__(self) -> None:
        if sorted(self.team_a_ids) > sorted(self.team_b_ids):
            raise ValueError(
                "team_a_ids must be canonically <= team_b_ids; use Composition.from_sides()"
            )

    @classmethod
    def from_sides(
        cls, event_id: int, winning: frozenset[int], losing: frozenset[int]
    ) -> tuple["Composition", bool]:
        """Return (Composition, a_won) with canonical team ordering."""
        if sorted(winning) <= sorted(losing):
            return cls(event_id=event_id, team_a_ids=winning, team_b_ids=losing), True
        return cls(event_id=event_id, team_a_ids=losing, team_b_ids=winning), False


@dataclass
class WinLoss:
    a_wins: int = 0
    total: int = 0


@dataclass
class Dataset:
    seen_tags: set[str]
    frontier: set[str]
    seen_battle_ids: set[str]
    bad_tags: set[str] = field(default_factory=set)
    stats: dict[Composition, WinLoss] = field(default_factory=dict)

    @classmethod
    def from_seed(cls, seed_tags: set[str]) -> "Dataset":
        return cls(
            seen_tags=set(seed_tags),
            frontier=set(seed_tags),
            seen_battle_ids=set(),
        )

    def save(self, path: Path) -> None:
        data: _DatasetJson = {
            "seen_tags": sorted(self.seen_tags),
            "frontier": sorted(self.frontier),
            "seen_battle_ids": sorted(self.seen_battle_ids),
            "bad_tags": sorted(self.bad_tags),
            "stats": [
                {
                    "event_id": comp.event_id,
                    "team_a": sorted(comp.team_a_ids),
                    "team_b": sorted(comp.team_b_ids),
                    "a_wins": wl.a_wins,
                    "total": wl.total,
                }
                for comp, wl in self.stats.items()
            ],
        }
        _ = path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        data = cast(_DatasetJson, json.loads(path.read_text()))
        stats: dict[Composition, WinLoss] = {}
        for entry in data["stats"]:
            comp = Composition(
                event_id=entry["event_id"],
                team_a_ids=frozenset(entry["team_a"]),
                team_b_ids=frozenset(entry["team_b"]),
            )
            stats[comp] = WinLoss(a_wins=entry["a_wins"], total=entry["total"])
        return cls(
            seen_tags=set(data["seen_tags"]),
            frontier=set(data["frontier"]),
            seen_battle_ids=set(data["seen_battle_ids"]),
            bad_tags=set(data.get("bad_tags", [])),
            stats=stats,
        )
