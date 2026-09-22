import json
import math
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


def merge_stats(old: WinLoss, new: WinLoss, alpha: float) -> WinLoss:
    """Combine stats for one composition seen in both an older and a newer dataset.

    `total` doubles as the training loss weight downstream (see geneus/train.py),
    so the combined total must never exceed the true observed evidence
    (old.total + new.total). Rather than boosting new's weight, we discount
    old's: `alpha` in [0, 1] scales down old's counts before pooling by sample
    count, so alpha=0 is a plain count-weighted pool and alpha=1 fully trusts
    the new dataset's win rate (old is discounted to zero weight).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    old_weight = old.total * (1.0 - alpha)
    old_wins = old.a_wins * (1.0 - alpha)
    total_f = old_weight + new.total
    wins_f = old_wins + new.a_wins
    total = math.floor(total_f + 0.5)
    a_wins = min(total, max(0, math.floor(wins_f / total_f * total + 0.5)))
    return WinLoss(a_wins=a_wins, total=total)


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

    @classmethod
    def merge(cls, old: "Dataset", new: "Dataset", alpha: float = 0.5) -> "Dataset":
        """Merge two datasets crawled at different times into one.

        Compositions seen in only one dataset are carried over unchanged;
        compositions seen in both are combined via `merge_stats`, with `alpha`
        biasing the result toward the newer dataset's win rate.
        """
        stats: dict[Composition, WinLoss] = dict(old.stats)
        for comp, new_wl in new.stats.items():
            old_wl = stats.get(comp)
            stats[comp] = (
                merge_stats(old_wl, new_wl, alpha)
                if old_wl is not None
                else WinLoss(a_wins=new_wl.a_wins, total=new_wl.total)
            )

        return cls(
            seen_tags=old.seen_tags | new.seen_tags,
            frontier=old.frontier | new.frontier,
            seen_battle_ids=old.seen_battle_ids | new.seen_battle_ids,
            bad_tags=old.bad_tags | new.bad_tags,
            stats=stats,
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
