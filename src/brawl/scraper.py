from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brawl.api import Battle, BrawlStarsClient
from brawl.battles import filter_by_min_trophies, filter_since, filter_solo_ranked, to_battle_records, unique_player_tags
from brawl.dataset import Composition, Dataset, WinLoss


@dataclass
class ConsumeResult:
    new_battle_ids: set[str]
    stat_updates: dict[Composition, WinLoss]
    discovered_tags: set[str]


def consume(
    owner_tag: str,
    battles: list[Battle],
    seen_battle_ids: set[str],
    min_trophies: int = 0,
    since: datetime | None = None,
) -> ConsumeResult:
    """Pure: process one player's battlelog, returning new data without mutating anything."""
    filtered = filter_solo_ranked(battles)
    if since is not None:
        filtered = filter_since(filtered, since)
    filtered = filter_by_min_trophies(filtered, min_trophies, min_players=5)
    records = to_battle_records(filtered, owner_tag)
    new_battle_ids: set[str] = set()
    stat_updates: dict[Composition, WinLoss] = {}

    for record in records:
        if record.battle_id in seen_battle_ids:
            continue
        new_battle_ids.add(record.battle_id)
        comp, a_won = Composition.from_sides(
            event_id=record.event_id,
            winning=frozenset(record.winning_brawler_ids),
            losing=frozenset(record.losing_brawler_ids),
        )
        entry = stat_updates.setdefault(comp, WinLoss())
        entry.total += 1
        if a_won:
            entry.a_wins += 1

    return ConsumeResult(
        new_battle_ids=new_battle_ids,
        stat_updates=stat_updates,
        discovered_tags=unique_player_tags(filtered, min_trophies=min_trophies),
    )


class Scraper:
    _client: BrawlStarsClient
    _dataset: Dataset
    _frontier_cap: int
    _min_trophies: int
    _since: datetime | None

    def __init__(
        self,
        client: BrawlStarsClient,
        dataset: Dataset,
        frontier_cap: int = 200,
        min_trophies: int = 0,
        since: datetime | None = None,
    ) -> None:
        self._client = client
        self._dataset = dataset
        self._frontier_cap = frontier_cap
        self._min_trophies = min_trophies
        self._since = since

    def step(self) -> int:
        """One BFS extension step. Returns count of new unique battles seen."""
        tags = list(self._dataset.frontier)[: self._frontier_cap]
        all_discovered: set[str] = set()
        new_battle_count = 0

        for tag in tags:
            battles = self._client.get_battlelog(tag)
            result = consume(tag, battles, self._dataset.seen_battle_ids, self._min_trophies, self._since)

            self._dataset.seen_battle_ids |= result.new_battle_ids
            new_battle_count += len(result.new_battle_ids)

            for comp, wl in result.stat_updates.items():
                entry = self._dataset.stats.setdefault(comp, WinLoss())
                entry.a_wins += wl.a_wins
                entry.total += wl.total

            all_discovered |= result.discovered_tags

        next_frontier = all_discovered - self._dataset.seen_tags
        self._dataset.seen_tags |= next_frontier
        self._dataset.frontier = next_frontier
        return new_battle_count

    def run(self, steps: int, save_path: Path | None = None) -> None:
        """Run multiple extension steps, optionally saving after each."""
        for i in range(steps):
            n = self.step()
            print(f"step {i + 1}/{steps}: +{n} battles, frontier={len(self._dataset.frontier)}, compositions={len(self._dataset.stats)}")
            if save_path is not None:
                self._dataset.save(save_path)
