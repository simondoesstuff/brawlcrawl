import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from brawl.api import Battle, BrawlStarsClient, TagInaccessibleError
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
    _request_interval: float
    _max_bad_per_step: int

    def __init__(
        self,
        client: BrawlStarsClient,
        dataset: Dataset,
        frontier_cap: int = 200,
        min_trophies: int = 0,
        since: datetime | None = None,
        request_interval: float = 0.0,
        max_bad_per_step: int = 10,
    ) -> None:
        self._client = client
        self._dataset = dataset
        self._frontier_cap = frontier_cap
        self._min_trophies = min_trophies
        self._since = since
        self._request_interval = request_interval
        self._max_bad_per_step = max_bad_per_step

    def step(self) -> int:
        """One BFS extension step. Returns count of new unique battles seen.

        Crash-safe: if get_battlelog raises mid-loop, seen_battle_ids and stats
        are already partially updated but the frontier drain (below) hasn't run,
        so processed tags remain in frontier. On the next run they'll be re-queried
        and their battles deduped via seen_battle_ids — no double-counting.
        """
        tags = list(self._dataset.frontier)[: self._frontier_cap]
        all_discovered: set[str] = set()
        new_battle_count = 0
        new_bad_count = 0

        for tag in tqdm(tags, desc="scraping", unit="tag", leave=False):
            if self._request_interval > 0:
                time.sleep(self._request_interval)
            try:
                battles = self._client.get_battlelog(tag)
            except TagInaccessibleError:
                self._dataset.bad_tags.add(tag)
                self._dataset.seen_tags.add(tag)
                new_bad_count += 1
                if new_bad_count > self._max_bad_per_step:
                    raise RuntimeError(
                        f"{new_bad_count} inaccessible tags in one step — possible API block. Halting."
                    )
                continue
            result = consume(tag, battles, self._dataset.seen_battle_ids, self._min_trophies, self._since)

            self._dataset.seen_battle_ids |= result.new_battle_ids
            new_battle_count += len(result.new_battle_ids)

            for comp, wl in result.stat_updates.items():
                entry = self._dataset.stats.setdefault(comp, WinLoss())
                entry.a_wins += wl.a_wins
                entry.total += wl.total

            all_discovered |= result.discovered_tags

        new_tags = all_discovered - self._dataset.seen_tags
        self._dataset.frontier -= set(tags)
        self._dataset.frontier |= new_tags
        self._dataset.seen_tags |= new_tags
        return new_battle_count

    def run(self, steps: int, save_path: Path | None = None) -> None:
        """Run multiple extension steps, optionally saving after each."""
        for i in range(steps):
            n = self.step()
            print(f"step {i + 1}/{steps}: +{n} battles, frontier={len(self._dataset.frontier)}, compositions={len(self._dataset.stats)}")
            if save_path is not None:
                self._dataset.save(save_path)
