# Dataset Notes

- `crawl_leg1_20260827.json` legendary 1+ battles collected between 2026/07/25 and 2026/08/27

# Dataset Generation

The scraper builds a training dataset by BFS-crawling player battlelogs. Each step queries a frontier of player tags, extracts soloRanked battles, and discovers new tags from the players encountered.

## Quick start

```python
from pathlib import Path
from crawl.api import BrawlStarsClient
from crawl.dataset import Dataset
from crawl.scraper import Scraper

path = Path("data/dataset.json")

# First run: seed from a known high-rank player tag
with BrawlStarsClient() as client:
    db = Dataset.from_seed({"#YLQQY2U0P"})
    scraper = Scraper(client, db, min_trophies=16, request_interval=0.5)
    scraper.run(steps=5, save_path=path)

# Resume from disk
with BrawlStarsClient() as client:
    db = Dataset.load(path)
    scraper = Scraper(client, db, min_trophies=16, request_interval=0.5)
    scraper.run(steps=10, save_path=path)
```

`BSTOK` must be set in the environment. `save_path` writes the dataset after each step so a crash loses at most one step.

## Parameters

| Parameter          | Default | Description                                                                                                                                                                                                                 |
| ------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontier_cap`     | 200     | Max tags queried per step. Uncapped tags stay in the frontier for the next step.                                                                                                                                            |
| `min_trophies`     | 0       | Only count battles where ≥5 players have at least this many trophies. Tags below this threshold are also excluded from discovery. For soloRanked, this field holds rank number — rank 16 (legendary 1) ≈ `min_trophies=16`. |
| `since`            | None    | Drop battles older than this datetime.                                                                                                                                                                                      |
| `request_interval` | 0.0     | Seconds to sleep before each API call. Use ≥0.5 to avoid rate limiting.                                                                                                                                                     |

## Dataset structure

`Dataset` holds four fields, all serialized to a single JSON file:

- **`seen_tags`** — every tag ever added to the frontier; monotonically grows
- **`frontier`** — tags queued for the next step; drains as steps run, refills with discoveries
- **`seen_battle_ids`** — SHA-256 hashes of processed battles; the dedup store
- **`stats`** — `Composition → WinLoss` mapping; the actual training data

`Composition` is a canonical matchup key: `(event_id, team_a_brawler_ids, team_b_brawler_ids)` where `team_a` is always the lexicographically smaller set of brawler IDs. `WinLoss.a_wins` counts wins for the `team_a` side; `WinLoss.total` counts all battles for that matchup.

## Growth characteristics

Each player's battlelog holds ~25 battles; soloRanked filtering and trophy gating leave ~8 qualifying battles per player. Each battle has 6 players, yielding ~14 net new tags per frontier tag after dedup. Frontier growth is roughly 14× per step before saturation.

## Merging datasets

Two datasets crawled at different times can be combined with `Dataset.merge(old, new, alpha)` (CLI: `dataset-merge`). Compositions seen in only one dataset carry over unchanged. Compositions seen in both are combined via `merge_stats`, which pools `a_wins`/`total` by sample count and uses `alpha` in `[0, 1]` to discount the older dataset's counts before pooling: `alpha=0` is a plain count-weighted pool, `alpha=1` fully trusts the newer dataset's win rate for that matchup and drops the old evidence entirely.

`total` doubles as the training loss weight (see `geneus/train.py`), so old is discounted rather than new being boosted — the combined total can only shrink toward (never exceed) the true `old.total + new.total`, keeping the merged confidence weight honest.

## Crash safety

If the process dies mid-step, the dataset on disk is consistent: `seen_battle_ids` guards against double-counting on resume, and unfinished frontier tags are re-queried cleanly. Any `BrawlApiError` (malformed 200 response) propagates immediately — do not catch it silently.
