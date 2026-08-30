# Overview

- Goal: build an ML model to predict Brawl Stars match outcomes from brawler picks
- Input signals: event (map/mode), brawler composition per team, player trophy level as a skill proxy
- Output: win probability for a given team composition

## Layout

- `workflow/` — Snakemake pipeline; fetches raw JSON from the Brawl Stars API (`$BSTOK`) into `data/`
- `src/crawl/` — REST client, battle log scraper; fetches and aggregates battle data
- `src/geneus/` — ML model and training loop (`BrawlModel`, `train` CLI)
- `src/pick/` — interactive draft-assist REPL (`pick` CLI)
  - `fuzzy.py` — subsequence fuzzy match with shortest-name tiebreak (see `docs/misc_notes.md`)
  - `score.py` — checkpoint loading; `get_q_values` builds draft observations (non-pool brawlers marked `LOCALLY_BANNED`) and calls `DraftQNetwork`; `get_terminal_pick6_scores` vmaps BrawlModel over pick-6 candidates and returns `P(team B wins)`
  - `display.py` — columnar terminal grid with rarity-colored brawler names
  - `main.py` — full draft loop: coin flip → 3 ally bans → 3 enemy bans → picks 1–5 (live Q-value grid) → pick-6 win-probability grid (in-place, terminal model); Q-values are z-scored over the post-filter available brawlers; after pick 5 any Enter exits; `$BRAWL_FILTER` (comma-separated names) hides and `LOCALLY_BANNED`-marks non-owned brawlers; filter is toggled with `/` (works in all phases including pick-6); filter defaults per phase: OFF during bans, ON during ally picks, OFF during enemy picks
- `tests/` — unit tests (no network); `tests/integration/` — API client tests using mocked HTTP
- `docs/dataset_generation.md` — usage guide for the dataset scraper

## Key domain notes (see `battlelog_api_notes.md`)

- `type: "ranked"` = trophy matchmaking; `type: "soloRanked"` = true ranked (trophies field holds rank number, not trophies)
- Battle logs use `teams[][]` for 3v3 modes and `players[]` for showdown
- The `result` field is always from the perspective of the battlelog owner
