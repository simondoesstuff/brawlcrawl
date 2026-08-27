# Overview

- Goal: build an ML model to predict Brawl Stars match outcomes from brawler picks
- Input signals: event (map/mode), brawler composition per team, player trophy level as a skill proxy
- Output: win probability for a given team composition

## Layout

- `workflow/` — Snakemake pipeline; fetches raw JSON from the Brawl Stars API (`$BSTOK`) into `data/`
- `src/brawl/api.py` — REST client; the only place that touches HTTP or JSON; returns typed dataclasses
- `src/brawl/battles.py` — composable filters and converters over battle lists
- `src/brawl/dataset.py` — `Dataset` state (seen tags, frontier, stats, battle hash store); serializes to JSON
- `src/brawl/scraper.py` — pure `consume()` function; `Scraper` drives iterative BFS extension steps
- `tests/` — unit tests (no network); `tests/integration/` — API client tests using mocked HTTP

## Key domain notes (see `battlelog_api_notes.md`)

- `type: "ranked"` = trophy matchmaking; `type: "soloRanked"` = true ranked (trophies field holds rank number, not trophies)
- Battle logs use `teams[][]` for 3v3 modes and `players[]` for showdown
- The `result` field is always from the perspective of the battlelog owner
