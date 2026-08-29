# Overview

- Goal: build an ML model to predict Brawl Stars match outcomes from brawler picks
- Input signals: event (map/mode), brawler composition per team, player trophy level as a skill proxy
- Output: win probability for a given team composition

## Layout

- `workflow/` — Snakemake pipeline; fetches raw JSON from the Brawl Stars API (`$BSTOK`) into `data/`
- `src/crawl/` — REST client, battle log scraper; fetches and aggregates battle data
- `src/geneus/` — ML model and training loop (`BrawlModel`, `train` CLI)
- `src/pick/` — interactive draft-assist REPL (`pick` CLI); ranks map picks and scores 6th picks
  - `fuzzy.py` — subsequence fuzzy match with shortest-name tiebreak (see `docs/misc_notes.md`)
  - `score.py` — checkpoint loading, per-brawler z-score ranking, 6th-pick win probability
  - `display.py` — columnar terminal grid with class-colored brawler names
  - `main.py` — interactive loop: map → rankings → picks → 6th-pick candidates → repeat
- `tests/` — unit tests (no network); `tests/integration/` — API client tests using mocked HTTP
- `docs/dataset_generation.md` — usage guide for the dataset scraper

## Key domain notes (see `battlelog_api_notes.md`)

- `type: "ranked"` = trophy matchmaking; `type: "soloRanked"` = true ranked (trophies field holds rank number, not trophies)
- Battle logs use `teams[][]` for 3v3 modes and `players[]` for showdown
- The `result` field is always from the perspective of the battlelog owner
