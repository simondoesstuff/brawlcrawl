# Overview

- Goal: build an ML model to predict Brawl Stars match outcomes from brawler picks
- Input signals: event (map/mode), brawler composition per team, player trophy level as a skill proxy
- Output: win probability for a given team composition

## Layout

- `workflow/` — Snakemake pipeline; fetches raw JSON from the Brawl Stars API (`$BSTOK`) into `data/`, plus `export_onnx` (`just export-onnx`) which traces the trained checkpoints to ONNX for `web/`, and `tier_lists` (`uv run tier-list`) which Monte Carlo-samples per-map first-pick tier lists via the terminal `BrawlModel` into `data/tier_lists.json`
- `src/crawl/` — REST client, battle log scraper; fetches and aggregates battle data
- `src/geneus/` — ML model and training loop (`BrawlModel`, `train` CLI)
- `src/pick/` — interactive draft-assist REPL (`pick` CLI)
  - `fuzzy.py` — subsequence fuzzy match with shortest-name tiebreak (see `docs/misc_notes.md`)
  - `score.py` — checkpoint loading; `get_q_values` builds draft observations (non-pool brawlers marked `LOCALLY_BANNED`) and calls `DraftQNetwork`; `get_terminal_pick6_scores` vmaps BrawlModel over pick-6 candidates and returns `P(team B wins)`
  - `display.py` — columnar terminal grid with rarity-colored brawler names
  - `main.py` — full draft loop: coin flip → 3 ally bans → 3 enemy bans → picks 1–5 (live Q-value grid) → pick-6 win-probability grid (in-place, terminal model); Q-values are z-scored over the post-filter available brawlers; after pick 5 any Enter exits; `$BRAWL_FILTER` (comma-separated names) hides and `LOCALLY_BANNED`-marks non-owned brawlers; filter is toggled with `/` (works in all phases including pick-6); filter defaults per phase: OFF during bans, ON during ally picks, OFF during enemy picks
  - `tier_list.py` — `tier-list` CLI (wrapped by the Snakemake `tier_lists` rule): per map, Monte Carlo-samples random full drafts (random permutation of all brawlers; first two fill the first-picking team's other two seats, next three fill the opposing team, everyone else is scored as that team's first pick against the same fixed continuation via the terminal `BrawlModel`) and keeps sampling until the win-rate ranking's Spearman correlation against the previous checkpoint clears a threshold for several consecutive checks (or a trial cap is hit); answers "if I pick this brawler first, what's my win rate against a fully random rest-of-draft" — unbiased by the real pick-order skew baked into `winrates_*.json`
  - `export_onnx.py` — `export-onnx` CLI (wrapped by the Snakemake `export_onnx` rule; run via `just export-onnx`): traces `DraftQNetwork` and `BrawlModel`'s pick-6 forward pass to static-shape ONNX graphs via `jax2onnx`, validates each against the JAX model (value + top-15 rank match), and writes `char_encs_all` (frozen per-event character encodings — internal to `BrawlModel`, not reachable via its public `__call__`) plus a `manifest.json` (shapes, event→row index, draft-state constants) into `web/static/`; both graphs are single-example (no batch dim) — the pick-6 candidate loop happens client-side, one inference call per candidate, matching the CLI's `vmap`. Also writes `web/tests/fixtures/onnx_fixture.json` (JAX-vs-ONNX numeric parity cases) and `web/tests/fixtures/logic_fixture.json` (`export_logic_fixture`: real-Python-generated cases for the pure draft-logic helpers — `_build_obs`, `_z_score_q`, `_pick6_team_split`, `_ban_excluded`, `_phase_default_filter`, `pick_annotations`, `overview_annotations` — so the TS ports in `web/src/lib/pick/` are checked against the real implementation, not just eyeballed)
- `web/` — SvelteKit site (web port of the `pick` CLI), keybind- and mobile-first; consumes the ONNX graphs and data exported by `export-onnx` from `web/static/models/` and `web/static/data/`
  - `src/lib/onnx/` — thin `onnxruntime-web` wrapper (`session.ts`) plus types for `manifest.json`/`metadata.json` (`manifest.ts`, `metadata.ts`)
  - `src/lib/pick/` — TS ports of the CLI's pure logic (`fuzzy.ts`, `obs.ts`, `phase.ts`, `annotations.ts`, `constants.ts`), each checked in `*.test.ts` against `logic_fixture.json`; `engine.ts` orchestrates the ONNX sessions into `scoreMap`/`getQValues`/`getTerminalPick6Scores` (mirrors `score.py`); `draftState.svelte.ts` is the Svelte 5 runes state machine mirroring `_run_draft`'s phase progression (map select → coin flip → bans → picks → pick-6), plus a web-only undo stack
  - `src/lib/components/` — `MapSelect`, `CoinFlip`, `Draft` (search input + live-scored `BrawlerGrid` + `DraftBoard`), `FilterPanel` (owned-brawlers filter, persisted to `localStorage`), `Legend`
  - Filter (`$BRAWL_FILTER` in the CLI) is a model input during bans/picks (`local_pool` → `LOCALLY_BANNED`, feeding back into the Q-network) but a display-only filter of already-computed candidates during pick-6 — `draftState.svelte.ts` routes it accordingly and resets `filterOn` to the phase default on every phase transition, matching `_phase_default_filter`
  - Keybinds: `/` focuses the search box, `Enter`/`Escape`/arrow keys drive it and its suggestion list, `Backspace` on an empty box or `u` anywhere (not while typing) undoes the last ban/pick; on touch devices the box does not autofocus (tapping a grid card submits directly instead) while desktop autofocuses it — the search box itself stays available in every ban/pick phase (not just what the grid displays) since submittable brawlers and displayed brawlers diverge during the ban phases (`_ban_excluded` vs `_display_excluded`)
- `tests/` — unit tests (no network); `tests/integration/` — API client tests using mocked HTTP
- `docs/dataset_generation.md` — usage guide for the dataset scraper

## Key domain notes (see `battlelog_api_notes.md`)

- `type: "ranked"` = trophy matchmaking; `type: "soloRanked"` = true ranked (trophies field holds rank number, not trophies)
- Battle logs use `teams[][]` for 3v3 modes and `players[]` for showdown
- The `result` field is always from the perspective of the battlelog owner
