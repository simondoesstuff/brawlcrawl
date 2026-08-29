# Brawl Crawl

Brawl Crawl predicts match-ups with a deep learning model trained on all battles fought by real players within the month.

With no public dataset previously available, brawl crawl queries individual players for their battlelogs, then all the players in those battlelogs... forming recursive chains that spread out until all active players have been discovered.

[data/crawl_leg1_20260827.json](data/crawl_leg1_20260827.json) ~350k battles among ~70k legendary 1+ players from July to August, 2026.

The model, "[Geneus](src/geneus)", understands counters, synergies, and map interaction. Geneus can predict optimal 6th picks in arbitrary match-ups.

## CLI

**`pick`** — interactive draft assistant

- Loads a trained model checkpoint and presents a REPL for map selection and 6th-pick scoring

**`crawl`** — dataset scraping and management

- `dataset-init <tags...>` — seed a new dataset from player tags via BFS
- `dataset-extend <dataset>` — continue BFS on an existing dataset
- `dataset-audit <dataset>` — print stats and distributions for a dataset
- `events-crawl <dataset>` — fetch mode/map metadata for all event IDs in a dataset
