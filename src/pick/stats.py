"""Per-(character, event) win-rate / pick-rate z-scores from raw crawl stats.

Pure aggregation over an already-crawled `data/crawl_{leg}.json` dataset --
no network calls, no model inference. Cheap enough to regenerate on every
build rather than commit the derived output.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

typer_app = typer.Typer(add_completion=False, help="Win-rate / pick-rate z-scores from crawl stats.")


def _zscore_by_event(rates: dict[tuple[int, int], float]) -> list[dict[str, object]]:
    event_rates: dict[int, list[float]] = defaultdict(list)
    for (_char_id, event_id), rate in rates.items():
        event_rates[event_id].append(rate)

    event_stats: dict[int, tuple[float, float]] = {}
    for event_id, values in event_rates.items():
        mu = sum(values) / len(values)
        sigma = (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5
        event_stats[event_id] = (mu, sigma)

    results = []
    for (char_id, event_id), rate in sorted(rates.items()):
        mu, sigma = event_stats[event_id]
        z = (rate - mu) / sigma if sigma > 0 else 0.0
        results.append({"char_id": char_id, "event_id": event_id, "z_score": z})
    return results


@typer_app.command()
def winrates(
    input: Annotated[Path, typer.Option(help="Path to data/crawl_{leg}.json")],
    output: Annotated[Path, typer.Option(help="Output winrates JSON path")],
) -> None:
    """Z-scored win rate per (char, event), relative to other chars on the same map."""
    raw = json.loads(input.read_text())
    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

    for entry in raw["stats"]:
        event_id = entry["event_id"]
        a_wins = entry["a_wins"]
        total = entry["total"]
        for char_id in entry["team_a"]:
            counts[(char_id, event_id)][0] += a_wins
            counts[(char_id, event_id)][1] += total
        for char_id in entry["team_b"]:
            counts[(char_id, event_id)][0] += total - a_wins
            counts[(char_id, event_id)][1] += total

    wr = {k: v[0] / v[1] for k, v in counts.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_zscore_by_event(wr), indent=2))


@typer_app.command()
def pickrates(
    input: Annotated[Path, typer.Option(help="Path to data/crawl_{leg}.json")],
    output: Annotated[Path, typer.Option(help="Output pickrates JSON path")],
) -> None:
    """Z-scored pick rate per (char, event), relative to other chars on the same map."""
    raw = json.loads(input.read_text())
    appearances: dict[tuple[int, int], int] = defaultdict(int)
    event_totals: dict[int, int] = defaultdict(int)

    for entry in raw["stats"]:
        event_id = entry["event_id"]
        total = entry["total"]
        event_totals[event_id] += total
        for char_id in entry["team_a"] + entry["team_b"]:
            appearances[(char_id, event_id)] += total

    pick_rates = {
        (char_id, event_id): count / (event_totals[event_id] * 6)
        for (char_id, event_id), count in appearances.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_zscore_by_event(pick_rates), indent=2))


def entrypoint() -> None:
    typer_app()
