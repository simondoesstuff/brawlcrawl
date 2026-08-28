from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@app.command()
def dataset_init(
    seed_tags: Annotated[
        list[str],
        typer.Argument(help="Player tags to seed the dataset (e.g. #YLQQY2U0P)"),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Path to write dataset JSON")
    ] = None,
    steps: Annotated[int, typer.Option(help="Number of BFS steps to run")] = 1,
    min_trophies: Annotated[
        int, typer.Option(help="Minimum rank/trophies to include a battle")
    ] = 16,
    frontier_cap: Annotated[int, typer.Option(help="Max tags queried per step")] = 200,
    request_interval: Annotated[
        float, typer.Option(help="Seconds between API requests")
    ] = 0.5,
    since: Annotated[
        datetime | None, typer.Option(help="Drop battles older than this datetime (ISO 8601)")
    ] = None,
) -> None:
    """Initialize a new dataset from seed player tags and run BFS steps."""
    from crawl.api import BrawlStarsClient
    from crawl.dataset import Dataset
    from crawl.scraper import Scraper

    save_path = output or Path("data/dataset.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with BrawlStarsClient() as client:
        db = Dataset.from_seed(set(seed_tags))
        scraper = Scraper(
            client,
            db,
            frontier_cap=frontier_cap,
            min_trophies=min_trophies,
            request_interval=request_interval,
            since=_as_utc(since),
        )
        scraper.run(steps=steps, save_path=save_path)


@app.command()
def dataset_extend(
    dataset: Annotated[Path, typer.Argument(help="Path to existing dataset JSON")],
    steps: Annotated[int, typer.Option(help="Number of BFS steps to run")] = 1,
    min_trophies: Annotated[
        int, typer.Option(help="Minimum rank/trophies to include a battle")
    ] = 16,
    frontier_cap: Annotated[int, typer.Option(help="Max tags queried per step")] = 200,
    request_interval: Annotated[
        float, typer.Option(help="Seconds between API requests")
    ] = 0.5,
    since: Annotated[
        datetime | None, typer.Option(help="Drop battles older than this datetime (ISO 8601)")
    ] = None,
) -> None:
    """Extend an existing dataset with more BFS steps."""
    from crawl.api import BrawlStarsClient
    from crawl.dataset import Dataset
    from crawl.scraper import Scraper

    with BrawlStarsClient() as client:
        db = Dataset.load(dataset)
        scraper = Scraper(
            client,
            db,
            frontier_cap=frontier_cap,
            min_trophies=min_trophies,
            request_interval=request_interval,
            since=_as_utc(since),
        )
        scraper.run(steps=steps, save_path=dataset)


@app.command()
def dataset_audit(
    dataset: Annotated[Path, typer.Argument(help="Path to dataset JSON")],
) -> None:
    """Print stats and metrics for an existing dataset."""
    import statistics

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    from crawl.dataset import Dataset

    console = Console()
    db = Dataset.load(dataset)

    # --- derived stats ---
    samples = [wl.total for wl in db.stats.values()]
    total_battles = sum(samples)
    n_comps = len(db.stats)

    unique_brawlers: set[int] = set()
    for comp in db.stats:
        unique_brawlers |= comp.team_a_ids | comp.team_b_ids

    # per-event aggregates
    from collections import defaultdict
    event_comps: dict[int, int] = defaultdict(int)
    event_battles: dict[int, int] = defaultdict(int)
    for comp, wl in db.stats.items():
        event_comps[comp.event_id] += 1
        event_battles[comp.event_id] += wl.total

    # sample-count buckets
    buckets = {"1": 0, "2–5": 0, "6–10": 0, "11–50": 0, "51+": 0}
    for s in samples:
        if s == 1:
            buckets["1"] += 1
        elif s <= 5:
            buckets["2–5"] += 1
        elif s <= 10:
            buckets["6–10"] += 1
        elif s <= 50:
            buckets["11–50"] += 1
        else:
            buckets["51+"] += 1

    # win-rate distribution (only comps with ≥2 observations)
    win_rates = [
        wl.a_wins / wl.total for wl in db.stats.values() if wl.total >= 2
    ]
    wr_balanced = sum(1 for w in win_rates if 0.4 <= w <= 0.6)
    wr_skewed   = sum(1 for w in win_rates if (0.2 <= w < 0.4) or (0.6 < w <= 0.8))
    wr_extreme  = sum(1 for w in win_rates if w < 0.2 or w > 0.8)

    # --- Overview panel ---
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold cyan")
    overview.add_column(justify="right")
    overview.add_row("Player tags seen",   f"{len(db.seen_tags):,}")
    overview.add_row("Frontier remaining", f"{len(db.frontier):,}")
    overview.add_row("Battles recorded",   f"{total_battles:,}")
    overview.add_row("Compositions",       f"{n_comps:,}")
    overview.add_row("Bad tags",           f"{len(db.bad_tags):,}")
    overview.add_row("Unique brawlers",    f"{len(unique_brawlers):,}")
    overview.add_row("Unique events",      f"{len(event_comps):,}")

    console.print(Panel(overview, title=f"[bold]Dataset: {dataset}[/bold]", box=box.ROUNDED))

    # --- Samples-per-composition table ---
    if samples:
        samp_table = Table(title="Samples per Composition", box=box.SIMPLE_HEAD, show_edge=False)
        samp_table.add_column("Bucket", style="cyan")
        samp_table.add_column("Count", justify="right")
        samp_table.add_column("Share", justify="right")
        for label, count in buckets.items():
            pct = count / n_comps * 100 if n_comps else 0
            samp_table.add_row(label, f"{count:,}", f"{pct:.1f}%")

        stat_table = Table.grid(padding=(0, 2))
        stat_table.add_column(style="bold cyan")
        stat_table.add_column(justify="right")
        stat_table.add_row("Min",    f"{min(samples):,}")
        stat_table.add_row("Median", f"{statistics.median(samples):,.1f}")
        stat_table.add_row("Mean",   f"{statistics.mean(samples):,.2f}")
        stat_table.add_row("Max",    f"{max(samples):,}")

        console.print(samp_table)
        console.print(Panel(stat_table, title="Sample Count Stats", box=box.ROUNDED))

    # --- Win-rate distribution ---
    if win_rates:
        wr_table = Table(title="Win-Rate Distribution (comps with ≥2 samples)", box=box.SIMPLE_HEAD, show_edge=False)
        wr_table.add_column("Category", style="cyan")
        wr_table.add_column("Range", justify="center")
        wr_table.add_column("Count", justify="right")
        wr_table.add_column("Share", justify="right")
        n_wr = len(win_rates)
        wr_table.add_row("Balanced", "40–60%", f"{wr_balanced:,}", f"{wr_balanced/n_wr*100:.1f}%")
        wr_table.add_row("Skewed",   "20–40% / 60–80%", f"{wr_skewed:,}", f"{wr_skewed/n_wr*100:.1f}%")
        wr_table.add_row("Extreme",  "<20% / >80%", f"{wr_extreme:,}", f"{wr_extreme/n_wr*100:.1f}%")
        console.print(wr_table)

    # --- Per-event breakdown ---
    if event_comps:
        ev_table = Table(title="Per-Event Breakdown", box=box.SIMPLE_HEAD, show_edge=False)
        ev_table.add_column("Event ID", style="cyan", justify="right")
        ev_table.add_column("Compositions", justify="right")
        ev_table.add_column("Battles", justify="right")
        ev_table.add_column("Avg Samples/Comp", justify="right")
        for eid in sorted(event_comps, key=lambda e: event_battles[e], reverse=True):
            avg = event_battles[eid] / event_comps[eid]
            ev_table.add_row(str(eid), f"{event_comps[eid]:,}", f"{event_battles[eid]:,}", f"{avg:.2f}")
        console.print(ev_table)


@app.command()
def events_crawl(
    dataset: Annotated[Path, typer.Argument(help="Path to dataset JSON")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Path to write events JSON")
    ] = None,
    request_interval: Annotated[
        float, typer.Option(help="Seconds between API requests")
    ] = 0.5,
    tag_cap: Annotated[
        int | None, typer.Option(help="Max number of player tags to query")
    ] = None,
) -> None:
    """Crawl event details (mode, modeId, map) for all event IDs in the dataset."""
    from crawl.api import BrawlStarsClient, Event
    from crawl.dataset import Dataset
    from crawl.events import crawl_events, load_events, save_events

    save_path = output or Path("data/events.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    db = Dataset.load(dataset)
    all_event_ids = {comp.event_id for comp in db.stats} - {0}

    events: dict[int, Event] = load_events(save_path) if save_path.exists() else {}

    needed = all_event_ids - set(events)
    if not needed:
        print(f"All {len(all_event_ids)} event IDs already known. Nothing to do.")
        return

    print(f"Discovering {len(needed)} event IDs ({len(events)} already known).")
    tags = sorted(db.seen_tags - db.bad_tags)

    with BrawlStarsClient() as client:
        for event in crawl_events(client, tags, needed, request_interval=request_interval, tag_cap=tag_cap):
            events[event.id] = event
            save_events(events, save_path)

    missing = needed - set(events)
    print(f"Found {len(needed) - len(missing)}/{len(needed)} event IDs. Saved to {save_path}.")
    if missing:
        print(f"Could not find {len(missing)} event IDs: {sorted(missing)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
