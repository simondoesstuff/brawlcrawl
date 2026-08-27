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
    from brawl.api import BrawlStarsClient
    from brawl.dataset import Dataset
    from brawl.scraper import Scraper

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
    from brawl.api import BrawlStarsClient
    from brawl.dataset import Dataset
    from brawl.scraper import Scraper

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
