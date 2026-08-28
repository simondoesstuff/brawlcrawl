import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, cast

from tqdm import tqdm

from crawl.api import BrawlStarsClient, Event, TagInaccessibleError


class _EventJsonRequired(TypedDict):
    id: int
    mode: str
    map: str


class _EventJson(_EventJsonRequired, total=False):
    modeId: int


def crawl_events(
    client: BrawlStarsClient,
    tags: list[str],
    needed_ids: set[int],
    request_interval: float = 0.0,
    tag_cap: int | None = None,
) -> Iterator[Event]:
    """Yield one Event per discovered ID, scanning tags until all needed_ids are found."""
    remaining = set(needed_ids)
    capped = tags[:tag_cap] if tag_cap is not None else tags
    with tqdm(capped, desc="scanning", unit="tag") as bar:
        for tag in bar:
            if not remaining:
                break
            bar.set_postfix(remaining=len(remaining))  # pyright: ignore[reportUnknownMemberType]
            if request_interval > 0:
                time.sleep(request_interval)
            try:
                battles = client.get_battlelog(tag)
            except TagInaccessibleError:
                continue
            for battle in battles:
                eid = battle.event.id
                if eid in remaining:
                    remaining.discard(eid)
                    yield battle.event


def load_events(path: Path) -> dict[int, Event]:
    items = cast(list[_EventJson], json.loads(path.read_text()))
    return {
        item["id"]: Event(
            id=item["id"],
            mode=item["mode"],
            map=item["map"],
            mode_id=item.get("modeId", 0),
        )
        for item in items
    }


def save_events(events: dict[int, Event], path: Path) -> None:
    data = [
        {"id": e.id, "mode": e.mode, "modeId": e.mode_id, "map": e.map}
        for e in sorted(events.values(), key=lambda e: e.id)
    ]
    _ = path.write_text(json.dumps(data, indent=2))
