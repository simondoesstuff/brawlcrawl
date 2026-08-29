"""Columnar terminal display with rarity-colored brawler names."""

from collections.abc import Callable

from rich.console import Console
from rich.table import Table
from rich.text import Text

from pick.score import BrawlerInfo

RARITY_COLORS: dict[str, str] = {
    "Starting Brawler": "white",
    "Rare": "bright_green",
    "Super Rare": "deep_sky_blue1",
    "Epic": "purple",
    "Mythic": "bright_red",
    "Legendary": "yellow",
    "Ultra Legendary": "bright_yellow",
}

# prompt_toolkit style strings, parallel to RARITY_COLORS
RARITY_PT_STYLES: dict[str, str] = {
    "Starting Brawler": "fg:ansiwhite bold",
    "Rare": "fg:ansibrightgreen bold",
    "Super Rare": "fg:#00afff bold",
    "Epic": "fg:ansipurple bold",
    "Mythic": "fg:ansibrightred bold",
    "Legendary": "fg:ansiyellow bold",
    "Ultra Legendary": "fg:ansibrightyellow bold",
}


_ANNOTATION_LEGEND: list[tuple[str, str]] = [
    ("🤫", "secret pick"),
    ("🧠", "genius AI pick"),
    ("✗ ", "popular, but bad"),
]


def render_rarity_legend(console: Console) -> None:
    """Print a one-line key mapping rarity tiers to their display colors."""
    t = Text("  ")
    for i, (rarity, color) in enumerate(RARITY_COLORS.items()):
        if i:
            t.append("   ")
        t.append("■ ", style=f"bold {color}")
        t.append(rarity, style=f"bold {color}")
    console.print(t)


def render_annotation_legend(console: Console) -> None:
    """Print a one-line key explaining annotation symbols."""
    t = Text("  ")
    for i, (symbol, label) in enumerate(_ANNOTATION_LEGEND):
        if i:
            t.append("   ")
        t.append(symbol, style="bold")
        t.append(f" {label}", style="dim")
    console.print(t)
    console.print()


_MAX_NAME_LEN = 14  # "LARRY & LAWRIE"
_RANK_WIDTH = 3  # up to 999 rows; in practice ≤ 106
_SCORE_WIDTH = 7  # e.g. " +2.34" or " 56.3%"
_COL_WIDTH = (
    _RANK_WIDTH + 2 + _MAX_NAME_LEN + _SCORE_WIDTH
)  # "  1. NAME          +2.34"
_COL_GAP = 2


def _cell(
    rank: int,
    brawler: BrawlerInfo,
    score_str: str,
    annotation: str = "",
    annotation_slot: int = 0,
) -> Text:
    t = Text()
    t.append(f"{rank:>{_RANK_WIDTH}}. ", style="dim")
    color = RARITY_COLORS.get(brawler.rarity, "white")
    t.append(f"{brawler.name:<{_MAX_NAME_LEN}}", style=f"bold {color}")
    if annotation_slot > 0:
        # Annotation sits between name and score; pad with spaces when absent.
        t.append(annotation if annotation else " " * annotation_slot)
        t.append(f"{score_str:>{_SCORE_WIDTH}}")
    else:
        t.append(f" {score_str:>{_SCORE_WIDTH - 1}}")
    return t


def render_brawler_table(
    console: Console,
    entries: list[tuple[BrawlerInfo, float]],
    score_fmt: Callable[[float], str],
    annotations: dict[int, str] | None = None,
    divider: float | None = None,
    annotation_slot: int = 0,
) -> None:
    """Render a ranked list of brawlers in a column-major grid.

    entries: (BrawlerInfo, score) sorted best-first.
    score_fmt: formats the float score into a display string.
    annotations: optional per-brawler-id annotation suffix (e.g. " 🧠").
    divider: if set, split into two grids at this score threshold with a rule between them.
    annotation_slot: terminal columns reserved after each cell for annotation.
    """
    if not entries:
        return

    ann = annotations or {}
    col_w = _COL_WIDTH + annotation_slot
    n_cols = max(1, (console.width + _COL_GAP) // (col_w + _COL_GAP))

    if divider is not None:
        above = [(b, s) for b, s in entries if s >= divider]
        below = [(b, s) for b, s in entries if s < divider]
    else:
        above = entries
        below = []

    def _render_grid(
        section: list[tuple[BrawlerInfo, float]], rank_offset: int
    ) -> None:
        n = len(section)
        if not n:
            return
        n_rows = (n + n_cols - 1) // n_cols

        table = Table(show_header=False, show_edge=False, padding=(0, 1), box=None)
        for _ in range(n_cols):
            table.add_column(no_wrap=True, min_width=col_w)

        for row in range(n_rows):
            cells: list[Text | str] = []
            for col in range(n_cols):
                idx = row + col * n_rows
                if idx < n:
                    brawler, score = section[idx]
                    cells.append(
                        _cell(
                            rank_offset + idx + 1,
                            brawler,
                            score_fmt(score),
                            ann.get(brawler.id, ""),
                            annotation_slot,
                        )
                    )
                else:
                    cells.append("")
            table.add_row(*cells)

        console.print(table)

    _render_grid(above, 0)
    if below:
        if above:
            console.rule(style="dim")
        _render_grid(below, len(above))
