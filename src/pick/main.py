"""Interactive full-draft REPL with DraftQNetwork recommendations."""

import os
import shutil
from math import ceil
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from prompt_toolkit import Application, PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion, DynamicCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from rich.console import Console

from pick.constants import (
    ANN_BRAIN, ANN_SECRET, ANN_X, ANNOTATION_SLOT,
    PICKRATE_HIGH_Z, PICKRATE_LOW_Z, Q_THRESHOLD,
)
from pick.display import (
    RARITY_COLORS,
    RARITY_PT_STYLES,
    _COL_GAP,
    _COL_WIDTH,
    _MAX_NAME_LEN,
    _RANK_WIDTH,
    _SCORE_WIDTH,
    render_annotation_legend,
    render_brawler_table,
    render_rarity_legend,
)
from pick.fuzzy import _normalize, _subseq_match, fuzzy_find
from pick.score import (
    BrawlerInfo,
    DraftContext,
    EventInfo,
    get_q_values,
    get_terminal_pick6_scores,
    load_context,
    score_map,
)

typer_app = typer.Typer(add_completion=False, help="Interactive draft-assist REPL.")
console = Console()

_DARK_STYLE = Style.from_dict({
    "completion-menu.completion":               "bg:#111111 fg:ansiwhite",
    "completion-menu.completion.current":       "bg:#3a3a3a fg:ansiwhite bold",
    "completion-menu.meta.completion":          "bg:#111111 fg:#666666",
    "completion-menu.meta.completion.current":  "bg:#3a3a3a fg:#aaaaaa",
    "completion-menu.border":                   "bg:#111111 fg:#444444",
    "scrollbar.background":                     "bg:#111111",
    "scrollbar.button":                         "bg:#444444",
})

# Background colours for brawler states in the draft board
_BG_BAN = "bg:#1a1a1a"
_BG_ALLY_PICK = "bg:#002a00"
_BG_ENEMY_PICK = "bg:#2a0000"


# ── helpers ───────────────────────────────────────────────────────────────────

def _z_score_q(q_vals: np.ndarray, available_char_idxs: list[int]) -> np.ndarray:
    """Normalize Q-values to z-scores over currently available brawlers.

    Z > 0 means above-average advantage relative to the current choice pool.
    """
    if not available_char_idxs:
        return q_vals
    vals = q_vals[available_char_idxs]
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-8:
        return np.zeros_like(q_vals)
    return (q_vals - mean) / std


def _ban_excluded(
    phase: int,
    ally_bans: list[BrawlerInfo],
    enemy_bans: list[BrawlerInfo],
    picks: list[tuple[bool, BrawlerInfo]],
) -> set[int]:
    """Brawlers excluded from submission at the given phase.

    During each team's ban phase only their own bans are excluded — the other
    team's bans remain submittable (inter-team duplicate ban is allowed).
    During the pick phase all bans and prior picks are excluded.
    """
    excl = {b.id for _, b in picks}
    if phase < 3:
        excl |= {b.id for b in ally_bans}
    elif phase < 6:
        excl |= {b.id for b in enemy_bans}
    else:
        excl |= {b.id for b in ally_bans} | {b.id for b in enemy_bans}
    return excl


def _parse_brawl_filter(ctx: DraftContext) -> set[int]:
    """Parse $BRAWL_FILTER (comma-separated names) into a set of brawler ids.

    Uses fuzzy (subsequence) matching so partial names work.  Silently skips
    names with no match — callers that want user-visible feedback should call
    _brawl_filter_names() first and print the results.
    """
    raw = os.environ.get("BRAWL_FILTER", "").strip()
    if not raw:
        return set()
    names = [n.strip() for n in raw.replace(",", "\n").splitlines() if n.strip()]
    ids: set[int] = set()
    for name in names:
        matched = fuzzy_find(name, ctx.brawler_names)
        if matched is not None:
            ids.add(ctx._brawler_by_name[matched].id)
    return ids


def _brawl_filter_names(ctx: DraftContext) -> tuple[list[str], list[str]]:
    """Return (matched_brawler_names, unmatched_raw_names) from $BRAWL_FILTER."""
    raw = os.environ.get("BRAWL_FILTER", "").strip()
    if not raw:
        return [], []
    names = [n.strip() for n in raw.replace(",", "\n").splitlines() if n.strip()]
    matched: list[str] = []
    unmatched: list[str] = []
    for name in names:
        m = fuzzy_find(name, ctx.brawler_names)
        if m is not None:
            if m not in matched:
                matched.append(m)
        else:
            unmatched.append(name)
    return matched, unmatched


# ── fuzzy completer ───────────────────────────────────────────────────────────

class _FuzzyCompleter(Completer):
    def __init__(self, candidates: list[str], meta: dict[str, str], style: dict[str, str]) -> None:
        self._candidates = candidates
        self._meta = meta
        self._style = style

    def get_completions(self, document, _complete_event):
        text = document.text_before_cursor
        if not text.strip():
            return
        norm = _normalize(text)
        if not norm:
            return
        matches = [c for c in self._candidates if _subseq_match(norm, _normalize(c))]
        matches.sort(key=lambda c: (len(_normalize(c)), c))
        for c in matches:
            yield Completion(
                c,
                start_position=-len(text),
                display=c,
                display_meta=self._meta.get(c, ""),
                style=self._style.get(c, ""),
            )


def _brawler_completer(ctx: DraftContext, excluded: set[int]) -> _FuzzyCompleter:
    available = [b for b in ctx.brawlers if b.id not in excluded]
    return _FuzzyCompleter(
        candidates=[b.name for b in available],
        meta={b.name: b.brawler_class for b in available},
        style={b.name: RARITY_PT_STYLES.get(b.rarity, "bold") for b in available},
    )


def _map_completer(ctx: DraftContext) -> _FuzzyCompleter:
    mode_map = {e.map_name: e.mode for e in ctx.events}
    return _FuzzyCompleter(
        candidates=ctx.map_names,
        meta=mode_map,
        style={},
    )


# ── rprompt helpers ───────────────────────────────────────────────────────────

def _rprompt_map(session: PromptSession, ctx: DraftContext):
    def _rp() -> FormattedText:
        text = session.default_buffer.text
        if not text.strip():
            return FormattedText([])
        matched = fuzzy_find(text, ctx.map_names)
        if matched is None:
            return FormattedText([("fg:ansired", "  ✗")])
        event = next(e for e in ctx.events if e.map_name == matched)
        return FormattedText([
            ("", "  → "), ("bold", matched), ("fg:ansibrightblack", f"  {event.mode}"),
        ])
    return _rp


# ── annotation helpers ────────────────────────────────────────────────────────

def pick_annotations(
    ctx: DraftContext,
    event: EventInfo,
    q_scores: list[tuple[BrawlerInfo, float]],
) -> dict[int, str]:
    """Annotate available brawlers based on Q-value and pickrate.

    Brain: Q > 0 and below-average pickrate (hidden gem for this state).
    X:     Q ≤ 0 and high pickrate (overrated by players).
    """
    result: dict[int, str] = {}
    for brawler, q_val in q_scores:
        pz = ctx.pickrates.get(brawler.id, {}).get(event.id)
        if pz is None:
            continue
        if q_val > Q_THRESHOLD and pz < PICKRATE_LOW_Z:
            result[brawler.id] = ANN_BRAIN
        elif q_val <= Q_THRESHOLD and pz >= PICKRATE_HIGH_Z:
            result[brawler.id] = ANN_X
    return result


def overview_annotations(
    ctx: DraftContext,
    event: EventInfo,
    map_scores: list[tuple[BrawlerInfo, float]],
) -> dict[int, str]:
    """Annotate brawlers for the winrate overview.

    Secret: above-average winrate z-score and below-average pickrate.
    X:      below-average winrate z-score and high pickrate.
    """
    result: dict[int, str] = {}
    for brawler, win_score in map_scores:
        pz = ctx.pickrates.get(brawler.id, {}).get(event.id)
        if pz is None:
            continue
        if win_score >= 0 and pz < PICKRATE_LOW_Z:
            result[brawler.id] = ANN_SECRET
        elif win_score < 0 and pz >= PICKRATE_HIGH_Z:
            result[brawler.id] = ANN_X
    return result


# ── pick order helpers ────────────────────────────────────────────────────────

def _pick_is_ally(pick_idx: int, ally_first: bool) -> bool:
    """True if pick_idx (0-5) belongs to the ally team."""
    from geneus.draft.env import TURN_SCHEDULE
    _, team, _ = TURN_SCHEDULE[6 + pick_idx]
    return (team == "A") == ally_first


def _phase_default_filter(phase: int, ally_first: bool) -> bool:
    """Default filter-on state for the given draft phase.

    Ban phases (0-5): OFF.  Pick phases: ON when it's the ally's turn.
    Phase 11 (terminal/6th pick) follows pick_idx 5 — team B.
    """
    if phase < 6:
        return False
    return _pick_is_ally(phase - 6, ally_first)


def _phase_label(phase: int, ally_first: bool) -> str:
    if phase < 3:
        return f"Ally ban {phase + 1}"
    if phase < 6:
        return f"Enemy ban {phase - 3 + 1}"
    pick_idx = phase - 6
    label = "Ally" if _pick_is_ally(pick_idx, ally_first) else "Enemy"
    return f"Pick {pick_idx + 1}  ({label})"


# ── draft Application ─────────────────────────────────────────────────────────

def _run_draft(
    ctx: DraftContext,
    event: EventInfo,
    map_scores: list[tuple[BrawlerInfo, float]],
    ann: dict[int, str],
    ally_first: bool,
    filter_ids: set[int],
) -> tuple[list[BrawlerInfo], list[BrawlerInfo], list[tuple[bool, BrawlerInfo]]] | None:
    """Run the full draft; shows pick-6 recommendations in-place before exit."""
    phase = [0]
    ally_bans: list[BrawlerInfo] = []
    enemy_bans: list[BrawlerInfo] = []
    picks: list[tuple[bool, BrawlerInfo]] = []
    cancelled = [False]
    error = [""]
    q_cache: list[np.ndarray | None] = [None]
    terminal_cache: list[list[tuple[BrawlerInfo, float]]] = [[]]
    final_mode = [False]

    filter_on = [False]  # phase 0 = ally ban: default OFF

    def _local_pool() -> set[int] | None:
        return filter_ids if (filter_on[0] and filter_ids) else None

    def _refresh_q_cache() -> None:
        q_cache[0] = get_q_values(
            ctx, event, ally_bans, enemy_bans, picks, phase[0], ally_first,
            local_pool=_local_pool(),
        )

    _refresh_q_cache()

    # Phase-aware exclusion for submission/completions (inter-team ban sharing)
    def _submit_excluded() -> set[int]:
        return _ban_excluded(phase[0], ally_bans, enemy_bans, picks)

    # Display exclusion: remove all banned/picked from the ranked grid
    def _display_excluded() -> set[int]:
        return {b.id for b in ally_bans} | {b.id for b in enemy_bans} | {b.id for _, b in picks}

    buf = Buffer(
        name="draft",
        multiline=False,
        completer=DynamicCompleter(lambda: _brawler_completer(ctx, _submit_excluded())),
        complete_while_typing=True,
    )

    # ── grid content ────────────────────────────────────────────────────────
    def _grid_content() -> FormattedText:
        try:
            term_w = shutil.get_terminal_size().columns
        except Exception:
            term_w = 80

        current_text = buf.text
        submit_excl = _submit_excluded()
        display_excl = _display_excluded()

        best_match_id: int | None = None
        if current_text.strip() and not final_mode[0]:
            avail = [b.name for b in ctx.brawlers if b.id not in submit_excl]
            m = fuzzy_find(current_text, avail)
            if m:
                best_match_id = ctx._brawler_by_name[m].id

        result: list[tuple[str, str]] = []

        # Header
        p_label = _phase_label(phase[0], ally_first)
        if final_mode[0]:
            p_label += "  (/ filter  ↵ finish)"
        flip_label = "Ally 1st" if ally_first else "Enemy 1st"
        filter_label = "  [filter]" if (filter_ids and filter_on[0]) else ""
        result += [
            ("bold", f"\n  {event.map_name}"),
            ("fg:ansibrightblack", f"  {event.mode}"),
            ("fg:ansibrightblack", f"  [{flip_label}]"),
            ("bold", f"   {p_label}"),
            ("fg:#555555", filter_label),
            ("bold", "\n\n"),
        ]

        # Draft board: bans
        result.append(("fg:ansibrightblack", "  Bans  "))
        ally_ban_cells = [
            (f"{RARITY_PT_STYLES.get(b.rarity, '')} {_BG_BAN}", f" {b.name} ")
            for b in ally_bans
        ]
        enemy_ban_cells = [
            (f"{RARITY_PT_STYLES.get(b.rarity, '')} {_BG_BAN}", f" {b.name} ")
            for b in enemy_bans
        ]
        result.append(("fg:#00afff bold", "Ally "))
        for style, txt in ally_ban_cells:
            result += [(style, txt), ("", " ")]
        for _ in range(3 - len(ally_bans)):
            result.append(("fg:ansibrightblack", " □ "))
        result.append(("fg:ansibrightblack", "  "))
        result.append(("fg:ansired bold", "Enemy "))
        for style, txt in enemy_ban_cells:
            result += [(style, txt), ("", " ")]
        for _ in range(3 - len(enemy_bans)):
            result.append(("fg:ansibrightblack", " □ "))
        result.append(("", "\n"))

        # Draft board: picks
        if picks or phase[0] >= 6:
            result.append(("fg:ansibrightblack", "  Picks "))
            for slot_idx in range(6):
                is_ally_slot = _pick_is_ally(slot_idx, ally_first)
                if slot_idx < len(picks):
                    _, b = picks[slot_idx]
                    bg = _BG_ALLY_PICK if is_ally_slot else _BG_ENEMY_PICK
                    rar_s = RARITY_PT_STYLES.get(b.rarity, "")
                    result.append((f"{rar_s} {bg}", f" {b.name} "))
                elif slot_idx == phase[0] - 6 and not final_mode[0]:
                    team_c = "fg:#00afff bold" if is_ally_slot else "fg:ansired bold"
                    result.append((team_c, " ? "))
                else:
                    result.append(("fg:ansibrightblack", " □ "))
                result.append(("", " "))
            result.append(("", "\n"))

        result.append(("", "\n"))

        # Brawler grid
        col_w = _COL_WIDTH + ANNOTATION_SLOT
        n_cols = max(1, (term_w + _COL_GAP) // (col_w + _COL_GAP))

        if final_mode[0]:
            # Pick-6: terminal model win probabilities, filtered if filter is on
            scored = terminal_cache[0]
            if filter_on[0] and filter_ids:
                scored = [(b, s) for b, s in scored if b.id in filter_ids]
            score_str_fn = lambda s: f"{s:.1%}"
            split_threshold = 0.5
            p_ann: dict[int, str] = {}
        elif q_cache[0] is not None:
            if filter_on[0] and filter_ids:
                q_norm_idxs = [b.char_idx for b in ctx.brawlers
                               if b.id not in display_excl and b.id in filter_ids]
            else:
                q_norm_idxs = [b.char_idx for b in ctx.brawlers if b.id not in display_excl]
            q_z = _z_score_q(q_cache[0], q_norm_idxs)
            scored = [
                (ctx._brawler_by_char_idx[b.char_idx], float(q_z[b.char_idx]))
                for b in ctx.brawlers
                if b.char_idx in ctx._brawler_by_char_idx and b.id not in display_excl
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            if filter_on[0] and filter_ids:
                scored = [(b, s) for b, s in scored if b.id in filter_ids]
            score_str_fn = lambda s: f"{s:+.2f}"
            split_threshold = Q_THRESHOLD
            p_ann = pick_annotations(ctx, event, scored) if phase[0] >= 6 else ann
        else:
            scored = [(b, s) for b, s in map_scores if b.id not in display_excl]
            if filter_on[0] and filter_ids:
                scored = [(b, s) for b, s in scored if b.id in filter_ids]
            score_str_fn = lambda s: f"{s:+.2f}"
            split_threshold = Q_THRESHOLD
            p_ann = ann

        above = [(b, s) for b, s in scored if s >= split_threshold]
        below = [(b, s) for b, s in scored if s < split_threshold]

        def _render_section(section: list[tuple[BrawlerInfo, float]], rank_offset: int) -> None:
            n = len(section)
            if not n:
                return
            n_rows = ceil(n / n_cols)
            for row in range(n_rows):
                for col in range(n_cols):
                    idx = row + col * n_rows
                    if idx >= n:
                        pad = col_w + (2 if col < n_cols - 1 else 0)
                        result.append(("", " " * pad))
                        continue

                    brawler, score = section[idx]
                    score_str = score_str_fn(score)
                    rar_style = RARITY_PT_STYLES.get(brawler.rarity, "")
                    rank_sep = "."

                    if brawler.id == best_match_id:
                        rank_sep = "▸"
                        rank_s = "fg:ansibrightblack bold bg:#1e1e00"
                        name_s = f"{rar_style} bold bg:#1e1e00"
                        ann_s  = "fg:ansibrightblack bold bg:#1e1e00"
                        score_s = "bold bg:#1e1e00"
                    else:
                        rank_s = "fg:ansibrightblack"
                        name_s = rar_style
                        ann_s  = "fg:ansibrightblack"
                        score_s = ""

                    result.append((rank_s, f"{rank_offset + idx + 1:>{_RANK_WIDTH}}{rank_sep} "))
                    result.append((name_s, f"{brawler.name:<{_MAX_NAME_LEN}}"))
                    result.append((ann_s, p_ann.get(brawler.id, " " * ANNOTATION_SLOT)))
                    result.append((score_s, f"{score_str:>{_SCORE_WIDTH}}"))
                    if col < n_cols - 1:
                        result.append(("", "  "))

                result.append(("", "\n"))

        _render_section(above, 0)
        if above and below:
            result.append(("fg:ansiwhite", "─" * term_w + "\n"))
        _render_section(below, len(above))

        if error[0]:
            result += [("", "\n"), ("fg:ansired", f"  {error[0]}\n")]

        result.append(("", "\n"))
        return FormattedText(result)

    # ── input prompt prefix ──────────────────────────────────────────────────
    def _line_prefix(_line_num, _wrap_count) -> FormattedText:
        if final_mode[0]:
            return FormattedText([("fg:ansibrightblack", "  (/ filter  ↵ finish)  ")])
        label = _phase_label(phase[0], ally_first)
        return FormattedText([("bold", f"  {label}> ")])

    # ── key bindings ─────────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(ev):
        text = buf.text.strip()

        # "/" toggles filter in any mode; no exit
        if text == "/":
            if filter_ids:
                filter_on[0] = not filter_on[0]
                if not final_mode[0]:
                    _refresh_q_cache()
            buf.reset()
            return

        # In final mode any other Enter exits
        if final_mode[0]:
            ev.app.exit()
            return

        if not text:
            return

        p = phase[0]
        avail = [b.name for b in ctx.brawlers if b.id not in _submit_excluded()]
        matched = fuzzy_find(text, avail)
        if matched is None:
            error[0] = f"No brawler matching '{text}'"
            buf.reset()
            return

        error[0] = ""
        brawler = ctx._brawler_by_name[matched]

        if p < 3:
            ally_bans.append(brawler)
        elif p < 6:
            enemy_bans.append(brawler)
        else:
            pick_idx = p - 6
            is_ally = _pick_is_ally(pick_idx, ally_first)
            picks.append((is_ally, brawler))

        phase[0] += 1
        buf.reset()

        if phase[0] >= 11:
            filter_on[0] = _phase_default_filter(11, ally_first)
            terminal_cache[0] = get_terminal_pick6_scores(
                ctx, event, picks, _display_excluded(), ally_first=ally_first,
            )
            final_mode[0] = True
            return

        filter_on[0] = _phase_default_filter(phase[0], ally_first)
        _refresh_q_cache()

    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(ev):
        cancelled[0] = True
        ev.app.exit()

    # ── layout ───────────────────────────────────────────────────────────────
    grid_win = Window(
        content=FormattedTextControl(_grid_content, focusable=False),
        wrap_lines=False,
    )
    input_win = Window(
        content=BufferControl(buf, include_default_input_processors=True, focusable=True),
        get_line_prefix=_line_prefix,
        height=1,
    )
    layout = Layout(
        FloatContainer(
            content=HSplit([grid_win, input_win]),
            floats=[Float(CompletionsMenu(max_height=10, scroll_offset=2), xcursor=True, ycursor=True)],
        ),
        focused_element=input_win,
    )

    Application(
        layout=layout,
        key_bindings=kb,
        style=_DARK_STYLE,
        full_screen=False,
        mouse_support=False,
    ).run()

    if cancelled[0]:
        return None
    return ally_bans, enemy_bans, picks


# ── main loop ─────────────────────────────────────────────────────────────────

@typer_app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    terminal_ckpt: Annotated[Path, typer.Option(help="Frozen BrawlModel checkpoint (.eqx)")] = Path("data/model/model.eqx"),
    draft_ckpt: Annotated[Path, typer.Option(help="Draft Q-network weights (.eqx)")] = Path("data/draft_model/draft_q.eqx"),
    embed_dim: Annotated[int, typer.Option(help="BrawlModel embedding dimension")] = 32,
    hidden_dim: Annotated[int, typer.Option(help="BrawlModel hidden dimension")] = 64,
    d_model: Annotated[int, typer.Option(help="DraftQNetwork d_model")] = 64,
    n_heads: Annotated[int, typer.Option(help="DraftQNetwork attention heads")] = 4,
    n_layers: Annotated[int, typer.Option(help="DraftQNetwork transformer layers")] = 2,
) -> None:
    console.print(f"Loading terminal model [dim]{terminal_ckpt}[/dim]...")
    console.print(f"Loading draft model    [dim]{draft_ckpt}[/dim]...")

    ctx = load_context(
        data_dir, terminal_ckpt, draft_ckpt,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    )
    console.print(
        f"Ready — [bold]{len(ctx.brawlers)}[/bold] brawlers, "
        f"[bold]{len(ctx.events)}[/bold] maps. "
        f"[dim]Ctrl-C to quit.[/dim]\n"
    )
    render_rarity_legend(console)
    render_annotation_legend(console)

    filter_ids = _parse_brawl_filter(ctx)
    matched_names, unmatched_names = _brawl_filter_names(ctx)
    for name in unmatched_names:
        console.print(f"  [red]No brawler matching '{name}' in $BRAWL_FILTER[/red]")
    if filter_ids:
        console.print(f"  Filter ({len(filter_ids)}): {', '.join(sorted(matched_names))}\n")

    session: PromptSession = PromptSession(style=_DARK_STYLE)
    last_event: EventInfo | None = None

    try:
        while True:
            # ── Map selection ──────────────────────────────────────────────
            map_raw = session.prompt(
                "Map> ",
                rprompt=_rprompt_map(session, ctx),
                completer=_map_completer(ctx),
                complete_while_typing=True,
            ).strip()

            if not map_raw:
                if last_event is None:
                    continue
                event = last_event
            else:
                matched_map = fuzzy_find(map_raw, ctx.map_names)
                if matched_map is None:
                    console.print(f"  [red]No map matching '{map_raw}'[/red]\n")
                    continue
                event = next(e for e in ctx.events if e.map_name == matched_map)

            last_event = event

            map_scores = score_map(ctx, event)
            ov_ann = overview_annotations(ctx, event, map_scores)

            # Show winrate overview
            console.print(f"\n  [bold]{event.map_name}[/bold]  [dim]{event.mode}[/dim]")
            render_brawler_table(
                console, map_scores,
                score_fmt=lambda s: f"{s:+.2f}",
                annotations=ov_ann,
                divider=0.0,
                annotation_slot=ANNOTATION_SLOT,
                filter_ids=filter_ids or None,
            )
            console.print()

            # ── Coin flip ──────────────────────────────────────────────────
            while True:
                flip_raw = session.prompt("  Ally first? [y/n]> ").strip().lower()
                if flip_raw in ("y", "yes"):
                    ally_first = True
                    break
                if flip_raw in ("n", "no"):
                    ally_first = False
                    break
                console.print("  [red]Type y or n[/red]")

            # ── Full draft (bans + picks 1–5; pick-6 shown in Application) ─
            result = _run_draft(ctx, event, map_scores, ov_ann, ally_first, filter_ids)
            if result is None:
                console.print("  [dim](cancelled)[/dim]\n")
                continue

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Goodbye.[/dim]")


def entrypoint() -> None:
    typer_app()
