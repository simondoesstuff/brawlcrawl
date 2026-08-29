"""Interactive draft-assist REPL: rank map picks and score 6th picks."""

import shutil
from math import ceil
from pathlib import Path
from typing import Annotated

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

from pick.constants import ANN_BRAIN, ANN_SECRET, ANN_X, ANNOTATION_SLOT, PICKRATE_HIGH_Z, PICKRATE_LOW_Z, WIN_PROB_THRESHOLD
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
from pick.score import BrawlerInfo, DraftContext, EventInfo, load_context, score_map, score_sixth_pick

typer_app = typer.Typer(add_completion=False, help="Interactive draft-assist REPL.")
console = Console()

_DEFAULT_EPOCH = 400

_DARK_STYLE = Style.from_dict({
    "completion-menu.completion":               "bg:#111111 fg:ansiwhite",
    "completion-menu.completion.current":       "bg:#3a3a3a fg:ansiwhite bold",
    "completion-menu.meta.completion":          "bg:#111111 fg:#666666",
    "completion-menu.meta.completion.current":  "bg:#3a3a3a fg:#aaaaaa",
    "completion-menu.border":                   "bg:#111111 fg:#444444",
    "scrollbar.background":                     "bg:#111111",
    "scrollbar.button":                         "bg:#444444",
})

_PHASES = [
    ("Enemy 1", "enemy"),
    ("Enemy 2", "enemy"),
    ("Enemy 3", "enemy"),
    ("Ally  1", "ally"),
    ("Ally  2", "ally"),
]


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

def sixth_pick_annotations(
    ctx: DraftContext,
    event: EventInfo,
    sixth_scores: list[tuple[BrawlerInfo, float]],
) -> dict[int, str]:
    """Return per-brawler-id annotation for the 6th-pick table.

    Brain icon: model ≥50% win, below-average pickrate (hidden gem).
    X icon:     model <50% win, high pickrate (overrated by players).
    """
    result: dict[int, str] = {}
    for brawler, win_prob in sixth_scores:
        pz = ctx.pickrates.get(brawler.id, {}).get(event.id)
        if pz is None:
            continue
        if win_prob >= WIN_PROB_THRESHOLD and pz < PICKRATE_LOW_Z:
            result[brawler.id] = ANN_BRAIN
        elif win_prob < WIN_PROB_THRESHOLD and pz >= PICKRATE_HIGH_Z:
            result[brawler.id] = ANN_X
    return result


def overview_annotations(
    ctx: DraftContext,
    event: EventInfo,
    map_scores: list[tuple[BrawlerInfo, float]],
) -> dict[int, str]:
    """Return per-brawler-id annotation for the overview table.

    Secret icon: good map score (≥0), below-average pickrate (hidden gem).
    X icon:      poor map score (<0), high pickrate (overrated by players).
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


# ── team assembly Application ─────────────────────────────────────────────────

def _run_team_assembly(
    ctx: DraftContext,
    event: EventInfo,
    map_scores: list[tuple[BrawlerInfo, float]],
    ann: dict[int, str] | None = None,
) -> tuple[list[BrawlerInfo], list[BrawlerInfo]] | None:
    """Interactive team assembly with live grid highlights.

    Returns (allies, enemies) or None if cancelled.
    """
    submitted: list[tuple[str, BrawlerInfo]] = []  # (role, brawler)
    cancelled = [False]
    error_msg = [""]
    _ann = ann or {}

    def _excluded() -> set[int]:
        return {b.id for _, b in submitted}

    def _available() -> list[str]:
        excl = _excluded()
        return [b.name for b in ctx.brawlers if b.id not in excl]

    buf = Buffer(
        name="pick",
        multiline=False,
        completer=DynamicCompleter(lambda: _brawler_completer(ctx, _excluded())),
        complete_while_typing=True,
    )

    # ── grid content (re-rendered on every keystroke) ──────────────────────
    def _grid_content() -> FormattedText:
        try:
            term_w = shutil.get_terminal_size().columns
        except Exception:
            term_w = 80

        current_text = buf.text
        best_match_id: int | None = None
        if current_text.strip():
            m = fuzzy_find(current_text, _available())
            if m:
                best_match_id = ctx._brawler_by_name[m].id

        submitted_roles: dict[int, str] = {b.id: role for role, b in submitted}

        col_w = _COL_WIDTH + ANNOTATION_SLOT
        n_cols = max(1, (term_w + _COL_GAP) // (col_w + _COL_GAP))

        above = [(b, s) for b, s in map_scores if s >= 0]
        below = [(b, s) for b, s in map_scores if s < 0]

        result: list[tuple[str, str]] = []

        # Header
        phase_label = ""
        if len(submitted) < len(_PHASES):
            label, _ = _PHASES[len(submitted)]
            phase_label = f"   entering {label.lower()}"
        result += [
            ("bold", f"\n  {event.map_name}"),
            ("fg:ansibrightblack", f"  {event.mode}{phase_label}\n\n"),
        ]

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
                    score_str = f"{score:+.2f}"
                    rar_style = RARITY_PT_STYLES.get(brawler.rarity, "")

                    rank_sep = "."
                    if brawler.id in submitted_roles:
                        role = submitted_roles[brawler.id]
                        bg = "bg:#2a0000" if role == "enemy" else "bg:#002a00"
                        rank_s = f"fg:ansibrightblack {bg}"
                        name_s = f"{rar_style} {bg}"
                        ann_s  = f"fg:ansibrightblack {bg}"
                        score_s = bg
                    elif brawler.id == best_match_id:
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
                    result.append((ann_s, _ann.get(brawler.id, " " * ANNOTATION_SLOT)))
                    result.append((score_s, f"{score_str:>{_SCORE_WIDTH}}"))

                    if col < n_cols - 1:
                        result.append(("", "  "))

                result.append(("", "\n"))

        _render_section(above, 0)
        if above and below:
            result.append(("fg:ansiwhite", "─" * term_w + "\n"))
        _render_section(below, len(above))

        # Submitted summary
        if submitted:
            result.append(("", "\n"))
            for role, b in submitted:
                role_label = "Enemy" if role == "enemy" else "Ally "
                bg = "bg:#2a0000" if role == "enemy" else "bg:#002a00"
                rar_s = RARITY_PT_STYLES.get(b.rarity, "")
                result += [
                    ("fg:ansibrightblack", f"  {role_label}  "),
                    (f"{rar_s} {bg}", f"{b.name:<{_MAX_NAME_LEN}}"),
                    ("fg:ansibrightblack", f"  {b.brawler_class}\n"),
                ]

        # Error
        if error_msg[0]:
            result += [("", "\n"), ("fg:ansired", f"  {error_msg[0]}\n")]

        result.append(("", "\n"))
        return FormattedText(result)

    # ── input prompt prefix ────────────────────────────────────────────────
    def _line_prefix(_line_num, _wrap_count) -> FormattedText:
        if len(submitted) < len(_PHASES):
            label, _ = _PHASES[len(submitted)]
            return FormattedText([("bold", f"  {label}> ")])
        return FormattedText([])

    # ── key bindings ───────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(ev):
        text = buf.text.strip()
        if not text:
            return
        idx = len(submitted)
        if idx >= len(_PHASES):
            return

        available = _available()
        matched = fuzzy_find(text, available)
        if matched is None:
            error_msg[0] = f"No brawler matching '{text}'"
            return

        error_msg[0] = ""
        _, role = _PHASES[idx]
        submitted.append((role, ctx._brawler_by_name[matched]))
        buf.reset()

        if len(submitted) >= len(_PHASES):
            ev.app.exit()

    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(ev):
        cancelled[0] = True
        ev.app.exit()

    # ── layout ─────────────────────────────────────────────────────────────
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

    app_inst = Application(
        layout=layout,
        key_bindings=kb,
        style=_DARK_STYLE,
        full_screen=False,
        mouse_support=False,
    )
    app_inst.run()

    if cancelled[0]:
        return None

    enemies = [b for role, b in submitted if role == "enemy"]
    allies = [b for role, b in submitted if role == "ally"]
    return allies, enemies


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _resolve_checkpoint(data_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    ckpt_dir = data_dir / "model" / "checkpoints"
    preferred = ckpt_dir / f"model_epoch_{_DEFAULT_EPOCH:04d}.eqx"
    if preferred.exists():
        return preferred
    candidates = sorted(ckpt_dir.glob("model_epoch_*.eqx"))
    if not candidates:
        console.print(f"[red]No checkpoints found in {ckpt_dir}[/red]")
        raise SystemExit(1)
    return candidates[-1]


# ── main loop ─────────────────────────────────────────────────────────────────

@typer_app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    checkpoint: Annotated[Path | None, typer.Option(help="Checkpoint .eqx file")] = None,
    embed_dim: Annotated[int, typer.Option(help="Model embedding dimension")] = 32,
    hidden_dim: Annotated[int, typer.Option(help="Model hidden dimension")] = 64,
) -> None:
    ckpt = _resolve_checkpoint(data_dir, checkpoint)
    console.print(f"Loading [dim]{ckpt.name}[/dim]...")

    ctx = load_context(data_dir, ckpt, embed_dim=embed_dim, hidden_dim=hidden_dim)
    console.print(
        f"Ready — [bold]{len(ctx.brawlers)}[/bold] brawlers, "
        f"[bold]{len(ctx.events)}[/bold] maps. "
        f"[dim]Ctrl-C to quit.[/dim]\n"
    )
    render_rarity_legend(console)
    render_annotation_legend(console)

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

            # ── Team assembly (with live grid) ─────────────────────────────
            result = _run_team_assembly(ctx, event, map_scores, ann=ov_ann)
            if result is None:
                console.print("  [dim](cancelled)[/dim]\n")
                continue
            allies, enemies = result

            # ── 6th pick ───────────────────────────────────────────────────
            def _colored(b: BrawlerInfo) -> str:
                c = RARITY_COLORS.get(b.rarity, "white")
                return f"[bold {c}]{b.name}[/bold {c}]"

            console.print(
                f"\n  Enemy: {'  +  '.join(_colored(b) for b in enemies)}\n"
                f"  Ally:  {'  +  '.join(_colored(b) for b in allies)}  +  [bold dim]?[/bold dim]\n"
            )

            sixth_scores = score_sixth_pick(ctx, event, allies, enemies)
            annotations = sixth_pick_annotations(ctx, event, sixth_scores)
            render_brawler_table(
                console, sixth_scores,
                score_fmt=lambda s: f"{s * 100:.1f}%",
                annotations=annotations,
                divider=WIN_PROB_THRESHOLD,
                annotation_slot=ANNOTATION_SLOT,
            )
            console.print()

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Goodbye.[/dim]")


def entrypoint() -> None:
    typer_app()
