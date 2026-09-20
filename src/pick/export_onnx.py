"""Export the trained BrawlModel and DraftQNetwork to ONNX for the web port.

Two static-shape graphs are produced (no batching in the graph itself — the
draft Q-network is already a single-example forward pass over all characters,
and the terminal model is exported as a single team-vs-team forward pass; the
web client loops over pick-6 candidates the same way the CLI's vmap does):

  draft_q.onnx        char_encs [n_chars, h], player_char_states [n_chars],
                       turn_token [1]  ->  q_values [n_chars]
  terminal_pick6.onnx event_idx [1], mode_idx [1], team_a_chars [3],
                       team_a_meta [3, 3], team_b_chars [3], team_b_meta [3, 3]
                       ->  logit []  (team-A-wins logit)

`char_encs_all` (the terminal model's frozen per-event character encodings,
used as the draft Q-network's input) is not reachable through either model's
public `__call__` — it comes from `BrawlModel._encode_char`, internal to the
model — so it is shipped alongside the graphs as a raw float32 binary plus a
manifest describing shapes and the draft-state constants the client needs to
rebuild `_build_obs` in JS. Brawler/event display metadata and
winrate/pickrate z-scores (everything else the CLI's `load_context` reads
from data_dir) go to a separate metadata.json.
"""

import json
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import typer
from jax2onnx import to_onnx
from rich.console import Console

from geneus.draft.env import (
    AVAILABLE,
    BAN_PHASE,
    GLOBALLY_BANNED,
    LOCALLY_BANNED,
    N_CHAR_STATES,
    N_DRAFT_TOKENS,
    PICKED_A,
    PICKED_B,
    TURN_SCHEDULE,
)
from pick.score import BrawlerInfo, DraftContext, load_context

typer_app = typer.Typer(add_completion=False, help="Export ONNX graphs + companion data for the web port.")
console = Console()

_RANK_K = 15  # top-k ordering checked against the JAX reference during validation


def _draft_q_fn(q_net):
    def fn(char_encs: jax.Array, player_states: jax.Array, turn_token: jax.Array) -> jax.Array:
        return q_net(char_encs, player_states, turn_token[0])
    return fn


def _terminal_pick_fn(terminal_model):
    def fn(
        event_idx: jax.Array, mode_idx: jax.Array,
        team_a_chars: jax.Array, team_a_meta: jax.Array,
        team_b_chars: jax.Array, team_b_meta: jax.Array,
    ) -> jax.Array:
        return terminal_model(event_idx[0], mode_idx[0], team_a_chars, team_a_meta, team_b_chars, team_b_meta)
    return fn


def export_draft_q(ctx: DraftContext, output_path: Path) -> None:
    h = ctx.char_encs_all.shape[-1]
    to_onnx(
        _draft_q_fn(ctx.q_net),
        inputs=[
            jax.ShapeDtypeStruct((ctx.n_chars, h), jnp.float32),
            jax.ShapeDtypeStruct((ctx.n_chars,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ],
        input_names=["char_encs", "player_char_states", "turn_token"],
        output_names=["q_values"],
        return_mode="file",
        output_path=str(output_path),
    )


def export_terminal_pick6(ctx: DraftContext, output_path: Path) -> None:
    to_onnx(
        _terminal_pick_fn(ctx.terminal_model),
        inputs=[
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((3, 3), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((3, 3), jnp.int32),
        ],
        input_names=["event_idx", "mode_idx", "team_a_chars", "team_a_meta", "team_b_chars", "team_b_meta"],
        output_names=["logit"],
        return_mode="file",
        output_path=str(output_path),
    )


def validate_draft_q(ctx: DraftContext, onnx_path: Path, n_trials: int = 8, seed: int = 0) -> None:
    """Check ONNX output matches the JAX model: values (atol 1e-4) and top-k ranking."""
    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.default_rng(seed)
    n_events = ctx.char_encs_all.shape[0]
    for _ in range(n_trials):
        row = int(rng.integers(0, n_events))
        char_encs = ctx.char_encs_all[row].astype(np.float32)
        states = rng.integers(0, N_CHAR_STATES, size=ctx.n_chars).astype(np.int32)
        turn = np.array([rng.integers(0, N_DRAFT_TOKENS)], dtype=np.int32)

        jax_out = np.array(ctx.q_net(jnp.array(char_encs), jnp.array(states), jnp.array(turn[0])))
        (onnx_out,) = sess.run(
            None,
            {"char_encs": char_encs, "player_char_states": states, "turn_token": turn},
        )
        onnx_out = np.asarray(onnx_out)

        if not np.allclose(jax_out, onnx_out, atol=1e-4):
            raise ValueError(f"draft_q ONNX output diverges from JAX (max abs diff {np.abs(jax_out - onnx_out).max():.2e})")
        jax_top = np.argsort(-jax_out)[:_RANK_K]
        onnx_top = np.argsort(-onnx_out)[:_RANK_K]
        if jax_top.tolist() != onnx_top.tolist():
            raise ValueError("draft_q ONNX top-k ranking diverges from JAX")


def validate_terminal_pick6(ctx: DraftContext, onnx_path: Path, n_trials: int = 8, seed: int = 0) -> None:
    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.default_rng(seed)
    n_events = len(ctx.events)
    for _ in range(n_trials):
        event = ctx.events[int(rng.integers(0, n_events))]
        chars = rng.choice(ctx.n_chars, size=6, replace=False)
        team_a_chars, team_b_chars = chars[:3].astype(np.int32), chars[3:].astype(np.int32)
        team_a_meta = ctx.char_meta_table[team_a_chars]
        team_b_meta = ctx.char_meta_table[team_b_chars]
        inputs = {
            "event_idx": np.array([event.event_idx], dtype=np.int32),
            "mode_idx": np.array([event.mode_idx], dtype=np.int32),
            "team_a_chars": team_a_chars,
            "team_a_meta": team_a_meta,
            "team_b_chars": team_b_chars,
            "team_b_meta": team_b_meta,
        }

        jax_out = float(ctx.terminal_model(
            jnp.array(event.event_idx), jnp.array(event.mode_idx),
            jnp.array(team_a_chars), jnp.array(team_a_meta),
            jnp.array(team_b_chars), jnp.array(team_b_meta),
        ))
        (onnx_out,) = sess.run(None, inputs)
        onnx_val = float(np.asarray(onnx_out))

        if not np.allclose(jax_out, onnx_val, atol=1e-4):
            raise ValueError(f"terminal_pick6 ONNX output diverges from JAX ({jax_out!r} vs {onnx_val!r})")


def export_data(ctx: DraftContext, data_out: Path) -> None:
    """Write char_encs_all as a raw float32 binary plus a manifest.

    Kept minimal: only what's needed to actually call the two ONNX graphs —
    char_encs_all (draft_q's per-event character encodings) and
    char_meta_table (terminal_pick6's per-character class/range/destruct
    indices) — plus the draft-state constants needed to rebuild `_build_obs`
    in JS. Brawler/event/winrate/pickrate display metadata (names, rarities,
    map names, ...) is a separate, later step.
    """
    data_out.mkdir(parents=True, exist_ok=True)
    char_encs = ctx.char_encs_all.astype(np.float32)
    (data_out / "char_encs.bin").write_bytes(char_encs.tobytes())

    manifest = {
        "n_chars": ctx.n_chars,
        "h_terminal": int(char_encs.shape[-1]),
        "char_encs": {
            "path": "char_encs.bin",
            "dtype": "float32",
            "shape": list(char_encs.shape),  # [n_events, n_chars, h]
        },
        "char_meta_table": ctx.char_meta_table.tolist(),  # [n_chars][3]: (class_idx, range_idx, destruct_idx)
        "event_idx_to_row": {str(k): v for k, v in ctx.event_enc_row.items()},
        "draft_state": {
            "AVAILABLE": AVAILABLE,
            "GLOBALLY_BANNED": GLOBALLY_BANNED,
            "LOCALLY_BANNED": LOCALLY_BANNED,
            "PICKED_A": PICKED_A,
            "PICKED_B": PICKED_B,
            "N_CHAR_STATES": N_CHAR_STATES,
            "BAN_PHASE": BAN_PHASE,
            "N_DRAFT_TOKENS": N_DRAFT_TOKENS,
            "TURN_SCHEDULE": [list(t) for t in TURN_SCHEDULE],
        },
        "models": {
            "draft_q": {
                "path": "models/draft_q.onnx",
                "inputs": {
                    "char_encs": [ctx.n_chars, int(char_encs.shape[-1])],
                    "player_char_states": [ctx.n_chars],
                    "turn_token": [1],
                },
                "outputs": {"q_values": [ctx.n_chars]},
            },
            "terminal_pick6": {
                "path": "models/terminal_pick6.onnx",
                "inputs": {
                    "event_idx": [1], "mode_idx": [1],
                    "team_a_chars": [3], "team_a_meta": [3, 3],
                    "team_b_chars": [3], "team_b_meta": [3, 3],
                },
                "outputs": {"logit": []},
                "note": "single team-vs-team forward pass (team-A-wins logit); call once per pick-6 candidate",
            },
        },
    }
    (data_out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def export_metadata(ctx: DraftContext, output_path: Path) -> None:
    """Write brawler/event display metadata + winrate/pickrate z-scores.

    `brawlers[i]` is the brawler at char_idx i, so it zips 1:1 with
    draft_q's q_values output and manifest.json's char_meta_table. Kept
    separate from manifest.json, which is only about how to call the ONNX
    graphs — this is display + `score_map`/`pick_annotations` data, mirroring
    what `load_context` reads from data_dir for the CLI.
    """
    by_char_idx: list[BrawlerInfo | None] = [None] * ctx.n_chars
    for b in ctx.brawlers:
        by_char_idx[b.char_idx] = b
    missing = [i for i, b in enumerate(by_char_idx) if b is None]
    if missing:
        raise ValueError(f"char_idx gap in ctx.brawlers (vocab/brawler_class.json mismatch): {missing}")

    metadata = {
        "brawlers": [
            {"id": b.id, "name": b.name, "class": b.brawler_class, "rarity": b.rarity, "char_idx": b.char_idx}
            for b in by_char_idx
            if b is not None
        ],
        "events": [
            {
                "id": e.id, "mode": e.mode, "mode_id": e.mode_id,
                "map_name": e.map_name, "event_idx": e.event_idx, "mode_idx": e.mode_idx,
            }
            for e in ctx.events
        ],
        "winrates": {str(cid): {str(eid): z for eid, z in ev.items()} for cid, ev in ctx.winrates.items()},
        "pickrates": {str(cid): {str(eid): z for eid, z in ev.items()} for cid, ev in ctx.pickrates.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2))


def export_test_fixture(ctx: DraftContext, output_path: Path, n_cases: int = 3, seed: int = 0) -> None:
    """Write fixed (input, output) pairs computed from the real JAX models.

    Consumed by the TS test suite (bun test, onnxruntime-web) to check the
    exported ONNX graphs still produce the same numbers in the browser
    runtime. Not shipped to web/static — this is a test-only artifact.
    """
    rng = np.random.default_rng(seed)
    row_to_event = {row: event_idx for event_idx, row in ctx.event_enc_row.items()}
    n_events = ctx.char_encs_all.shape[0]

    draft_q_cases = []
    for _ in range(n_cases):
        row = int(rng.integers(0, n_events))
        char_encs = ctx.char_encs_all[row].astype(np.float32)
        states = rng.integers(0, N_CHAR_STATES, size=ctx.n_chars).astype(np.int32)
        turn = int(rng.integers(0, N_DRAFT_TOKENS))
        q_values = np.array(ctx.q_net(jnp.array(char_encs), jnp.array(states), jnp.array(turn)))
        draft_q_cases.append({
            "event_idx": row_to_event[row],
            "player_char_states": states.tolist(),
            "turn_token": turn,
            "q_values": q_values.tolist(),
        })

    terminal_pick6_cases = []
    for _ in range(n_cases):
        event = ctx.events[int(rng.integers(0, len(ctx.events)))]
        chars = rng.choice(ctx.n_chars, size=6, replace=False)
        team_a_chars, team_b_chars = chars[:3].astype(np.int32), chars[3:].astype(np.int32)
        team_a_meta = ctx.char_meta_table[team_a_chars]
        team_b_meta = ctx.char_meta_table[team_b_chars]
        logit = float(ctx.terminal_model(
            jnp.array(event.event_idx), jnp.array(event.mode_idx),
            jnp.array(team_a_chars), jnp.array(team_a_meta),
            jnp.array(team_b_chars), jnp.array(team_b_meta),
        ))
        terminal_pick6_cases.append({
            "event_idx": event.event_idx,
            "mode_idx": event.mode_idx,
            "team_a_chars": team_a_chars.tolist(),
            "team_a_meta": team_a_meta.tolist(),
            "team_b_chars": team_b_chars.tolist(),
            "team_b_meta": team_b_meta.tolist(),
            "logit": logit,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"draft_q": draft_q_cases, "terminal_pick6": terminal_pick6_cases}, indent=2))


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
    models_out: Annotated[Path, typer.Option(help="ONNX output directory")] = Path("web/static/models"),
    data_out: Annotated[Path, typer.Option(help="Companion data output directory")] = Path("web/static/data"),
    metadata_out: Annotated[Path, typer.Option(help="Brawler/event/winrate/pickrate metadata output path")] = Path("web/static/data/metadata.json"),
    fixture_out: Annotated[Path, typer.Option(help="TS test fixture output path (not shipped to web/static)")] = Path("web/tests/fixtures/onnx_fixture.json"),
    validate: Annotated[bool, typer.Option(help="Cross-check ONNX outputs against the JAX models")] = True,
) -> None:
    console.print(f"Loading terminal model [dim]{terminal_ckpt}[/dim]...")
    console.print(f"Loading draft model    [dim]{draft_ckpt}[/dim]...")
    ctx = load_context(
        data_dir, terminal_ckpt, draft_ckpt,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    )

    models_out.mkdir(parents=True, exist_ok=True)

    console.print("Exporting draft_q.onnx...")
    export_draft_q(ctx, models_out / "draft_q.onnx")
    if validate:
        validate_draft_q(ctx, models_out / "draft_q.onnx")
        console.print("  [green]validated against JAX (values + top-15 ranking)[/green]")

    console.print("Exporting terminal_pick6.onnx...")
    export_terminal_pick6(ctx, models_out / "terminal_pick6.onnx")
    if validate:
        validate_terminal_pick6(ctx, models_out / "terminal_pick6.onnx")
        console.print("  [green]validated against JAX[/green]")

    console.print(f"Writing companion data to [dim]{data_out}[/dim]...")
    export_data(ctx, data_out)

    console.print(f"Writing display metadata to [dim]{metadata_out}[/dim]...")
    export_metadata(ctx, metadata_out)

    console.print(f"Writing TS test fixture to [dim]{fixture_out}[/dim]...")
    export_test_fixture(ctx, fixture_out)

    console.print("[bold green]Done.[/bold green]")


def entrypoint() -> None:
    typer_app()
