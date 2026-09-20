"""Tests for pick.export_onnx — ONNX export + validation, on freshly-initialized
small models (no dependency on trained checkpoints in data/)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geneus.draft.env import N_CHAR_STATES, N_DRAFT_TOKENS
from geneus.draft.model import DraftQNetwork
from geneus.model import BrawlModel
from pick.export_onnx import (
    export_data,
    export_draft_q,
    export_metadata,
    export_terminal_pick6,
    export_test_fixture,
    validate_draft_q,
    validate_terminal_pick6,
)
from pick.score import BrawlerInfo, DraftContext, EventInfo

N_CHARS = 20
N_EVENTS = 3
H = 16          # BrawlModel hidden_dim == terminal encoding dim
D_MODEL = 12
N_HEADS = 2
N_LAYERS = 1

N_CLASSES = 3
N_RANGES = 2
N_DESTRUCTS = 2
DROPOUT_P = 0.3  # nonzero: exercises the inference no-op path


@pytest.fixture
def ctx() -> DraftContext:
    terminal_model = BrawlModel(
        n_events=N_EVENTS, n_modes=2, n_chars=N_CHARS,
        n_classes=N_CLASSES, n_ranges=N_RANGES, n_destructs=N_DESTRUCTS,
        embed_dim=8, hidden_dim=H, dropout_p=DROPOUT_P,
        key=jax.random.PRNGKey(0),
    )
    q_net = DraftQNetwork(h_terminal=H, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, key=jax.random.PRNGKey(1))

    rng = np.random.default_rng(0)
    char_encs_all = rng.normal(size=(N_EVENTS, N_CHARS, H)).astype(np.float32)
    char_meta_table = np.stack([
        rng.integers(0, N_CLASSES, N_CHARS),
        rng.integers(0, N_RANGES, N_CHARS),
        rng.integers(0, N_DESTRUCTS, N_CHARS),
    ], axis=1).astype(np.int32)

    events = [EventInfo(id=i, mode="bounty", mode_id=0, map_name=f"map{i}", event_idx=i, mode_idx=0) for i in range(N_EVENTS)]
    brawlers = [
        BrawlerInfo(id=1000 + i, name=f"Brawler{i}", brawler_class="Damage Dealer", rarity="Rare", char_idx=i)
        for i in range(N_CHARS)
    ]
    winrates = {b.id: {e.id: float(rng.normal()) for e in events} for b in brawlers}
    pickrates = {b.id: {e.id: float(rng.normal()) for e in events} for b in brawlers}

    return DraftContext(
        q_net=q_net,
        terminal_model=terminal_model,
        char_meta_table=char_meta_table,
        char_encs_all=char_encs_all,
        event_enc_row={i: i for i in range(N_EVENTS)},
        brawlers=brawlers,
        events=events,
        brawler_names=sorted(b.name for b in brawlers),
        map_names=[e.map_name for e in events],
        n_chars=N_CHARS,
        winrates=winrates,
        pickrates=pickrates,
        _brawler_by_char_idx={b.char_idx: b for b in brawlers},
        _brawler_by_name={b.name: b for b in brawlers},
    )


def test_export_draft_q_validates(ctx, tmp_path):
    onnx_path = tmp_path / "draft_q.onnx"
    export_draft_q(ctx, onnx_path)
    assert onnx_path.exists()
    validate_draft_q(ctx, onnx_path, n_trials=3)


def test_export_terminal_pick6_validates(ctx, tmp_path):
    onnx_path = tmp_path / "terminal_pick6.onnx"
    export_terminal_pick6(ctx, onnx_path)
    assert onnx_path.exists()
    validate_terminal_pick6(ctx, onnx_path, n_trials=3)


def test_export_data_writes_manifest_and_binary(ctx, tmp_path):
    export_data(ctx, tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_chars"] == N_CHARS
    assert manifest["h_terminal"] == H
    assert manifest["char_encs"]["shape"] == [N_EVENTS, N_CHARS, H]
    assert manifest["char_meta_table"] == ctx.char_meta_table.tolist()
    assert manifest["event_idx_to_row"] == {"0": 0, "1": 1, "2": 2}
    assert manifest["draft_state"]["N_CHAR_STATES"] == N_CHAR_STATES
    assert manifest["draft_state"]["N_DRAFT_TOKENS"] == N_DRAFT_TOKENS
    assert len(manifest["draft_state"]["TURN_SCHEDULE"]) == 12

    blob = (tmp_path / "char_encs.bin").read_bytes()
    assert len(blob) == N_EVENTS * N_CHARS * H * 4  # float32
    restored = np.frombuffer(blob, dtype=np.float32).reshape(N_EVENTS, N_CHARS, H)
    np.testing.assert_array_equal(restored, ctx.char_encs_all.astype(np.float32))


def test_export_metadata(ctx, tmp_path):
    metadata_path = tmp_path / "metadata.json"
    export_metadata(ctx, metadata_path)

    metadata = json.loads(metadata_path.read_text())
    assert len(metadata["brawlers"]) == N_CHARS
    for i, b in enumerate(metadata["brawlers"]):
        assert b["char_idx"] == i
        assert b["id"] == ctx.brawlers[i].id
        assert b["name"] == ctx.brawlers[i].name

    assert len(metadata["events"]) == N_EVENTS
    assert {e["event_idx"] for e in metadata["events"]} == {0, 1, 2}

    first_brawler_id = ctx.brawlers[0].id
    first_event_id = ctx.events[0].id
    assert metadata["winrates"][str(first_brawler_id)][str(first_event_id)] == pytest.approx(
        ctx.winrates[first_brawler_id][first_event_id]
    )
    assert metadata["pickrates"][str(first_brawler_id)][str(first_event_id)] == pytest.approx(
        ctx.pickrates[first_brawler_id][first_event_id]
    )


def test_export_test_fixture_matches_jax(ctx, tmp_path):
    fixture_path = tmp_path / "onnx_fixture.json"
    export_test_fixture(ctx, fixture_path, n_cases=2)

    fixture = json.loads(fixture_path.read_text())
    assert len(fixture["draft_q"]) == 2
    assert len(fixture["terminal_pick6"]) == 2

    for case in fixture["draft_q"]:
        row = ctx.event_enc_row[case["event_idx"]]
        char_encs = jnp.array(ctx.char_encs_all[row], dtype=jnp.float32)
        states = jnp.array(case["player_char_states"], dtype=jnp.int32)
        turn = jnp.array(case["turn_token"], dtype=jnp.int32)
        q_values = ctx.q_net(char_encs, states, turn)
        np.testing.assert_allclose(np.array(q_values), case["q_values"], atol=1e-6)

    for case in fixture["terminal_pick6"]:
        logit = ctx.terminal_model(
            jnp.array(case["event_idx"]), jnp.array(case["mode_idx"]),
            jnp.array(case["team_a_chars"]), jnp.array(case["team_a_meta"]),
            jnp.array(case["team_b_chars"]), jnp.array(case["team_b_meta"]),
        )
        assert float(logit) == pytest.approx(case["logit"], abs=1e-6)
