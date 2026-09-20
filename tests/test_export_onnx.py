"""Tests for pick.export_onnx — ONNX export + validation, on freshly-initialized
small models (no dependency on trained checkpoints in data/)."""

import json

import jax
import numpy as np
import pytest

from geneus.draft.env import N_CHAR_STATES, N_DRAFT_TOKENS
from geneus.draft.model import DraftQNetwork
from geneus.model import BrawlModel
from pick.export_onnx import (
    export_data,
    export_draft_q,
    export_terminal_pick6,
    validate_draft_q,
    validate_terminal_pick6,
)
from pick.score import DraftContext, EventInfo

N_CHARS = 20
N_EVENTS = 3
H = 16          # BrawlModel hidden_dim == terminal encoding dim
D_MODEL = 12
N_HEADS = 2
N_LAYERS = 1

VOCAB = dict(
    n_events=N_EVENTS, n_modes=2, n_chars=N_CHARS,
    n_classes=3, n_ranges=2, n_destructs=2,
    embed_dim=8, hidden_dim=H, dropout_p=0.3,  # nonzero dropout: exercises the inference no-op path
)


@pytest.fixture
def ctx() -> DraftContext:
    terminal_model = BrawlModel(**VOCAB, key=jax.random.PRNGKey(0))
    q_net = DraftQNetwork(h_terminal=H, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, key=jax.random.PRNGKey(1))

    rng = np.random.default_rng(0)
    char_encs_all = rng.normal(size=(N_EVENTS, N_CHARS, H)).astype(np.float32)
    char_meta_table = np.stack([
        rng.integers(0, VOCAB["n_classes"], N_CHARS),
        rng.integers(0, VOCAB["n_ranges"], N_CHARS),
        rng.integers(0, VOCAB["n_destructs"], N_CHARS),
    ], axis=1).astype(np.int32)

    events = [EventInfo(id=i, mode="bounty", mode_id=0, map_name=f"map{i}", event_idx=i, mode_idx=0) for i in range(N_EVENTS)]

    return DraftContext(
        q_net=q_net,
        terminal_model=terminal_model,
        char_meta_table=char_meta_table,
        char_encs_all=char_encs_all,
        event_enc_row={i: i for i in range(N_EVENTS)},
        brawlers=[],
        events=events,
        brawler_names=[],
        map_names=[e.map_name for e in events],
        n_chars=N_CHARS,
        winrates={},
        pickrates={},
        _brawler_by_char_idx={},
        _brawler_by_name={},
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
    assert manifest["event_idx_to_row"] == {"0": 0, "1": 1, "2": 2}
    assert manifest["draft_state"]["N_CHAR_STATES"] == N_CHAR_STATES
    assert manifest["draft_state"]["N_DRAFT_TOKENS"] == N_DRAFT_TOKENS
    assert len(manifest["draft_state"]["TURN_SCHEDULE"]) == 12

    blob = (tmp_path / "char_encs.bin").read_bytes()
    assert len(blob) == N_EVENTS * N_CHARS * H * 4  # float32
    restored = np.frombuffer(blob, dtype=np.float32).reshape(N_EVENTS, N_CHARS, H)
    np.testing.assert_array_equal(restored, ctx.char_encs_all.astype(np.float32))
