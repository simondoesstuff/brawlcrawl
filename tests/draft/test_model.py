"""Tests for the DraftQNetwork."""

import jax
import jax.numpy as jnp
import pytest

from geneus.draft.env import N_DRAFT_TOKENS, N_CHAR_STATES, BAN_PHASE_FIRST_PICK, PICK_1
from geneus.draft.model import DraftQNetwork

N_CHARS = 20
H = 32       # fake terminal encoder dim
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 1


@pytest.fixture(scope="module")
def q_net():
    return DraftQNetwork(
        h_terminal=H,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        key=jax.random.PRNGKey(0),
    )


def test_output_shape(q_net):
    char_encs = jnp.zeros((N_CHARS, H))
    states = jnp.zeros(N_CHARS, dtype=jnp.int32)
    turn = jnp.array(BAN_PHASE_FIRST_PICK, dtype=jnp.int32)
    out = q_net(char_encs, states, turn)
    assert out.shape == (N_CHARS,)


def test_output_shape_all_tokens(q_net):
    char_encs = jnp.ones((N_CHARS, H))
    states = jnp.zeros(N_CHARS, dtype=jnp.int32)
    for tok in range(N_DRAFT_TOKENS):
        turn = jnp.array(tok, dtype=jnp.int32)
        out = q_net(char_encs, states, turn)
        assert out.shape == (N_CHARS,), f"Shape mismatch for token {tok}"


def test_different_states_differ(q_net):
    char_encs = jax.random.normal(jax.random.PRNGKey(1), (N_CHARS, H))
    turn = jnp.array(PICK_1, dtype=jnp.int32)

    states_a = jnp.zeros(N_CHARS, dtype=jnp.int32)
    states_b = jnp.ones(N_CHARS, dtype=jnp.int32)  # all GLOBALLY_BANNED

    out_a = q_net(char_encs, states_a, turn)
    out_b = q_net(char_encs, states_b, turn)
    assert not jnp.allclose(out_a, out_b), "Different states should produce different Q values"


def test_different_turn_tokens_differ(q_net):
    char_encs = jax.random.normal(jax.random.PRNGKey(2), (N_CHARS, H))
    states = jnp.zeros(N_CHARS, dtype=jnp.int32)

    out_ban = q_net(char_encs, states, jnp.array(BAN_PHASE_FIRST_PICK, dtype=jnp.int32))
    out_pick = q_net(char_encs, states, jnp.array(PICK_1, dtype=jnp.int32))
    assert not jnp.allclose(out_ban, out_pick)


def test_vmap_over_turns(q_net):
    char_encs = jnp.ones((N_CHARS, H))
    states_seq = jnp.zeros((12, N_CHARS), dtype=jnp.int32)
    tokens_seq = jnp.arange(12, dtype=jnp.int32) % N_DRAFT_TOKENS

    def q_for_turn(ps, tt):
        return q_net(char_encs, ps, tt)

    out = jax.vmap(q_for_turn)(states_seq, tokens_seq)
    assert out.shape == (12, N_CHARS)


def test_no_nan(q_net):
    key = jax.random.PRNGKey(99)
    char_encs = jax.random.normal(key, (N_CHARS, H))
    states = jax.random.randint(key, (N_CHARS,), 0, N_CHAR_STATES)
    turn = jnp.array(BAN_PHASE_FIRST_PICK, dtype=jnp.int32)
    out = q_net(char_encs, states, turn)
    assert not jnp.any(jnp.isnan(out))
