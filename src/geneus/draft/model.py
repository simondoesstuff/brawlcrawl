"""Transformer Q-network for the draft phase."""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Any

from geneus.draft.env import N_CHAR_STATES, N_DRAFT_TOKENS
from geneus.model import MLP


class TransformerBlock(eqx.Module):
    """Pre-LN self-attention block (no positional encoding)."""

    attn: eqx.nn.MultiheadAttention
    attn_norm: eqx.nn.LayerNorm
    ffn: MLP
    ffn_norm: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, *, key: Array) -> None:
        super().__init__()
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=k1)
        self.attn_norm = eqx.nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, d_model * 2, d_model, depth=1, key=k2)
        self.ffn_norm = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: Float[Array, "seq d"]) -> Float[Array, "seq d"]:
        normed = jax.vmap(self.attn_norm)(x)
        x = x + self.attn(normed, normed, normed)
        x = x + jax.vmap(lambda xi: self.ffn(self.ffn_norm(xi)))(x)
        return x


class DraftQNetwork(eqx.Module):
    """Transformer Q-network for the draft phase.

    Architecture:
        seq[0]   = draft_turn_emb(turn_token)
        seq[1:n] = char_proj(char_encs) + char_state_emb(player_char_states)
        encoded  = TransformerEncoder(seq)   — no positional encoding (set-based)
        Q_local  = q_head(encoded[1:]).squeeze()  — one scalar per character

    Q_local is in logit space from the acting team's perspective:
      positive → current team expected to win
      negative → current team expected to lose

    The draft turn token encodes which phase, team, and pick position this is.
    The character state embeddings encode the observable draft history.
    Together they give the transformer full context to distinguish all 12 turns.
    """

    char_proj: eqx.nn.Linear          # h_terminal → d_model
    char_state_emb: eqx.nn.Embedding  # N_CHAR_STATES → d_model
    draft_turn_emb: eqx.nn.Embedding  # N_DRAFT_TOKENS → d_model
    transformer: list[TransformerBlock]
    q_head: eqx.nn.Linear             # d_model → 1

    def __init__(
        self,
        h_terminal: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        *,
        key: Array,
    ) -> None:
        super().__init__()
        keys = jax.random.split(key, n_layers + 4)
        self.char_proj = eqx.nn.Linear(h_terminal, d_model, use_bias=False, key=keys[0])
        self.char_state_emb = eqx.nn.Embedding(N_CHAR_STATES, d_model, key=keys[1])
        self.draft_turn_emb = eqx.nn.Embedding(N_DRAFT_TOKENS, d_model, key=keys[2])
        self.transformer = [
            TransformerBlock(d_model, n_heads, key=keys[3 + i]) for i in range(n_layers)
        ]
        self.q_head = eqx.nn.Linear(d_model, 1, key=keys[3 + n_layers])

    def __call__(
        self,
        char_encs: Float[Array, "n_chars h"],
        player_char_states: Int[Array, "n_chars"],
        turn_token: Int[Array, ""],
    ) -> Float[Array, "n_chars"]:
        char_tokens = (
            jax.vmap(self.char_proj)(char_encs)
            + jax.vmap(self.char_state_emb)(player_char_states)
        )
        turn_tok = self.draft_turn_emb(turn_token)
        seq = jnp.concatenate([turn_tok[None], char_tokens], axis=0)  # [1+n_chars, d]

        for block in self.transformer:
            seq = block(seq)

        return jax.vmap(self.q_head)(seq[1:]).squeeze(-1)  # [n_chars]
