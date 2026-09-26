"""Team-vs-team transformer model for Brawl Stars win probability prediction."""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int


class Swish(eqx.Module):
    """Parameterized Swish / SiLU: x * sigmoid(β * x), β initialized to 1."""

    beta: Array

    def __init__(self) -> None:
        self.beta = jnp.ones(())

    def __call__(self, x: Array) -> Array:
        return x * jax.nn.sigmoid(self.beta * x)


class MLP(eqx.Module):
    """MLP with `depth` hidden layers (Linear → Swish → Dropout each) and a linear output."""

    layers: list

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        *,
        dropout_p: float = 0.0,
        key: Array,
    ) -> None:
        keys = jax.random.split(key, depth + 1)
        layers: list = []
        prev = in_dim
        for i in range(depth):
            layers.append(eqx.nn.Linear(prev, hidden_dim, key=keys[i]))
            layers.append(Swish())
            if dropout_p > 0.0:
                layers.append(eqx.nn.Dropout(p=dropout_p))
            prev = hidden_dim
        layers.append(eqx.nn.Linear(prev, out_dim, key=keys[depth]))
        self.layers = layers

    def __call__(self, x: Array, *, key: Array | None = None) -> Array:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Dropout):
                if key is not None:
                    key, subkey = jax.random.split(key)
                    x = layer(x, key=subkey)
                # else: inference — dropout is a no-op, skip the call entirely
            else:
                x = layer(x)
        return x


class FiLM(eqx.Module):
    """Feature-wise linear modulation: gamma(cond) * x + beta(cond)."""

    proj: eqx.nn.Linear

    def __init__(self, cond_dim: int, feature_dim: int, *, key: Array) -> None:
        self.proj = eqx.nn.Linear(cond_dim, feature_dim * 2, key=key)

    def __call__(self, cond: Float[Array, " c"], x: Float[Array, " d"]) -> Float[Array, " d"]:
        gamma, beta = jnp.split(self.proj(cond), 2)
        return gamma * x + beta


class SymmetricBlock(eqx.Module):
    """Pre-LN self-attn → cross-attn → FFN.

    Applied with identical weights to both teams (roles swapped for
    cross-attention), so the block commutes with a team-label swap:
    `block(X, ctx=Y) `and `block(Y, ctx=X)` are the same function evaluated on
    swapped arguments. That commutation is what lets `BrawlModel`'s head stay
    exactly antisymmetric without an explicit `f(A,B) - f(B,A)` computation.
    """

    self_attn: eqx.nn.MultiheadAttention
    self_norm: eqx.nn.LayerNorm
    cross_attn: eqx.nn.MultiheadAttention
    cross_norm_q: eqx.nn.LayerNorm
    cross_norm_kv: eqx.nn.LayerNorm
    ffn: MLP
    ffn_norm: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, dropout_p: float, *, key: Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.self_attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=k1)
        self.self_norm = eqx.nn.LayerNorm(d_model)
        self.cross_attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=k2)
        self.cross_norm_q = eqx.nn.LayerNorm(d_model)
        self.cross_norm_kv = eqx.nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, d_model * 2, d_model, depth=1, dropout_p=dropout_p, key=k3)
        self.ffn_norm = eqx.nn.LayerNorm(d_model)

    def self_stage(self, x: Float[Array, "3 d"]) -> Float[Array, "3 d"]:
        normed = jax.vmap(self.self_norm)(x)
        return x + self.self_attn(normed, normed, normed)

    def cross_stage(self, x: Float[Array, "3 d"], ctx: Float[Array, "3 d"]) -> Float[Array, "3 d"]:
        xn = jax.vmap(self.cross_norm_q)(x)
        cn = jax.vmap(self.cross_norm_kv)(ctx)
        return x + self.cross_attn(xn, cn, cn)

    def ffn_stage(self, x: Float[Array, "3 d"], *, key: Array | None = None) -> Float[Array, "3 d"]:
        if key is not None:
            keys = jax.random.split(key, x.shape[0])
            return x + jax.vmap(lambda xi, k: self.ffn(self.ffn_norm(xi), key=k))(x, keys)
        return x + jax.vmap(lambda xi: self.ffn(self.ffn_norm(xi)))(x)


class SymmetricEncoder(eqx.Module):
    """Stacks `SymmetricBlock`s so cross-attention in each block reads the
    sibling team's latest *self-attended* state (both teams finish self-attn
    before either one cross-attends), matching:

        A -> transformer -> A'
        B -> ... -> B'
        A' -> CrossAttn(Q=A', KV=B') -> A''
        B' -> CrossAttn(Q=B', KV=A') -> B''

    per block. Every branch (team A vs team B) runs through the exact same
    `SymmetricBlock` instances — required for `BrawlModel`'s antisymmetric head.
    """

    blocks: list

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, dropout_p: float, *, key: Array) -> None:
        keys = jax.random.split(key, n_blocks)
        self.blocks = [SymmetricBlock(d_model, n_heads, dropout_p, key=k) for k in keys]

    def __call__(
        self,
        team_a: Float[Array, "3 d"],
        team_b: Float[Array, "3 d"],
        *,
        key: Array | None = None,
    ) -> tuple[Float[Array, "3 d"], Float[Array, "3 d"]]:
        block_keys = jax.random.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for block, k in zip(self.blocks, block_keys):
            a_self = block.self_stage(team_a)
            b_self = block.self_stage(team_b)
            a_cross = block.cross_stage(a_self, b_self)
            b_cross = block.cross_stage(b_self, a_self)
            # Same FFN dropout key for both teams: keeps the encoder exactly
            # equivariant under a team-label swap, even mid-training.
            team_a = block.ffn_stage(a_cross, key=k)
            team_b = block.ffn_stage(b_cross, key=k)
        return team_a, team_b


class BrawlModel(eqx.Module):
    """
    Team-vs-team transformer, anti-symmetric by construction.

    Architecture (all operations on single examples; vmap for batching):

        map     = embed_mapID(event) + embed_mapMode(mode)
        brawler = FiLM(map)([embed_char | embed_class + embed_range + embed_destruct])
        A, B    = SymmetricEncoder(brawlers_A, brawlers_B)   # self-attn + cross-attn, shared weights
        a       = [mean_i(A_i) | max_i(A_i)],  b = [mean_i(B_i) | max_i(B_i)]
        logit   = W^T (a - b)                                # bias-free linear -> exactly antisymmetric

    `_encode_char` (the FiLM'd per-brawler vector, before any team-vs-team
    attention) stays a pure function of (map, char, meta) — no teammate or
    opponent context — because `geneus.draft.train.precompute_char_encs`
    calls it directly to build the frozen per-(event, char) encodings the
    draft Q-network is trained on. Only the win-probability head above models
    cross-team synergies/counters; `_encode_char`'s signature and semantics
    must not change.

    Pass key=<PRNGKey> to __call__ to enable dropout during training. `__call__`
    and `predict_winrate` share a single top-level key across both teams'
    branches (not independent per-team subkeys) so that swapping team A/B
    labels exactly negates the logit, even with dropout active.
    """

    embed_map_id: eqx.nn.Embedding
    embed_map_mode: eqx.nn.Embedding
    embed_char: eqx.nn.Embedding
    embed_class: eqx.nn.Embedding
    embed_range: eqx.nn.Embedding
    embed_destruct: eqx.nn.Embedding
    char_dropout: eqx.nn.Dropout
    brawler_proj: eqx.nn.Linear
    film: FiLM
    encoder: SymmetricEncoder
    head: eqx.nn.Linear
    mlp_winrate: MLP

    def __init__(
        self,
        n_events: int,
        n_modes: int,
        n_chars: int,
        n_classes: int,
        n_ranges: int,
        n_destructs: int,
        d_model: int = 48,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout_p: float = 0.1,
        *,
        key: Array,
    ) -> None:
        k = jax.random.split(key, 11)
        self.embed_map_id = eqx.nn.Embedding(n_events, d_model, key=k[0])
        self.embed_map_mode = eqx.nn.Embedding(n_modes, d_model, key=k[1])
        self.embed_char = eqx.nn.Embedding(n_chars, d_model, key=k[2])
        self.embed_class = eqx.nn.Embedding(n_classes, d_model, key=k[3])
        self.embed_range = eqx.nn.Embedding(n_ranges, d_model, key=k[4])
        self.embed_destruct = eqx.nn.Embedding(n_destructs, d_model, key=k[5])
        self.char_dropout = eqx.nn.Dropout(p=dropout_p)
        self.brawler_proj = eqx.nn.Linear(d_model * 2, d_model, key=k[6])
        self.film = FiLM(d_model, d_model, key=k[7])
        self.encoder = SymmetricEncoder(d_model, n_heads, n_blocks, dropout_p, key=k[8])
        self.head = eqx.nn.Linear(d_model * 2, 1, use_bias=False, key=k[9])
        self.mlp_winrate = MLP(d_model, d_model, 1, depth=1, dropout_p=dropout_p, key=k[10])

    def _encode_char(
        self,
        char_idx: Int[Array, ""],
        meta_idxs: Int[Array, "3"],
        map_emb: Float[Array, " d"],
        *,
        key: Array | None = None,
    ) -> Float[Array, " d"]:
        char_emb = self.embed_char(char_idx)
        if key is not None:
            char_emb = self.char_dropout(char_emb, key=key)
        meta_sum = (
            self.embed_class(meta_idxs[0])
            + self.embed_range(meta_idxs[1])
            + self.embed_destruct(meta_idxs[2])
        )
        brawler_base = self.brawler_proj(jnp.concatenate([char_emb, meta_sum]))
        return self.film(map_emb, brawler_base)

    def _encode_team(
        self,
        char_idxs: Int[Array, "3"],
        meta_idxs: Int[Array, "3 3"],
        map_emb: Float[Array, " d"],
        *,
        key: Array | None = None,
    ) -> Float[Array, "3 d"]:
        if key is not None:
            char_keys = jax.random.split(key, 3)
            return jax.vmap(
                lambda c, m, k: self._encode_char(c, m, map_emb, key=k)
            )(char_idxs, meta_idxs, char_keys)
        return jax.vmap(lambda c, m: self._encode_char(c, m, map_emb))(char_idxs, meta_idxs)

    def __call__(
        self,
        event_idx: Int[Array, ""],
        mode_idx: Int[Array, ""],
        team_a_chars: Int[Array, "3"],
        team_a_meta: Int[Array, "3 3"],
        team_b_chars: Int[Array, "3"],
        team_b_meta: Int[Array, "3 3"],
        *,
        key: Array | None = None,
    ) -> Float[Array, ""]:
        map_emb = self.embed_map_id(event_idx) + self.embed_map_mode(mode_idx)
        if key is not None:
            k_enc, k_team = jax.random.split(key)
        else:
            k_enc = k_team = None
        # Same key for both teams' `_encode_team` calls: see class docstring.
        team_a = self._encode_team(team_a_chars, team_a_meta, map_emb, key=k_enc)
        team_b = self._encode_team(team_b_chars, team_b_meta, map_emb, key=k_enc)
        a_f, b_f = self.encoder(team_a, team_b, key=k_team)
        a = jnp.concatenate([a_f.mean(axis=0), a_f.max(axis=0)])
        b = jnp.concatenate([b_f.mean(axis=0), b_f.max(axis=0)])
        return self.head(a - b).squeeze()

    def predict_winrate(
        self,
        event_idx: Int[Array, ""],
        mode_idx: Int[Array, ""],
        char_idx: Int[Array, ""],
        char_meta: Int[Array, "3"],
        *,
        key: Array | None = None,
    ) -> Float[Array, ""]:
        """Predict the win probability for a single character on a given map."""
        map_emb = self.embed_map_id(event_idx) + self.embed_map_mode(mode_idx)
        if key is not None:
            k_char, k_wr = jax.random.split(key)
        else:
            k_char = k_wr = None
        char_emb = self._encode_char(char_idx, char_meta, map_emb, key=k_char)
        return self.mlp_winrate(char_emb, key=k_wr).squeeze()
