"""Deep-sets anti-symmetric model for Brawl Stars win probability prediction."""

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
                else:
                    x = layer(x, inference=True)
            else:
                x = layer(x)
        return x


class BrawlModel(eqx.Module):
    """
    Anti-symmetric deep-sets model.

    Architecture (all operations on single examples; vmap for batching):

        map  = embed_mapID(event) + embed_mapMode(mode)
        meta = embed_class + embed_range + embed_destruct   # per brawler
        char_i = MLP_ctx([map | MLP_inner([dropout(embed_char) | meta])])
        T    = [max_i(char_i) | sum_i(char_i)]             # per team
        logit = MLP_team([T_A | T_B | map]) - MLP_team([T_B | T_A | map])

    Pass key=<PRNGKey> to __call__ to enable dropout during training.
    """

    embed_map_id: eqx.nn.Embedding
    embed_map_mode: eqx.nn.Embedding
    embed_char: eqx.nn.Embedding
    embed_class: eqx.nn.Embedding
    embed_range: eqx.nn.Embedding
    embed_destruct: eqx.nn.Embedding
    char_dropout: eqx.nn.Dropout
    mlp_char_inner: MLP
    mlp_char_ctx: MLP
    mlp_team: MLP
    mlp_winrate: MLP

    def __init__(
        self,
        n_events: int,
        n_modes: int,
        n_chars: int,
        n_classes: int,
        n_ranges: int,
        n_destructs: int,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        dropout_p: float = 0.1,
        *,
        key: Array,
    ) -> None:
        k = jax.random.split(key, 10)
        self.embed_map_id = eqx.nn.Embedding(n_events, embed_dim, key=k[0])
        self.embed_map_mode = eqx.nn.Embedding(n_modes, embed_dim, key=k[1])
        self.embed_char = eqx.nn.Embedding(n_chars, embed_dim, key=k[2])
        self.embed_class = eqx.nn.Embedding(n_classes, embed_dim, key=k[3])
        self.embed_range = eqx.nn.Embedding(n_ranges, embed_dim, key=k[4])
        self.embed_destruct = eqx.nn.Embedding(n_destructs, embed_dim, key=k[5])
        self.char_dropout = eqx.nn.Dropout(p=dropout_p)
        self.mlp_char_inner = MLP(embed_dim * 2, hidden_dim, hidden_dim, depth=1, dropout_p=dropout_p, key=k[6])
        self.mlp_char_ctx = MLP(embed_dim + hidden_dim, hidden_dim, hidden_dim, depth=1, dropout_p=dropout_p, key=k[7])
        self.mlp_team = MLP(hidden_dim * 4 + embed_dim, hidden_dim, 1, depth=1, dropout_p=dropout_p, key=k[8])
        self.mlp_winrate = MLP(hidden_dim, hidden_dim, 1, depth=1, dropout_p=dropout_p, key=k[9])

    def _encode_char(
        self,
        char_idx: Int[Array, ""],
        meta_idxs: Int[Array, "3"],
        map_emb: Float[Array, " d"],
        *,
        key: Array | None = None,
    ) -> Float[Array, " h"]:
        char_emb = self.embed_char(char_idx)
        if key is not None:
            k_drop, k_inner, k_ctx = jax.random.split(key, 3)
            char_emb = self.char_dropout(char_emb, key=k_drop)
        else:
            k_inner = k_ctx = None
        meta_sum = (
            self.embed_class(meta_idxs[0])
            + self.embed_range(meta_idxs[1])
            + self.embed_destruct(meta_idxs[2])
        )
        inner = self.mlp_char_inner(jnp.concatenate([char_emb, meta_sum]), key=k_inner)
        return self.mlp_char_ctx(jnp.concatenate([map_emb, inner]), key=k_ctx)

    def _encode_team(
        self,
        char_idxs: Int[Array, "3"],
        meta_idxs: Int[Array, "3 3"],
        map_emb: Float[Array, " d"],
        *,
        key: Array | None = None,
    ) -> Float[Array, " two_h"]:
        if key is not None:
            char_keys = jax.random.split(key, 3)
            chars = jax.vmap(
                lambda c, m, k: self._encode_char(c, m, map_emb, key=k)
            )(char_idxs, meta_idxs, char_keys)
        else:
            chars = jax.vmap(
                lambda c, m: self._encode_char(c, m, map_emb)
            )(char_idxs, meta_idxs)
        return jnp.concatenate([chars.max(axis=0), chars.sum(axis=0)])

    def _score(
        self,
        t_a: Float[Array, " two_h"],
        t_b: Float[Array, " two_h"],
        map_emb: Float[Array, " d"],
        *,
        key: Array | None = None,
    ) -> Float[Array, ""]:
        return self.mlp_team(jnp.concatenate([t_a, t_b, map_emb]), key=key).squeeze()

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
            k_a, k_b, k_s1, k_s2 = jax.random.split(key, 4)
            t_a = self._encode_team(team_a_chars, team_a_meta, map_emb, key=k_a)
            t_b = self._encode_team(team_b_chars, team_b_meta, map_emb, key=k_b)
            return self._score(t_a, t_b, map_emb, key=k_s1) - self._score(t_b, t_a, map_emb, key=k_s2)
        else:
            t_a = self._encode_team(team_a_chars, team_a_meta, map_emb)
            t_b = self._encode_team(team_b_chars, team_b_meta, map_emb)
            return self._score(t_a, t_b, map_emb) - self._score(t_b, t_a, map_emb)

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
