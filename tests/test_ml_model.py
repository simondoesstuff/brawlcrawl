"""Tests for geneus.model — shapes, anti-symmetry, permutation invariance, jit."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geneus.model import BrawlModel, MLP, Swish
from geneus.train import winrate_loss
from geneus.data import WinrateArrays

VOCAB = dict(
    n_events=31,
    n_modes=6,
    n_chars=106,
    n_classes=7,
    n_ranges=4,
    n_destructs=4,
    embed_dim=8,
    hidden_dim=16,
    dropout_p=0.1,
)


@pytest.fixture
def model() -> BrawlModel:
    return BrawlModel(**VOCAB, key=jax.random.PRNGKey(0))


@pytest.fixture
def sample_inputs() -> tuple:
    event_idx = jnp.array(0)
    mode_idx = jnp.array(0)
    team_a_chars = jnp.array([0, 1, 2])
    team_a_meta = jnp.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    team_b_chars = jnp.array([3, 4, 5])
    team_b_meta = jnp.array([[3, 3, 3], [0, 0, 0], [1, 1, 1]])
    return event_idx, mode_idx, team_a_chars, team_a_meta, team_b_chars, team_b_meta


class TestSwish:
    def test_beta_initialized_to_one(self) -> None:
        s = Swish()
        assert float(s.beta) == pytest.approx(1.0)

    def test_output_at_zero(self) -> None:
        s = Swish()
        assert float(s(jnp.zeros(()))) == pytest.approx(0.0)

    def test_positive_output_for_positive_input(self) -> None:
        s = Swish()
        assert float(s(jnp.array(1.0))) > 0.0

    def test_beta_is_leaf(self) -> None:
        s = Swish()
        leaves = jax.tree.leaves(s)
        assert any(jnp.array_equal(l, jnp.ones(())) for l in leaves)


class TestMLP:
    def test_output_shape_depth_1(self) -> None:
        m = MLP(4, 8, 2, depth=1, key=jax.random.PRNGKey(0))
        out = m(jnp.ones(4))
        assert out.shape == (2,)

    def test_output_shape_depth_2(self) -> None:
        m = MLP(4, 8, 1, depth=2, key=jax.random.PRNGKey(0))
        out = m(jnp.ones(4))
        assert out.shape == (1,)

    def test_depth_zero_is_linear(self) -> None:
        m = MLP(4, 8, 3, depth=0, key=jax.random.PRNGKey(0))
        out = m(jnp.ones(4))
        assert out.shape == (3,)


class TestBrawlModel:
    def test_forward_returns_scalar(self, model: BrawlModel, sample_inputs: tuple) -> None:
        logit = model(*sample_inputs)
        assert logit.shape == ()

    def test_anti_symmetry(self, model: BrawlModel, sample_inputs: tuple) -> None:
        event_idx, mode_idx, ta_c, ta_m, tb_c, tb_m = sample_inputs
        logit_ab = model(event_idx, mode_idx, ta_c, ta_m, tb_c, tb_m)
        logit_ba = model(event_idx, mode_idx, tb_c, tb_m, ta_c, ta_m)
        np.testing.assert_allclose(float(logit_ab), -float(logit_ba), atol=1e-5)

    def test_permutation_invariance_team_a(self, model: BrawlModel, sample_inputs: tuple) -> None:
        event_idx, mode_idx, ta_c, ta_m, tb_c, tb_m = sample_inputs
        # Permute brawlers within team A
        perm = jnp.array([2, 0, 1])
        ta_c_perm = ta_c[perm]
        ta_m_perm = ta_m[perm]
        logit_orig = model(event_idx, mode_idx, ta_c, ta_m, tb_c, tb_m)
        logit_perm = model(event_idx, mode_idx, ta_c_perm, ta_m_perm, tb_c, tb_m)
        np.testing.assert_allclose(float(logit_orig), float(logit_perm), atol=1e-5)

    def test_permutation_invariance_team_b(self, model: BrawlModel, sample_inputs: tuple) -> None:
        event_idx, mode_idx, ta_c, ta_m, tb_c, tb_m = sample_inputs
        perm = jnp.array([1, 2, 0])
        tb_c_perm = tb_c[perm]
        tb_m_perm = tb_m[perm]
        logit_orig = model(event_idx, mode_idx, ta_c, ta_m, tb_c, tb_m)
        logit_perm = model(event_idx, mode_idx, ta_c, ta_m, tb_c_perm, tb_m_perm)
        np.testing.assert_allclose(float(logit_orig), float(logit_perm), atol=1e-5)

    def test_jit_compatible(self, model: BrawlModel, sample_inputs: tuple) -> None:
        jitted = eqx.filter_jit(model)
        logit = jitted(*sample_inputs)
        assert logit.shape == ()

    def test_vmap_over_batch(self, model: BrawlModel) -> None:
        B = 4
        event_idx = jnp.zeros(B, dtype=jnp.int32)
        mode_idx = jnp.zeros(B, dtype=jnp.int32)
        team_a_chars = jnp.zeros((B, 3), dtype=jnp.int32)
        team_a_meta = jnp.zeros((B, 3, 3), dtype=jnp.int32)
        team_b_chars = jnp.ones((B, 3), dtype=jnp.int32)
        team_b_meta = jnp.zeros((B, 3, 3), dtype=jnp.int32)
        logits = jax.vmap(model)(
            event_idx, mode_idx, team_a_chars, team_a_meta, team_b_chars, team_b_meta
        )
        assert logits.shape == (B,)

    def test_dropout_with_key(self, model: BrawlModel, sample_inputs: tuple) -> None:
        logit_inf = model(*sample_inputs)
        logit_train = model(*sample_inputs, key=jax.random.PRNGKey(1))
        assert logit_inf.shape == ()
        assert logit_train.shape == ()

    def test_dropout_stochastic(self, model: BrawlModel, sample_inputs: tuple) -> None:
        logit_k1 = model(*sample_inputs, key=jax.random.PRNGKey(1))
        logit_k2 = model(*sample_inputs, key=jax.random.PRNGKey(2))
        assert not jnp.array_equal(logit_k1, logit_k2)

    def test_predict_winrate_scalar(self, model: BrawlModel) -> None:
        logit = model.predict_winrate(
            jnp.array(0), jnp.array(0), jnp.array(0), jnp.array([0, 0, 0])
        )
        assert logit.shape == ()

    def test_predict_winrate_with_key(self, model: BrawlModel) -> None:
        logit = model.predict_winrate(
            jnp.array(0), jnp.array(0), jnp.array(0), jnp.array([0, 0, 0]),
            key=jax.random.PRNGKey(0),
        )
        assert logit.shape == ()

    def test_predict_winrate_vmap(self, model: BrawlModel) -> None:
        B = 4
        logits = jax.vmap(model.predict_winrate)(
            jnp.zeros(B, dtype=jnp.int32),
            jnp.zeros(B, dtype=jnp.int32),
            jnp.zeros(B, dtype=jnp.int32),
            jnp.zeros((B, 3), dtype=jnp.int32),
        )
        assert logits.shape == (B,)

    def test_winrate_loss_runs(self, model: BrawlModel) -> None:
        B = 4
        batch = WinrateArrays(
            event_idx=jnp.zeros(B, dtype=jnp.int32),
            mode_idx=jnp.zeros(B, dtype=jnp.int32),
            char_idx=jnp.zeros(B, dtype=jnp.int32),
            char_meta=jnp.zeros((B, 3), dtype=jnp.int32),
            z_scores=jnp.zeros(B, dtype=jnp.float32),
        )
        loss = winrate_loss(model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_vmap_with_keys(self, model: BrawlModel) -> None:
        B = 4
        event_idx = jnp.zeros(B, dtype=jnp.int32)
        mode_idx = jnp.zeros(B, dtype=jnp.int32)
        team_a_chars = jnp.zeros((B, 3), dtype=jnp.int32)
        team_a_meta = jnp.zeros((B, 3, 3), dtype=jnp.int32)
        team_b_chars = jnp.ones((B, 3), dtype=jnp.int32)
        team_b_meta = jnp.zeros((B, 3, 3), dtype=jnp.int32)
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        logits = jax.vmap(
            lambda ev, mo, tac, tam, tbc, tbm, k: model(ev, mo, tac, tam, tbc, tbm, key=k)
        )(event_idx, mode_idx, team_a_chars, team_a_meta, team_b_chars, team_b_meta, keys)
        assert logits.shape == (B,)
