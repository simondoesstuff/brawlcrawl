"""Training loop and CLI for the BrawlModel."""

from pathlib import Path
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as mplt
import numpy as np
import optax
import plotext as plt
import typer
from tqdm import tqdm

from geneus.data import BattleArrays, load_battles, load_vocabs, train_val_split
from geneus.model import BrawlModel

app = typer.Typer(add_completion=False)

_MODEL_PATH = Path("data/model/model.eqx")


def _to_jax(batch: BattleArrays) -> BattleArrays:
    return BattleArrays(
        event_idx=jnp.asarray(batch.event_idx),
        mode_idx=jnp.asarray(batch.mode_idx),
        team_a_chars=jnp.asarray(batch.team_a_chars),
        team_a_meta=jnp.asarray(batch.team_a_meta),
        team_b_chars=jnp.asarray(batch.team_b_chars),
        team_b_meta=jnp.asarray(batch.team_b_meta),
        a_wins=jnp.asarray(batch.a_wins),
        totals=jnp.asarray(batch.totals),
    )


def bce_loss(
    model: BrawlModel,
    batch: BattleArrays,
    key: jax.Array | None = None,
) -> jax.Array:
    """Total-weighted BCE. Reconstructs the full Bernoulli likelihood."""
    if key is not None:
        batch_keys = jax.random.split(key, batch.event_idx.shape[0])
        logits = jax.vmap(
            lambda ev, mo, tac, tam, tbc, tbm, k: model(ev, mo, tac, tam, tbc, tbm, key=k)
        )(
            batch.event_idx,
            batch.mode_idx,
            batch.team_a_chars,
            batch.team_a_meta,
            batch.team_b_chars,
            batch.team_b_meta,
            batch_keys,
        )
    else:
        logits = jax.vmap(model)(
            batch.event_idx,
            batch.mode_idx,
            batch.team_a_chars,
            batch.team_a_meta,
            batch.team_b_chars,
            batch.team_b_meta,
        )
    totals = batch.totals.astype(jnp.float32)
    p = batch.a_wins.astype(jnp.float32) / totals
    bce = -(jax.nn.log_sigmoid(logits) * p + jax.nn.log_sigmoid(-logits) * (1.0 - p))
    return (bce * totals).sum() / totals.sum()


def accuracy(model: BrawlModel, batch: BattleArrays) -> jax.Array:
    logits = jax.vmap(model)(
        batch.event_idx,
        batch.mode_idx,
        batch.team_a_chars,
        batch.team_a_meta,
        batch.team_b_chars,
        batch.team_b_meta,
    )
    p = batch.a_wins.astype(jnp.float32) / batch.totals.astype(jnp.float32)
    return ((logits > 0) == (p > 0.5)).mean()


def make_step_fn(optimizer: optax.GradientTransformation):
    """Return a jit-compiled training step that closes over the optimizer."""

    @eqx.filter_jit
    def step(
        model: BrawlModel,
        opt_state: optax.OptState,
        batch: BattleArrays,
        key: jax.Array,
    ) -> tuple[BrawlModel, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(bce_loss)(model, batch, key)
        updates, new_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_state, loss

    return step


def _iter_batches(
    arrays: BattleArrays,
    batch_size: int,
    rng: np.random.Generator,
) -> list[BattleArrays]:
    perm = rng.permutation(len(arrays))
    batches = []
    for start in range(0, len(arrays), batch_size):
        idx = perm[start : start + batch_size]
        # Randomly flip which team is "A" — exact since model is anti-symmetric.
        flip = rng.integers(0, 2, size=len(idx)).astype(bool)
        ta_c = arrays.team_a_chars[idx]
        ta_m = arrays.team_a_meta[idx]
        tb_c = arrays.team_b_chars[idx]
        tb_m = arrays.team_b_meta[idx]
        a_wins = arrays.a_wins[idx]
        totals = arrays.totals[idx]
        batches.append(_to_jax(BattleArrays(
            event_idx=arrays.event_idx[idx],
            mode_idx=arrays.mode_idx[idx],
            team_a_chars=np.where(flip[:, None], tb_c, ta_c),
            team_a_meta=np.where(flip[:, None, None], tb_m, ta_m),
            team_b_chars=np.where(flip[:, None], ta_c, tb_c),
            team_b_meta=np.where(flip[:, None, None], ta_m, tb_m),
            a_wins=np.where(flip, totals - a_wins, a_wins),
            totals=totals,
        )))
    return batches


def _colored_marker(color: str) -> plt.colorize:
    return plt.colorize("━", pixel=plt.pixel(color))


def _render_dashboard(
    epoch_lines: list[str],
    epochs_x: list[int],
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
) -> None:
    fig = plt.figure
    fig.clear()
    fig.subplots(1, 2)

    sp_loss = fig.subplot(1, 1)
    sp_loss.title("BCE Loss")
    s_train = sp_loss.signal(epochs_x, train_losses, marker=_colored_marker("cyan"))
    s_val   = sp_loss.signal(epochs_x, val_losses,   marker=_colored_marker("red"))
    s_train.label("train")
    s_val.label("val")
    sp_loss.draw(s_train)
    sp_loss.draw(s_val)
    sp_loss.legend()
    sp_loss.plot_size(55, 18)

    sp_acc = fig.subplot(1, 2)
    sp_acc.title("Val Accuracy %")
    s_acc = sp_acc.signal(epochs_x, [a * 100 for a in val_accs], marker=_colored_marker("green"))
    s_acc.label("val acc")
    sp_acc.draw(s_acc)
    sp_acc.legend()
    sp_acc.plot_size(55, 18)

    plt.terminal.clean(-1)  # clear screen, keep history scrollable
    print(fig.build().string())
    for line in epoch_lines[-20:]:
        print(line)


def _save_curves(
    path: Path,
    epochs_x: list[int],
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
) -> None:
    fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_x, train_losses, label="train", color="tab:blue")
    ax1.plot(epochs_x, val_losses, label="val", color="tab:red")
    ax1.set_title("BCE Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(epochs_x, [a * 100 for a in val_accs], color="tab:green")
    ax2.set_title("Val Accuracy %")
    ax2.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    mplt.close(fig)


@app.command()
def train(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    out: Annotated[Path, typer.Option(help="Best-model output path")] = _MODEL_PATH,
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 512,
    lr: Annotated[float, typer.Option(help="Peak learning rate")] = 1e-3,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay")] = 1e-4,
    embed_dim: Annotated[int, typer.Option(help="Embedding dimension")] = 16,
    hidden_dim: Annotated[int, typer.Option(help="MLP hidden dimension")] = 64,
    dropout_p: Annotated[float, typer.Option(help="Dropout probability (char + MLP)")] = 0.3,
    val_frac: Annotated[float, typer.Option(help="Validation fraction")] = 0.1,
    checkpoint_every: Annotated[int, typer.Option(help="Periodic checkpoint interval in epochs (0=off)")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    typer.echo("Loading data...")
    vocabs = load_vocabs(data_dir)
    all_battles = load_battles(data_dir, vocabs=vocabs)
    train_data, val_data = train_val_split(all_battles, val_frac=val_frac, seed=seed)
    val_jax = _to_jax(val_data)
    typer.echo(
        f"  {len(train_data)} train / {len(val_data)} val compositions  "
        f"({int(train_data.totals.sum()):,} + {int(val_data.totals.sum()):,} battles)"
    )

    key = jax.random.PRNGKey(seed)
    model = BrawlModel(
        n_events=vocabs.n_events,
        n_modes=vocabs.n_modes,
        n_chars=vocabs.n_chars,
        n_classes=vocabs.n_classes,
        n_ranges=vocabs.n_ranges,
        n_destructs=vocabs.n_destructs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        key=key,
    )
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    typer.echo(f"  {n_params:,} parameters")

    steps_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch  # one epoch warm-up
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step = make_step_fn(optimizer)

    ckpt_dir = out.parent / "checkpoints"
    if checkpoint_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    train_key = jax.random.PRNGKey(seed + 1)
    best_val_loss = float("inf")

    epoch_lines: list[str] = []
    epochs_x: list[int] = []
    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_acc_hist: list[float] = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        batches = _iter_batches(train_data, batch_size, rng)
        batch_losses: list[float] = []
        for batch in batches:
            train_key, step_key = jax.random.split(train_key)
            model, opt_state, loss = step(model, opt_state, batch, step_key)
            batch_losses.append(float(loss))

        val_loss = float(bce_loss(model, val_jax))
        val_acc = float(accuracy(model, val_jax))
        mean_train = float(np.mean(batch_losses))
        marker = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(out, model)
            marker = " ✓"

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch:04d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, model)

        epochs_x.append(epoch)
        train_loss_hist.append(mean_train)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        epoch_lines.append(
            f"Epoch {epoch:3d}  train={mean_train:.4f}"
            f"  val={val_loss:.4f}  acc={val_acc:.3f}{marker}"
        )

        _render_dashboard(epoch_lines, epochs_x, train_loss_hist, val_loss_hist, val_acc_hist)

    curves_path = out.parent / "train_curves.png"
    _save_curves(curves_path, epochs_x, train_loss_hist, val_loss_hist, val_acc_hist)
    print(f"\nBest val loss: {best_val_loss:.4f}  → {out}")
    print(f"Training curves  → {curves_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
