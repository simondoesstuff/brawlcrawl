"""Training loop and CLI for the BrawlModel."""

from datetime import datetime
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
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm import tqdm

from geneus.data import BattleArrays, WinrateArrays, load_battles, load_vocabs, load_winrates, train_val_split
from geneus.model import BrawlModel

app = typer.Typer(add_completion=False)

# Modules whose params should never be weight-decayed: embedding tables (decay
# would uniformly shrink learned per-brawler/per-category vectors) and
# LayerNorm (decay fights the normalization it's meant to provide). Matched by
# attribute name since eqx.nn.Linear and eqx.nn.LayerNorm both name their
# array fields "weight"/"bias" -- ancestry, not the leaf's own name, is what
# distinguishes a LayerNorm's weight from a Linear's.
_NO_DECAY_MODULES = frozenset({
    "embed_map_id", "embed_map_mode", "embed_char",
    "embed_class", "embed_range", "embed_destruct",
    "self_norm", "cross_norm_q", "cross_norm_kv", "ffn_norm",
})
# Leaf names excluded regardless of ancestry: every Linear/attention bias
# (standard AdamW practice) and Swish's beta (a lone learned scalar).
_NO_DECAY_LEAF_NAMES = frozenset({"bias", "beta"})


def _decay_mask(params: object) -> object:
    """AdamW weight-decay mask: True on weight matrices (attention/FFN/
    projection Linear.weight), False on embeddings, LayerNorm weight/bias,
    Linear biases, and Swish.beta."""

    def leaf_mask(path: tuple, leaf: object) -> bool:
        names = {k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)}
        if names & _NO_DECAY_MODULES:
            return False
        last = path[-1]
        if isinstance(last, jax.tree_util.GetAttrKey) and last.name in _NO_DECAY_LEAF_NAMES:
            return False
        return True

    return jax.tree_util.tree_map_with_path(leaf_mask, params)

_DATA_DIR = Path("data")
_MODEL_FILENAME = "model.eqx"


def _put(x: np.ndarray, sharding: jax.sharding.Sharding | None) -> jax.Array:
    return jax.device_put(x, sharding) if sharding is not None else jnp.asarray(x)


def _to_jax(batch: BattleArrays, sharding: jax.sharding.Sharding | None = None) -> BattleArrays:
    return BattleArrays(
        event_idx=_put(batch.event_idx, sharding),
        mode_idx=_put(batch.mode_idx, sharding),
        team_a_chars=_put(batch.team_a_chars, sharding),
        team_a_meta=_put(batch.team_a_meta, sharding),
        team_b_chars=_put(batch.team_b_chars, sharding),
        team_b_meta=_put(batch.team_b_meta, sharding),
        a_wins=_put(batch.a_wins, sharding),
        totals=_put(batch.totals, sharding),
    )


def _to_jax_winrates(w: WinrateArrays, sharding: jax.sharding.Sharding | None = None) -> WinrateArrays:
    return WinrateArrays(
        event_idx=_put(w.event_idx, sharding),
        mode_idx=_put(w.mode_idx, sharding),
        char_idx=_put(w.char_idx, sharding),
        char_meta=_put(w.char_meta, sharding),
        z_scores=_put(w.z_scores, sharding),
    )


def _grow_vocab_filter_spec(f, like_leaf: object) -> object:
    """`--init-from` deserialisation filter: left-align a checkpoint leaf into a
    larger `like_leaf`, growing vocab embedding tables when the roster (brawlers,
    events, ...) has grown since the checkpoint was saved. New rows keep `like_leaf`'s
    freshly-initialized values; any other shape mismatch still raises.

    This is only correct because `geneus.data.load_vocabs` assigns every vocab index
    by ascending raw id (its docstring/comments explain why) — growth always appends
    new ids at the tail, so a checkpoint's rows are always a left-aligned prefix of
    the current vocab's rows. If that monotonic-id assumption is ever violated (an id
    reused, entries resorted, a vocab shrinks), row alignment breaks silently instead
    of raising, since a same-shape leaf is indistinguishable from a correctly-grown one.
    """
    loaded = eqx.default_deserialise_filter_spec(f, like_leaf)
    if (
        isinstance(loaded, jax.Array)
        and isinstance(like_leaf, jax.Array)
        and loaded.ndim >= 1
        and loaded.shape != like_leaf.shape
    ):
        if loaded.shape[1:] != like_leaf.shape[1:] or loaded.shape[0] > like_leaf.shape[0]:
            raise RuntimeError(
                f"--init-from checkpoint leaf shape {loaded.shape} is not a "
                f"prefix-compatible vocab growth of the target shape {like_leaf.shape} "
                "(trailing dims must match and the checkpoint must not be larger)."
            )
        typer.echo(f"  extending vocab embedding {loaded.shape} -> {like_leaf.shape} (new rows randomly initialized)")
        return like_leaf.at[: loaded.shape[0]].set(loaded)
    return loaded


def bce_loss(
    model: BrawlModel,
    batch: BattleArrays,
    key: jax.Array | None = None,
    label_smoothing: float = 0.0,
) -> jax.Array:
    """Total-weighted BCE. Reconstructs the full Bernoulli likelihood.

    `label_smoothing` (alpha) applies add-alpha (Laplace) smoothing to each
    composition's empirical win rate: `p = (a_wins + alpha) / (totals + 2*alpha)`
    instead of the raw MLE `a_wins / totals`. Most compositions here have only
    1-3 recorded battles (median 2), so the raw rate is closer to a coin flip
    than a real estimate -- alpha > 0 pulls low-count rows toward 0.5 in
    proportion to how little evidence they carry, without ever flipping a
    unanimous result across the 0.5 decision boundary.
    """
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
    p = (batch.a_wins.astype(jnp.float32) + label_smoothing) / (totals + 2.0 * label_smoothing)
    bce = -(jax.nn.log_sigmoid(logits) * p + jax.nn.log_sigmoid(-logits) * (1.0 - p))
    return (bce * totals).sum() / totals.sum()


def winrate_loss(
    model: BrawlModel,
    batch: WinrateArrays,
    key: jax.Array | None = None,
) -> jax.Array:
    """MSE between predicted and z-score-normalized per-(character, map) win rates."""
    if key is not None:
        batch_keys = jax.random.split(key, batch.event_idx.shape[0])
        preds = jax.vmap(
            lambda ev, mo, ch, cm, k: model.predict_winrate(ev, mo, ch, cm, key=k)
        )(batch.event_idx, batch.mode_idx, batch.char_idx, batch.char_meta, batch_keys)
    else:
        preds = jax.vmap(model.predict_winrate)(
            batch.event_idx, batch.mode_idx, batch.char_idx, batch.char_meta
        )
    return jnp.mean((preds - batch.z_scores) ** 2)


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


def make_step_fn(
    optimizer: optax.GradientTransformation,
    winrate_weight: float = 0.1,
    label_smoothing: float = 0.0,
):
    """Return a jit-compiled training step combining BCE and winrate auxiliary losses."""

    @eqx.filter_jit
    def step(
        model: BrawlModel,
        opt_state: optax.OptState,
        battle_batch: BattleArrays,
        winrate_batch: WinrateArrays,
        key: jax.Array,
    ) -> tuple[BrawlModel, optax.OptState, jax.Array, jax.Array]:
        def loss_fn(m: BrawlModel, k: jax.Array) -> jax.Array:
            k1, k2 = jax.random.split(k)
            return bce_loss(m, battle_batch, k1, label_smoothing) + winrate_weight * winrate_loss(m, winrate_batch, k2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, key)
        updates, new_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_state, loss

    return step


def _iter_batches(
    arrays: BattleArrays,
    winrates: WinrateArrays,
    batch_size: int,
    rng: np.random.Generator,
    n_devices: int = 1,
    sharding: jax.sharding.Sharding | None = None,
) -> list[tuple[BattleArrays, WinrateArrays]]:
    """Cut `arrays` into `batch_size`-ish chunks and place them on `sharding`.

    Each chunk's size is truncated down to a multiple of `n_devices` (dropping
    a few leftover rows at most) so it shards evenly across the data-parallel
    mesh axis — `jax.device_put` with a sharded `PartitionSpec` requires exact
    divisibility. A chunk that truncates to 0 (only possible on the final,
    smallest chunk) is dropped.
    """
    perm = rng.permutation(len(arrays))
    batches = []
    for start in range(0, len(arrays), batch_size):
        idx = perm[start : start + batch_size]
        n = len(idx) - (len(idx) % n_devices)
        if n == 0:
            continue
        idx = idx[:n]
        battle_batch = _to_jax(BattleArrays(
            event_idx=arrays.event_idx[idx],
            mode_idx=arrays.mode_idx[idx],
            team_a_chars=arrays.team_a_chars[idx],
            team_a_meta=arrays.team_a_meta[idx],
            team_b_chars=arrays.team_b_chars[idx],
            team_b_meta=arrays.team_b_meta[idx],
            a_wins=arrays.a_wins[idx],
            totals=arrays.totals[idx],
        ), sharding)
        # Sample a winrate mini-batch with replacement (winrate data << steps × batch_size).
        wr_idx = rng.integers(0, len(winrates.event_idx), size=n)
        wr_batch = _to_jax_winrates(WinrateArrays(
            event_idx=winrates.event_idx[wr_idx],
            mode_idx=winrates.mode_idx[wr_idx],
            char_idx=winrates.char_idx[wr_idx],
            char_meta=winrates.char_meta[wr_idx],
            z_scores=winrates.z_scores[wr_idx],
        ), sharding)
        batches.append((battle_batch, wr_batch))
    return batches


def _colored_marker(color: str) -> plt.colorize:
    return plt.colorize("━", pixel=plt.pixel(color))


def _render_dashboard(
    epoch_lines: list[str],
    epochs_x: list[int],
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    winrate_losses: list[float],
) -> None:
    fig = plt.figure
    fig.clear()
    fig.subplots(1, 2)

    sp_loss = fig.subplot(1, 1)
    sp_loss.title("BCE Loss")
    s_train = sp_loss.signal(epochs_x, train_losses,   marker=_colored_marker("cyan"))
    s_val   = sp_loss.signal(epochs_x, val_losses,     marker=_colored_marker("red"))
    s_wr    = sp_loss.signal(epochs_x, winrate_losses, marker=_colored_marker("yellow"))
    s_train.label("train")
    s_val.label("val")
    s_wr.label("winrate")
    sp_loss.draw(s_train)
    sp_loss.draw(s_val)
    sp_loss.draw(s_wr)
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
    winrate_losses: list[float],
) -> None:
    fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_x, train_losses,   label="train",   color="tab:blue")
    ax1.plot(epochs_x, val_losses,     label="val",     color="tab:red")
    ax1.plot(epochs_x, winrate_losses, label="winrate", color="tab:orange")
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
    *,
    data_dir: Annotated[Path, typer.Option(help="Data directory (vocab/reference files: events.json, brawler_class.json, etc.)")] = _DATA_DIR,
    battles_file: Annotated[Path, typer.Option(help="Path to the crawl battles JSON file")],
    winrates_file: Annotated[
        Path | None,
        typer.Option(help="Path to the per-char winrate JSON file (default: battles_file with its 'crawl_' prefix swapped for 'winrates_')"),
    ] = None,
    out: Annotated[Path, typer.Option(help="Output directory for the best model, checkpoints, and training curves (e.g. data/terminal_myt2_20260916)")],
    init_from: Annotated[
        Path | None,
        typer.Option(help="Warm-start model weights from an existing .eqx checkpoint before training on --battles-file. Optimizer/scheduler state is always reinitialized, not restored."),
    ] = None,
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 80,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 512,
    lr: Annotated[float, typer.Option(help="Peak learning rate")] = 1e-3,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay")] = 1e-4,
    d_model: Annotated[int, typer.Option(help="Embedding / transformer dimension")] = 56,
    n_heads: Annotated[int, typer.Option(help="Attention heads per block")] = 4,
    n_blocks: Annotated[int, typer.Option(help="Self-attn + cross-attn block pairs")] = 2,
    dropout_p: Annotated[float, typer.Option(help="Dropout probability (char + FFN)")] = 0.4,
    val_frac: Annotated[float, typer.Option(help="Validation fraction")] = 0.15,
    checkpoint_every: Annotated[int, typer.Option(help="Periodic checkpoint interval in epochs (0=off)")] = 20,
    log_every: Annotated[
        int,
        typer.Option(help="Log running train loss + val loss/acc every N steps within an epoch, in addition to the once-per-epoch summary (0=off). Each log runs a full val-set forward pass, so smaller values cost real throughput."),
    ] = 100,
    winrate_weight: Annotated[float, typer.Option(help="Weight of the per-char winrate auxiliary loss")] = 0.1,
    label_smoothing: Annotated[
        float,
        typer.Option(help="Add-alpha (Laplace) smoothing on each composition's empirical win rate: p = (a_wins + alpha) / (totals + 2*alpha). Most compositions have only 1-3 recorded battles, so alpha > 0 pulls those targets toward 0.5 instead of letting the model fit near-coin-flip noise."),
    ] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    if winrates_file is None:
        winrates_file = battles_file.with_name(battles_file.name.replace("crawl_", "winrates_", 1))

    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(devices, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    typer.echo(f"Devices: {n_devices} × {devices[0].platform} ({[str(d) for d in devices]})")

    typer.echo(f"Loading data ({battles_file})...")
    vocabs = load_vocabs(data_dir)
    all_battles = load_battles(data_dir, vocabs=vocabs, battles_file=battles_file)
    train_data, val_data = train_val_split(all_battles, val_frac=val_frac, seed=seed)
    val_jax = _to_jax(val_data)
    winrate_data = load_winrates(data_dir, vocabs=vocabs, winrates_file=winrates_file)
    winrate_jax = _to_jax_winrates(winrate_data)
    typer.echo(
        f"  {len(train_data)} train / {len(val_data)} val compositions  "
        f"({int(train_data.totals.sum()):,} + {int(val_data.totals.sum()):,} battles)"
    )
    typer.echo(f"  {len(winrate_data)} (char, map) winrate targets")

    key = jax.random.PRNGKey(seed)
    model = BrawlModel(
        n_events=vocabs.n_events,
        n_modes=vocabs.n_modes,
        n_chars=vocabs.n_chars,
        n_classes=vocabs.n_classes,
        n_ranges=vocabs.n_ranges,
        n_destructs=vocabs.n_destructs,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout_p=dropout_p,
        key=key,
    )
    if init_from is not None:
        typer.echo(f"Warm-starting weights from {init_from} (optimizer/scheduler reset)...")
        model = eqx.tree_deserialise_leaves(init_from, model, filter_spec=_grow_vocab_filter_spec)
    # Replicate the model across every device in the mesh: data parallelism
    # shards batches, not parameters, so every device needs its own full copy.
    params, static = eqx.partition(model, eqx.is_array)
    model = eqx.combine(jax.device_put(params, replicated_sharding), static)
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
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay, mask=_decay_mask)
    opt_state = jax.device_put(optimizer.init(eqx.filter(model, eqx.is_array)), replicated_sharding)
    step = make_step_fn(optimizer, winrate_weight=winrate_weight, label_smoothing=label_smoothing)

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / _MODEL_FILENAME
    ckpt_dir = out / "checkpoints"
    if checkpoint_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Appended across runs (e.g. --init-from resumes into the same --out), so
    # a header marks where each run starts. Flushed after every write so the
    # file stays tail-able live during a long run.
    log_fh = (out / "train.log").open("a")
    log_fh.write(
        f"\n=== run started {datetime.now().isoformat(timespec='seconds')}  "
        f"d_model={d_model} n_heads={n_heads} n_blocks={n_blocks} dropout_p={dropout_p}  "
        f"epochs={epochs} batch_size={batch_size} lr={lr} weight_decay={weight_decay}  "
        f"label_smoothing={label_smoothing} devices={n_devices} ===\n"
    )
    log_fh.flush()

    has_val = len(val_data) > 0
    rng = np.random.default_rng(seed)
    train_key = jax.random.PRNGKey(seed + 1)
    best_val_loss = float("inf")

    epoch_lines: list[str] = []
    epochs_x: list[int] = []
    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_acc_hist: list[float] = []
    winrate_loss_hist: list[float] = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        batches = _iter_batches(train_data, winrate_data, batch_size, rng, n_devices, data_sharding)
        batch_losses: list[float] = []
        for step_idx, (battle_batch, wr_batch) in enumerate(batches, start=1):
            train_key, step_key = jax.random.split(train_key)
            step_key = jax.device_put(step_key, replicated_sharding)
            model, opt_state, loss = step(model, opt_state, battle_batch, wr_batch, step_key)
            batch_losses.append(float(loss))

            if log_every > 0 and step_idx % log_every == 0:
                running_train = float(np.mean(batch_losses[-log_every:]))
                if has_val:
                    running_val = float(bce_loss(model, val_jax, label_smoothing=label_smoothing))
                    running_acc = float(accuracy(model, val_jax))
                    step_line = (
                        f"  epoch {epoch:3d} step {step_idx:5d}/{len(batches)}  "
                        f"train={running_train:.4f}  val={running_val:.4f}  acc={running_acc:.3f}"
                    )
                else:
                    step_line = f"  epoch {epoch:3d} step {step_idx:5d}/{len(batches)}  train={running_train:.4f}"
                tqdm.write(step_line)
                log_fh.write(step_line + "\n")
                log_fh.flush()

        mean_train = float(np.mean(batch_losses))
        wr_loss = float(winrate_loss(model, winrate_jax))
        marker = ""

        if has_val:
            val_loss = float(bce_loss(model, val_jax, label_smoothing=label_smoothing))
            val_acc = float(accuracy(model, val_jax))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                eqx.tree_serialise_leaves(model_path, model)
                marker = " ✓"
            log_suffix = f"  val={val_loss:.4f}  acc={val_acc:.3f}  wr={wr_loss:.4f}{marker}"
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            eqx.tree_serialise_leaves(model_path, model)
            log_suffix = f"  wr={wr_loss:.4f}"

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch:04d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, model)

        epochs_x.append(epoch)
        train_loss_hist.append(mean_train)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        winrate_loss_hist.append(wr_loss)
        epoch_line = f"Epoch {epoch:3d}  train={mean_train:.4f}{log_suffix}"
        epoch_lines.append(epoch_line)
        log_fh.write(epoch_line + "\n")
        log_fh.flush()

        _render_dashboard(epoch_lines, epochs_x, train_loss_hist, val_loss_hist, val_acc_hist, winrate_loss_hist)

    curves_path = out / "train_curves.png"
    _save_curves(curves_path, epochs_x, train_loss_hist, val_loss_hist, val_acc_hist, winrate_loss_hist)
    if has_val:
        print(f"\nBest val loss: {best_val_loss:.4f}  → {model_path}")
    else:
        print(f"\nFinal train loss: {mean_train:.4f}  → {model_path}")
    print(f"Training curves  → {curves_path}")
    print(f"Log                → {log_fh.name}")
    log_fh.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
