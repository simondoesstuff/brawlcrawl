"""Model loading and inference for draft scoring."""

import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from geneus.data import Vocabs, _build_char_meta_lookup, load_vocabs
from geneus.model import BrawlModel


@dataclass(frozen=True)
class BrawlerInfo:
    id: int
    name: str
    brawler_class: str
    char_idx: int
    meta: tuple[int, int, int]  # (class_idx, range_idx, destruct_idx)


@dataclass(frozen=True)
class EventInfo:
    id: int
    mode: str
    mode_id: int
    map_name: str
    event_idx: int
    mode_idx: int


@dataclass
class DraftContext:
    model: BrawlModel
    brawlers: list[BrawlerInfo]          # sorted by char_idx
    events: list[EventInfo]
    brawler_names: list[str]             # all names, sorted alphabetically, for fuzzy lookup
    map_names: list[str]                 # all map names, for fuzzy lookup
    pickrates: dict[int, dict[int, float]]  # char_id → event_id → pickrate z-score
    _char_idxs: jax.Array               # [n_chars] — parallel with brawlers
    _char_metas: jax.Array              # [n_chars, 3]
    _brawler_by_char_idx: dict[int, BrawlerInfo]
    _brawler_by_name: dict[str, BrawlerInfo]


def load_context(
    data_dir: Path,
    checkpoint_path: Path,
    embed_dim: int = 32,
    hidden_dim: int = 64,
) -> DraftContext:
    vocabs = load_vocabs(data_dir)
    char_meta_lookup = _build_char_meta_lookup(data_dir, vocabs)

    bclass: list[dict] = json.loads((data_dir / "brawler_class.json").read_text())
    brawlers: list[BrawlerInfo] = []
    for b in bclass:
        bid = b["id"]
        if bid in vocabs.char_to_idx and bid in char_meta_lookup:
            brawlers.append(BrawlerInfo(
                id=bid,
                name=b["name"],
                brawler_class=b["class"],
                char_idx=vocabs.char_to_idx[bid],
                meta=char_meta_lookup[bid],
            ))
    brawlers.sort(key=lambda b: b.char_idx)

    events_raw: list[dict] = json.loads((data_dir / "events.json").read_text())
    events: list[EventInfo] = []
    for e in events_raw:
        if e["id"] in vocabs.event_to_idx:
            events.append(EventInfo(
                id=e["id"],
                mode=e["mode"],
                mode_id=e["modeId"],
                map_name=e["map"],
                event_idx=vocabs.event_to_idx[e["id"]],
                mode_idx=vocabs.mode_to_idx[e["modeId"]],
            ))

    pickrates_raw: list[dict] = json.loads(
        (data_dir / "pickrates_leg1_20260827.json").read_text()
    )
    pickrates: dict[int, dict[int, float]] = {}
    for entry in pickrates_raw:
        pickrates.setdefault(entry["char_id"], {})[entry["event_id"]] = entry["z_score"]

    key = jax.random.PRNGKey(0)
    model = BrawlModel(
        n_events=vocabs.n_events,
        n_modes=vocabs.n_modes,
        n_chars=vocabs.n_chars,
        n_classes=vocabs.n_classes,
        n_ranges=vocabs.n_ranges,
        n_destructs=vocabs.n_destructs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout_p=0.3,
        key=key,
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)

    char_idxs = jnp.array([b.char_idx for b in brawlers])
    char_metas = jnp.array([list(b.meta) for b in brawlers])

    return DraftContext(
        model=model,
        brawlers=brawlers,
        events=events,
        brawler_names=sorted(b.name for b in brawlers),
        map_names=[e.map_name for e in events],
        pickrates=pickrates,
        _char_idxs=char_idxs,
        _char_metas=char_metas,
        _brawler_by_char_idx={b.char_idx: b for b in brawlers},
        _brawler_by_name={b.name: b for b in brawlers},
    )


def score_map(ctx: DraftContext, event: EventInfo) -> list[tuple[BrawlerInfo, float]]:
    """Predict per-brawler z-scores on a map. Returns sorted descending (best first)."""
    e_idx = jnp.array(event.event_idx)
    m_idx = jnp.array(event.mode_idx)

    scores = jax.vmap(
        lambda ci, cm: ctx.model.predict_winrate(e_idx, m_idx, ci, cm)
    )(ctx._char_idxs, ctx._char_metas)

    results = [
        (ctx._brawler_by_char_idx[int(ci)], float(s))
        for ci, s in zip(ctx._char_idxs, scores)
    ]
    return sorted(results, key=lambda x: x[1], reverse=True)


def score_sixth_pick(
    ctx: DraftContext,
    event: EventInfo,
    partial_team: list[BrawlerInfo],
    full_team: list[BrawlerInfo],
) -> list[tuple[BrawlerInfo, float]]:
    """Score each candidate as the 6th pick that completes partial_team.

    partial_team (2 brawlers) is treated as team A; full_team (3) as team B.
    Returns (brawler, win_prob) sorted descending — best picks for partial_team first.
    """
    excluded_ids = {b.id for b in partial_team + full_team}
    candidates = [b for b in ctx.brawlers if b.id not in excluded_ids]

    e_idx = jnp.array(event.event_idx)
    m_idx = jnp.array(event.mode_idx)
    b_chars = jnp.array([b.char_idx for b in full_team])
    b_meta = jnp.array([list(b.meta) for b in full_team])
    p_base_chars = jnp.array([b.char_idx for b in partial_team])
    p_base_meta = jnp.array([list(b.meta) for b in partial_team])

    def _score(ci: jax.Array, cm: jax.Array) -> jax.Array:
        a_chars = jnp.concatenate([p_base_chars, ci[None]])
        a_meta = jnp.concatenate([p_base_meta, cm[None]])
        logit = ctx.model(e_idx, m_idx, a_chars, a_meta, b_chars, b_meta)
        return jax.nn.sigmoid(logit)

    cand_chars = jnp.array([b.char_idx for b in candidates])
    cand_metas = jnp.array([list(b.meta) for b in candidates])
    probs = jax.vmap(_score)(cand_chars, cand_metas)

    results = [(b, float(p)) for b, p in zip(candidates, probs)]
    return sorted(results, key=lambda x: x[1], reverse=True)
