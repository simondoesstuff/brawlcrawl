"""Data loading, feature extraction, and train/val splitting for the ML pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

_WINRATES_FILE = "winrates_leg1_20260827.json"


@dataclass(frozen=True)
class Vocabs:
    event_to_idx: dict[int, int]
    mode_to_idx: dict[int, int]
    char_to_idx: dict[int, int]
    class_to_idx: dict[str, int]
    range_to_idx: dict[str, int]
    destruct_to_idx: dict[str, int]

    @property
    def n_events(self) -> int:
        return len(self.event_to_idx)

    @property
    def n_modes(self) -> int:
        return len(self.mode_to_idx)

    @property
    def n_chars(self) -> int:
        return len(self.char_to_idx)

    @property
    def n_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def n_ranges(self) -> int:
        return len(self.range_to_idx)

    @property
    def n_destructs(self) -> int:
        return len(self.destruct_to_idx)


@dataclass
class BattleArrays:
    """Parallel numpy arrays; index i is one unique team composition."""

    event_idx: np.ndarray   # [N] int32
    mode_idx: np.ndarray    # [N] int32
    team_a_chars: np.ndarray  # [N, 3] int32
    team_a_meta: np.ndarray   # [N, 3, 3] int32  — axis-2: (class, range, destruct)
    team_b_chars: np.ndarray  # [N, 3] int32
    team_b_meta: np.ndarray   # [N, 3, 3] int32
    a_wins: np.ndarray      # [N] int32
    totals: np.ndarray      # [N] int32

    def __len__(self) -> int:
        return len(self.event_idx)


# Register as a JAX pytree so BattleArrays can cross jit/vmap boundaries.
jax.tree_util.register_pytree_node(
    BattleArrays,
    lambda b: (
        [b.event_idx, b.mode_idx, b.team_a_chars, b.team_a_meta,
         b.team_b_chars, b.team_b_meta, b.a_wins, b.totals],
        None,
    ),
    lambda _, xs: BattleArrays(*xs),
)


@dataclass
class WinrateArrays:
    """Per-(character, event) z-scored win rates for the auxiliary objective.

    z_scores are normalized within each event: z = (wr - μ_event) / σ_event,
    capturing relative character advantage on a map rather than absolute win rate.
    """

    event_idx: np.ndarray  # [M] int32
    mode_idx: np.ndarray   # [M] int32
    char_idx: np.ndarray   # [M] int32
    char_meta: np.ndarray  # [M, 3] int32 — (class_idx, range_idx, destruct_idx)
    z_scores: np.ndarray   # [M] float32

    def __len__(self) -> int:
        return len(self.event_idx)


jax.tree_util.register_pytree_node(
    WinrateArrays,
    lambda w: (
        [w.event_idx, w.mode_idx, w.char_idx, w.char_meta, w.z_scores],
        None,
    ),
    lambda _, xs: WinrateArrays(*xs),
)


def load_vocabs(data_dir: Path = _DATA_DIR) -> Vocabs:
    """Build index vocabs from the current data files.

    `char_to_idx`/`event_to_idx` are assigned by ascending raw brawler/event id, so a
    growing roster appends new ids at the tail without shifting existing indices —
    *only* as long as new ids are always numerically greater than every existing one
    (true for Brawl Stars' monotonically-increasing id assignment so far). This is
    what makes `geneus.train`'s `--init-from` vocab growth
    (`_grow_vocab_filter_spec`) safe: it left-aligns a checkpoint's embedding rows
    into a larger table, which only lines up correctly if indices never get
    reassigned. `class_to_idx`/`range_to_idx`/`destruct_to_idx` sort by *string*
    category value instead — a new category name that sorts alphabetically before an
    existing one would shift every index after it, silently breaking that same
    alignment.
    """
    events: list[dict] = json.loads((data_dir / "events.json").read_text())
    event_to_idx = {e["id"]: i for i, e in enumerate(sorted(events, key=lambda x: x["id"]))}
    mode_to_idx = {m: i for i, m in enumerate(sorted({e["modeId"] for e in events}))}

    brawler_class: list[dict] = json.loads((data_dir / "brawler_class.json").read_text())
    char_to_idx = {x["id"]: i for i, x in enumerate(sorted(brawler_class, key=lambda x: x["id"]))}
    class_to_idx = {c: i for i, c in enumerate(sorted({x["class"] for x in brawler_class}))}

    brawler_range: list[dict] = json.loads((data_dir / "brawler_effective_range.json").read_text())
    range_to_idx = {r: i for i, r in enumerate(sorted({x["range"] for x in brawler_range}))}

    brawler_destruct: list[dict] = json.loads((data_dir / "brawler_destruction.json").read_text())
    destruct_to_idx = {d: i for i, d in enumerate(sorted({x["destruction"] for x in brawler_destruct}))}

    return Vocabs(
        event_to_idx=event_to_idx,
        mode_to_idx=mode_to_idx,
        char_to_idx=char_to_idx,
        class_to_idx=class_to_idx,
        range_to_idx=range_to_idx,
        destruct_to_idx=destruct_to_idx,
    )


def _build_char_meta_lookup(
    data_dir: Path,
    vocabs: Vocabs,
) -> dict[int, tuple[int, int, int]]:
    """Map brawler_id -> (class_idx, range_idx, destruct_idx). Raises if any are missing."""
    brawler_class: list[dict] = json.loads((data_dir / "brawler_class.json").read_text())
    brawler_range: list[dict] = json.loads((data_dir / "brawler_effective_range.json").read_text())
    brawler_destruct: list[dict] = json.loads((data_dir / "brawler_destruction.json").read_text())

    class_map = {x["id"]: vocabs.class_to_idx[x["class"]] for x in brawler_class}
    range_map = {x["id"]: vocabs.range_to_idx[x["range"]] for x in brawler_range}
    destruct_map = {x["id"]: vocabs.destruct_to_idx[x["destruction"]] for x in brawler_destruct}

    all_ids = set(class_map) | set(range_map) | set(destruct_map)
    missing = {
        bid for bid in all_ids
        if bid not in class_map or bid not in range_map or bid not in destruct_map
    }
    if missing:
        raise ValueError(f"Brawlers missing metadata: {missing}")

    return {bid: (class_map[bid], range_map[bid], destruct_map[bid]) for bid in all_ids}


def _check_events_known(event_ids: set[int], vocabs: Vocabs, source: Path, data_dir: Path) -> None:
    """Raise if `source` references event IDs missing from events.json.

    Loaders used to filter these out silently, which meant a crawl dataset
    could get ahead of events.json (e.g. a newly-discovered event ID not yet
    resolved to mode/map) and training would quietly drop those battles
    instead of failing loudly. See docs/dataset_generation.md.
    """
    unknown = (event_ids - {0}) - set(vocabs.event_to_idx)
    if unknown:
        raise ValueError(
            f"{source} references {len(unknown)} event ID(s) not in {data_dir / 'events.json'}: "
            f"{sorted(unknown)}. Run `crawl events-crawl` to refresh events.json before training."
        )


def _check_chars_known(char_ids: set[int], vocabs: Vocabs, source: Path, data_dir: Path) -> None:
    """Raise if `source` references brawler IDs missing from brawler_class.json.

    Same failure mode as `_check_events_known`: a stats source can get ahead of
    brawler_class.json (a newly-released brawler not yet added), and filtering
    those entries out silently would quietly drop training data instead of
    failing loudly.
    """
    unknown = char_ids - set(vocabs.char_to_idx)
    if unknown:
        raise ValueError(
            f"{source} references {len(unknown)} brawler ID(s) not in {data_dir / 'brawler_class.json'}: "
            f"{sorted(unknown)}. Update brawler_class.json (and brawler_effective_range.json / "
            "brawler_destruction.json) before training."
        )


def load_battles(
    data_dir: Path = _DATA_DIR,
    *,
    vocabs: Vocabs | None = None,
    battles_file: Path = _DATA_DIR / "crawl_leg1_20260827.json",
) -> BattleArrays:
    if vocabs is None:
        vocabs = load_vocabs(data_dir)

    char_meta = _build_char_meta_lookup(data_dir, vocabs)

    events: list[dict] = json.loads((data_dir / "events.json").read_text())
    event_mode: dict[int, int] = {e["id"]: e["modeId"] for e in events}

    raw: dict = json.loads(battles_file.read_text())
    _check_events_known({b["event_id"] for b in raw["stats"]}, vocabs, battles_file, data_dir)
    battles: list[dict] = raw["stats"]

    for b in battles:
        for cid in b["team_a"] + b["team_b"]:
            if cid not in char_meta:
                raise ValueError(f"Brawler {cid} has no metadata")
            if cid not in vocabs.char_to_idx:
                raise ValueError(f"Brawler {cid} not in char vocab")

    N = len(battles)
    event_idx = np.empty(N, dtype=np.int32)
    mode_idx = np.empty(N, dtype=np.int32)
    team_a_chars = np.empty((N, 3), dtype=np.int32)
    team_a_meta = np.empty((N, 3, 3), dtype=np.int32)
    team_b_chars = np.empty((N, 3), dtype=np.int32)
    team_b_meta = np.empty((N, 3, 3), dtype=np.int32)
    a_wins_arr = np.empty(N, dtype=np.int32)
    totals_arr = np.empty(N, dtype=np.int32)

    for i, b in enumerate(battles):
        eid = b["event_id"]
        event_idx[i] = vocabs.event_to_idx[eid]
        mode_idx[i] = vocabs.mode_to_idx[event_mode[eid]]

        for j, cid in enumerate(b["team_a"]):
            team_a_chars[i, j] = vocabs.char_to_idx[cid]
            team_a_meta[i, j] = char_meta[cid]

        for j, cid in enumerate(b["team_b"]):
            team_b_chars[i, j] = vocabs.char_to_idx[cid]
            team_b_meta[i, j] = char_meta[cid]

        a_wins_arr[i] = b["a_wins"]
        totals_arr[i] = b["total"]

    return BattleArrays(
        event_idx=event_idx,
        mode_idx=mode_idx,
        team_a_chars=team_a_chars,
        team_a_meta=team_a_meta,
        team_b_chars=team_b_chars,
        team_b_meta=team_b_meta,
        a_wins=a_wins_arr,
        totals=totals_arr,
    )


def load_winrates(
    data_dir: Path = _DATA_DIR,
    *,
    vocabs: Vocabs | None = None,
    winrates_file: Path = _DATA_DIR / _WINRATES_FILE,
) -> WinrateArrays:
    if vocabs is None:
        vocabs = load_vocabs(data_dir)

    char_meta = _build_char_meta_lookup(data_dir, vocabs)

    events: list[dict] = json.loads((data_dir / "events.json").read_text())
    event_mode: dict[int, int] = {e["id"]: e["modeId"] for e in events}

    raw: list[dict] = json.loads(winrates_file.read_text())
    _check_events_known({e["event_id"] for e in raw}, vocabs, winrates_file, data_dir)
    _check_chars_known({e["char_id"] for e in raw}, vocabs, winrates_file, data_dir)

    M = len(raw)
    event_idx = np.empty(M, dtype=np.int32)
    mode_idx = np.empty(M, dtype=np.int32)
    char_idx = np.empty(M, dtype=np.int32)
    char_meta_arr = np.empty((M, 3), dtype=np.int32)
    z_scores = np.empty(M, dtype=np.float32)

    for i, e in enumerate(raw):
        eid = e["event_id"]
        cid = e["char_id"]
        event_idx[i] = vocabs.event_to_idx[eid]
        mode_idx[i] = vocabs.mode_to_idx[event_mode[eid]]
        char_idx[i] = vocabs.char_to_idx[cid]
        char_meta_arr[i] = char_meta[cid]
        z_scores[i] = e["z_score"]

    return WinrateArrays(
        event_idx=event_idx,
        mode_idx=mode_idx,
        char_idx=char_idx,
        char_meta=char_meta_arr,
        z_scores=z_scores,
    )


def train_val_split(
    arrays: BattleArrays,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[BattleArrays, BattleArrays]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arrays))
    n_val = int(len(arrays) * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _slice(idx: np.ndarray) -> BattleArrays:
        return BattleArrays(
            event_idx=arrays.event_idx[idx],
            mode_idx=arrays.mode_idx[idx],
            team_a_chars=arrays.team_a_chars[idx],
            team_a_meta=arrays.team_a_meta[idx],
            team_b_chars=arrays.team_b_chars[idx],
            team_b_meta=arrays.team_b_meta[idx],
            a_wins=arrays.a_wins[idx],
            totals=arrays.totals[idx],
        )

    return _slice(train_idx), _slice(val_idx)
