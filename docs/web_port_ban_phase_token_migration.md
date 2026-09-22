# Web port migration: two-state ban-phase token

## What changed on the Python side

`src/geneus/draft/env.py` used to give every ban turn the same `BAN_PHASE`
turn token (index 0), regardless of which team was banning. That was a bug:
the acting player always knows which team they are (and therefore whether
they're about to get the first pick or the sixth), but the character
observation during bans never reveals team identity either (own bans only,
never which team they belong to) — so the network had no way to learn
different ban strategies for "I pick first" vs "I pick last".

Fix: `BAN_PHASE` is now split into two tokens, keyed by team:

```python
BAN_PHASE_FIRST_PICK = 0   # Team A bans — this team gets the first pick
BAN_PHASE_SIXTH_PICK = 1   # Team B bans — this team gets the sixth (last) pick
PICK_1 = 2
PICK_2 = 3
PICK_3 = 4
PICK_4 = 5
PICK_5 = 6
PICK_6 = 7
N_DRAFT_TOKENS = 8   # was 7
```

`TURN_SCHEDULE`'s six ban entries now carry `BAN_PHASE_FIRST_PICK` for team
A's three bans and `BAN_PHASE_SIXTH_PICK` for team B's three bans (previously
all six were `BAN_PHASE`). All six pick tokens shifted by +1.

`src/pick/score.py::_build_obs` (the human-facing CLI's turn-token builder,
which the web port's `buildObs` in `web/src/lib/pick/obs.ts` is a line-for-line
TS port of) was updated to pick the right ban token using `ally_first` (the
coin-flip result), the same way it already resolves `PICKED_A`/`PICKED_B`:

```python
if phase < 3:      # ally bans
    acting_is_team_a = ally_first
    turn_token = BAN_PHASE_FIRST_PICK if acting_is_team_a else BAN_PHASE_SIXTH_PICK
elif phase < 6:     # enemy bans
    acting_is_team_a = not ally_first
    turn_token = BAN_PHASE_FIRST_PICK if acting_is_team_a else BAN_PHASE_SIXTH_PICK
```

`src/pick/export_onnx.py` was updated minimally so Python's own test suite
still collects and passes (it emits both `BAN_PHASE_FIRST_PICK` and
`BAN_PHASE_SIXTH_PICK` into the manifest `draft_state` block instead of the
old single `BAN_PHASE` key). **This does not touch anything under `web/`.**

## Blocking dependency: existing checkpoints are now invalid

`DraftQNetwork.draft_turn_emb` is an `Embedding(N_DRAFT_TOKENS, d_model)`
table, so this change resizes it from 7 to 8 rows. Every previously trained
artifact will fail to deserialize against the new code:

- `data/draft_model/draft_q.eqx` and everything under
  `data/draft_model/checkpoints/`
- any other draft checkpoint dir (e.g. `data/draft_myt2_20260916/` if it's a
  draft-model checkpoint)
- `web/static/models/draft_q.onnx` (the embedding table is baked into the
  ONNX graph)

`eqx.tree_deserialise_leaves` hard-fails on shape mismatch, so
`train --resume-from`, `eval`, `eval-sweep`, and `export_onnx`'s
`load_context` (which eagerly deserializes `draft_ckpt`) are all dead until a
**fresh draft-model training run** produces a checkpoint with the new
8-token embedding table. Practically: this whole migration is gated on
running `uv run python -m geneus.draft.train train ...` to completion before
`export_onnx.py` can regenerate anything.

## What needs to change under `web/`

1. **`web/src/lib/onnx/manifest.ts`** — the `Manifest['draft_state']` type
   still declares `BAN_PHASE: number`. Replace with:
   ```ts
   BAN_PHASE_FIRST_PICK: number;
   BAN_PHASE_SIXTH_PICK: number;
   ```
   (matches the new `export_data()` manifest shape in `export_onnx.py`.)

2. **`web/src/lib/pick/obs.ts`** (`buildObs`, lines ~21-67) — this is a
   direct port of `score.py::_build_obs` and has the exact same bug: it reads
   `BAN_PHASE` off `draftState` and uses it unconditionally for both the
   `phase < 3` (ally bans) and `phase < 6` (enemy bans) branches, ignoring
   `allyFirst`. Port the fixed Python logic:
   ```ts
   const { AVAILABLE, GLOBALLY_BANNED, PICKED_A, PICKED_B, LOCALLY_BANNED,
           BAN_PHASE_FIRST_PICK, BAN_PHASE_SIXTH_PICK, TURN_SCHEDULE } = draftState;
   ...
   if (phase < 3) {
       for (const ci of allyBanCharIdxs) obs[ci] = GLOBALLY_BANNED;
       turnToken = allyFirst ? BAN_PHASE_FIRST_PICK : BAN_PHASE_SIXTH_PICK;
   } else if (phase < 6) {
       for (const ci of enemyBanCharIdxs) obs[ci] = GLOBALLY_BANNED;
       turnToken = allyFirst ? BAN_PHASE_SIXTH_PICK : BAN_PHASE_FIRST_PICK;
   } else {
       ...unchanged...
   }
   ```

3. **`web/src/lib/stress-test/simulate.ts`** (comment only, lines 6-15) — the
   header comment explicitly documents the old (buggy) behavior: *"BAN_PHASE
   is one shared token for all six ban turns... ban order carries no
   information the model can see"*. That claim is no longer true once
   `obs.ts` is fixed — the token now encodes team identity during bans. The
   simulation logic itself needs no change (it already threads `allyFirst`
   into `buildObs` at line 88-100); just correct the comment.

4. **Checked-in generated artifacts are stale and must be regenerated (not
   hand-patched) via `export_onnx.py`, after retraining**:
   - `web/static/models/draft_q.onnx`
   - `web/static/data/manifest.json` (currently has `"BAN_PHASE": 0,
     "N_DRAFT_TOKENS": 7`)
   - `web/tests/fixtures/onnx_fixture.json` (`turn_token` values assume the
     old 7-token scheme)
   - `web/tests/fixtures/logic_fixture.json` (`build_obs` cases have
     `turn_token` values computed by the old, buggy `_build_obs` — every ban
     case needs recomputation, and the fixture-generation code in
     `export_onnx.py::export_logic_fixture` already calls the fixed
     `_build_obs`, so regenerating is sufficient once a retrained checkpoint
     exists)

5. **Tests that will need re-running against regenerated fixtures**:
   `web/src/lib/onnx/onnx.test.ts` and `web/src/lib/pick/obs.test.ts` both
   assert against the fixture files above — expect them to fail until the
   fixtures are regenerated, and then to pass again once `obs.ts` matches
   the new logic.

## Suggested order of operations

1. Retrain the draft Q-network (`uv run python -m geneus.draft.train train
   ...`) to get a checkpoint with the new 8-row embedding table. Consider
   using the new `--eval-every`/`--eval-episodes`/`--eval-temp` options to
   watch win-rate-vs-random during the run.
2. Fix `web/src/lib/onnx/manifest.ts` and `web/src/lib/pick/obs.ts` (source
   changes, independent of having a checkpoint).
3. Run `uv run python -m pick.export_onnx` (or whatever `just export-onnx`
   wraps) against the new checkpoint to regenerate `web/static/models/*.onnx`,
   `web/static/data/manifest.json`, and the two `web/tests/fixtures/*.json`
   files.
4. Fix the stale comment in `simulate.ts`.
5. Run the web test suite (`bun test` or equivalent) to confirm
   `onnx.test.ts` and `obs.test.ts` pass against the regenerated fixtures.
