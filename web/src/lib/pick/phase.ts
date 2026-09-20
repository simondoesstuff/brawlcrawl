// Port of the pick-order / phase helpers in src/pick/main.py (`_pick_is_ally`,
// `_phase_default_filter`, `_phase_label`, `_ban_excluded`) and
// src/pick/score.py (`_pick6_team_split`). TURN_SCHEDULE is read from
// manifest.json (draft_state.TURN_SCHEDULE) rather than duplicated here, so
// the client can never drift from the model's training-time schedule.

import type { Manifest } from '../onnx/manifest';

export type TurnSchedule = Manifest['draft_state']['TURN_SCHEDULE'];

export interface PickEntry<Id extends number = number> {
	isAlly: boolean;
	id: Id;
}

/** True if pickIdx (0-5) belongs to the ally team. */
export function pickIsAlly(pickIdx: number, allyFirst: boolean, turnSchedule: TurnSchedule): boolean {
	const [, team] = turnSchedule[6 + pickIdx];
	return (team === 'A') === allyFirst;
}

/**
 * Default filter-on state for the given draft phase.
 * Ban phases (0-5): OFF. Pick phases: ON when it's the ally's turn.
 */
export function phaseDefaultFilter(phase: number, allyFirst: boolean, turnSchedule: TurnSchedule): boolean {
	if (phase < 6) return false;
	return pickIsAlly(phase - 6, allyFirst, turnSchedule);
}

export function phaseLabel(phase: number, allyFirst: boolean, turnSchedule: TurnSchedule): string {
	if (phase < 3) return `Ally ban ${phase + 1}`;
	if (phase < 6) return `Enemy ban ${phase - 3 + 1}`;
	const pickIdx = phase - 6;
	const label = pickIsAlly(pickIdx, allyFirst, turnSchedule) ? 'Ally' : 'Enemy';
	return `Pick ${pickIdx + 1}  (${label})`;
}

/**
 * Brawlers excluded from submission at the given phase (id-space, matches
 * `_ban_excluded`). During each team's ban phase only their own bans are
 * excluded — the other team's bans remain submittable (inter-team duplicate
 * ban is allowed). During the pick phase all bans and prior picks are excluded.
 */
export function banExcluded(
	phase: number,
	allyBanIds: number[],
	enemyBanIds: number[],
	picks: PickEntry[]
): Set<number> {
	const excl = new Set<number>(picks.map((p) => p.id));
	if (phase < 3) {
		for (const id of allyBanIds) excl.add(id);
	} else if (phase < 6) {
		for (const id of enemyBanIds) excl.add(id);
	} else {
		for (const id of allyBanIds) excl.add(id);
		for (const id of enemyBanIds) excl.add(id);
	}
	return excl;
}

/** Display exclusion: all banned/picked brawlers, hidden from the ranked grid. */
export function displayExcluded(allyBanIds: number[], enemyBanIds: number[], picks: PickEntry[]): Set<number> {
	const excl = new Set<number>(allyBanIds);
	for (const id of enemyBanIds) excl.add(id);
	for (const p of picks) excl.add(p.id);
	return excl;
}

/**
 * Split 5 picks into model team A (first-picking) and team B by TURN_SCHEDULE.
 * Returns teamA (3 char_idxs) and teamBPartial (2 char_idxs); the 6th pick
 * (TURN_SCHEDULE[11] = team B seat 2) is always the missing slot.
 */
export function pick6TeamSplit(
	picks: { charIdx: number }[],
	turnSchedule: TurnSchedule
): { teamA: number[]; teamBPartial: number[] } {
	const teamA: number[] = [];
	const teamBPartial: number[] = [];
	picks.forEach((p, pickIdx) => {
		const [, team] = turnSchedule[6 + pickIdx];
		(team === 'A' ? teamA : teamBPartial).push(p.charIdx);
	});
	return { teamA, teamBPartial };
}
