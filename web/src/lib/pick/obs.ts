// Port of src/pick/score.py::_build_obs — builds the character observation
// vector and turn token for the acting player. Operates entirely in
// char_idx-space except `localPoolIds`, which (matching the Python) is
// expressed in brawler-id-space: the caller passes `brawlers` (id + charIdx
// pairs, e.g. metadata.brawlers) so this function can translate.

import type { Manifest } from '../onnx/manifest';

type DraftStateConsts = Manifest['draft_state'];

export interface PickCharEntry {
	isAlly: boolean;
	charIdx: number;
}

export interface BrawlerRef {
	id: number;
	charIdx: number;
}

export function buildObs(
	draftState: DraftStateConsts,
	nChars: number,
	allyBanCharIdxs: number[],
	enemyBanCharIdxs: number[],
	picks: PickCharEntry[],
	phase: number,
	allyFirst: boolean,
	opts: { localPoolIds?: Set<number> | null; brawlers?: BrawlerRef[] } = {}
): { obs: Int32Array; turnToken: number } {
	const { AVAILABLE, GLOBALLY_BANNED, PICKED_A, PICKED_B, LOCALLY_BANNED, BAN_PHASE, TURN_SCHEDULE } = draftState;
	const obs = new Int32Array(nChars); // defaults to AVAILABLE (0)
	let turnToken: number;

	if (phase < 3) {
		for (const ci of allyBanCharIdxs) obs[ci] = GLOBALLY_BANNED;
		turnToken = BAN_PHASE;
	} else if (phase < 6) {
		for (const ci of enemyBanCharIdxs) obs[ci] = GLOBALLY_BANNED;
		turnToken = BAN_PHASE;
	} else {
		for (const ci of [...allyBanCharIdxs, ...enemyBanCharIdxs]) obs[ci] = GLOBALLY_BANNED;
		for (const { isAlly, charIdx } of picks) obs[charIdx] = isAlly === allyFirst ? PICKED_A : PICKED_B;
		const pickIdx = phase - 6;
		turnToken = TURN_SCHEDULE[6 + pickIdx][0];
	}

	const { localPoolIds, brawlers } = opts;
	if (localPoolIds != null && brawlers != null) {
		for (const b of brawlers) {
			if (obs[b.charIdx] === AVAILABLE && !localPoolIds.has(b.id)) {
				obs[b.charIdx] = LOCALLY_BANNED;
			}
		}
	}

	return { obs, turnToken };
}
