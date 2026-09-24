// Runs one full 12-turn draft (6 bans + 6 picks) between a "player" pool and
// an "adversary" pool through the trained models, then scores the finished
// 3v3 with the terminal model — the atomic unit of the Monte Carlo
// stress-test in ./stressTest.ts. "Ally" always means "player" here, matching
// src/lib/pick's convention (the human ally in the interactive tool).
//
// Turn order: player bans at phases 0-2, adversary bans at phases 3-5,
// *regardless* of which team ends up first-picking — only the turn token fed
// to the model (BAN_PHASE_FIRST_PICK vs BAN_PHASE_SIXTH_PICK, resolved from
// allyFirst in buildObs) encodes which side is banning; the observed
// character states during bans still only ever reveal each side's own bans
// (env.py's get_player_observed_states), never which team they belong to.
// Picks (phase 6+) *do* care about allyFirst — team A/B there is resolved
// against it via phase.ts's pickIsAlly, matching buildObs's
// `isAlly === allyFirst` rule.
//
// Pick 6 is chosen with the same draftQ policy as every other turn, rather
// than scoring every pick-6 candidate with the terminal model (which is what
// the interactive UI does, at the cost of one session.run per candidate —
// fine for a human picking once, too expensive for thousands of simulated
// drafts). This is faithful to training: the Q-network is trained to
// regress the terminal logit at the last turn (see TERMINAL_SIGN in
// env.py), so draftQ's pick-6 output is already an approximation of the
// same terminal judgment. Only the finished composition is scored with the
// terminal model, once, to get the trial's win-probability outcome.

import { charEncsRowForEvent, flattenCharMeta } from '../onnx/manifest';
import type { EventMeta } from '../onnx/metadata';
import { runDraftQ, runTerminalPick6 } from '../onnx/session';
import type { DraftEngine } from '../pick/engine';
import { buildObs, type BrawlerRef, type PickCharEntry } from '../pick/obs';
import { pickIsAlly } from '../pick/phase';
import { banValidMask, pickValidMask } from './mask';
import { sampleMaskedSoftmax, type Rng } from './rng';

export interface DraftSimParams {
	engine: DraftEngine;
	event: EventMeta;
	allyFirst: boolean;
	/** Brawler ids the player is willing to draft from. */
	playerPoolIds: ReadonlySet<number>;
	/** Brawler ids the simulated adversary is willing to draft from. */
	adversaryPoolIds: ReadonlySet<number>;
	/** Softmax temperature for masked action sampling — low = near-optimal. */
	temperature: number;
	rng: Rng;
}

export interface DraftSimResult {
	/** P(player team wins), from the terminal model on the finished 3v3. */
	pPlayerWin: number;
	playerPickIds: number[];
	adversaryPickIds: number[];
}

export async function simulateOneDraft(params: DraftSimParams): Promise<DraftSimResult> {
	const { engine, event, allyFirst, playerPoolIds, adversaryPoolIds, temperature, rng } = params;
	const { manifest, metadata } = engine;
	const nChars = manifest.n_chars;
	const h = manifest.h_terminal;
	const turnSchedule = manifest.draft_state.TURN_SCHEDULE;
	const charEncsRow = charEncsRowForEvent(engine.charEncsAll, manifest, event.event_idx);

	// metadata.brawlers[i] is the brawler at char_idx i (export_metadata's invariant).
	const brawlers: BrawlerRef[] = metadata.brawlers.map((b) => ({ id: b.id, charIdx: b.char_idx }));
	const charIdxOf = new Map(brawlers.map((b) => [b.id, b.charIdx]));
	const toCharIdxs = (ids: ReadonlySet<number>): Set<number> => {
		const out = new Set<number>();
		for (const id of ids) {
			const ci = charIdxOf.get(id);
			if (ci === undefined) throw new Error(`simulateOneDraft: unknown brawler id ${id}`);
			out.add(ci);
		}
		return out;
	};
	const playerPoolCharIdxs = toCharIdxs(playerPoolIds);
	const adversaryPoolCharIdxs = toCharIdxs(adversaryPoolIds);

	const playerBans: number[] = [];
	const adversaryBans: number[] = [];
	const picks: PickCharEntry[] = []; // char_idx space; isAlly === player

	async function chooseAction(
		phase: number,
		actingLocalPoolIds: ReadonlySet<number>,
		validMask: Uint8Array
	): Promise<number> {
		const { obs, turnToken } = buildObs(
			manifest.draft_state,
			nChars,
			playerBans,
			adversaryBans,
			picks,
			phase,
			allyFirst,
			{
				localPoolIds: actingLocalPoolIds,
				brawlers
			}
		);
		const qValues = await runDraftQ(engine.draftQSession, charEncsRow, obs, turnToken, nChars, h);
		return sampleMaskedSoftmax(qValues, validMask, temperature, rng);
	}

	for (let phase = 0; phase < 3; phase++) {
		const ci = await chooseAction(phase, playerPoolIds, banValidMask(nChars, playerBans));
		playerBans.push(ci);
	}
	for (let phase = 3; phase < 6; phase++) {
		const ci = await chooseAction(phase, adversaryPoolIds, banValidMask(nChars, adversaryBans));
		adversaryBans.push(ci);
	}

	for (let pickIdx = 0; pickIdx < 6; pickIdx++) {
		const phase = 6 + pickIdx;
		const actingIsPlayer = pickIsAlly(pickIdx, allyFirst, turnSchedule);
		const actingPoolIds = actingIsPlayer ? playerPoolIds : adversaryPoolIds;
		const actingPoolCharIdxs = actingIsPlayer ? playerPoolCharIdxs : adversaryPoolCharIdxs;
		const bannedCharIdxs = [...playerBans, ...adversaryBans];
		const pickedCharIdxs = picks.map((p) => p.charIdx);
		const mask = pickValidMask(nChars, bannedCharIdxs, pickedCharIdxs, actingPoolCharIdxs);
		const ci = await chooseAction(phase, actingPoolIds, mask);
		picks.push({ isAlly: actingIsPlayer, charIdx: ci });
	}

	const teamACharIdxs: number[] = [];
	const teamBCharIdxs: number[] = [];
	picks.forEach((p, pickIdx) => {
		const [, team] = turnSchedule[6 + pickIdx];
		(team === 'A' ? teamACharIdxs : teamBCharIdxs).push(p.charIdx);
	});

	const logit = await runTerminalPick6(
		engine.terminalSession,
		event.event_idx,
		event.mode_idx,
		Int32Array.from(teamACharIdxs),
		flattenCharMeta(teamACharIdxs.map((ci) => manifest.char_meta_table[ci])),
		Int32Array.from(teamBCharIdxs),
		flattenCharMeta(teamBCharIdxs.map((ci) => manifest.char_meta_table[ci]))
	);
	// logit is the team-A-wins logit (session.ts convention); flip for a team-B ally.
	const pPlayerWin = allyFirst ? 1 / (1 + Math.exp(-logit)) : 1 / (1 + Math.exp(logit));

	const idOfCharIdx = (ci: number): number => metadata.brawlers[ci].id;
	return {
		pPlayerWin,
		playerPickIds: picks.filter((p) => p.isAlly).map((p) => idOfCharIdx(p.charIdx)),
		adversaryPickIds: picks.filter((p) => !p.isAlly).map((p) => idOfCharIdx(p.charIdx))
	};
}
