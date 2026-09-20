// Orchestrates the ONNX sessions + companion data into the same operations
// src/pick/score.py exposes to the CLI (load_context, score_map,
// get_q_values, get_terminal_pick6_scores), but backed by onnxruntime-web
// instead of JAX.

import type * as ort from 'onnxruntime-web';
import { charEncsRowForEvent, flattenCharMeta, type Manifest } from '../onnx/manifest';
import { winrateZ, type Brawler, type EventMeta, type Metadata } from '../onnx/metadata';
import { createSession, runDraftQ, runTerminalPick6 } from '../onnx/session';
import { buildObs, type BrawlerRef, type PickCharEntry } from './obs';
import { pick6TeamSplit } from './phase';

export interface DraftEngine {
	manifest: Manifest;
	metadata: Metadata;
	charEncsAll: Float32Array;
	draftQSession: ort.InferenceSession;
	terminalSession: ort.InferenceSession;
	byId: Map<number, Brawler>;
}

export async function loadEngine(fetchImpl: typeof fetch = fetch): Promise<DraftEngine> {
	const [manifest, metadata, charEncsBuf] = await Promise.all([
		fetchImpl('/data/manifest.json').then((r) => r.json()) as Promise<Manifest>,
		fetchImpl('/data/metadata.json').then((r) => r.json()) as Promise<Metadata>,
		fetchImpl('/data/char_encs.bin').then((r) => r.arrayBuffer())
	]);
	const charEncsAll = new Float32Array(charEncsBuf);
	const [draftQSession, terminalSession] = await Promise.all([
		createSession('/models/draft_q.onnx'),
		createSession('/models/terminal_pick6.onnx')
	]);
	const byId = new Map(metadata.brawlers.map((b) => [b.id, b]));
	return { manifest, metadata, charEncsAll, draftQSession, terminalSession, byId };
}

/** Rank brawlers by winrate z-score for this event (port of `score_map`). */
export function scoreMap(engine: DraftEngine, event: EventMeta): { brawler: Brawler; score: number }[] {
	const results = engine.metadata.brawlers.map((b) => ({
		brawler: b,
		score: winrateZ(engine.metadata, b.id, event.id) ?? 0
	}));
	return results.sort((a, b) => b.score - a.score);
}

export interface QValueParams {
	event: EventMeta;
	allyBanIds: number[];
	enemyBanIds: number[];
	picks: { isAlly: boolean; id: number }[];
	phase: number;
	allyFirst: boolean;
	localPoolIds?: Set<number> | null;
}

/**
 * Compute Q-values [n_chars] for the current draft phase (port of
 * `get_q_values`). Indexed by char_idx, zip with metadata.brawlers to
 * display — `metadata.brawlers[i].char_idx === i`.
 */
export async function getQValues(engine: DraftEngine, params: QValueParams): Promise<Float32Array> {
	const { manifest, metadata, byId } = engine;
	const allyBanCharIdxs = params.allyBanIds.map((id) => byId.get(id)!.char_idx);
	const enemyBanCharIdxs = params.enemyBanIds.map((id) => byId.get(id)!.char_idx);
	const picks: PickCharEntry[] = params.picks.map((p) => ({
		isAlly: p.isAlly,
		charIdx: byId.get(p.id)!.char_idx
	}));
	const brawlers: BrawlerRef[] = metadata.brawlers.map((b) => ({ id: b.id, charIdx: b.char_idx }));

	const { obs, turnToken } = buildObs(
		manifest.draft_state,
		manifest.n_chars,
		allyBanCharIdxs,
		enemyBanCharIdxs,
		picks,
		params.phase,
		params.allyFirst,
		{ localPoolIds: params.localPoolIds ?? null, brawlers }
	);

	const charEncsRow = charEncsRowForEvent(engine.charEncsAll, manifest, params.event.event_idx);
	return runDraftQ(engine.draftQSession, charEncsRow, obs, turnToken, manifest.n_chars, manifest.h_terminal);
}

export interface TerminalPick6Params {
	event: EventMeta;
	/** Exactly 5 picks (the completed draft minus the final pick-6 slot). */
	picks: { isAlly: boolean; id: number }[];
	excludedIds: Set<number>;
	allyFirst: boolean;
}

/**
 * Score pick-6 candidates with the terminal model (port of
 * `get_terminal_pick6_scores`). Returns (brawler, prob) sorted descending by
 * P(ally wins) — always from the ally's perspective, regardless of which
 * team actually picks last (see TURN_SCHEDULE: team B always picks 6th, so
 * when the ally is team B this list *is* the ally's own recommendation, but
 * when the ally is team A it describes what the enemy could pick against
 * the locked-in ally roster — callers should label the header accordingly).
 */
export async function getTerminalPick6Scores(
	engine: DraftEngine,
	params: TerminalPick6Params
): Promise<{ brawler: Brawler; prob: number }[]> {
	const { manifest, metadata, byId } = engine;
	const picks = params.picks.map((p) => ({ isAlly: p.isAlly, charIdx: byId.get(p.id)!.char_idx }));
	const { teamA, teamBPartial } = pick6TeamSplit(picks, manifest.draft_state.TURN_SCHEDULE);

	const teamAChars = Int32Array.from(teamA);
	const teamAMeta = flattenCharMeta(teamA.map((ci) => manifest.char_meta_table[ci]));
	const teamBPartialMetaRows = teamBPartial.map((ci) => manifest.char_meta_table[ci]);

	const candidates = metadata.brawlers.filter((b) => !params.excludedIds.has(b.id));
	const results: { brawler: Brawler; prob: number }[] = [];

	for (const cand of candidates) {
		const teamBChars = Int32Array.from([...teamBPartial, cand.char_idx]);
		const teamBMeta = flattenCharMeta([...teamBPartialMetaRows, manifest.char_meta_table[cand.char_idx]]);
		const logit = await runTerminalPick6(
			engine.terminalSession,
			params.event.event_idx,
			params.event.mode_idx,
			teamAChars,
			teamAMeta,
			teamBChars,
			teamBMeta
		);
		// sigmoid(-logit) = P(team B wins); team B always picks last.
		let prob = 1 / (1 + Math.exp(logit));
		if (params.allyFirst) prob = 1 - prob; // ally is team A -> flip to P(ally wins)
		results.push({ brawler: cand, prob });
	}

	results.sort((a, b) => b.prob - a.prob);
	return results;
}
