// Per-trial setup: draws the trial's event, coin flip, and simulated
// adversary's owned-brawler pool — the values that must line up between a
// baseline run and every candidate-extended run at the same trial index
// (common random numbers), so the *only* thing that can differ between arms
// is the drafted state itself, once the player's own decisions start to
// diverge because their pool grew by one brawler.
//
// Pure aside from reading `engine.metadata` — no ONNX session involved — so
// it's directly unit-testable against a minimal fake DraftEngine.

import type { EventMeta } from '../onnx/metadata';
import type { DraftEngine } from '../pick/engine';
import { makeTrialRng, sampleWithoutReplacement, type Rng } from './rng';

export interface TrialConfig {
	events: EventMeta[];
	/** Fixed adversary pool size, or an inclusive [min, max] range drawn per trial. */
	adversaryPoolSize: number | [number, number];
	seed: number;
}

export interface TrialSetup {
	event: EventMeta;
	allyFirst: boolean;
	adversaryPoolIds: Set<number>;
	/** Advanced past the setup draws — continue consuming it for action sampling. */
	rng: Rng;
}

function resolvePoolSize(size: number | [number, number], nBrawlers: number, rng: Rng): number {
	const [min, max] = typeof size === 'number' ? [size, size] : size;
	const lo = Math.max(0, Math.min(min, nBrawlers));
	const hi = Math.max(lo, Math.min(max, nBrawlers));
	return lo + Math.floor(rng.next() * (hi - lo + 1));
}

export function setupTrial(
	engine: DraftEngine,
	config: TrialConfig,
	trialIndex: number
): TrialSetup {
	if (config.events.length === 0) throw new Error('setupTrial: events must be non-empty');
	const rng = makeTrialRng(config.seed, trialIndex);

	const event = config.events[Math.floor(rng.next() * config.events.length)];
	const allyFirst = rng.next() < 0.5;

	const brawlers = engine.metadata.brawlers;
	const k = resolvePoolSize(config.adversaryPoolSize, brawlers.length, rng);
	const idxs = sampleWithoutReplacement(brawlers.length, k, rng);
	const adversaryPoolIds = new Set(idxs.map((i) => brawlers[i].id));

	return { event, allyFirst, adversaryPoolIds, rng };
}
