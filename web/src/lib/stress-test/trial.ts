// Per-trial setup: assigns the trial's event and coin flip. Stratified, not
// drawn — trialIndex is mapped deterministically onto every (event,
// allyFirst) cell in turn (see setupTrial below) so map/side variance, which
// dominates the outcome far more than action-sampling noise, is eliminated
// by design rather than left to i.i.d. chance. The returned `rng` carries no
// consumed draws from this step; it's handed back purely so the caller can
// keep consuming it for reproducible-per-(seed, trialIndex) action sampling
// in simulate.ts.
//
// The adversary's brawler pool is no longer set up here: the deployed
// draft_q model is now trained on the full local-pool size range up to
// every brawler (see train.py's `pool_max` defaulting to `vocabs.n_chars`),
// so the stress test assumes a near-optimal adversary with the entire
// roster available rather than marginalizing over a sampled adversary pool
// — see stressTest.ts's header for the full rationale.
//
// Pure aside from reading `config.events` — no ONNX session involved — so
// it's directly unit-testable.

import type { EventMeta } from '../onnx/metadata';
import { makeTrialRng, type Rng } from './rng';

export interface TrialConfig {
	events: EventMeta[];
	seed: number;
}

export interface TrialSetup {
	event: EventMeta;
	allyFirst: boolean;
	/** Untouched by setup — consume it for action sampling. */
	rng: Rng;
}

export function setupTrial(config: TrialConfig, trialIndex: number): TrialSetup {
	const nEvents = config.events.length;
	if (nEvents === 0) throw new Error('setupTrial: events must be non-empty');
	const rng = makeTrialRng(config.seed, trialIndex);

	const cell = trialIndex % (nEvents * 2);
	const event = config.events[Math.floor(cell / 2)];
	const allyFirst = cell % 2 === 0;

	return { event, allyFirst, rng };
}
