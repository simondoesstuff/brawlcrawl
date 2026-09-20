// Port of the z-score and annotation helpers in src/pick/main.py
// (`_z_score_q`, `pick_annotations`, `overview_annotations`).

import { ANN_BRAIN, ANN_SECRET, ANN_X, PICKRATE_HIGH_Z, PICKRATE_LOW_Z, Q_THRESHOLD } from './constants';
import { pickrateZ, type EventMeta, type Metadata } from '../onnx/metadata';

/**
 * Normalize Q-values to z-scores over the currently available brawlers
 * (population std, matching numpy's default ddof=0). Z > 0 means
 * above-average advantage relative to the current choice pool.
 */
export function zScoreQ(qVals: ArrayLike<number>, availableCharIdxs: number[]): Float64Array {
	const n = qVals.length;
	if (availableCharIdxs.length === 0) {
		return Float64Array.from(qVals as ArrayLike<number>);
	}
	let sum = 0;
	for (const i of availableCharIdxs) sum += qVals[i];
	const mean = sum / availableCharIdxs.length;
	let sqSum = 0;
	for (const i of availableCharIdxs) {
		const d = qVals[i] - mean;
		sqSum += d * d;
	}
	const std = Math.sqrt(sqSum / availableCharIdxs.length);
	if (std < 1e-8) return new Float64Array(n);
	const out = new Float64Array(n);
	for (let i = 0; i < n; i++) out[i] = (qVals[i] - mean) / std;
	return out;
}

export interface ScoredBrawler {
	id: number;
	score: number;
}

/**
 * Annotate available brawlers based on Q-value and pickrate.
 * "brain": Q > 0 and below-average pickrate (hidden gem for this state).
 * "overrated": Q <= 0 and high pickrate (overrated by players).
 */
export function pickAnnotations(
	metadata: Metadata,
	event: EventMeta,
	qScores: ScoredBrawler[]
): Map<number, string> {
	const result = new Map<number, string>();
	for (const { id, score } of qScores) {
		const pz = pickrateZ(metadata, id, event.id);
		if (pz === undefined) continue;
		if (score > Q_THRESHOLD && pz < PICKRATE_LOW_Z) result.set(id, ANN_BRAIN);
		else if (score <= Q_THRESHOLD && pz >= PICKRATE_HIGH_Z) result.set(id, ANN_X);
	}
	return result;
}

/**
 * Annotate brawlers for the winrate overview.
 * "secret": above-average winrate z-score and below-average pickrate.
 * "overrated": below-average winrate z-score and high pickrate.
 */
export function overviewAnnotations(
	metadata: Metadata,
	event: EventMeta,
	mapScores: ScoredBrawler[]
): Map<number, string> {
	const result = new Map<number, string>();
	for (const { id, score } of mapScores) {
		const pz = pickrateZ(metadata, id, event.id);
		if (pz === undefined) continue;
		if (score >= 0 && pz < PICKRATE_LOW_Z) result.set(id, ANN_SECRET);
		else if (score < 0 && pz >= PICKRATE_HIGH_Z) result.set(id, ANN_X);
	}
	return result;
}
