// Buckets a map's first-pick win rates into letter tiers by z-score
// (population std), the same normalization convention used elsewhere in the
// app (see pick/annotations.ts::zScoreQ). 0.5 win rate is a meaningful
// baseline here — these are Monte Carlo probabilities against uniformly
// random teammates/opponents — but the z-score spread still varies per map,
// so tiers are relative to that map's own distribution.

export type Tier = 'S' | 'A' | 'B' | 'C' | 'D';

export const TIER_ORDER: Tier[] = ['S', 'A', 'B', 'C', 'D'];

const TIER_THRESHOLDS: [Tier, number][] = [
	['S', 1.25],
	['A', 0.4],
	['B', -0.4],
	['C', -1.25]
];

export function winRateZScores(winRates: ArrayLike<number>): Float64Array {
	const n = winRates.length;
	const out = new Float64Array(n);
	if (n === 0) return out;
	let sum = 0;
	for (let i = 0; i < n; i++) sum += winRates[i];
	const mean = sum / n;
	let sqSum = 0;
	for (let i = 0; i < n; i++) sqSum += (winRates[i] - mean) ** 2;
	const std = Math.sqrt(sqSum / n);
	if (std < 1e-9) return out;
	for (let i = 0; i < n; i++) out[i] = (winRates[i] - mean) / std;
	return out;
}

export function tierForZ(z: number): Tier {
	for (const [tier, threshold] of TIER_THRESHOLDS) {
		if (z >= threshold) return tier;
	}
	return 'D';
}

export interface TieredEntry<T> {
	tier: Tier;
	z: number;
	entry: T;
}

/** Assigns a letter tier to each entry, in input order. */
export function assignTiers<T>(entries: T[], winRateOf: (entry: T) => number): TieredEntry<T>[] {
	const z = winRateZScores(entries.map(winRateOf));
	return entries.map((entry, i) => ({ tier: tierForZ(z[i]), z: z[i], entry }));
}

/** Groups tiered entries by tier, in TIER_ORDER, dropping empty tiers. Preserves input order within a tier. */
export function groupByTier<T>(tiered: TieredEntry<T>[]): { tier: Tier; entries: TieredEntry<T>[] }[] {
	const byTier = new Map<Tier, TieredEntry<T>[]>();
	for (const t of tiered) {
		const group = byTier.get(t.tier);
		if (group) group.push(t);
		else byTier.set(t.tier, [t]);
	}
	return TIER_ORDER.filter((tier) => byTier.has(tier)).map((tier) => ({ tier, entries: byTier.get(tier)! }));
}
