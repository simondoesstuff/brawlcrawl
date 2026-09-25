// Aggregates per-map tier lists into a single synthetic "Overall" tier list:
// each brawler's overall win rate is the samples-weighted mean of its win
// rate across all maps, so maps with more converged Monte Carlo trials count
// for more. Pure function of the already-loaded per-map data — no extra
// fetch or backend computation needed.

import type { MapTierList, TierEntry } from './types';

export const OVERALL_EVENT_ID = -1;

export function computeOverallTierList(maps: MapTierList[]): MapTierList {
	const totals = new Map<number, { name: string; weightedSum: number; weight: number }>();

	for (const map of maps) {
		for (const entry of map.tier_list) {
			const t = totals.get(entry.brawler_id) ?? { name: entry.name, weightedSum: 0, weight: 0 };
			t.weightedSum += entry.win_rate * entry.samples;
			t.weight += entry.samples;
			totals.set(entry.brawler_id, t);
		}
	}

	const tier_list: TierEntry[] = Array.from(totals, ([brawler_id, t]) => ({
		brawler_id,
		name: t.name,
		win_rate: t.weight > 0 ? t.weightedSum / t.weight : 0,
		samples: t.weight
	})).sort((a, b) => b.win_rate - a.win_rate);

	return {
		event_id: OVERALL_EVENT_ID,
		map_name: 'Overall',
		mode: 'overall',
		trials: maps.reduce((sum, m) => sum + m.trials, 0),
		converged: maps.length > 0 && maps.every((m) => m.converged),
		tier_list
	};
}
