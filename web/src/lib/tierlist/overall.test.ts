import { describe, expect, test } from 'bun:test';
import { computeOverallTierList, OVERALL_EVENT_ID } from './overall';
import type { MapTierList } from './types';

function map(partial: Partial<MapTierList> & Pick<MapTierList, 'tier_list'>): MapTierList {
	return {
		event_id: 1,
		map_name: 'Test Map',
		mode: 'gemGrab',
		trials: 100,
		converged: true,
		...partial
	};
}

describe('computeOverallTierList', () => {
	test('empty input', () => {
		const overall = computeOverallTierList([]);
		expect(overall.event_id).toBe(OVERALL_EVENT_ID);
		expect(overall.tier_list).toEqual([]);
		expect(overall.trials).toBe(0);
		expect(overall.converged).toBe(false);
	});

	test('weights by samples, not by map count', () => {
		const maps = [
			map({
				trials: 100,
				tier_list: [{ brawler_id: 1, name: 'A', win_rate: 0.9, samples: 900 }]
			}),
			map({
				trials: 10,
				tier_list: [{ brawler_id: 1, name: 'A', win_rate: 0.1, samples: 100 }]
			})
		];
		const overall = computeOverallTierList(maps);
		expect(overall.tier_list).toHaveLength(1);
		// (0.9*900 + 0.1*100) / 1000 = 0.82
		expect(overall.tier_list[0].win_rate).toBeCloseTo(0.82, 10);
		expect(overall.tier_list[0].samples).toBe(1000);
		expect(overall.trials).toBe(110);
	});

	test('sorted by win_rate desc and converged only when every map converged', () => {
		const maps = [
			map({
				converged: true,
				tier_list: [
					{ brawler_id: 1, name: 'Low', win_rate: 0.3, samples: 100 },
					{ brawler_id: 2, name: 'High', win_rate: 0.7, samples: 100 }
				]
			}),
			map({ converged: false, tier_list: [] })
		];
		const overall = computeOverallTierList(maps);
		expect(overall.tier_list.map((e) => e.name)).toEqual(['High', 'Low']);
		expect(overall.converged).toBe(false);
	});
});
