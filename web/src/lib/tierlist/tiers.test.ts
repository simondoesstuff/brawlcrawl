import { describe, expect, test } from 'bun:test';
import { assignTiers, groupByTier, tierForZ, winRateZScores, type Tier } from './tiers';

describe('winRateZScores', () => {
	test('empty input', () => {
		expect(winRateZScores([])).toEqual(new Float64Array(0));
	});

	test('constant input has zero std -> all zeros', () => {
		expect(winRateZScores([0.5, 0.5, 0.5])).toEqual(new Float64Array(3));
	});

	test('mean maps to z=0, symmetric spread', () => {
		const z = winRateZScores([0.4, 0.5, 0.6]);
		expect(z[1]).toBeCloseTo(0, 10);
		expect(z[0]).toBeCloseTo(-z[2], 10);
		expect(z[2]).toBeGreaterThan(0);
	});
});

describe('tierForZ', () => {
	test.each([
		[2.0, 'S'],
		[1.25, 'S'],
		[1.0, 'A'],
		[0.4, 'A'],
		[0.0, 'B'],
		[-0.4, 'B'],
		[-1.0, 'C'],
		[-1.25, 'C'],
		[-2.0, 'D']
	] satisfies [number, Tier][])('z=%p -> %p', (z, expected) => {
		expect(tierForZ(z)).toBe(expected);
	});
});

describe('assignTiers / groupByTier', () => {
	test('groups preserve input order within a tier and skip empty tiers', () => {
		const entries = [
			{ name: 'best', wr: 0.9 },
			{ name: 'mid1', wr: 0.5 },
			{ name: 'mid2', wr: 0.5 },
			{ name: 'worst', wr: 0.1 }
		];
		const tiered = assignTiers(entries, (e) => e.wr);
		expect(tiered.map((t) => t.entry.name)).toEqual(['best', 'mid1', 'mid2', 'worst']);

		const groups = groupByTier(tiered);
		const groupNames = groups.map((g) => ({ tier: g.tier, names: g.entries.map((e) => e.entry.name) }));
		// 'best' and 'worst' are >1 std from the mean of 4 points; mid1/mid2 sit together.
		expect(groupNames[0].tier).toBe('S');
		expect(groupNames[0].names).toEqual(['best']);
		expect(groupNames.at(-1)!.tier).toBe('D');
		expect(groupNames.at(-1)!.names).toEqual(['worst']);
		const midGroup = groupNames.find((g) => g.names.includes('mid1'));
		expect(midGroup?.names).toEqual(['mid1', 'mid2']);
	});
});
