import { describe, expect, test } from 'bun:test';
import { parseOwnedNames, resolveOwnedBrawlers } from './ownedBrawlers';
import type { Brawler } from '../onnx/metadata';

function brawler(id: number, name: string): Brawler {
	return { id, name, class: 'Damage Dealer', rarity: 'Rare', char_idx: id };
}

const ROSTER: Brawler[] = [brawler(1, 'Shelly'), brawler(2, 'Colt'), brawler(3, 'Bull'), brawler(4, 'Bo'), brawler(5, '8-Bit')];

describe('parseOwnedNames', () => {
	test('splits on commas, trims, and drops empty tokens', () => {
		expect(parseOwnedNames(' shelly ,, colt ,  ')).toEqual(['shelly', 'colt']);
	});

	test('empty string yields no tokens', () => {
		expect(parseOwnedNames('')).toEqual([]);
		expect(parseOwnedNames('   ')).toEqual([]);
	});
});

describe('resolveOwnedBrawlers', () => {
	test('matches full names case- and punctuation-insensitively', () => {
		const r = resolveOwnedBrawlers('shelly, 8bit', ROSTER);
		expect(r.tokens).toEqual([
			{ raw: 'shelly', matched: 'Shelly' },
			{ raw: '8bit', matched: '8-Bit' }
		]);
		expect(r.ids).toEqual(new Set([1, 5]));
		expect(r.unmatchedRaw).toEqual([]);
	});

	test('matches partial (subsequence) names', () => {
		const r = resolveOwnedBrawlers('col', ROSTER);
		expect(r.tokens).toEqual([{ raw: 'col', matched: 'Colt' }]);
	});

	test('reports unmatched tokens without matching anything', () => {
		const r = resolveOwnedBrawlers('shelly, zzz', ROSTER);
		expect(r.tokens).toEqual([
			{ raw: 'shelly', matched: 'Shelly' },
			{ raw: 'zzz', matched: null }
		]);
		expect(r.ids).toEqual(new Set([1]));
		expect(r.unmatchedRaw).toEqual(['zzz']);
	});

	test('two tokens colliding on the same shortest match both resolve, but only count once', () => {
		// 'b' and 'bo' both subsequence-match 'Bo' (shortest) before 'Bull'.
		const r = resolveOwnedBrawlers('b, bo', ROSTER);
		expect(r.tokens).toEqual([
			{ raw: 'b', matched: 'Bo' },
			{ raw: 'bo', matched: 'Bo' }
		]);
		expect(r.ids).toEqual(new Set([4]));
	});

	test('empty input resolves to nothing', () => {
		const r = resolveOwnedBrawlers('', ROSTER);
		expect(r.tokens).toEqual([]);
		expect(r.ids.size).toBe(0);
		expect(r.unmatchedRaw).toEqual([]);
	});
});
