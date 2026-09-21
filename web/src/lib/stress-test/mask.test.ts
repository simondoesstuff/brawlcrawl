import { describe, expect, test } from 'bun:test';
import { banValidMask, pickValidMask } from './mask';

describe('banValidMask', () => {
	test('everything valid with no prior bans', () => {
		expect(Array.from(banValidMask(4, []))).toEqual([1, 1, 1, 1]);
	});

	test("excludes only the acting team's own bans (cross-team overlap allowed)", () => {
		expect(Array.from(banValidMask(4, [0, 2]))).toEqual([0, 1, 0, 1]);
	});
});

describe('pickValidMask', () => {
	test('only the local pool is eligible, minus bans and picks', () => {
		const mask = pickValidMask(6, [1], [4], new Set([0, 1, 2, 4]));
		// pool {0,1,2,4} minus banned {1} minus picked {4} -> {0,2}
		expect(Array.from(mask)).toEqual([1, 0, 1, 0, 0, 0]);
	});

	test('chars outside the local pool are never eligible even if unbanned/unpicked', () => {
		const mask = pickValidMask(4, [], [], new Set([0]));
		expect(Array.from(mask)).toEqual([1, 0, 0, 0]);
	});

	test('empty local pool -> nothing eligible', () => {
		const mask = pickValidMask(3, [], [], new Set());
		expect(Array.from(mask)).toEqual([0, 0, 0]);
	});
});
