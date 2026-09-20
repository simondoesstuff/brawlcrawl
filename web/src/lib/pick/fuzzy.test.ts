import { describe, expect, test } from 'bun:test';
import { fuzzyFind, normalize, subseqMatch } from './fuzzy';

describe('normalize', () => {
	test('uppercases and strips non-alphanumerics', () => {
		expect(normalize('8-Bit')).toBe('8BIT');
		expect(normalize("Mr. P")).toBe('MRP');
		expect(normalize('Larry & Lawrie')).toBe('LARRYLAWRIE');
	});
});

describe('subseqMatch', () => {
	test('matches in-order subsequences', () => {
		expect(subseqMatch('SH', 'SHELLY')).toBe(true);
		expect(subseqMatch('SY', 'SHELLY')).toBe(true);
		expect(subseqMatch('YS', 'SHELLY')).toBe(false);
		expect(subseqMatch('Z', 'SHELLY')).toBe(false);
	});
});

describe('fuzzyFind', () => {
	const names = ['SHELLY', 'COLT', 'BULL', 'BROCK', 'BO', '8-BIT'];

	test('prefers shortest normalized match', () => {
		expect(fuzzyFind('B', names)).toBe('BO');
	});

	test('is case-insensitive and ignores punctuation', () => {
		expect(fuzzyFind('8bit', names)).toBe('8-BIT');
	});

	test('returns null with no match', () => {
		expect(fuzzyFind('ZZZZ', names)).toBeNull();
	});

	test('returns null for empty query', () => {
		expect(fuzzyFind('', names)).toBeNull();
	});
});
