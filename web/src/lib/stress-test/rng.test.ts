import { describe, expect, test } from 'bun:test';
import {
	makeTrialRng,
	sampleCategorical,
	sampleMaskedSoftmax,
	sampleWithoutReplacement
} from './rng';

describe('makeTrialRng', () => {
	test('same (seed, trialIndex) reproduces the same draw sequence', () => {
		const a = makeTrialRng(7, 3);
		const b = makeTrialRng(7, 3);
		const drawsA = Array.from({ length: 10 }, () => a.next());
		const drawsB = Array.from({ length: 10 }, () => b.next());
		expect(drawsA).toEqual(drawsB);
	});

	test('different trialIndex diverges', () => {
		const a = makeTrialRng(7, 3);
		const b = makeTrialRng(7, 4);
		expect(a.next()).not.toBe(b.next());
	});

	test('different seed diverges', () => {
		const a = makeTrialRng(7, 3);
		const b = makeTrialRng(8, 3);
		expect(a.next()).not.toBe(b.next());
	});

	test('draws land in [0, 1)', () => {
		const rng = makeTrialRng(1, 1);
		for (let i = 0; i < 1000; i++) {
			const x = rng.next();
			expect(x).toBeGreaterThanOrEqual(0);
			expect(x).toBeLessThan(1);
		}
	});
});

describe('sampleCategorical', () => {
	test('always picks the only positive-weight option', () => {
		const rng = makeTrialRng(1, 1);
		for (let i = 0; i < 20; i++) {
			expect(sampleCategorical([0, 0, 5, 0], rng)).toBe(2);
		}
	});

	test('respects weight proportions over many draws', () => {
		const rng = makeTrialRng(1, 1);
		const counts = [0, 0];
		for (let i = 0; i < 20000; i++) counts[sampleCategorical([1, 3], rng)]++;
		const frac0 = counts[0] / 20000;
		expect(frac0).toBeGreaterThan(0.2);
		expect(frac0).toBeLessThan(0.3);
	});

	test('throws when no option has positive weight', () => {
		const rng = makeTrialRng(1, 1);
		expect(() => sampleCategorical([0, 0, 0], rng)).toThrow();
	});
});

describe('sampleMaskedSoftmax', () => {
	test('low temperature is near-argmax over valid positions', () => {
		const rng = makeTrialRng(2, 2);
		const logits = [1, 5, 3, 100, 2]; // index 3 is best but masked out
		const mask = [1, 1, 1, 0, 1];
		let hits = 0;
		for (let i = 0; i < 200; i++) {
			if (sampleMaskedSoftmax(logits, mask, 0.01, rng) === 1) hits++;
		}
		expect(hits).toBe(200);
	});

	test('never returns a masked-out index', () => {
		const rng = makeTrialRng(3, 3);
		const logits = [10, 10, 10, 10];
		const mask = [0, 1, 0, 1];
		for (let i = 0; i < 200; i++) {
			const idx = sampleMaskedSoftmax(logits, mask, 1, rng);
			expect(mask[idx]).toBe(1);
		}
	});

	test('throws when every position is masked out', () => {
		const rng = makeTrialRng(4, 4);
		expect(() => sampleMaskedSoftmax([1, 2, 3], [0, 0, 0], 0.5, rng)).toThrow();
	});
});

describe('sampleWithoutReplacement', () => {
	test('returns k distinct indices within range', () => {
		const rng = makeTrialRng(5, 5);
		const out = sampleWithoutReplacement(20, 7, rng);
		expect(out.length).toBe(7);
		expect(new Set(out).size).toBe(7);
		for (const i of out) {
			expect(i).toBeGreaterThanOrEqual(0);
			expect(i).toBeLessThan(20);
		}
	});

	test('k = n returns a permutation of everything', () => {
		const rng = makeTrialRng(6, 6);
		const out = sampleWithoutReplacement(5, 5, rng);
		expect([...out].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
	});

	test('k = 0 returns empty', () => {
		const rng = makeTrialRng(7, 7);
		expect(sampleWithoutReplacement(5, 0, rng)).toEqual([]);
	});

	test('throws when k > n', () => {
		const rng = makeTrialRng(8, 8);
		expect(() => sampleWithoutReplacement(3, 4, rng)).toThrow();
	});
});
