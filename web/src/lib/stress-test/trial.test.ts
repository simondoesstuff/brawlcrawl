// Pure logic test — uses a minimal fake DraftEngine (just the `metadata`
// shape setupTrial reads) rather than loading real ONNX artifacts.

import { describe, expect, test } from 'bun:test';
import type { EventMeta } from '../onnx/metadata';
import type { DraftEngine } from '../pick/engine';
import { setupTrial } from './trial';

function fakeEngine(nBrawlers: number, events: EventMeta[]): DraftEngine {
	const brawlers = Array.from({ length: nBrawlers }, (_, i) => ({
		id: 1000 + i,
		name: `B${i}`,
		class: 'Damage Dealer',
		rarity: 'Rare',
		char_idx: i
	}));
	return {
		metadata: { brawlers, events, winrates: {}, pickrates: {} }
	} as unknown as DraftEngine;
}

const EVENTS: EventMeta[] = [
	{ id: 1, mode: 'Gem Grab', mode_id: 1, map_name: 'Map A', event_idx: 10, mode_idx: 1 },
	{ id: 2, mode: 'Brawl Ball', mode_id: 2, map_name: 'Map B', event_idx: 20, mode_idx: 2 }
];

describe('setupTrial', () => {
	test('same (seed, trialIndex) reproduces identical event/allyFirst/adversary pool', () => {
		const engine = fakeEngine(30, EVENTS);
		const config = { events: EVENTS, adversaryPoolSize: 12 as number, seed: 42 };
		const a = setupTrial(engine, config, 5);
		const b = setupTrial(engine, config, 5);
		expect(a.event).toEqual(b.event);
		expect(a.allyFirst).toBe(b.allyFirst);
		expect([...a.adversaryPoolIds].sort()).toEqual([...b.adversaryPoolIds].sort());
	});

	test('different trialIndex generally differs', () => {
		const engine = fakeEngine(30, EVENTS);
		const config = { events: EVENTS, adversaryPoolSize: 12 as number, seed: 42 };
		const results = Array.from({ length: 10 }, (_, i) => setupTrial(engine, config, i));
		const distinctPools = new Set(results.map((r) => [...r.adversaryPoolIds].sort().join(',')));
		expect(distinctPools.size).toBeGreaterThan(1);
	});

	test('fixed adversaryPoolSize always draws exactly that many distinct brawlers', () => {
		const engine = fakeEngine(30, EVENTS);
		for (let i = 0; i < 20; i++) {
			const { adversaryPoolIds } = setupTrial(
				engine,
				{ events: EVENTS, adversaryPoolSize: 12, seed: 1 },
				i
			);
			expect(adversaryPoolIds.size).toBe(12);
		}
	});

	test('[min, max] adversaryPoolSize stays within range', () => {
		const engine = fakeEngine(30, EVENTS);
		for (let i = 0; i < 50; i++) {
			const { adversaryPoolIds } = setupTrial(
				engine,
				{ events: EVENTS, adversaryPoolSize: [10, 15], seed: 2 },
				i
			);
			expect(adversaryPoolIds.size).toBeGreaterThanOrEqual(10);
			expect(adversaryPoolIds.size).toBeLessThanOrEqual(15);
		}
	});

	test('drawn event is always one of the configured events', () => {
		const engine = fakeEngine(30, EVENTS);
		for (let i = 0; i < 20; i++) {
			const { event } = setupTrial(engine, { events: EVENTS, adversaryPoolSize: 5, seed: 3 }, i);
			expect(EVENTS).toContainEqual(event);
		}
	});

	test('throws on empty events list', () => {
		const engine = fakeEngine(30, []);
		expect(() => setupTrial(engine, { events: [], adversaryPoolSize: 5, seed: 1 }, 0)).toThrow();
	});
});
