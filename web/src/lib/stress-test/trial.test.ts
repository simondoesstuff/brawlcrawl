// Pure logic test — no DraftEngine/ONNX involved, setupTrial only reads the
// configured events list.

import { describe, expect, test } from 'bun:test';
import type { EventMeta } from '../onnx/metadata';
import { setupTrial } from './trial';

const EVENTS: EventMeta[] = [
	{ id: 1, mode: 'Gem Grab', mode_id: 1, map_name: 'Map A', event_idx: 10, mode_idx: 1 },
	{ id: 2, mode: 'Brawl Ball', mode_id: 2, map_name: 'Map B', event_idx: 20, mode_idx: 2 }
];

describe('setupTrial', () => {
	test('same (seed, trialIndex) reproduces identical event/allyFirst', () => {
		const config = { events: EVENTS, seed: 42 };
		const a = setupTrial(config, 5);
		const b = setupTrial(config, 5);
		expect(a.event).toEqual(b.event);
		expect(a.allyFirst).toBe(b.allyFirst);
	});

	test('stratifies exactly: every (event, allyFirst) cell is hit exactly once over 2 * events.length trials', () => {
		const config = { events: EVENTS, seed: 1 };
		const nTrials = 2 * EVENTS.length;
		const cells = Array.from({ length: nTrials }, (_, i) => {
			const { event, allyFirst } = setupTrial(config, i);
			return `${event.id}:${allyFirst}`;
		});
		expect(new Set(cells).size).toBe(nTrials);
		for (const event of EVENTS) {
			expect(cells).toContain(`${event.id}:true`);
			expect(cells).toContain(`${event.id}:false`);
		}
	});

	test('stratification wraps around past 2 * events.length trials', () => {
		const config = { events: EVENTS, seed: 1 };
		const nTrials = 2 * EVENTS.length;
		for (let i = 0; i < nTrials; i++) {
			const a = setupTrial(config, i);
			const b = setupTrial(config, i + nTrials);
			expect(a.event).toEqual(b.event);
			expect(a.allyFirst).toBe(b.allyFirst);
		}
	});

	test('drawn event is always one of the configured events', () => {
		const config = { events: EVENTS, seed: 3 };
		for (let i = 0; i < 20; i++) {
			const { event } = setupTrial(config, i);
			expect(EVENTS).toContainEqual(event);
		}
	});

	test('the returned rng is independent per (seed, trialIndex) for action sampling', () => {
		const config = { events: EVENTS, seed: 42 };
		const a = setupTrial(config, 5);
		const b = setupTrial(config, 6);
		expect(a.rng.next()).not.toBe(b.rng.next());
	});

	test('throws on empty events list', () => {
		expect(() => setupTrial({ events: [], seed: 1 }, 0)).toThrow();
	});
});
