import { describe, expect, test } from 'bun:test';
import type { Metadata } from '../onnx/metadata';
import { rankCandidatesByWinrate } from './candidates';

function meta(ids: number[], winrates: Record<string, Record<string, number>>): Metadata {
	return {
		brawlers: ids.map((id) => ({ id, name: `B${id}`, class: 'x', rarity: 'Rare', char_idx: id })),
		events: [{ id: 1, mode: 'm', mode_id: 1, map_name: 'Map', event_idx: 0, mode_idx: 0 }],
		winrates,
		pickrates: {}
	};
}

describe('rankCandidatesByWinrate', () => {
	test('ranks by descending mean winrate z-score', () => {
		const metadata = meta([1, 2, 3], { '1': { '1': 0.5 }, '2': { '1': 1.5 }, '3': { '1': -0.5 } });
		expect(rankCandidatesByWinrate(metadata, new Set())).toEqual([2, 1, 3]);
	});

	test('excludes owned ids', () => {
		const metadata = meta([1, 2], { '1': { '1': 0.5 }, '2': { '1': 1.5 } });
		expect(rankCandidatesByWinrate(metadata, new Set([2]))).toEqual([1]);
	});

	test('treats missing winrate entries as zero', () => {
		const metadata = meta([1, 2], { '1': { '1': 1.0 } });
		expect(rankCandidatesByWinrate(metadata, new Set())).toEqual([1, 2]);
	});
});
