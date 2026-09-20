// Checks the TS ports of _ban_excluded and _phase_default_filter against
// fixtures generated from the real Python implementation (see obs.test.ts
// for the shared fixture-loading rationale).

import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { banExcluded, phaseDefaultFilter } from './phase';
import type { Manifest } from '../onnx/manifest';

const STATIC_DIR = join(import.meta.dir, '../../../static');
const FIXTURE_PATH = join(import.meta.dir, '../../../tests/fixtures/logic_fixture.json');

function requireFile(path: string): Buffer {
	if (!existsSync(path)) {
		throw new Error(`logic fixture missing: ${path}\nRun \`just export-onnx\` from the repo root first.`);
	}
	return readFileSync(path);
}

interface BanExcludedCase {
	phase: number;
	ally_ban_ids: number[];
	enemy_ban_ids: number[];
	pick_ids: { is_ally: boolean; id: number }[];
	excluded_ids: number[];
}

interface PhaseDefaultFilterCase {
	phase: number;
	ally_first: boolean;
	filter_on: boolean;
}

interface Fixture {
	ban_excluded: BanExcludedCase[];
	phase_default_filter: PhaseDefaultFilterCase[];
}

const fixture: Fixture = JSON.parse(requireFile(FIXTURE_PATH).toString('utf-8'));
const manifest: Manifest = JSON.parse(requireFile(join(STATIC_DIR, 'data/manifest.json')).toString('utf-8'));

describe('banExcluded', () => {
	test('matches the Python reference for every fixture case', () => {
		expect(fixture.ban_excluded.length).toBeGreaterThan(0);
		for (const c of fixture.ban_excluded) {
			const picks = c.pick_ids.map((p) => ({ isAlly: p.is_ally, id: p.id }));
			const excl = banExcluded(c.phase, c.ally_ban_ids, c.enemy_ban_ids, picks);
			expect(Array.from(excl).sort((a, b) => a - b)).toEqual(c.excluded_ids);
		}
	});
});

describe('phaseDefaultFilter', () => {
	test('matches the Python reference for every phase/ally_first combination', () => {
		expect(fixture.phase_default_filter.length).toBeGreaterThan(0);
		for (const c of fixture.phase_default_filter) {
			expect(phaseDefaultFilter(c.phase, c.ally_first, manifest.draft_state.TURN_SCHEDULE)).toBe(c.filter_on);
		}
	});
});
