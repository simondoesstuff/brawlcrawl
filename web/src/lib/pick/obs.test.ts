// Checks the TS ports of _build_obs and _pick6_team_split against fixtures
// generated from the real Python implementation (src/pick/export_onnx.py ::
// export_logic_fixture). Run `just export-onnx` from the repo root if the
// fixture is missing or stale.

import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildObs } from './obs';
import { pick6TeamSplit } from './phase';
import type { Manifest } from '../onnx/manifest';

const STATIC_DIR = join(import.meta.dir, '../../../static');
const FIXTURE_PATH = join(import.meta.dir, '../../../tests/fixtures/logic_fixture.json');

function requireFile(path: string): Buffer {
	if (!existsSync(path)) {
		throw new Error(`logic fixture missing: ${path}\nRun \`just export-onnx\` from the repo root first.`);
	}
	return readFileSync(path);
}

interface BuildObsCase {
	phase: number;
	ally_first: boolean;
	ally_ban_char_idxs: number[];
	enemy_ban_char_idxs: number[];
	picks: { is_ally: boolean; char_idx: number }[];
	local_pool_ids: number[] | null;
	obs: number[];
	turn_token: number;
}

interface TeamSplitCase {
	ally_first: boolean;
	picks: { is_ally: boolean; char_idx: number }[];
	team_a_char_idxs: number[];
	team_b_partial_char_idxs: number[];
}

interface Fixture {
	build_obs: BuildObsCase[];
	pick6_team_split: TeamSplitCase[];
}

const fixture: Fixture = JSON.parse(requireFile(FIXTURE_PATH).toString('utf-8'));
const manifest: Manifest = JSON.parse(requireFile(join(STATIC_DIR, 'data/manifest.json')).toString('utf-8'));
const metadata: { brawlers: { id: number; char_idx: number }[] } = JSON.parse(
	requireFile(join(STATIC_DIR, 'data/metadata.json')).toString('utf-8')
);

describe('buildObs', () => {
	test('matches the Python reference for every fixture case', () => {
		expect(fixture.build_obs.length).toBeGreaterThan(0);
		for (const c of fixture.build_obs) {
			const localPoolIds = c.local_pool_ids ? new Set(c.local_pool_ids) : null;
			const brawlers = metadata.brawlers.map((b) => ({ id: b.id, charIdx: b.char_idx }));
			const { obs, turnToken } = buildObs(
				manifest.draft_state,
				manifest.n_chars,
				c.ally_ban_char_idxs,
				c.enemy_ban_char_idxs,
				c.picks.map((p) => ({ isAlly: p.is_ally, charIdx: p.char_idx })),
				c.phase,
				c.ally_first,
				{ localPoolIds, brawlers }
			);
			expect(turnToken).toBe(c.turn_token);
			expect(Array.from(obs)).toEqual(c.obs);
		}
	});
});

describe('pick6TeamSplit', () => {
	test('matches the Python reference for every fixture case', () => {
		expect(fixture.pick6_team_split.length).toBeGreaterThan(0);
		for (const c of fixture.pick6_team_split) {
			const picks = c.picks.map((p) => ({ isAlly: p.is_ally, charIdx: p.char_idx }));
			const { teamA, teamBPartial } = pick6TeamSplit(picks, manifest.draft_state.TURN_SCHEDULE);
			expect(teamA).toEqual(c.team_a_char_idxs);
			expect(teamBPartial).toEqual(c.team_b_partial_char_idxs);
		}
	});
});
