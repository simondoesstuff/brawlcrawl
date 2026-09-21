// Integration tests against the real exported artifacts (web/static/models,
// web/static/data) — same gating pattern as onnx.test.ts: run
// `just export-onnx` from the repo root first if these are missing.

import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Manifest } from '../onnx/manifest';
import type { Metadata } from '../onnx/metadata';
import { createSession } from '../onnx/session';
import type { DraftEngine } from '../pick/engine';
import { makeTrialRng } from './rng';
import { simulateOneDraft } from './simulate';
import { estimateCallCount, runAdversarialStressTest } from './stressTest';

const STATIC_DIR = join(import.meta.dir, '../../../static');

function requireFile(path: string): Buffer {
	if (!existsSync(path)) {
		throw new Error(
			`stress-test artifact missing: ${path}\nRun \`just export-onnx\` from the repo root first.`
		);
	}
	return readFileSync(path);
}

async function loadRealEngine(): Promise<DraftEngine> {
	const manifest: Manifest = JSON.parse(
		requireFile(join(STATIC_DIR, 'data/manifest.json')).toString('utf-8')
	);
	const metadata: Metadata = JSON.parse(
		requireFile(join(STATIC_DIR, 'data/metadata.json')).toString('utf-8')
	);
	const charEncsBytes = requireFile(join(STATIC_DIR, 'data/char_encs.bin'));
	const charEncsAll = new Float32Array(
		charEncsBytes.buffer,
		charEncsBytes.byteOffset,
		charEncsBytes.byteLength / Float32Array.BYTES_PER_ELEMENT
	);
	const [draftQSession, terminalSession] = await Promise.all([
		createSession(requireFile(join(STATIC_DIR, 'models/draft_q.onnx'))),
		createSession(requireFile(join(STATIC_DIR, 'models/terminal_pick6.onnx')))
	]);
	const byId = new Map(metadata.brawlers.map((b) => [b.id, b]));
	return { manifest, metadata, charEncsAll, draftQSession, terminalSession, byId };
}

describe('simulateOneDraft', () => {
	test('produces a valid, legal, reproducible draft', async () => {
		const engine = await loadRealEngine();
		const brawlers = engine.metadata.brawlers;
		const playerPoolIds = new Set(brawlers.slice(0, 15).map((b) => b.id));
		const adversaryPoolIds = new Set(brawlers.slice(10, 28).map((b) => b.id));
		const event = engine.metadata.events[0];

		const run = () =>
			simulateOneDraft({
				engine,
				event,
				allyFirst: true,
				playerPoolIds,
				adversaryPoolIds,
				temperature: 0.15,
				rng: makeTrialRng(123, 0)
			});

		const result = await run();

		expect(result.pPlayerWin).toBeGreaterThanOrEqual(0);
		expect(result.pPlayerWin).toBeLessThanOrEqual(1);
		expect(result.playerPickIds).toHaveLength(3);
		expect(result.adversaryPickIds).toHaveLength(3);

		const allPicks = [...result.playerPickIds, ...result.adversaryPickIds];
		expect(new Set(allPicks).size).toBe(6); // no duplicate picks

		for (const id of result.playerPickIds) expect(playerPoolIds.has(id)).toBe(true);
		for (const id of result.adversaryPickIds) expect(adversaryPoolIds.has(id)).toBe(true);

		const again = await run();
		expect(again.pPlayerWin).toBeCloseTo(result.pPlayerWin, 6);
		expect(again.playerPickIds).toEqual(result.playerPickIds);
		expect(again.adversaryPickIds).toEqual(result.adversaryPickIds);
	}, 30000);
});

describe('runAdversarialStressTest', () => {
	const TRIALS = 16;

	test('baseline + candidate arms are well-formed and paired-comparable', async () => {
		const engine = await loadRealEngine();
		const brawlers = engine.metadata.brawlers;
		const ownedIds = brawlers.slice(0, 12).map((b) => b.id);
		const candidateIds = brawlers.slice(12, 15).map((b) => b.id);

		expect(estimateCallCount(engine, { ownedIds, candidateIds, trials: TRIALS })).toBe(
			(1 + 3) * TRIALS * 13
		);

		const result = await runAdversarialStressTest(engine, {
			ownedIds,
			candidateIds,
			trials: TRIALS,
			adversaryPoolSize: [10, 15],
			concurrency: 2,
			seed: 9
		});

		expect(result.baseline.trials).toBe(TRIALS);
		expect(result.baseline.winRate).toBeGreaterThanOrEqual(0);
		expect(result.baseline.winRate).toBeLessThanOrEqual(1);

		expect(result.candidates).toHaveLength(3);
		const ids = result.candidates.map((c) => c.brawler.id).sort();
		expect(ids).toEqual([...candidateIds].sort());

		for (const c of result.candidates) {
			expect(c.trials).toBe(TRIALS);
			expect(c.winRate).toBeGreaterThanOrEqual(0);
			expect(c.winRate).toBeLessThanOrEqual(1);
			expect(c.deltaStderr).toBeGreaterThanOrEqual(0);
		}
		for (let i = 1; i < result.candidates.length; i++) {
			expect(result.candidates[i - 1].delta).toBeGreaterThanOrEqual(result.candidates[i].delta);
		}

		// Common-random-numbers check: the paired delta's stderr should, on
		// average, beat the naive independent-arms estimate
		// sqrt(baseline.stderr^2 + candidate.stderr^2) — otherwise the shared
		// per-trial draws (event, coin flip, adversary pool; see trial.ts)
		// aren't actually reducing variance and the pairing is pointless.
		const naive = result.candidates.map((c) =>
			Math.sqrt(result.baseline.stderr ** 2 + c.stderr ** 2)
		);
		const avgPaired =
			result.candidates.reduce((s, c) => s + c.deltaStderr, 0) / result.candidates.length;
		const avgNaive = naive.reduce((s, x) => s + x, 0) / naive.length;
		expect(avgPaired).toBeLessThan(avgNaive);
	}, 90000);

	test('is invariant to concurrency (no cross-talk between parallel session.run calls)', async () => {
		const engine = await loadRealEngine();
		const brawlers = engine.metadata.brawlers;
		const ownedIds = brawlers.slice(0, 10).map((b) => b.id);
		const candidateIds = brawlers.slice(10, 12).map((b) => b.id);
		const base = {
			ownedIds,
			candidateIds,
			trials: 8,
			adversaryPoolSize: [10, 15] as [number, number],
			seed: 17
		};

		const sequential = await runAdversarialStressTest(engine, { ...base, concurrency: 1 });
		const parallel = await runAdversarialStressTest(engine, { ...base, concurrency: 4 });

		expect(parallel.baseline.winRate).toBe(sequential.baseline.winRate);
		expect(parallel.baseline.stderr).toBe(sequential.baseline.stderr);
		for (let i = 0; i < sequential.candidates.length; i++) {
			const s = sequential.candidates[i];
			const p = parallel.candidates.find((c) => c.brawler.id === s.brawler.id)!;
			expect(p.winRate).toBe(s.winRate);
			expect(p.delta).toBe(s.delta);
			expect(p.deltaStderr).toBe(s.deltaStderr);
		}
	}, 60000);

	test('rejects an empty owned set', async () => {
		const engine = await loadRealEngine();
		let threw = false;
		try {
			await runAdversarialStressTest(engine, { ownedIds: [] });
		} catch {
			threw = true;
		}
		expect(threw).toBe(true);
	});
});
