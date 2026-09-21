// Adversarial stress-test for an owned brawler set: "how strong is this
// account, and which brawler should I get next?"
//
// For the owned set, Monte-Carlo-simulates many full drafts (simulate.ts)
// against a randomly-sampled adversary — the adversary's pool *size* is a
// hyperparameter (adversaryPoolSize), but which brawlers fill it is drawn
// fresh per trial — and averages the resulting P(player wins) into a
// baseline win rate. Then, for every brawler not owned, reruns the same
// trials with that one brawler added to the pool.
//
// Each candidate's trials reuse the *same* per-trial random draws (event,
// coin flip, adversary pool — see trial.ts) as the baseline's trial of the
// same index, so `delta = winRate(owned + candidate) - winRate(owned)` is a
// paired comparison: the map, the coin flip, and who the random opponent is
// always match between a baseline trial and the corresponding
// candidate-extended trial. That's a real but partial variance reduction —
// the extra owned brawler changes what the player's own local pool looks
// like from turn 0 (bans are pool-aware too, see simulate.ts), so the two
// arms' *action* draws can diverge as early as the first ban, not just at
// whichever pick the new brawler actually gets drafted in. Still strictly
// better than two independent arms, since the shared draws never desync.
//
// Cost: each simulated draft is 13 ONNX session.run calls (12 draftQ turns +
// 1 terminal call). A full sweep is therefore roughly
// `(1 + candidates) * trials * 13` calls — see estimateCallCount. With the
// defaults below (60 trials) and a ~70-brawler roster that's on the order of
// 55k calls; budget seconds-to-low-minutes in a browser tab, and prefer
// passing a smaller `candidateIds` shortlist (e.g. pre-screened by
// score_map/pickrate) over sweeping the full roster at high trial counts.
//
// adversaryPoolSize defaults to env.py's DraftConfig training range
// (pool_min=10, pool_max=40) — owned sets much larger than 40 push the
// model's local-pool encoding outside what it saw in training.

import type { Brawler, EventMeta } from '../onnx/metadata';
import type { DraftEngine } from '../pick/engine';
import { simulateOneDraft } from './simulate';
import { setupTrial } from './trial';

export interface StressTestConfig {
	/** Brawler ids the player owns and is willing to play. */
	ownedIds: number[];
	/** Brawlers to test adding one at a time; default: every brawler not in ownedIds. */
	candidateIds?: number[];
	/** Events (maps) to sample per trial; default: every event in metadata. */
	events?: EventMeta[];
	/** Adversary's owned-pool size, fixed or an inclusive [min, max] range drawn per trial. */
	adversaryPoolSize?: number | [number, number];
	/** Monte Carlo trials per arm (baseline, and each candidate). */
	trials?: number;
	/** Softmax temperature for "nearly optimal" action sampling — lower = closer to argmax. */
	temperature?: number;
	/** RNG seed — same seed + config reproduces the same result. */
	seed?: number;
	/** Max simulated drafts to run concurrently. */
	concurrency?: number;
	signal?: AbortSignal;
	/** Called after each simulated draft completes, across all arms. */
	onProgress?: (done: number, total: number) => void;
}

const DEFAULT_ADVERSARY_POOL_SIZE: [number, number] = [10, 40]; // env.py DraftConfig.pool_min/pool_max
const DEFAULT_TRIALS = 60;
const DEFAULT_TEMPERATURE = 0.15;
const DEFAULT_SEED = 1;
const DEFAULT_CONCURRENCY = 4;
const CALLS_PER_DRAFT = 13; // 12 draftQ turns + 1 terminal call

export interface ArmSummary {
	/** Mean P(player wins) over trials. */
	winRate: number;
	/** Standard error of that mean. */
	stderr: number;
	trials: number;
}

export interface CandidateSummary extends ArmSummary {
	brawler: Brawler;
	/** Paired mean(candidate winRate - baseline winRate) at matching trial indices. */
	delta: number;
	/** Standard error of that paired mean. */
	deltaStderr: number;
}

export interface StressTestResult {
	baseline: ArmSummary;
	/** Sorted descending by delta — best acquisition first. */
	candidates: CandidateSummary[];
}

function defaultCandidateIds(engine: DraftEngine, ownedIds: readonly number[]): number[] {
	const owned = new Set(ownedIds);
	return engine.metadata.brawlers.filter((b) => !owned.has(b.id)).map((b) => b.id);
}

/** Rough ONNX session.run call count a config will issue — see the cost note above. */
export function estimateCallCount(
	engine: DraftEngine,
	config: Pick<StressTestConfig, 'ownedIds' | 'candidateIds' | 'trials'>
): number {
	const trials = config.trials ?? DEFAULT_TRIALS;
	const nCandidates = (config.candidateIds ?? defaultCandidateIds(engine, config.ownedIds)).length;
	return (1 + nCandidates) * trials * CALLS_PER_DRAFT;
}

function meanStderr(xs: ArrayLike<number>): { mean: number; stderr: number } {
	const n = xs.length;
	let sum = 0;
	for (let i = 0; i < n; i++) sum += xs[i];
	const mean = n > 0 ? sum / n : NaN;
	if (n < 2) return { mean, stderr: 0 };
	let sqDiff = 0;
	for (let i = 0; i < n; i++) sqDiff += (xs[i] - mean) ** 2;
	const sd = Math.sqrt(sqDiff / (n - 1));
	return { mean, stderr: sd / Math.sqrt(n) };
}

function summarizeArm(probs: Float64Array, trials: number): ArmSummary {
	const { mean, stderr } = meanStderr(probs);
	return { winRate: mean, stderr, trials };
}

async function mapPool<T>(
	items: T[],
	limit: number,
	fn: (item: T) => Promise<void>
): Promise<void> {
	let next = 0;
	const workerCount = Math.max(1, Math.min(limit, items.length));
	const workers = Array.from({ length: workerCount }, async () => {
		while (next < items.length) {
			await fn(items[next++]);
		}
	});
	await Promise.all(workers);
}

interface ResolvedConfig {
	events: EventMeta[];
	adversaryPoolSize: number | [number, number];
	trials: number;
	temperature: number;
	seed: number;
	concurrency: number;
}

async function runArm(
	engine: DraftEngine,
	poolIds: ReadonlySet<number>,
	config: ResolvedConfig,
	signal: AbortSignal | undefined,
	onTrialDone: () => void
): Promise<Float64Array> {
	const probs = new Float64Array(config.trials);
	const trialIndices = Array.from({ length: config.trials }, (_, i) => i);
	await mapPool(trialIndices, config.concurrency, async (i) => {
		if (signal?.aborted) throw new DOMException('Stress test aborted', 'AbortError');
		const { event, allyFirst, adversaryPoolIds, rng } = setupTrial(
			engine,
			{ events: config.events, adversaryPoolSize: config.adversaryPoolSize, seed: config.seed },
			i
		);
		const result = await simulateOneDraft({
			engine,
			event,
			allyFirst,
			playerPoolIds: poolIds,
			adversaryPoolIds,
			temperature: config.temperature,
			rng
		});
		probs[i] = result.pPlayerWin;
		onTrialDone();
	});
	return probs;
}

export async function runAdversarialStressTest(
	engine: DraftEngine,
	userConfig: StressTestConfig
): Promise<StressTestResult> {
	if (userConfig.ownedIds.length === 0) {
		throw new Error('runAdversarialStressTest: ownedIds must be non-empty');
	}
	const config: ResolvedConfig = {
		events: userConfig.events ?? engine.metadata.events,
		adversaryPoolSize: userConfig.adversaryPoolSize ?? DEFAULT_ADVERSARY_POOL_SIZE,
		trials: userConfig.trials ?? DEFAULT_TRIALS,
		temperature: userConfig.temperature ?? DEFAULT_TEMPERATURE,
		seed: userConfig.seed ?? DEFAULT_SEED,
		concurrency: userConfig.concurrency ?? DEFAULT_CONCURRENCY
	};
	const candidateIds = userConfig.candidateIds ?? defaultCandidateIds(engine, userConfig.ownedIds);

	const totalUnits = (1 + candidateIds.length) * config.trials;
	let done = 0;
	const tick = () => userConfig.onProgress?.(++done, totalUnits);

	const ownedSet = new Set(userConfig.ownedIds);
	const baselineProbs = await runArm(engine, ownedSet, config, userConfig.signal, tick);
	const baseline = summarizeArm(baselineProbs, config.trials);

	const candidates: CandidateSummary[] = [];
	for (const id of candidateIds) {
		const extendedSet = new Set(ownedSet);
		extendedSet.add(id);
		const probs = await runArm(engine, extendedSet, config, userConfig.signal, tick);

		const diffs = new Float64Array(config.trials);
		for (let i = 0; i < config.trials; i++) diffs[i] = probs[i] - baselineProbs[i];
		const { mean: delta, stderr: deltaStderr } = meanStderr(diffs);

		const brawler = engine.byId.get(id);
		if (!brawler) throw new Error(`runAdversarialStressTest: unknown candidate brawler id ${id}`);

		candidates.push({ ...summarizeArm(probs, config.trials), brawler, delta, deltaStderr });
	}
	candidates.sort((a, b) => b.delta - a.delta);

	return { baseline, candidates };
}
