// Adversarial stress-test for an owned brawler set: "how strong is this
// account, and which brawler should I get next?"
//
// The adversary is assumed near-optimal with the *entire* roster available
// — not a randomly-sampled pool of a hyperparameter size. This matches how
// draft_q is now trained: `pool_max` defaults to every brawler (see
// train.py), so the model has seen full-roster locals during training and a
// full-roster adversary is in-distribution, not an extrapolation. Bans were
// already never pool-restricted (see mask.ts), so concretely this only
// widens the adversary's *pick* legality from their sampled pool to
// everyone, and removes the `LOCALLY_BANNED` signal they'd otherwise see in
// their own observation. Practically the player's owned set will usually
// lose to this adversary — that's expected and fine, since only the
// relative delta between arms (below) is being measured, not the baseline
// win rate itself.
//
// Dropping the adversary-pool draw also removes a source of between-trial
// variance without touching the two that matter more: which map is played,
// and which side picks first. Rather than drawing those i.i.d. and relying
// on enough trials to average them out, `trial.ts` stratifies them —
// trialIndex is walked deterministically over every (event, allyFirst) cell
// — so map/side variance is eliminated by design. The only remaining
// per-trial randomness is near-optimal action-sampling noise (temperature),
// which makes each trial carry more information than an i.i.d. draw did —
// see DEFAULT_STRATIFICATION_CYCLES and the honesty caveat below for what
// the default trial count actually buys you in practice.
//
// For the owned set, Monte-Carlo-simulates many full drafts (simulate.ts)
// against that adversary and averages the resulting P(player wins) into a
// baseline win rate. Then, for every brawler not owned, reruns the same
// trials with that one brawler added to the pool.
//
// Each candidate's trials reuse the *same* per-trial random draws (the
// action-sampling RNG stream — see trial.ts/rng.ts) as the baseline's trial
// of the same index, so `delta = winRate(owned + candidate) - winRate(owned)`
// is a paired comparison: the map and the coin flip always match (they're
// stratified identically for every arm), and the action-sampling noise
// starts from the same seed. That's a real but partial variance reduction —
// the extra owned brawler changes what the player's own local pool looks
// like from turn 0 (bans are pool-aware too, see simulate.ts), so the two
// arms' *action* draws can diverge as early as the first ban, not just at
// whichever pick the new brawler actually gets drafted in. Still strictly
// better than two independent arms, since the shared draws never desync.
//
// Two honesty caveats:
//
// - `meanStderr` over trials/diffs still treats stratification cells as
//   i.i.d. draws, so the reported stderr/deltaStderr *overestimates*
//   uncertainty (it counts map-to-map variation that stratification has
//   already eliminated). That's a conservative bias, not an optimistic one.
// - The default (DEFAULT_STRATIFICATION_CYCLES samples per cell) was
//   calibrated to the point where the #1 candidate and the top/bottom-3
//   split were stable across seeds in a spot-check, with only minor
//   reshuffling *within* those groups — not to exact full-ranking
//   convergence. Treat close deltas (within a stderr or two of each other)
//   as tied rather than meaningfully ordered; pass a larger `trials` (an
//   integer multiple of `2 * events.length`, so every cell keeps equal
//   weight — see the `trials` field's doc) if you need the tail order
//   itself to be reliable, not just the shortlist.
//
// Cost: each simulated draft is 13 ONNX session.run calls (12 draftQ turns +
// 1 terminal call). A full sweep is therefore roughly
// `(1 + candidates) * trials * 13` calls — see estimateCallCount. The
// default trials is now 8x what one stratification cycle alone would need
// (see DEFAULT_STRATIFICATION_CYCLES), so a handful of candidates is still
// quick but sweeping the full roster at the default is minutes, not
// seconds. Prefer passing a smaller `candidateIds` shortlist (e.g.
// pre-screened by score_map/pickrate) over sweeping the full roster at the
// default trial count, or pass a smaller explicit `trials` (still a multiple
// of `2 * events.length`) for a faster, rougher first pass. Also worth
// knowing: onnxruntime-web's node/WASM backend has been observed to crash
// with an out-of-bounds memory error after tens of thousands of cumulative
// session.run calls within one long-lived process — nothing in this repo's
// tests or a single browser-tab sweep gets close, but a full-roster sweep
// at the default trial count does (hundreds of thousands of calls), so
// don't assume that ceiling doesn't exist if this ever runs unattended for
// many sweeps in a row (e.g. a server-side batch job).

import type { Brawler, EventMeta } from '../onnx/metadata';
import type { DraftEngine } from '../pick/engine';
import { simulateOneDraft } from './simulate';
import { setupTrial } from './trial';

export interface StressTestConfig {
	/** Brawler ids the player owns and is willing to play. */
	ownedIds: number[];
	/** Brawlers to test adding one at a time; default: every brawler not in ownedIds. */
	candidateIds?: number[];
	/** Events (maps) stratified per trial; default: every event in metadata. */
	events?: EventMeta[];
	/**
	 * Monte Carlo trials per arm (baseline, and each candidate); default:
	 * `DEFAULT_STRATIFICATION_CYCLES * 2 * events.length` — see that
	 * constant for how many samples per (event, allyFirst) cell that is and
	 * why. Must be a multiple of `2 * events.length` — trial.ts assigns
	 * cells by `trialIndex % (2 * events.length)`, so any other count would
	 * silently under-weight or skip some maps instead of cycling through
	 * all of them evenly.
	 */
	trials?: number;
	/** Softmax temperature for near-optimal action sampling (both sides) — lower = closer to argmax; default matches env.py's deployed `eval_temp`. */
	temperature?: number;
	/** RNG seed — same seed + config reproduces the same result. */
	seed?: number;
	/** Max simulated drafts to run concurrently. */
	concurrency?: number;
	signal?: AbortSignal;
	/** Called after each simulated draft completes, across all arms. */
	onProgress?: (done: number, total: number) => void;
}

const DEFAULT_TEMPERATURE = 0.1; // matches train.py's eval_temp (the deployed near-optimal temperature)
const DEFAULT_SEED = 1;
const DEFAULT_CONCURRENCY = 4;
const CALLS_PER_DRAFT = 13; // 12 draftQ turns + 1 terminal call

// Empirically calibrated (2026-09-24): a 3-seed spot-check on the real
// model (6 held-out candidates, all 31 events) measured the candidate-delta
// ranking's average pairwise Spearman correlation across seeds at increasing
// sample counts per stratification cell:
//   1 cycle -> 0.28, 2 -> 0.70, 4 -> 0.70, 8 -> 0.81, 16 -> 0.85
// — a steep climb through 8 cycles, then diminishing returns (doubling to
// 16 only bought +0.04). At 8 cycles the #1 candidate and the top/bottom-3
// partition were already stable across all 3 seeds, with only minor
// reshuffling *within* those groups; that's the "good enough to act on"
// point this default targets, not exact full-ranking convergence (see the
// honesty caveat above). Re-run this spot-check if the roster size or model
// changes enough that the calibration might no longer hold: for a few
// candidates and a handful of seeds, run `runAdversarialStressTest` at
// increasing multiples of `2 * events.length` trials and compare the
// resulting candidate-delta rankings' pairwise Spearman correlation across
// seeds — pick the multiple past which it stops climbing meaningfully.
const DEFAULT_STRATIFICATION_CYCLES = 8;

/** `DEFAULT_STRATIFICATION_CYCLES` samples per (event, allyFirst) stratification cell — see that constant. */
function defaultTrials(events: readonly EventMeta[]): number {
	return DEFAULT_STRATIFICATION_CYCLES * 2 * events.length;
}

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
	config: Pick<StressTestConfig, 'ownedIds' | 'candidateIds' | 'trials' | 'events'>
): number {
	const events = config.events ?? engine.metadata.events;
	const trials = config.trials ?? defaultTrials(events);
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
	trials: number;
	temperature: number;
	seed: number;
	concurrency: number;
}

async function runArm(
	engine: DraftEngine,
	poolIds: ReadonlySet<number>,
	adversaryPoolIds: ReadonlySet<number>,
	config: ResolvedConfig,
	signal: AbortSignal | undefined,
	onTrialDone: () => void
): Promise<Float64Array> {
	const probs = new Float64Array(config.trials);
	const trialIndices = Array.from({ length: config.trials }, (_, i) => i);
	await mapPool(trialIndices, config.concurrency, async (i) => {
		if (signal?.aborted) throw new DOMException('Stress test aborted', 'AbortError');
		const { event, allyFirst, rng } = setupTrial({ events: config.events, seed: config.seed }, i);
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
	const events = userConfig.events ?? engine.metadata.events;
	const cellCount = 2 * events.length;
	if (userConfig.trials !== undefined && userConfig.trials % cellCount !== 0) {
		throw new Error(
			`runAdversarialStressTest: trials (${userConfig.trials}) must be a multiple of ` +
				`2 * events.length (${cellCount}) — see setupTrial's stratification`
		);
	}
	const config: ResolvedConfig = {
		events,
		trials: userConfig.trials ?? defaultTrials(events),
		temperature: userConfig.temperature ?? DEFAULT_TEMPERATURE,
		seed: userConfig.seed ?? DEFAULT_SEED,
		concurrency: userConfig.concurrency ?? DEFAULT_CONCURRENCY
	};
	const candidateIds = userConfig.candidateIds ?? defaultCandidateIds(engine, userConfig.ownedIds);
	const adversaryPoolIds = new Set(engine.metadata.brawlers.map((b) => b.id));

	const totalUnits = (1 + candidateIds.length) * config.trials;
	let done = 0;
	const tick = () => userConfig.onProgress?.(++done, totalUnits);

	const ownedSet = new Set(userConfig.ownedIds);
	const baselineProbs = await runArm(
		engine,
		ownedSet,
		adversaryPoolIds,
		config,
		userConfig.signal,
		tick
	);
	const baseline = summarizeArm(baselineProbs, config.trials);

	const candidates: CandidateSummary[] = [];
	for (const id of candidateIds) {
		const extendedSet = new Set(ownedSet);
		extendedSet.add(id);
		const probs = await runArm(
			engine,
			extendedSet,
			adversaryPoolIds,
			config,
			userConfig.signal,
			tick
		);

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
