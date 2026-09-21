// Deterministic PRNG for the Monte Carlo stress-test. Not cryptographic —
// its only job is reproducible, *paired* random draws: seeding two RNGs
// with the same (seed, trialIndex) reproduces the same event / coin-flip /
// adversary-pool draws for a baseline set and a candidate-extended set, so
// stressTest.ts can compare them as a paired sample (common random numbers)
// instead of two independent ones.

export interface Rng {
	/** Uniform float in [0, 1). */
	next(): number;
}

// mulberry32: small, fast, decent-quality 32-bit PRNG.
function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// Mixes two 32-bit integers into one well-distributed 32-bit state
// (murmur3 finalizer), so nearby seeds/trialIndexes don't produce
// correlated mulberry32 streams.
function hash32(a: number, b: number): number {
	let h = (a ^ 0x9e3779b9) >>> 0;
	h = Math.imul(h ^ b, 0x85ebca6b) >>> 0;
	h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
	return (h ^ (h >>> 16)) >>> 0;
}

/**
 * Deterministic RNG for one simulation trial. `makeTrialRng(seed, i)`
 * always reproduces the same draw sequence — call it once per arm (baseline,
 * or one candidate-extended set) at the same trial index `i` to get paired
 * draws; each call returns an independent, freshly-seeded instance (no
 * shared mutable state), so the two calls' later draws can diverge safely
 * once the simulated decisions themselves diverge.
 */
export function makeTrialRng(seed: number, trialIndex: number): Rng {
	const next = mulberry32(hash32(seed | 0, trialIndex | 0));
	next(); // discard the first draw — decorrelates small (seed, trialIndex) pairs
	return { next };
}

/**
 * Categorical sample over non-negative `weights`, using one draw from `rng`.
 * Throws if every weight is zero (or the array is empty).
 */
export function sampleCategorical(weights: ArrayLike<number>, rng: Rng): number {
	let total = 0;
	for (let i = 0; i < weights.length; i++) total += weights[i];
	if (!(total > 0)) throw new Error('sampleCategorical: no positive-weight options');
	let r = rng.next() * total;
	for (let i = 0; i < weights.length; i++) {
		r -= weights[i];
		if (r <= 0) return i;
	}
	return weights.length - 1; // floating-point fallback
}

/**
 * Masked-softmax categorical sample: picks one index among the positions
 * where `mask` is truthy, weighted by softmax(logits / temperature) over
 * just those positions. Low temperature -> near-argmax ("nearly optimal");
 * higher temperature -> more exploration. Throws if no position is valid.
 */
export function sampleMaskedSoftmax(
	logits: ArrayLike<number>,
	mask: ArrayLike<boolean | number>,
	temperature: number,
	rng: Rng
): number {
	let maxLogit = -Infinity;
	for (let i = 0; i < logits.length; i++) {
		if (mask[i] && logits[i] > maxLogit) maxLogit = logits[i];
	}
	if (maxLogit === -Infinity) throw new Error('sampleMaskedSoftmax: no valid actions');
	const weights = new Float64Array(logits.length);
	for (let i = 0; i < logits.length; i++) {
		weights[i] = mask[i] ? Math.exp((logits[i] - maxLogit) / temperature) : 0;
	}
	return sampleCategorical(weights, rng);
}

/** `k` distinct indices from [0, n), uniformly, without replacement (partial Fisher-Yates). */
export function sampleWithoutReplacement(n: number, k: number, rng: Rng): number[] {
	if (k < 0 || k > n) throw new Error(`sampleWithoutReplacement: k=${k} out of range for n=${n}`);
	const pool = Array.from({ length: n }, (_, i) => i);
	const out: number[] = [];
	for (let i = 0; i < k; i++) {
		const j = i + Math.floor(rng.next() * (n - i));
		[pool[i], pool[j]] = [pool[j], pool[i]];
		out.push(pool[i]);
	}
	return out;
}
