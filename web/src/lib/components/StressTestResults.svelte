<script lang="ts">
	import type { StressTestResult } from '$lib/stress-test/stressTest';
	import BrawlerGrid from './BrawlerGrid.svelte';

	interface Props {
		result: StressTestResult;
	}
	let { result }: Props = $props();

	const winPct = $derived(result.baseline.winRate * 100);
	const stderrPct = $derived(result.baseline.stderr * 100);

	// candidates arrives sorted descending by delta (stressTest.ts) — splitting
	// preserves that order within each half.
	const above = $derived(
		result.candidates.filter((c) => c.delta > 0).map((c) => ({ brawler: c.brawler, score: c.delta }))
	);
	const below = $derived(
		result.candidates.filter((c) => c.delta <= 0).map((c) => ({ brawler: c.brawler, score: c.delta }))
	);

	function fmtDelta(d: number): string {
		const pp = d * 100;
		return `${pp >= 0 ? '+' : ''}${pp.toFixed(1)}pp`;
	}
</script>

<div class="results">
	<div class="baseline">
		<span class="label">Win chance with your set</span>
		<span class="value">{winPct.toFixed(1)}%</span>
		<span class="stderr">± {stderrPct.toFixed(1)}pp over {result.baseline.trials} trials</span>
	</div>

	<div class="next">
		<h2>Which brawler should you get next?</h2>
		<p class="hint">
			Win-rate change from adding each brawler, paired against the same simulated drafts as the baseline above.
		</p>
		<BrawlerGrid {above} {below} scoreFmt={fmtDelta} emptyMessage="No candidates tested." />
	</div>
</div>

<style>
	.results {
		display: flex;
		flex-direction: column;
		gap: 1.1em;
	}
	.baseline {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.15em;
		padding: 1em 1.2em;
		border-radius: 0.6em;
		background: color-mix(in oklab, var(--color-primary) 10%, transparent);
	}
	.label {
		font-size: 0.85rem;
		opacity: 0.75;
		font-weight: 600;
	}
	.value {
		font-size: 2rem;
		font-weight: 800;
		line-height: 1.1;
	}
	.stderr {
		font-size: 0.75rem;
		opacity: 0.6;
	}
	.next {
		display: flex;
		flex-direction: column;
		gap: 0.6em;
	}
	h2 {
		font-size: 1rem;
		margin: 0;
	}
	.hint {
		margin: 0;
		font-size: 0.8rem;
		opacity: 0.65;
	}
</style>
