<script lang="ts">
	import { onMount } from 'svelte';
	import StressTestResults from '$lib/components/StressTestResults.svelte';
	import {
		StressTestState,
		type Shortlist,
		type Speed
	} from '$lib/stress-test/stressTestState.svelte';

	const state = new StressTestState();

	onMount(() => {
		void state.loadMetadata();
	});

	const SPEED_OPTIONS: { value: Speed; label: string }[] = [
		{ value: 'quick', label: 'Quick' },
		{ value: 'balanced', label: 'Balanced' },
		{ value: 'thorough', label: 'Thorough' }
	];
	const SHORTLIST_OPTIONS: { value: Shortlist; label: string }[] = [
		{ value: 8, label: 'Top 8' },
		{ value: 16, label: 'Top 16' },
		{ value: 30, label: 'Top 30' },
		{ value: 'all', label: 'Every brawler' }
	];
</script>

<svelte:head>
	<title>BrawlCrawl — stress test</title>
</svelte:head>

<section>
	<h1>Stress Test</h1>
	<p class="hint">
		Simulates full drafts between your <a href="/settings">owned brawler set</a> and a near-optimal
		adversary with the whole roster available, then checks how much each candidate brawler would raise
		your odds if you added it.
	</p>

	{#if state.metadataStatus === 'loading'}
		<p class="status">Loading brawler data…</p>
	{:else if state.metadataStatus === 'error'}
		<p class="status error">{state.metadataError}</p>
	{:else if state.ownedIds.size === 0}
		<p class="status">No owned brawlers set yet — add some in <a href="/settings">Settings</a> first.</p>
	{:else}
		<div class="config">
			<span class="owned-count">
				{state.ownedIds.size} owned brawler{state.ownedIds.size === 1 ? '' : 's'}
			</span>

			<label>
				Candidates
				<select bind:value={state.shortlist} disabled={state.busy}>
					{#each SHORTLIST_OPTIONS as opt (opt.value)}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>

			<label>
				Trials
				<select bind:value={state.speed} disabled={state.busy}>
					{#each SPEED_OPTIONS as opt (opt.value)}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>

			{#if state.busy}
				<button type="button" onclick={() => state.cancel()}>Cancel</button>
			{:else}
				<button type="button" class="run" onclick={() => state.run()}>Run stress test</button>
			{/if}
		</div>

		{#if state.estimatedCalls !== null && !state.busy}
			<p class="estimate">~{state.estimatedCalls.toLocaleString()} model calls at this setting.</p>
		{/if}

		{#if state.view === 'loading-models'}
			<p class="status">Loading models…</p>
		{:else if state.view === 'running'}
			<p class="status">
				Simulating drafts… {state.progress.done.toLocaleString()} / {state.progress.total.toLocaleString()}
			</p>
			<progress value={state.progress.done} max={Math.max(state.progress.total, 1)}></progress>
		{:else if state.view === 'error'}
			<p class="status error">{state.errorMsg}</p>
		{/if}

		{#if state.result}
			<StressTestResults result={state.result} />
		{/if}
	{/if}
</section>

<style>
	.status {
		opacity: 0.7;
		padding: 2rem 0;
		text-align: center;
	}
	.status.error {
		color: var(--color-failure);
	}
	section {
		display: flex;
		flex-direction: column;
		gap: 0.9em;
	}
	h1 {
		font-size: 1.2rem;
		margin: 0;
	}
	.hint {
		margin: 0;
		font-size: 0.85rem;
		opacity: 0.7;
	}
	.config {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 1em;
	}
	.owned-count {
		font-size: 0.85rem;
		font-weight: 600;
		opacity: 0.75;
	}
	label {
		display: flex;
		align-items: center;
		gap: 0.4em;
		font-size: 0.85rem;
		font-weight: 600;
		opacity: 0.85;
	}
	select {
		padding: 0.3em 0.5em;
		border-radius: 0.4em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		background: var(--color-bg);
		color: var(--color-fg);
		font: inherit;
	}
	button {
		padding: 0.5em 1em;
		border-radius: 0.5em;
		border: none;
		background: color-mix(in oklab, var(--color-fg) 8%, transparent);
		color: var(--color-fg);
		font: inherit;
		font-weight: 700;
		cursor: pointer;
	}
	button.run {
		background: var(--color-primary);
		color: var(--color-bg);
	}
	button:hover {
		filter: brightness(1.08);
	}
	.estimate {
		margin: 0;
		font-size: 0.78rem;
		opacity: 0.55;
	}
	progress {
		width: 100%;
		accent-color: var(--color-primary);
	}
</style>
