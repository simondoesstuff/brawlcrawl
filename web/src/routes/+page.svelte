<script lang="ts">
	import { onMount } from 'svelte';
	import { DraftState } from '$lib/pick/draftState.svelte';
	import MapSelect from '$lib/components/MapSelect.svelte';
	import CoinFlip from '$lib/components/CoinFlip.svelte';
	import Draft from '$lib/components/Draft.svelte';
	import FilterPanel from '$lib/components/FilterPanel.svelte';
	import Legend from '$lib/components/Legend.svelte';

	const draft = new DraftState();
	let filterOpen = $state(false);

	onMount(() => {
		void draft.init();
	});
</script>

<svelte:head>
	<title>BrawlCrawl — draft assist</title>
</svelte:head>

<main>
	<header class="app-header">
		<h1>BrawlCrawl</h1>
	</header>

	{#if draft.view === 'loading'}
		<p class="status">Loading models…</p>
	{:else if draft.view === 'error'}
		<p class="status error">{draft.errorMsg}</p>
	{:else}
		{#if draft.view === 'map-select'}
			<MapSelect {draft} />
		{:else if draft.view === 'coin-flip'}
			<CoinFlip {draft} />
		{:else if draft.view === 'drafting'}
			<Draft {draft} onOpenFilter={() => (filterOpen = true)} />
		{/if}

		<footer class="app-footer">
			<Legend />
		</footer>

		{#if filterOpen}
			<FilterPanel {draft} onClose={() => (filterOpen = false)} />
		{/if}
	{/if}
</main>

<style>
	:global(html) {
		color-scheme: light dark;
	}
	:global(body) {
		background: var(--color-bg);
		color: var(--color-fg);
		margin: 0;
		font-family:
			system-ui,
			-apple-system,
			'Segoe UI',
			sans-serif;
	}
	main {
		max-width: 64rem;
		margin: 0 auto;
		padding: 1rem;
		padding-bottom: 3rem;
		box-sizing: border-box;
		display: flex;
		flex-direction: column;
		gap: 1.1rem;
		min-height: 100dvh;
	}
	.app-header h1 {
		font-size: 1.1rem;
		margin: 0;
		opacity: 0.85;
		letter-spacing: 0.02em;
	}
	.status {
		opacity: 0.7;
		padding: 2rem 0;
		text-align: center;
	}
	.status.error {
		color: var(--color-failure);
	}
	.app-footer {
		margin-top: auto;
		padding-top: 1rem;
		border-top: 1px solid color-mix(in oklab, var(--color-fg) 10%, transparent);
	}
</style>
