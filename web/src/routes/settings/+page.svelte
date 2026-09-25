<script lang="ts">
	import { onMount } from 'svelte';
	import OwnedBrawlersEditor from '$lib/components/OwnedBrawlersEditor.svelte';
	import type { Metadata } from '$lib/onnx/metadata';

	type View = 'loading' | 'error' | 'ready';

	let view = $state<View>('loading');
	let errorMsg = $state('');
	let metadata = $state<Metadata | null>(null);

	onMount(async () => {
		try {
			const res = await fetch('/data/metadata.json');
			if (!res.ok) throw new Error(`Failed to load brawler data (${res.status})`);
			metadata = (await res.json()) as Metadata;
			view = 'ready';
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : String(e);
			view = 'error';
		}
	});
</script>

<svelte:head>
	<title>BrawlCrawl — settings</title>
</svelte:head>

<section>
	<h1>Settings</h1>

	{#if view === 'loading'}
		<p class="status">Loading brawler data…</p>
	{:else if view === 'error'}
		<p class="status error">{errorMsg}</p>
	{:else if metadata}
		<OwnedBrawlersEditor brawlers={metadata.brawlers} />
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
</style>
