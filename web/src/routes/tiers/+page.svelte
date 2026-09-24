<script lang="ts">
	import { onMount } from 'svelte';
	import TierBoard from '$lib/components/TierBoard.svelte';
	import TierMapPicker from '$lib/components/TierMapPicker.svelte';
	import type { Brawler, Metadata } from '$lib/onnx/metadata';
	import type { TierListsData } from '$lib/tierlist/types';

	type View = 'loading' | 'error' | 'ready';

	let view = $state<View>('loading');
	let errorMsg = $state<string | null>(null);
	let data = $state<TierListsData | null>(null);
	let brawlerMeta = $state<Map<number, Brawler> | null>(null);
	let selectedEventId = $state<number | null>(null);

	onMount(async () => {
		try {
			const res = await fetch('/data/tier_lists.json');
			if (!res.ok) throw new Error(`Failed to load tier lists (${res.status})`);
			data = (await res.json()) as TierListsData;
			selectedEventId = data.maps[0]?.event_id ?? null;
			view = 'ready';
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : String(e);
			view = 'error';
			return;
		}

		// Best-effort: rarity coloring needs metadata.json from `just export-onnx`.
		// Tier lists themselves don't depend on the ONNX pipeline, so a missing
		// or failed fetch here just means brawler names render uncolored.
		try {
			const metaRes = await fetch('/data/metadata.json');
			if (metaRes.ok) {
				const metadata = (await metaRes.json()) as Metadata;
				brawlerMeta = new Map(metadata.brawlers.map((b) => [b.id, b]));
			}
		} catch {
			/* rarity coloring stays neutral */
		}
	});

	const selectedMap = $derived(data?.maps.find((m) => m.event_id === selectedEventId) ?? null);
</script>

<svelte:head>
	<title>BrawlCrawl — tier lists</title>
</svelte:head>

{#if view === 'loading'}
	<p class="status">Loading tier lists…</p>
{:else if view === 'error'}
	<p class="status error">{errorMsg}</p>
{:else if data}
	<section>
		<h1>Tier Lists</h1>
		<p class="hint">
			First-pick win probability per map: Monte Carlo–simulated against uniformly random teammates
			and opponents, so 50% is the baseline brawler.
		</p>

		<TierMapPicker maps={data.maps} {selectedEventId} onSelect={(id) => (selectedEventId = id)} />

		{#if selectedMap}
			<TierBoard map={selectedMap} {brawlerMeta} />
		{/if}
	</section>
{/if}

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
		opacity: 0.6;
		font-size: 0.85rem;
	}
</style>
