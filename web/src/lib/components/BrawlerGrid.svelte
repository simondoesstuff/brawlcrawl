<script lang="ts">
	import type { Brawler } from '$lib/onnx/metadata';
	import BrawlerCard from './BrawlerCard.svelte';

	interface Entry {
		brawler: Brawler;
		score: number;
	}

	interface Props {
		above: Entry[];
		below: Entry[];
		scoreFmt: (n: number) => string;
		annotations?: Map<number, string>;
		dimmedIds?: Set<number>;
		highlightedId?: number | null;
		interactive?: boolean;
		onSelect?: (brawler: Brawler) => void;
		emptyMessage?: string;
	}

	let {
		above,
		below,
		scoreFmt,
		annotations = new Map(),
		dimmedIds,
		highlightedId = null,
		interactive = false,
		onSelect,
		emptyMessage = 'Nothing to show.'
	}: Props = $props();
</script>

{#if above.length === 0 && below.length === 0}
	<p class="empty">{emptyMessage}</p>
{:else}
	{#if above.length}
		<div class="grid">
			{#each above as entry (entry.brawler.id)}
				<BrawlerCard
					brawler={entry.brawler}
					rank={above.indexOf(entry) + 1}
					scoreStr={scoreFmt(entry.score)}
					annotation={annotations.get(entry.brawler.id)}
					dimmed={dimmedIds?.has(entry.brawler.id) ?? false}
					highlighted={highlightedId === entry.brawler.id}
					{interactive}
					{onSelect}
				/>
			{/each}
		</div>
	{/if}
	{#if above.length && below.length}
		<hr />
	{/if}
	{#if below.length}
		<div class="grid">
			{#each below as entry (entry.brawler.id)}
				<BrawlerCard
					brawler={entry.brawler}
					rank={above.length + below.indexOf(entry) + 1}
					scoreStr={scoreFmt(entry.score)}
					annotation={annotations.get(entry.brawler.id)}
					dimmed={dimmedIds?.has(entry.brawler.id) ?? false}
					highlighted={highlightedId === entry.brawler.id}
					{interactive}
					{onSelect}
				/>
			{/each}
		</div>
	{/if}
{/if}

<style>
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
		gap: 0.4em;
	}
	hr {
		margin: 0.75em 0;
		border: none;
		border-top: 1px solid color-mix(in oklab, var(--color-fg) 15%, transparent);
	}
	.empty {
		opacity: 0.6;
		padding: 1em 0;
	}
</style>
