<script lang="ts">
	import type { Brawler } from '$lib/onnx/metadata';
	import { ANNOTATION_LEGEND, RARITY_COLORS } from '$lib/pick/constants';

	interface Props {
		brawler: Brawler;
		rank: number;
		scoreStr: string;
		annotation?: string;
		highlighted?: boolean;
		dimmed?: boolean;
		interactive?: boolean;
		onSelect?: (brawler: Brawler) => void;
	}

	let { brawler, rank, scoreStr, annotation, highlighted = false, dimmed = false, interactive = false, onSelect }: Props = $props();

	const color = $derived(RARITY_COLORS[brawler.rarity] ?? '#c9ccd6');
	const symbol = $derived(ANNOTATION_LEGEND.find((a) => a.key === annotation)?.symbol);
</script>

<button
	type="button"
	class="card"
	class:highlighted
	class:dimmed
	class:interactive
	disabled={!interactive}
	onclick={() => onSelect?.(brawler)}
	style:--rarity-color={color}
>
	<span class="rank">{rank}</span>
	<span class="name">{brawler.name}</span>
	{#if symbol}<span class="ann" title={annotation}>{symbol}</span>{/if}
	<span class="score">{scoreStr}</span>
</button>

<style>
	.card {
		display: grid;
		grid-template-columns: 1.6em 1fr 1.4em auto;
		align-items: center;
		gap: 0.35em;
		width: 100%;
		padding: 0.45em 0.6em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 12%, transparent);
		border-radius: 0.5em;
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
		font: inherit;
		text-align: left;
		cursor: default;
		color: var(--color-fg);
	}
	.card.interactive {
		cursor: pointer;
	}
	.card.interactive:hover,
	.card.interactive:focus-visible {
		background: color-mix(in oklab, var(--rarity-color) 16%, transparent);
		border-color: var(--rarity-color);
		outline: none;
	}
	.card.highlighted {
		background: color-mix(in oklab, var(--rarity-color) 28%, transparent);
		border-color: var(--rarity-color);
	}
	.card.dimmed {
		opacity: 0.35;
	}
	.rank {
		font-size: 0.75em;
		opacity: 0.55;
		text-align: right;
	}
	.name {
		font-weight: 700;
		color: var(--rarity-color);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.ann {
		text-align: center;
	}
	.score {
		font-variant-numeric: tabular-nums;
		font-size: 0.85em;
		opacity: 0.85;
		text-align: right;
	}
</style>
