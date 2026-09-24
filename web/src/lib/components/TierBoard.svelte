<script lang="ts">
	import type { Brawler } from '$lib/onnx/metadata';
	import { RARITY_COLORS } from '$lib/pick/constants';
	import { assignTiers, groupByTier } from '$lib/tierlist/tiers';
	import { TIER_COLORS } from '$lib/tierlist/tierColors';
	import type { MapTierList } from '$lib/tierlist/types';

	interface Props {
		map: MapTierList;
		brawlerMeta: Map<number, Brawler> | null;
	}
	let { map, brawlerMeta }: Props = $props();

	const groups = $derived(groupByTier(assignTiers(map.tier_list, (e) => e.win_rate)));
</script>

<div class="board">
	{#each groups as group (group.tier)}
		<div class="row">
			<span class="badge" style:--tier-color={TIER_COLORS[group.tier]}>{group.tier}</span>
			<div class="chips">
				{#each group.entries as t (t.entry.brawler_id)}
					{@const rarity = brawlerMeta?.get(t.entry.brawler_id)?.rarity}
					<span class="chip" style:--rarity-color={rarity ? RARITY_COLORS[rarity] : 'var(--color-fg)'}>
						<span class="name">{t.entry.name}</span>
						<span class="wr">{(t.entry.win_rate * 100).toFixed(1)}%</span>
					</span>
				{/each}
			</div>
		</div>
	{/each}
</div>

<style>
	.board {
		display: flex;
		flex-direction: column;
		gap: 0.5em;
	}
	.row {
		display: flex;
		align-items: stretch;
		gap: 0.6em;
	}
	.badge {
		flex: none;
		width: 2.2em;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 1.1rem;
		font-weight: 800;
		border-radius: 0.5em;
		color: color-mix(in oklab, var(--tier-color) 85%, black);
		background: color-mix(in oklab, var(--tier-color) 30%, transparent);
	}
	.chips {
		flex: 1;
		display: flex;
		flex-wrap: wrap;
		align-content: flex-start;
		gap: 0.4em;
		padding: 0.4em;
		border-radius: 0.5em;
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
	}
	.chip {
		display: inline-flex;
		align-items: baseline;
		gap: 0.4em;
		padding: 0.35em 0.6em;
		border-radius: 0.4em;
		border: 1px solid color-mix(in oklab, var(--rarity-color) 40%, transparent);
		background: color-mix(in oklab, var(--rarity-color) 10%, transparent);
	}
	.name {
		font-weight: 700;
		font-size: 0.85em;
		color: var(--rarity-color);
	}
	.wr {
		font-variant-numeric: tabular-nums;
		font-size: 0.75em;
		opacity: 0.75;
	}
</style>
