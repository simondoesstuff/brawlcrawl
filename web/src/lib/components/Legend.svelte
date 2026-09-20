<script lang="ts">
	import { ANNOTATION_LEGEND, RARITY_COLORS, RARITY_ORDER } from '$lib/pick/constants';
</script>

<div class="legend">
	<div class="row">
		{#each RARITY_ORDER as rarity (rarity)}
			<span class="chip"><span class="dot" style:background={RARITY_COLORS[rarity]}></span>{rarity}</span>
		{/each}
	</div>
	<div class="row">
		{#each ANNOTATION_LEGEND as a (a.key)}
			<span class="chip"><span class="symbol">{a.symbol}</span>{a.label}</span>
		{/each}
	</div>
</div>

<style>
	.legend {
		display: flex;
		flex-direction: column;
		gap: 0.35em;
		font-size: 0.75rem;
		opacity: 0.75;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75em;
	}
	.chip {
		display: inline-flex;
		align-items: center;
		gap: 0.35em;
		white-space: nowrap;
	}
	.dot {
		width: 0.65em;
		height: 0.65em;
		border-radius: 50%;
		display: inline-block;
	}
	.symbol {
		font-size: 0.95em;
	}
</style>
