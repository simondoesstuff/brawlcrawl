<script lang="ts">
	import { fuzzyFind } from '$lib/pick/fuzzy';
	import type { MapTierList } from '$lib/tierlist/types';
	import SearchInput from './SearchInput.svelte';

	interface Props {
		maps: MapTierList[];
		selectedEventId: number | null;
		onSelect: (eventId: number) => void;
	}
	let { maps, selectedEventId, onSelect }: Props = $props();

	const mapNames = $derived(maps.map((m) => m.map_name));
	const byName = $derived(new Map(maps.map((m) => [m.map_name, m])));

	function selectByName(mapName: string) {
		const matched = fuzzyFind(mapName, mapNames);
		const m = matched ? byName.get(matched) : undefined;
		if (m) onSelect(m.event_id);
	}
</script>

<div class="picker">
	<SearchInput placeholder="Search maps…" candidates={mapNames} onSubmit={selectByName} />
	<ul class="maps">
		{#each maps as m (m.event_id)}
			<li>
				<button
					type="button"
					class:active={m.event_id === selectedEventId}
					onclick={() => onSelect(m.event_id)}
				>
					<span class="name">{m.map_name}</span>
					<span class="mode">{m.mode}</span>
				</button>
			</li>
		{/each}
	</ul>
</div>

<style>
	.picker {
		display: flex;
		flex-direction: column;
		gap: 0.6em;
	}
	.maps {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
		gap: 0.45em;
	}
	.maps button {
		width: 100%;
		display: flex;
		justify-content: space-between;
		gap: 0.5em;
		padding: 0.6em 0.8em;
		border-radius: 0.5em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 12%, transparent);
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
		color: var(--color-fg);
		font: inherit;
		cursor: pointer;
		text-align: left;
	}
	.maps button:hover,
	.maps button:focus-visible {
		border-color: var(--color-primary);
		background: color-mix(in oklab, var(--color-primary) 14%, transparent);
	}
	.maps button.active {
		border-color: var(--color-primary);
		background: color-mix(in oklab, var(--color-primary) 22%, transparent);
	}
	.name {
		font-weight: 600;
	}
	.mode {
		opacity: 0.55;
		font-size: 0.8em;
	}
</style>
