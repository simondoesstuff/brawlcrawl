<script lang="ts">
	import type { DraftState } from '$lib/pick/draftState.svelte';
	import SearchInput from './SearchInput.svelte';

	interface Props {
		draft: DraftState;
	}
	let { draft }: Props = $props();

	const events = $derived(draft.engine!.metadata.events);
</script>

<section>
	<h1>Pick a map</h1>
	<SearchInput
		placeholder="Search maps…"
		candidates={draft.mapNames}
		onSubmit={(text) => draft.selectMap(text)}
	/>
	{#if draft.submitError}<p class="error">{draft.submitError}</p>{/if}

	<ul class="maps">
		{#each events as e (e.id)}
			<li>
				<button type="button" onclick={() => draft.selectMap(e.map_name)}>
					<span class="name">{e.map_name}</span>
					<span class="mode">{e.mode}</span>
				</button>
			</li>
		{/each}
	</ul>
</section>

<style>
	section {
		display: flex;
		flex-direction: column;
		gap: 0.9em;
	}
	h1 {
		font-size: 1.2rem;
		margin: 0;
	}
	.error {
		color: var(--color-failure);
		margin: 0;
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
	.name {
		font-weight: 600;
	}
	.mode {
		opacity: 0.55;
		font-size: 0.8em;
	}
</style>
