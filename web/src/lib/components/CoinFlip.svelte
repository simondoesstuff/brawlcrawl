<script lang="ts">
	import type { DraftState } from '$lib/pick/draftState.svelte';
	import BrawlerGrid from './BrawlerGrid.svelte';

	interface Props {
		draft: DraftState;
	}
	let { draft }: Props = $props();

	const event = $derived(draft.event!);
	const above = $derived(draft.mapScores.filter((e) => e.score >= 0));
	const below = $derived(draft.mapScores.filter((e) => e.score < 0));

	function onKeydown(e: KeyboardEvent) {
		if (e.key.toLowerCase() === 'y') void draft.startDraft(true);
		else if (e.key.toLowerCase() === 'n') void draft.startDraft(false);
	}
</script>

<svelte:window onkeydown={onKeydown} />

<section>
	<div class="header">
		<button type="button" class="back" onclick={() => draft.backToMapSelect()}>← change map</button>
		<h1>{event.map_name} <span class="mode">{event.mode}</span></h1>
	</div>

	<BrawlerGrid {above} {below} scoreFmt={(s) => `${s >= 0 ? '+' : ''}${s.toFixed(2)}`} annotations={draft.overviewAnn} dimmedIds={draft.filterIds.size ? new Set(draft.mapScores.filter((e) => !draft.filterIds.has(e.brawler.id)).map((e) => e.brawler.id)) : undefined} />

	<div class="flip">
		<p>Ally picks first?</p>
		<div class="buttons">
			<button type="button" onclick={() => draft.startDraft(true)}>Yes <kbd>y</kbd></button>
			<button type="button" onclick={() => draft.startDraft(false)}>No <kbd>n</kbd></button>
		</div>
	</div>
</section>

<style>
	section {
		display: flex;
		flex-direction: column;
		gap: 1em;
	}
	.header {
		display: flex;
		flex-direction: column;
		gap: 0.3em;
	}
	.back {
		align-self: flex-start;
		border: none;
		background: none;
		color: var(--color-fg);
		opacity: 0.6;
		font: inherit;
		cursor: pointer;
		padding: 0;
	}
	h1 {
		font-size: 1.2rem;
		margin: 0;
	}
	.mode {
		opacity: 0.55;
		font-size: 0.8em;
		font-weight: 400;
	}
	.flip {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.6em;
		padding: 1em;
		border-radius: 0.6em;
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
	}
	.flip p {
		margin: 0;
		font-weight: 600;
	}
	.buttons {
		display: flex;
		gap: 0.6em;
	}
	.buttons button {
		display: flex;
		align-items: center;
		gap: 0.4em;
		padding: 0.6em 1.2em;
		border-radius: 0.5em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		background: var(--color-bg);
		color: var(--color-fg);
		font: inherit;
		font-weight: 600;
		cursor: pointer;
	}
	.buttons button:hover {
		border-color: var(--color-primary);
	}
	kbd {
		font-size: 0.75em;
		opacity: 0.6;
		border: 1px solid currentColor;
		border-radius: 0.3em;
		padding: 0 0.3em;
	}
</style>
