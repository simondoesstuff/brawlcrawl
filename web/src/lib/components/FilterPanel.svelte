<script lang="ts">
	import type { DraftState } from '$lib/pick/draftState.svelte';
	import { fuzzyMatches } from '$lib/pick/fuzzy';
	import { RARITY_COLORS } from '$lib/pick/constants';

	interface Props {
		draft: DraftState;
		onClose: () => void;
	}
	let { draft, onClose }: Props = $props();

	let search = $state('');

	const brawlers = $derived(draft.engine!.metadata.brawlers);
	const shown = $derived(search.trim() ? fuzzyMatches(search, brawlers.map((b) => b.name)) : brawlers.map((b) => b.name));
	const byName = $derived(new Map(brawlers.map((b) => [b.name, b])));

	function onBackdropKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onClose();
	}
</script>

<svelte:window onkeydown={onBackdropKeydown} />

<!-- svelte-ignore a11y_click_events_have_key_events -- decorative click-outside-to-close overlay; Escape is handled globally above -->
<div class="backdrop" onclick={onClose} role="presentation">
	<div
		class="panel"
		onclick={(e) => e.stopPropagation()}
		role="dialog"
		aria-modal="true"
		aria-label="Owned brawlers filter"
		tabindex="-1"
	>
		<header>
			<h2>Your brawlers</h2>
			<button type="button" class="close" onclick={onClose} aria-label="Close">✕</button>
		</header>
		<p class="hint">
			Select the brawlers you own. When enabled, recommendations only consider these — press
			<kbd>/</kbd> then <kbd>Enter</kbd> during a draft to toggle.
		</p>
		<input class="search" placeholder="Search…" bind:value={search} />
		<div class="grid">
			{#each shown as name (name)}
				{@const b = byName.get(name)!}
				<label class="chip" style:--c={RARITY_COLORS[b.rarity]}>
					<input type="checkbox" checked={draft.filterIds.has(b.id)} onchange={() => draft.toggleOwned(b.id)} />
					{name}
				</label>
			{/each}
		</div>
		<footer>
			<span>{draft.filterIds.size} owned</span>
			<button type="button" onclick={() => draft.clearOwned()}>Clear all</button>
		</footer>
	</div>
</div>

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: color-mix(in oklab, black 45%, transparent);
		display: flex;
		align-items: flex-end;
		justify-content: center;
		z-index: 50;
	}
	.panel {
		background: var(--color-bg);
		color: var(--color-fg);
		width: 100%;
		max-width: 40rem;
		max-height: 85vh;
		border-radius: 1em 1em 0 0;
		padding: 1em;
		display: flex;
		flex-direction: column;
		gap: 0.7em;
		box-sizing: border-box;
	}
	@media (min-width: 640px) {
		.backdrop {
			align-items: center;
		}
		.panel {
			border-radius: 1em;
			max-height: 75vh;
		}
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	h2 {
		margin: 0;
		font-size: 1.05rem;
	}
	.close {
		border: none;
		background: none;
		color: inherit;
		font-size: 1rem;
		cursor: pointer;
	}
	.hint {
		margin: 0;
		font-size: 0.78rem;
		opacity: 0.65;
	}
	kbd {
		border: 1px solid currentColor;
		border-radius: 0.3em;
		padding: 0 0.3em;
	}
	.search {
		padding: 0.6em 0.8em;
		border-radius: 0.5em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		background: var(--color-bg);
		color: var(--color-fg);
		font: inherit;
	}
	.grid {
		overflow-y: auto;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));
		gap: 0.35em;
	}
	.chip {
		display: flex;
		align-items: center;
		gap: 0.4em;
		padding: 0.4em 0.55em;
		border-radius: 0.4em;
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
		font-size: 0.85rem;
		font-weight: 600;
		color: var(--c);
		cursor: pointer;
	}
	footer {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 0.8rem;
		opacity: 0.75;
	}
	footer button {
		border: none;
		background: none;
		color: var(--color-fg);
		text-decoration: underline;
		cursor: pointer;
		font: inherit;
	}
</style>
