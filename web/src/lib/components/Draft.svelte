<script lang="ts">
	import type { DraftState } from '$lib/pick/draftState.svelte';
	import BrawlerGrid from './BrawlerGrid.svelte';
	import DraftBoard from './DraftBoard.svelte';
	import SearchInput from './SearchInput.svelte';
	import type { Brawler } from '$lib/onnx/metadata';

	interface Props {
		draft: DraftState;
		onOpenFilter: () => void;
	}
	let { draft, onOpenFilter }: Props = $props();

	let highlightedId = $state<number | null>(null);
	let confirmingCancel = $state(false);

	const grid = $derived(draft.visibleScores());

	function onSelect(b: Brawler) {
		void draft.submitBrawler(b);
	}

	function onCancelClick() {
		if (confirmingCancel) {
			draft.cancelDraft();
			confirmingCancel = false;
		} else {
			confirmingCancel = true;
		}
	}

	// Global undo shortcut, independent of the search input's own mount state
	// (the input disappears once pick-6 is reached, but undo should still work).
	function isTypingElsewhere(): boolean {
		const el = document.activeElement;
		if (!el) return false;
		const tag = el.tagName;
		return tag === 'INPUT' || tag === 'TEXTAREA' || (el as HTMLElement).isContentEditable;
	}
	function onWindowKeydown(e: KeyboardEvent) {
		if (e.key === 'u' && draft.canUndo && !isTypingElsewhere()) {
			e.preventDefault();
			void draft.undo();
		}
	}
</script>

<svelte:window onkeydown={onWindowKeydown} />

<section>
	<DraftBoard
		allyBans={draft.allyBans}
		enemyBans={draft.enemyBans}
		picks={draft.picks}
		phase={draft.phase}
		allyFirst={draft.allyFirst}
		finalMode={draft.finalMode}
		turnSchedule={draft.turnSchedule}
	/>

	<div class="toolbar">
		<span class="phase">{draft.phaseLabel}</span>
		<div class="actions">
			{#if draft.filterIds.size}
				<button type="button" class:active={draft.filterOn} onclick={() => draft.toggleFilter()}>
					filter {draft.filterOn ? 'on' : 'off'} <kbd>/</kbd>
				</button>
			{/if}
			<button type="button" onclick={onOpenFilter}>owned brawlers</button>
			<button type="button" disabled={!draft.canUndo} onclick={() => draft.undo()}>undo <kbd>u</kbd></button>
			<button type="button" class="danger" onclick={onCancelClick}>
				{confirmingCancel ? 'confirm cancel?' : 'cancel draft'}
			</button>
		</div>
	</div>

	{#if draft.finalMode}
		{#if draft.pick6Loading}
			<p class="loading">Scoring pick-6 candidates…</p>
		{:else}
			<p class="pick6-note">
				{draft.pick6IsAllyPick
					? 'Your pick — ranked by P(ally wins).'
					: "Enemy's last pick — ranked by P(ally wins), so the top of this list is their worst option against your comp."}
			</p>
			<div class="next-steps">
				<button type="button" onclick={() => draft.restartSameMap()}>draft again (same map)</button>
				<button type="button" onclick={() => draft.backToMapSelect()}>change map</button>
			</div>
		{/if}
	{:else}
		<SearchInput
			placeholder="Ban or pick a brawler…"
			candidates={draft.availableNames}
			onSubmit={(text) => void draft.submitText(text)}
			onInputChange={(text) => (highlightedId = draft.bestMatchId(text))}
			onUndo={() => void draft.undo()}
			hint={draft.qLoading ? 'scoring…' : undefined}
		/>
	{/if}

	{#if draft.submitError}<p class="error">{draft.submitError}</p>{/if}

	<BrawlerGrid
		above={grid.above}
		below={grid.below}
		scoreFmt={grid.scoreFmt}
		annotations={grid.annotations}
		highlightedId={grid.finalMode ? null : highlightedId}
		interactive={!grid.finalMode}
		{onSelect}
		emptyMessage={grid.finalMode ? 'No candidates left.' : 'Loading…'}
	/>
</section>

<style>
	section {
		display: flex;
		flex-direction: column;
		gap: 0.8em;
	}
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: 0.5em;
	}
	.phase {
		font-weight: 700;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4em;
	}
	.actions button {
		display: flex;
		align-items: center;
		gap: 0.3em;
		padding: 0.35em 0.6em;
		border-radius: 0.4em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 18%, transparent);
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
		color: var(--color-fg);
		font: inherit;
		font-size: 0.78rem;
		cursor: pointer;
	}
	.actions button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.actions button.active {
		border-color: var(--color-primary);
		background: color-mix(in oklab, var(--color-primary) 20%, transparent);
	}
	.actions button.danger {
		color: var(--color-failure);
	}
	kbd {
		font-size: 0.85em;
		opacity: 0.7;
		border: 1px solid currentColor;
		border-radius: 0.3em;
		padding: 0 0.3em;
	}
	.error {
		color: var(--color-failure);
		margin: 0;
	}
	.loading,
	.pick6-note {
		margin: 0;
		font-size: 0.85rem;
		opacity: 0.75;
	}
	.next-steps {
		display: flex;
		gap: 0.5em;
	}
	.next-steps button {
		padding: 0.55em 1em;
		border-radius: 0.5em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		background: var(--color-bg);
		color: var(--color-fg);
		font: inherit;
		font-weight: 600;
		cursor: pointer;
	}
	.next-steps button:hover {
		border-color: var(--color-primary);
	}
</style>
