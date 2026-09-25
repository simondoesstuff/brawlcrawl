<script lang="ts">
	import type { DraftState } from '$lib/pick/draftState.svelte';
	import OwnedBrawlersEditor from './OwnedBrawlersEditor.svelte';

	interface Props {
		draft: DraftState;
		onClose: () => void;
	}
	let { draft, onClose }: Props = $props();

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
		aria-label="Owned brawlers"
		tabindex="-1"
	>
		<header>
			<h2>Your brawlers</h2>
			<button type="button" class="close" onclick={onClose} aria-label="Close">✕</button>
		</header>
		<p class="hint">
			When you have any brawlers saved, recommendations only consider these — press <kbd>/</kbd> then
			<kbd>Enter</kbd> during a draft to toggle. Also editable from the Settings tab.
		</p>
		<OwnedBrawlersEditor brawlers={draft.engine!.metadata.brawlers} />
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
		overflow-y: auto;
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
</style>
