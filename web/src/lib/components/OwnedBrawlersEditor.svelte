<script lang="ts">
	import type { Brawler } from '$lib/onnx/metadata';
	import { ownedBrawlers } from '$lib/pick/ownedBrawlersStore.svelte';
	import { resolveOwnedBrawlers } from '$lib/pick/ownedBrawlers';
	import { RARITY_COLORS } from '$lib/pick/constants';

	interface Props {
		brawlers: Brawler[];
	}
	let { brawlers }: Props = $props();

	const byName = $derived(new Map(brawlers.map((b) => [b.name, b])));
	const resolution = $derived(resolveOwnedBrawlers(ownedBrawlers.text, brawlers));

	function onInput(e: Event) {
		ownedBrawlers.setText((e.currentTarget as HTMLTextAreaElement).value);
	}
</script>

<div class="editor">
	<label for="owned-text">Your brawlers</label>
	<textarea
		id="owned-text"
		rows="3"
		placeholder="Shelly, Colt, 8-Bit, Larry & Lawrie…"
		value={ownedBrawlers.text}
		oninput={onInput}
	></textarea>
	<p class="hint">
		Comma-separated, partial names OK — matched the same way as picks (e.g. "shel" → Shelly). Used to filter
		recommendations to brawlers you own during a draft.
	</p>

	{#if resolution.tokens.length}
		<ul class="matches">
			{#each resolution.tokens as t (t.raw)}
				{@const b = t.matched ? byName.get(t.matched) : undefined}
				<li class:ok={t.matched} class:bad={!t.matched}>
					<span class="mark">{t.matched ? '✓' : '✗'}</span>
					<span class="raw">{t.raw}</span>
					{#if t.matched}
						<span class="arrow">→</span>
						<span class="name" style:color={b ? RARITY_COLORS[b.rarity] : undefined}>{t.matched}</span>
					{:else}
						<span class="no-match">no match</span>
					{/if}
				</li>
			{/each}
		</ul>
		<p class="summary">
			{resolution.ids.size} owned brawler{resolution.ids.size === 1 ? '' : 's'} saved
			{#if resolution.unmatchedRaw.length}
				&middot; {resolution.unmatchedRaw.length} unmatched
			{/if}
		</p>
	{:else}
		<p class="summary empty">No brawlers entered yet — recommendations will use the full roster.</p>
	{/if}

	{#if ownedBrawlers.text}
		<button type="button" class="clear" onclick={() => ownedBrawlers.setText('')}>Clear all</button>
	{/if}
</div>

<style>
	.editor {
		display: flex;
		flex-direction: column;
		gap: 0.5em;
	}
	label {
		font-weight: 700;
		font-size: 0.9rem;
	}
	textarea {
		width: 100%;
		box-sizing: border-box;
		padding: 0.7em 0.9em;
		border-radius: 0.6em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		background: var(--color-bg);
		color: var(--color-fg);
		font: inherit;
		font-size: 0.95rem;
		resize: vertical;
	}
	.hint {
		margin: 0;
		font-size: 0.78rem;
		opacity: 0.65;
	}
	.matches {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 0.35em;
		max-height: 40vh;
		overflow-y: auto;
	}
	.matches li {
		display: flex;
		align-items: center;
		gap: 0.35em;
		padding: 0.3em 0.55em;
		border-radius: 0.4em;
		background: color-mix(in oklab, var(--color-fg) 4%, transparent);
		font-size: 0.82rem;
	}
	.matches li.bad {
		background: color-mix(in oklab, var(--color-failure) 12%, transparent);
	}
	.mark {
		font-weight: 700;
	}
	li.ok .mark {
		color: var(--color-success, #3ecf5f);
	}
	li.bad .mark {
		color: var(--color-failure);
	}
	.raw {
		opacity: 0.75;
	}
	.arrow {
		opacity: 0.5;
	}
	.name {
		font-weight: 700;
	}
	.no-match {
		color: var(--color-failure);
		font-size: 0.78rem;
	}
	.summary {
		margin: 0;
		font-size: 0.8rem;
		opacity: 0.75;
	}
	.summary.empty {
		opacity: 0.6;
	}
	.clear {
		align-self: flex-start;
		border: none;
		background: none;
		color: var(--color-fg);
		text-decoration: underline;
		cursor: pointer;
		font: inherit;
		font-size: 0.8rem;
		opacity: 0.75;
		padding: 0;
	}
</style>
