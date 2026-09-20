<script lang="ts">
	import { onMount } from 'svelte';
	import { fuzzyMatches } from '$lib/pick/fuzzy';

	interface Props {
		placeholder: string;
		candidates: string[];
		disabled?: boolean;
		hint?: string;
		onSubmit: (text: string) => void;
		onInputChange?: (text: string) => void;
		onUndo?: () => void;
	}

	let { placeholder, candidates, disabled = false, hint, onSubmit, onInputChange, onUndo }: Props = $props();

	let value = $state('');
	let inputEl = $state<HTMLInputElement | null>(null);
	let suggestionIdx = $state(-1);
	let focused = $state(false);
	let autofocusDesktop = $state(false);

	const suggestions = $derived(value.trim() ? fuzzyMatches(value, candidates).slice(0, 8) : []);

	function isTypingElsewhere(): boolean {
		const el = document.activeElement;
		if (!el || el === inputEl) return false;
		const tag = el.tagName;
		return tag === 'INPUT' || tag === 'TEXTAREA' || (el as HTMLElement).isContentEditable;
	}

	onMount(() => {
		autofocusDesktop = matchMedia('(pointer: fine)').matches;
		if (autofocusDesktop) inputEl?.focus();

		function onWindowKeydown(e: KeyboardEvent) {
			if (document.activeElement === inputEl) return;
			if (isTypingElsewhere()) return;
			if (e.key === '/') {
				e.preventDefault();
				inputEl?.focus();
			}
		}
		window.addEventListener('keydown', onWindowKeydown);
		return () => window.removeEventListener('keydown', onWindowKeydown);
	});

	function refocus() {
		if (autofocusDesktop) inputEl?.focus();
	}

	function commit(text: string) {
		onSubmit(text);
		value = '';
		suggestionIdx = -1;
		onInputChange?.('');
		refocus();
	}

	function handleInput() {
		suggestionIdx = -1;
		onInputChange?.(value);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			if (suggestionIdx >= 0 && suggestions[suggestionIdx]) {
				commit(suggestions[suggestionIdx]);
			} else {
				commit(value);
			}
		} else if (e.key === 'Escape') {
			e.preventDefault();
			value = '';
			suggestionIdx = -1;
			onInputChange?.('');
		} else if (e.key === 'ArrowDown') {
			if (suggestions.length) {
				e.preventDefault();
				suggestionIdx = Math.min(suggestionIdx + 1, suggestions.length - 1);
			}
		} else if (e.key === 'ArrowUp') {
			if (suggestions.length) {
				e.preventDefault();
				suggestionIdx = Math.max(suggestionIdx - 1, -1);
			}
		} else if (e.key === 'Backspace' && value === '' && onUndo) {
			e.preventDefault();
			onUndo();
		}
	}
</script>

<div class="wrap">
	<input
		bind:this={inputEl}
		bind:value
		oninput={handleInput}
		onkeydown={handleKeydown}
		onfocus={() => (focused = true)}
		onblur={() => (focused = false)}
		{disabled}
		{placeholder}
		autocomplete="off"
		autocorrect="off"
		autocapitalize="off"
		spellcheck="false"
		inputmode="search"
	/>
	{#if hint}<span class="hint">{hint}</span>{/if}
	{#if focused && suggestions.length}
		<ul class="suggestions">
			{#each suggestions as s, i (s)}
				<li>
					<button
						type="button"
						class:active={i === suggestionIdx}
						onmousedown={(e) => e.preventDefault()}
						onclick={() => commit(s)}
					>
						{s}
					</button>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
	}
	input {
		width: 100%;
		box-sizing: border-box;
		padding: 0.7em 0.9em;
		border-radius: 0.6em;
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		background: var(--color-bg);
		color: var(--color-fg);
		font-size: 1rem;
	}
	input:disabled {
		opacity: 0.5;
	}
	.hint {
		display: block;
		margin-top: 0.3em;
		font-size: 0.72rem;
		opacity: 0.55;
	}
	.suggestions {
		position: absolute;
		z-index: 10;
		top: calc(100% + 0.25em);
		left: 0;
		right: 0;
		margin: 0;
		padding: 0.25em;
		list-style: none;
		background: var(--color-bg);
		border: 1px solid color-mix(in oklab, var(--color-fg) 20%, transparent);
		border-radius: 0.6em;
		box-shadow: 0 8px 24px color-mix(in oklab, var(--color-fg) 15%, transparent);
		max-height: 60vh;
		overflow-y: auto;
	}
	.suggestions li button {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.55em 0.7em;
		border: none;
		background: none;
		border-radius: 0.4em;
		font: inherit;
		color: var(--color-fg);
		cursor: pointer;
	}
	.suggestions li button.active,
	.suggestions li button:hover {
		background: color-mix(in oklab, var(--color-primary) 35%, transparent);
	}
</style>
