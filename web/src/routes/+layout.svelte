<script lang="ts">
	import { page } from '$app/state';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();

	const tabs = [
		{ href: '/', label: 'Draft' },
		{ href: '/tiers', label: 'Tier Lists' },
		{ href: '/settings', label: 'Settings' }
	];
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<main>
	<nav class="app-nav">
		<span class="brand">BrawlCrawl</span>
		<div class="tabs">
			{#each tabs as tab (tab.href)}
				<a href={tab.href} class:active={page.url.pathname === tab.href}>{tab.label}</a>
			{/each}
		</div>
	</nav>

	{@render children()}
</main>

<style>
	:global(html) {
		color-scheme: light dark;
	}
	:global(body) {
		background: var(--color-bg);
		color: var(--color-fg);
		margin: 0;
		font-family:
			system-ui,
			-apple-system,
			'Segoe UI',
			sans-serif;
	}
	main {
		max-width: 64rem;
		margin: 0 auto;
		padding: 1rem;
		padding-bottom: 3rem;
		box-sizing: border-box;
		display: flex;
		flex-direction: column;
		gap: 1.1rem;
		min-height: 100dvh;
	}
	.app-nav {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1em;
		flex-wrap: wrap;
	}
	.brand {
		font-size: 1.1rem;
		font-weight: 700;
		opacity: 0.85;
		letter-spacing: 0.02em;
	}
	.tabs {
		display: flex;
		gap: 0.4em;
	}
	.tabs a {
		padding: 0.4em 0.8em;
		border-radius: 0.5em;
		color: var(--color-fg);
		text-decoration: none;
		font-size: 0.9rem;
		font-weight: 600;
		opacity: 0.65;
	}
	.tabs a:hover {
		opacity: 1;
		background: color-mix(in oklab, var(--color-fg) 6%, transparent);
	}
	.tabs a.active {
		opacity: 1;
		background: color-mix(in oklab, var(--color-primary) 18%, transparent);
		color: var(--color-primary);
	}
</style>
