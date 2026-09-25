<script lang="ts">
	import { onMount } from 'svelte';
	import TierBoard from '$lib/components/TierBoard.svelte';
	import TierMapPicker from '$lib/components/TierMapPicker.svelte';
	import type { Brawler, Metadata } from '$lib/onnx/metadata';
	import type { TierListsData } from '$lib/tierlist/types';

	type Policy = 'random' | 'optimal';
	type PolicyStatus = 'idle' | 'loading' | 'ready' | 'error';
	type View = 'loading' | 'error' | 'ready';

	const POLICIES: Policy[] = ['random', 'optimal'];

	const POLICY_FILES: Record<Policy, string> = {
		random: '/data/tier_lists.json',
		optimal: '/data/tier_lists_optimal.json'
	};
	const POLICY_LABELS: Record<Policy, string> = {
		random: 'Bad randoms',
		optimal: 'Optimal picks'
	};
	const POLICY_HINTS: Record<Policy, string> = {
		random:
			'First-pick win probability per map: Monte Carlo–simulated against uniformly random teammates and opponents, so 50% is the baseline brawler.',
		optimal:
			'First-pick win probability per map: Monte Carlo–simulated against teammates and opponents who draft well (DraftQNetwork), so 50% is the baseline brawler.'
	};

	let policy = $state<Policy>('random');
	let cache = $state<Partial<Record<Policy, TierListsData>>>({});
	let statusByPolicy = $state<Record<Policy, PolicyStatus>>({ random: 'idle', optimal: 'idle' });
	let errorByPolicy = $state<Partial<Record<Policy, string>>>({});
	let brawlerMeta = $state<Map<number, Brawler> | null>(null);
	let selectedEventId = $state<number | null>(null);

	async function loadPolicy(p: Policy) {
		if (cache[p] || statusByPolicy[p] === 'loading') return;
		statusByPolicy = { ...statusByPolicy, [p]: 'loading' };
		try {
			const res = await fetch(POLICY_FILES[p]);
			if (!res.ok) throw new Error(`Failed to load tier lists (${res.status})`);
			cache = { ...cache, [p]: (await res.json()) as TierListsData };
			statusByPolicy = { ...statusByPolicy, [p]: 'ready' };
		} catch (e) {
			errorByPolicy = { ...errorByPolicy, [p]: e instanceof Error ? e.message : String(e) };
			statusByPolicy = { ...statusByPolicy, [p]: 'error' };
		}
	}

	function selectPolicy(p: Policy) {
		policy = p;
		void loadPolicy(p);
	}

	onMount(() => {
		void loadPolicy('random');

		// Best-effort: rarity coloring needs metadata.json from `just export-onnx`.
		// Tier lists themselves don't depend on the ONNX pipeline, so a missing
		// or failed fetch here just means brawler names render uncolored.
		void (async () => {
			try {
				const metaRes = await fetch('/data/metadata.json');
				if (metaRes.ok) {
					const metadata = (await metaRes.json()) as Metadata;
					brawlerMeta = new Map(metadata.brawlers.map((b) => [b.id, b]));
				}
			} catch {
				/* rarity coloring stays neutral */
			}
		})();
	});

	const view = $derived<View>(
		statusByPolicy[policy] === 'ready'
			? 'ready'
			: statusByPolicy[policy] === 'error'
				? 'error'
				: 'loading'
	);
	const data = $derived(cache[policy] ?? null);

	// Keep the selection valid when switching datasets: same event universe,
	// but an in-progress optimal run may not yet cover every map.
	$effect(() => {
		if (data && !data.maps.some((m) => m.event_id === selectedEventId)) {
			selectedEventId = data.maps[0]?.event_id ?? null;
		}
	});

	const selectedMap = $derived(data?.maps.find((m) => m.event_id === selectedEventId) ?? null);
</script>

<svelte:head>
	<title>BrawlCrawl — tier lists</title>
</svelte:head>

<section>
	<h1>Tier Lists</h1>

	<div class="policy-toggle" role="tablist" aria-label="Draft continuation policy">
		{#each POLICIES as p (p)}
			<button
				type="button"
				role="tab"
				aria-selected={policy === p}
				class:active={policy === p}
				onclick={() => selectPolicy(p)}
			>
				{POLICY_LABELS[p]}
			</button>
		{/each}
	</div>

	<p class="hint">{POLICY_HINTS[policy]}</p>

	{#if view === 'loading'}
		<p class="status">Loading tier lists…</p>
	{:else if view === 'error'}
		<p class="status error">{errorByPolicy[policy]}</p>
	{:else if data}
		<TierMapPicker maps={data.maps} {selectedEventId} onSelect={(id) => (selectedEventId = id)} />

		{#if selectedMap}
			<TierBoard map={selectedMap} {brawlerMeta} />
		{/if}
	{/if}
</section>

<style>
	.status {
		opacity: 0.7;
		padding: 2rem 0;
		text-align: center;
	}
	.status.error {
		color: var(--color-failure);
	}
	section {
		display: flex;
		flex-direction: column;
		gap: 0.9em;
	}
	h1 {
		font-size: 1.2rem;
		margin: 0;
	}
	.policy-toggle {
		display: flex;
		gap: 0.4em;
	}
	.policy-toggle button {
		padding: 0.4em 0.8em;
		border-radius: 0.5em;
		border: none;
		background: none;
		color: var(--color-fg);
		font: inherit;
		font-size: 0.9rem;
		font-weight: 600;
		opacity: 0.65;
		cursor: pointer;
	}
	.policy-toggle button:hover {
		opacity: 1;
		background: color-mix(in oklab, var(--color-fg) 6%, transparent);
	}
	.policy-toggle button.active {
		opacity: 1;
		background: color-mix(in oklab, var(--color-primary) 18%, transparent);
		color: var(--color-primary);
	}
	.hint {
		margin: 0;
		opacity: 0.6;
		font-size: 0.85rem;
	}
</style>
