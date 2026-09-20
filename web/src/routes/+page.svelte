<script lang="ts">
	import { onMount } from 'svelte';
	import {
		createSession,
		runDraftQ,
		charEncsRowForEvent,
		type Manifest,
		type Metadata
	} from '$lib/onnx';

	// Diagnostic-only smoke test: proves the exported ONNX graphs + data load
	// and run correctly through onnxruntime-web in an actual browser (bun
	// test already covers the Node-like path). Real UX comes later.

	let status = $state('loading...');
	let error = $state<string | null>(null);
	let inputNames = $state<string[]>([]);
	let outputNames = $state<string[]>([]);
	let eventLabel = $state('');
	let topPicks = $state<{ name: string; q: number }[]>([]);

	onMount(async () => {
		try {
			status = 'fetching manifest + metadata...';
			const [manifest, metadata]: [Manifest, Metadata] = await Promise.all([
				fetch('/data/manifest.json').then((r) => r.json()),
				fetch('/data/metadata.json').then((r) => r.json())
			]);

			status = 'fetching char_encs.bin...';
			const charEncsBuf = await fetch('/data/char_encs.bin').then((r) => r.arrayBuffer());
			const charEncsAll = new Float32Array(charEncsBuf);

			status = 'loading draft_q.onnx...';
			const session = await createSession('/models/draft_q.onnx');
			inputNames = [...session.inputNames];
			outputNames = [...session.outputNames];

			status = 'running inference...';
			const event = metadata.events[0];
			eventLabel = `${event.map_name} (${event.mode})`;
			const charEncsRow = charEncsRowForEvent(charEncsAll, manifest, event.event_idx);
			const states = new Int32Array(manifest.n_chars).fill(manifest.draft_state.AVAILABLE);
			const q = await runDraftQ(
				session,
				charEncsRow,
				states,
				manifest.draft_state.BAN_PHASE,
				manifest.n_chars,
				manifest.h_terminal
			);

			topPicks = Array.from(q)
				.map((score, i) => ({ name: metadata.brawlers[i].name, q: score }))
				.sort((a, b) => b.q - a.q)
				.slice(0, 10);

			status = 'ok';
		} catch (e) {
			error = e instanceof Error ? (e.stack ?? e.message) : String(e);
			status = 'failed';
		}
	});
</script>

<main style="font-family: monospace; padding: 2rem;">
	<h1>onnx smoke test</h1>
	<p>status: {status}</p>
	{#if error}
		<pre style="color: red; white-space: pre-wrap;">{error}</pre>
	{/if}
	{#if inputNames.length}
		<p>draft_q.onnx inputs: {inputNames.join(', ')}</p>
		<p>draft_q.onnx outputs: {outputNames.join(', ')}</p>
	{/if}
	{#if topPicks.length}
		<h2>top ban-phase Q-values — {eventLabel}</h2>
		<ol>
			{#each topPicks as p (p.name)}
				<li>{p.name}: {p.q.toFixed(3)}</li>
			{/each}
		</ol>
	{/if}
</main>
