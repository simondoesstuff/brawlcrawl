// Svelte 5 runes state machine driving the "Stress Test" tab: loads the
// ONNX engine on demand (never eagerly — it's the same heavy sessions the
// draft page loads), runs runAdversarialStressTest against the owned set
// from settings, and exposes progress/results for the page to render.

import type { Metadata } from '../onnx/metadata';
import { loadEngine, type DraftEngine } from '../pick/engine';
import { resolveOwnedBrawlers } from '../pick/ownedBrawlers';
import { ownedBrawlers } from '../pick/ownedBrawlersStore.svelte';
import { rankCandidatesByWinrate } from './candidates';
import { estimateCallCount, runAdversarialStressTest, type StressTestResult } from './stressTest';

export type Speed = 'quick' | 'balanced' | 'thorough';

/** Stratification cycles per speed — see stressTest.ts's DEFAULT_STRATIFICATION_CYCLES for what this buys. */
export const SPEED_CYCLES: Record<Speed, number> = { quick: 1, balanced: 2, thorough: 8 };

export type Shortlist = 8 | 16 | 30 | 'all';

export type MetadataStatus = 'loading' | 'error' | 'ready';
export type View = 'idle' | 'loading-models' | 'running' | 'done' | 'error';

export class StressTestState {
	metadataStatus = $state<MetadataStatus>('loading');
	metadataError = $state<string | null>(null);
	metadata = $state<Metadata | null>(null);

	engine = $state<DraftEngine | null>(null);

	speed = $state<Speed>('balanced');
	shortlist = $state<Shortlist>(16);

	view = $state<View>('idle');
	errorMsg = $state<string | null>(null);
	progress = $state({ done: 0, total: 0 });
	result = $state<StressTestResult | null>(null);

	private abortCtrl: AbortController | null = null;

	async loadMetadata(): Promise<void> {
		this.metadataStatus = 'loading';
		try {
			const res = await fetch('/data/metadata.json');
			if (!res.ok) throw new Error(`Failed to load brawler data (${res.status})`);
			this.metadata = (await res.json()) as Metadata;
			this.metadataStatus = 'ready';
		} catch (e) {
			this.metadataError = e instanceof Error ? e.message : String(e);
			this.metadataStatus = 'error';
		}
	}

	/** Owned brawler ids, resolved live from the settings page's comma-separated text. */
	get ownedIds(): Set<number> {
		if (!this.metadata) return new Set();
		return resolveOwnedBrawlers(ownedBrawlers.text, this.metadata.brawlers).ids;
	}

	private candidateIds(): number[] {
		const ranked = rankCandidatesByWinrate(this.metadata!, this.ownedIds);
		return this.shortlist === 'all' ? ranked : ranked.slice(0, this.shortlist);
	}

	private trials(nEvents: number): number {
		return SPEED_CYCLES[this.speed] * 2 * nEvents;
	}

	/** Rough ONNX call count for the current config; null until the engine (and metadata) is loaded. */
	get estimatedCalls(): number | null {
		if (!this.engine || this.ownedIds.size === 0) return null;
		return estimateCallCount(this.engine, {
			ownedIds: [...this.ownedIds],
			candidateIds: this.candidateIds(),
			trials: this.trials(this.engine.metadata.events.length)
		});
	}

	get busy(): boolean {
		return this.view === 'loading-models' || this.view === 'running';
	}

	async run(): Promise<void> {
		if (this.ownedIds.size === 0 || this.busy) return;
		this.errorMsg = null;
		this.result = null;
		this.view = 'loading-models';
		try {
			if (!this.engine) this.engine = await loadEngine();
			const engine = this.engine;
			this.view = 'running';
			this.progress = { done: 0, total: 0 };
			this.abortCtrl = new AbortController();
			const result = await runAdversarialStressTest(engine, {
				ownedIds: [...this.ownedIds],
				candidateIds: this.candidateIds(),
				trials: this.trials(engine.metadata.events.length),
				signal: this.abortCtrl.signal,
				onProgress: (done, total) => {
					this.progress = { done, total };
				}
			});
			this.result = result;
			this.view = 'done';
		} catch (e) {
			if (e instanceof DOMException && e.name === 'AbortError') {
				this.view = 'idle';
				return;
			}
			this.errorMsg = e instanceof Error ? e.message : String(e);
			this.view = 'error';
		}
	}

	cancel(): void {
		this.abortCtrl?.abort();
	}
}
