// Svelte 5 runes state machine mirroring `_run_draft` in src/pick/main.py:
// map select -> coin flip -> 3 ally bans -> 3 enemy bans -> picks 1-5 (live
// Q-value grid) -> pick-6 (terminal win-probability grid). Undo is a web-only
// addition (the CLI only supports cancelling the whole draft via Ctrl-C).

import { Q_THRESHOLD } from './constants';
import { overviewAnnotations, pickAnnotations, zScoreQ } from './annotations';
import { loadEngine, getQValues, getTerminalPick6Scores, scoreMap, type DraftEngine } from './engine';
import { fuzzyFind } from './fuzzy';
import { resolveOwnedBrawlers } from './ownedBrawlers';
import { ownedBrawlers } from './ownedBrawlersStore.svelte';
import { banExcluded, displayExcluded, phaseDefaultFilter, phaseLabel, pickIsAlly } from './phase';
import type { Brawler, EventMeta } from '../onnx/metadata';

export interface DraftPick {
	isAlly: boolean;
	brawler: Brawler;
}

export type View = 'loading' | 'error' | 'map-select' | 'coin-flip' | 'drafting';

// Snapshots are only ever pushed from submitBrawler, which refuses to run
// once finalMode is true — so every snapshot is necessarily pre-final-mode.
interface Snapshot {
	phase: number;
	allyBans: Brawler[];
	enemyBans: Brawler[];
	picks: DraftPick[];
	filterOn: boolean;
}

export interface GridEntry {
	brawler: Brawler;
	score: number;
}

export interface GridView {
	above: GridEntry[];
	below: GridEntry[];
	splitThreshold: number;
	scoreFmt: (n: number) => string;
	annotations: Map<number, string>;
	finalMode: boolean;
}

export class DraftState {
	view = $state<View>('loading');
	errorMsg = $state<string | null>(null);

	engine = $state<DraftEngine | null>(null);

	event = $state<EventMeta | null>(null);
	mapScores = $state<GridEntry[]>([]);
	overviewAnn = $state<Map<number, string>>(new Map());

	allyFirst = $state(true);
	phase = $state(0);
	allyBans = $state<Brawler[]>([]);
	enemyBans = $state<Brawler[]>([]);
	picks = $state<DraftPick[]>([]);

	filterOn = $state(false);

	qValues = $state<Float32Array | null>(null);
	qLoading = $state(false);
	finalMode = $state(false);
	pick6Scores = $state<GridEntry[]>([]);
	pick6Loading = $state(false);

	submitError = $state<string | null>(null);

	private history: Snapshot[] = $state([]);

	async init(): Promise<void> {
		this.view = 'loading';
		try {
			this.engine = await loadEngine();
			this.view = 'map-select';
		} catch (e) {
			this.errorMsg = e instanceof Error ? e.message : String(e);
			this.view = 'error';
		}
	}

	get mapNames(): string[] {
		return this.engine?.metadata.events.map((e) => e.map_name) ?? [];
	}

	get turnSchedule() {
		return this.engine!.manifest.draft_state.TURN_SCHEDULE;
	}

	get canUndo(): boolean {
		return this.history.length > 0;
	}

	get phaseLabel(): string {
		if (this.finalMode) return 'Pick 6';
		return phaseLabel(this.phase, this.allyFirst, this.turnSchedule);
	}

	/** True if the pick-6 slot is the ally's own pick (vs. describing the enemy's best option). */
	get pick6IsAllyPick(): boolean {
		return pickIsAlly(5, this.allyFirst, this.turnSchedule);
	}

	/** Owned brawler ids, resolved live from the settings page's comma-separated text. */
	get filterIds(): Set<number> {
		if (!this.engine) return new Set();
		return resolveOwnedBrawlers(ownedBrawlers.text, this.engine.metadata.brawlers).ids;
	}

	selectMap(mapName: string): void {
		const engine = this.engine!;
		const matched = fuzzyFind(mapName, this.mapNames);
		if (!matched) {
			this.submitError = `No map matching '${mapName}'`;
			return;
		}
		const event = engine.metadata.events.find((e) => e.map_name === matched)!;
		this.event = event;
		this.submitError = null;

		const scored = scoreMap(engine, event);
		this.mapScores = scored.map(({ brawler, score }) => ({ brawler, score }));
		this.overviewAnn = overviewAnnotations(
			engine.metadata,
			event,
			this.mapScores.map((s) => ({ id: s.brawler.id, score: s.score }))
		);
		this.view = 'coin-flip';
	}

	backToMapSelect(): void {
		this.view = 'map-select';
	}

	/** Keep the current map/event, go back to asking who picks first. */
	restartSameMap(): void {
		this.view = 'coin-flip';
	}

	async startDraft(allyFirst: boolean): Promise<void> {
		this.allyFirst = allyFirst;
		this.phase = 0;
		this.allyBans = [];
		this.enemyBans = [];
		this.picks = [];
		this.finalMode = false;
		this.pick6Scores = [];
		this.history = [];
		this.submitError = null;
		this.filterOn = phaseDefaultFilter(0, allyFirst, this.turnSchedule);
		this.view = 'drafting';
		await this.refreshQValues();
	}

	private submitExcludedIds(): Set<number> {
		return banExcluded(
			this.phase,
			this.allyBans.map((b) => b.id),
			this.enemyBans.map((b) => b.id),
			this.picks.map((p) => ({ isAlly: p.isAlly, id: p.brawler.id }))
		);
	}

	private displayExcludedIds(): Set<number> {
		return displayExcluded(
			this.allyBans.map((b) => b.id),
			this.enemyBans.map((b) => b.id),
			this.picks.map((p) => ({ isAlly: p.isAlly, id: p.brawler.id }))
		);
	}

	/** Names available for fuzzy matching at the current phase (submit exclusion, not display exclusion). */
	get availableNames(): string[] {
		const excl = this.submitExcludedIds();
		return this.engine!.metadata.brawlers.filter((b) => !excl.has(b.id)).map((b) => b.name);
	}

	/** Live best-match preview for the given input text, for keyboard-driven highlighting. */
	bestMatchId(text: string): number | null {
		if (!text.trim() || this.finalMode) return null;
		const matched = fuzzyFind(text, this.availableNames);
		if (!matched) return null;
		return this.engine!.metadata.brawlers.find((b) => b.name === matched)?.id ?? null;
	}

	private async refreshQValues(): Promise<void> {
		const engine = this.engine!;
		this.qLoading = true;
		try {
			const localPoolIds = this.filterOn && this.filterIds.size ? this.filterIds : null;
			this.qValues = await getQValues(engine, {
				event: this.event!,
				allyBanIds: this.allyBans.map((b) => b.id),
				enemyBanIds: this.enemyBans.map((b) => b.id),
				picks: this.picks.map((p) => ({ isAlly: p.isAlly, id: p.brawler.id })),
				phase: this.phase,
				allyFirst: this.allyFirst,
				localPoolIds
			});
		} finally {
			this.qLoading = false;
		}
	}

	private pushHistory(): void {
		this.history.push({
			phase: this.phase,
			allyBans: [...this.allyBans],
			enemyBans: [...this.enemyBans],
			picks: [...this.picks],
			filterOn: this.filterOn
		});
	}

	/** Submit free-text input: "/" toggles the filter, otherwise fuzzy-matches and commits. */
	async submitText(text: string): Promise<void> {
		const trimmed = text.trim();
		if (trimmed === '/') {
			this.toggleFilter();
			return;
		}
		if (!trimmed || this.finalMode) return;

		const matched = fuzzyFind(trimmed, this.availableNames);
		if (!matched) {
			this.submitError = `No brawler matching '${trimmed}'`;
			return;
		}
		const brawler = this.engine!.metadata.brawlers.find((b) => b.name === matched)!;
		await this.submitBrawler(brawler);
	}

	/** Submit a specific brawler directly (e.g. a tapped grid card). */
	async submitBrawler(brawler: Brawler): Promise<void> {
		if (this.finalMode) return;
		if (this.submitExcludedIds().has(brawler.id)) {
			this.submitError = `${brawler.name} is not available`;
			return;
		}

		this.submitError = null;
		this.pushHistory();

		const p = this.phase;
		if (p < 3) {
			this.allyBans = [...this.allyBans, brawler];
		} else if (p < 6) {
			this.enemyBans = [...this.enemyBans, brawler];
		} else {
			const isAlly = pickIsAlly(p - 6, this.allyFirst, this.turnSchedule);
			this.picks = [...this.picks, { isAlly, brawler }];
		}
		this.phase = p + 1;

		if (this.phase >= 11) {
			this.filterOn = phaseDefaultFilter(11, this.allyFirst, this.turnSchedule);
			this.finalMode = true;
			await this.computePick6();
			return;
		}

		this.filterOn = phaseDefaultFilter(this.phase, this.allyFirst, this.turnSchedule);
		await this.refreshQValues();
	}

	private async computePick6(): Promise<void> {
		this.pick6Loading = true;
		try {
			const results = await getTerminalPick6Scores(this.engine!, {
				event: this.event!,
				picks: this.picks.map((p) => ({ isAlly: p.isAlly, id: p.brawler.id })),
				excludedIds: this.displayExcludedIds(),
				allyFirst: this.allyFirst
			});
			this.pick6Scores = results.map(({ brawler, prob }) => ({ brawler, score: prob }));
		} finally {
			this.pick6Loading = false;
		}
	}

	toggleFilter(): void {
		if (this.filterIds.size === 0) return;
		this.filterOn = !this.filterOn;
		if (!this.finalMode) void this.refreshQValues();
	}

	async undo(): Promise<void> {
		const snap = this.history.pop();
		if (!snap) return;
		this.phase = snap.phase;
		this.allyBans = snap.allyBans;
		this.enemyBans = snap.enemyBans;
		this.picks = snap.picks;
		this.filterOn = snap.filterOn;
		this.finalMode = false;
		this.pick6Scores = [];
		this.submitError = null;
		await this.refreshQValues();
	}

	cancelDraft(): void {
		this.view = 'map-select';
		this.event = null;
		this.history = [];
	}

	/** The scored, threshold-split, annotated grid for the current phase (port of `_grid_content`). */
	visibleScores(): GridView {
		if (this.finalMode) {
			let entries = this.pick6Scores;
			if (this.filterOn && this.filterIds.size) {
				entries = entries.filter((e) => this.filterIds.has(e.brawler.id));
			}
			const splitThreshold = 0.5;
			return {
				above: entries.filter((e) => e.score >= splitThreshold),
				below: entries.filter((e) => e.score < splitThreshold),
				splitThreshold,
				scoreFmt: (s) => `${(s * 100).toFixed(1)}%`,
				annotations: new Map(),
				finalMode: true
			};
		}

		const engine = this.engine!;
		const displayExcl = this.displayExcludedIds();
		const brawlers = engine.metadata.brawlers.filter((b) => !displayExcl.has(b.id));
		const qValues = this.qValues;

		let entries: GridEntry[];
		if (qValues) {
			const useFilter = this.filterOn && this.filterIds.size > 0;
			const normPool = useFilter ? brawlers.filter((b) => this.filterIds.has(b.id)) : brawlers;
			const z = zScoreQ(qValues, normPool.map((b) => b.char_idx));
			entries = brawlers.map((b) => ({ brawler: b, score: z[b.char_idx] }));
			if (useFilter) entries = entries.filter((e) => this.filterIds.has(e.brawler.id));
		} else {
			entries = this.mapScores.filter((e) => !displayExcl.has(e.brawler.id));
		}
		entries.sort((a, b) => b.score - a.score);

		const annotations =
			this.phase >= 6
				? pickAnnotations(
						engine.metadata,
						this.event!,
						entries.map((e) => ({ id: e.brawler.id, score: e.score }))
					)
				: this.overviewAnn;

		return {
			above: entries.filter((e) => e.score >= Q_THRESHOLD),
			below: entries.filter((e) => e.score < Q_THRESHOLD),
			splitThreshold: Q_THRESHOLD,
			scoreFmt: (s) => `${s >= 0 ? '+' : ''}${s.toFixed(2)}`,
			annotations,
			finalMode: false
		};
	}
}
