// Port of src/pick/main.py::_parse_brawl_filter / _brawl_filter_names — the
// web equivalent of $BRAWL_FILTER: a comma-separated list of owned brawler
// names, resolved with the same shortest-first subsequence match as the CLI.

import { fuzzyFind } from './fuzzy';
import type { Brawler } from '../onnx/metadata';

export interface OwnedToken {
	/** Raw, untrimmed-of-case text the user typed for this entry. */
	raw: string;
	/** Canonical matched brawler name, or null if nothing matched. */
	matched: string | null;
}

export interface OwnedResolution {
	/** One entry per non-empty comma-separated token, in the order typed. */
	tokens: OwnedToken[];
	/** Unique matched brawler ids. */
	ids: Set<number>;
	/** Raw tokens that matched no brawler. */
	unmatchedRaw: string[];
}

/** Split a comma-separated owned-brawlers string into trimmed, non-empty names. */
export function parseOwnedNames(text: string): string[] {
	return text
		.split(',')
		.map((n) => n.trim())
		.filter((n) => n.length > 0);
}

/** Resolve a comma-separated owned-brawlers string against the roster, with per-token feedback. */
export function resolveOwnedBrawlers(text: string, brawlers: Brawler[]): OwnedResolution {
	const names = brawlers.map((b) => b.name);
	const byName = new Map(brawlers.map((b) => [b.name, b]));

	const tokens: OwnedToken[] = [];
	const ids = new Set<number>();
	const unmatchedRaw: string[] = [];

	for (const raw of parseOwnedNames(text)) {
		const matched = fuzzyFind(raw, names);
		tokens.push({ raw, matched });
		if (matched) {
			ids.add(byName.get(matched)!.id);
		} else {
			unmatchedRaw.push(raw);
		}
	}

	return { tokens, ids, unmatchedRaw };
}
