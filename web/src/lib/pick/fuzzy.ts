// Port of src/pick/fuzzy.py — subsequence fuzzy matching for brawler/map names.
// See docs/misc_notes.md for the algorithm description.

/** Uppercase and strip non-alphanumeric chars for matching. */
export function normalize(s: string): string {
	return Array.from(s.toUpperCase())
		.filter((c) => /[A-Z0-9]/.test(c))
		.join('');
}

/** True if every char of `query` (already normalized) appears in `target` in order. */
export function subseqMatch(query: string, target: string): boolean {
	let ti = 0;
	for (const ch of query) {
		while (ti < target.length && target[ti] !== ch) ti++;
		if (ti >= target.length) return false;
		ti++;
	}
	return true;
}

/**
 * Best candidate matching `query` as a subsequence, or null if none match.
 * Among matches: prefer shortest normalized name, then lexicographic order.
 */
export function fuzzyFind(query: string, candidates: string[]): string | null {
	const normQuery = normalize(query);
	if (!normQuery) return null;
	const matches = candidates.filter((c) => subseqMatch(normQuery, normalize(c)));
	if (!matches.length) return null;
	return matches.reduce((best, c) => {
		const cn = normalize(c);
		const bn = normalize(best);
		if (cn.length !== bn.length) return cn.length < bn.length ? c : best;
		return c < best ? c : best;
	});
}

/** All candidates matching `query` as a subsequence, sorted like the CLI completer. */
export function fuzzyMatches(query: string, candidates: string[]): string[] {
	const normQuery = normalize(query);
	if (!normQuery) return [];
	return candidates
		.filter((c) => subseqMatch(normQuery, normalize(c)))
		.sort((a, b) => {
			const an = normalize(a);
			const bn = normalize(b);
			if (an.length !== bn.length) return an.length - bn.length;
			return a < b ? -1 : a > b ? 1 : 0;
		});
}
