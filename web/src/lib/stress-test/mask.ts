// Port of env.py::get_valid_action_mask — which characters are legal for the
// acting player at a given draft turn, in char_idx space. Bans are never
// pool-restricted (a team may ban any character, in or out of either side's
// owned set); picks are restricted to the acting player's own local pool.
// See env.py's docstring: cross-team ban overlap is allowed, so a ban mask
// only ever excludes the *acting* team's own prior bans.

/** Valid ban targets: every character not yet banned by the acting team itself. */
export function banValidMask(nChars: number, ownBanCharIdxs: Iterable<number>): Uint8Array {
	const mask = new Uint8Array(nChars).fill(1);
	for (const ci of ownBanCharIdxs) mask[ci] = 0;
	return mask;
}

/** Valid pick targets: in the acting player's local pool, and not yet banned or picked by anyone. */
export function pickValidMask(
	nChars: number,
	bannedCharIdxs: Iterable<number>,
	pickedCharIdxs: Iterable<number>,
	localPoolCharIdxs: ReadonlySet<number>
): Uint8Array {
	const mask = new Uint8Array(nChars);
	for (const ci of localPoolCharIdxs) {
		if (ci >= 0 && ci < nChars) mask[ci] = 1;
	}
	for (const ci of bannedCharIdxs) mask[ci] = 0;
	for (const ci of pickedCharIdxs) mask[ci] = 0;
	return mask;
}
