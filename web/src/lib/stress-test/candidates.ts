// Cheap (no ONNX) pre-screen for stress-test candidates: ranks unowned
// brawlers by mean winrate z-score across every event, so a UI can default
// to testing a fast shortlist instead of the full roster — see stressTest.ts's
// header note on a full-roster sweep at the default trial count being
// "minutes, not seconds", and its suggestion to pass a pre-screened
// `candidateIds` shortlist instead.

import { winrateZ, type Metadata } from '../onnx/metadata';

/** Unowned brawler ids ranked by mean winrate z-score across all events, best first. */
export function rankCandidatesByWinrate(
	metadata: Metadata,
	ownedIds: ReadonlySet<number>
): number[] {
	const scored = metadata.brawlers
		.filter((b) => !ownedIds.has(b.id))
		.map((b) => {
			let sum = 0;
			for (const e of metadata.events) sum += winrateZ(metadata, b.id, e.id) ?? 0;
			return { id: b.id, avgZ: metadata.events.length > 0 ? sum / metadata.events.length : 0 };
		});
	scored.sort((a, b) => b.avgZ - a.avgZ);
	return scored.map((s) => s.id);
}
