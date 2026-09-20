// Types for web/static/data/metadata.json, written by `just export-onnx`
// (src/pick/export_onnx.py::export_metadata). Keep in sync with that function.

export interface Brawler {
	id: number;
	name: string;
	class: string;
	rarity: string;
	/** index into draft_q's q_values output / manifest.json's char_meta_table */
	char_idx: number;
}

export interface EventMeta {
	id: number;
	mode: string;
	mode_id: number;
	map_name: string;
	/** vocab idx: key into manifest.json's event_idx_to_row */
	event_idx: number;
	/** required (with event_idx) as a terminal_pick6.onnx input */
	mode_idx: number;
}

export interface Metadata {
	/** brawlers[i] is the brawler at char_idx i */
	brawlers: Brawler[];
	events: EventMeta[];
	/** brawler id (as string) -> event id (as string) -> z-score */
	winrates: Record<string, Record<string, number>>;
	pickrates: Record<string, Record<string, number>>;
}

export function winrateZ(
	metadata: Metadata,
	brawlerId: number,
	eventId: number
): number | undefined {
	return metadata.winrates[String(brawlerId)]?.[String(eventId)];
}

export function pickrateZ(
	metadata: Metadata,
	brawlerId: number,
	eventId: number
): number | undefined {
	return metadata.pickrates[String(brawlerId)]?.[String(eventId)];
}
