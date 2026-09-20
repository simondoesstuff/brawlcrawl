// Types for web/static/data/manifest.json, written by `just export-onnx`
// (src/pick/export_onnx.py::export_data). Keep in sync with that function.

export interface Manifest {
	n_chars: number;
	h_terminal: number;
	char_encs: {
		path: string;
		dtype: 'float32';
		shape: [n_events: number, n_chars: number, h: number];
	};
	/** [n_chars][3]: (class_idx, range_idx, destruct_idx), indexed by char_idx */
	char_meta_table: number[][];
	/** event vocab idx (as string key) -> row in char_encs */
	event_idx_to_row: Record<string, number>;
	draft_state: {
		AVAILABLE: number;
		GLOBALLY_BANNED: number;
		LOCALLY_BANNED: number;
		PICKED_A: number;
		PICKED_B: number;
		N_CHAR_STATES: number;
		BAN_PHASE: number;
		N_DRAFT_TOKENS: number;
		/** 12 entries: (turn_token, "A" | "B", seat_within_team) */
		TURN_SCHEDULE: [turn_token: number, team: 'A' | 'B', seat: number][];
	};
	models: {
		draft_q: { path: string; inputs: Record<string, number[]>; outputs: Record<string, number[]> };
		terminal_pick6: {
			path: string;
			inputs: Record<string, number[]>;
			outputs: Record<string, number[]>;
			note: string;
		};
	};
}

/** Row-major slice of char_encs_all for one event: Float32Array[n_chars * h]. */
export function charEncsRowForEvent(
	charEncsAll: Float32Array,
	manifest: Manifest,
	eventIdx: number
): Float32Array {
	const row = manifest.event_idx_to_row[String(eventIdx)];
	if (row === undefined) throw new Error(`onnx: unknown event_idx ${eventIdx}`);
	const [, nChars, h] = manifest.char_encs.shape;
	const stride = nChars * h;
	return charEncsAll.subarray(row * stride, (row + 1) * stride);
}

/** Flatten [n][3] char_meta_table rows (e.g. for a team of 3 chars) into a [n*3] Int32Array. */
export function flattenCharMeta(rows: number[][]): Int32Array {
	const out = new Int32Array(rows.length * 3);
	rows.forEach((row, i) => out.set(row, i * 3));
	return out;
}
