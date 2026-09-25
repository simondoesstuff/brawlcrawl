// Thin wrapper around onnxruntime-web for the two exported graphs
// (draft_q.onnx, terminal_pick6.onnx — see src/pick/export_onnx.py).
// Both graphs are single-example (no batch dim): pick-6 candidates are
// scored with one `runTerminalPick6` call per candidate, same as the CLI's
// `vmap` over the JAX model.

import * as ort from 'onnxruntime-web';

export type OnnxSource = string | ArrayBufferLike | Uint8Array;

export async function createSession(source: OnnxSource): Promise<ort.InferenceSession> {
	return ort.InferenceSession.create(source as never);
}

// onnxruntime-web's wasm backend throws "Session already started" if a
// second run() is issued on the same session before the first resolves — it
// has no internal run queue, unlike the node backend the test suite runs
// against. The stress-test module intentionally runs several simulated
// drafts concurrently (see stressTest.ts's `concurrency`), and every draft
// shares the same two engine sessions, so calls onto one session must be
// serialized here rather than left to the backend.
const runQueues = new WeakMap<ort.InferenceSession, Promise<unknown>>();

function serialized<T>(session: ort.InferenceSession, fn: () => Promise<T>): Promise<T> {
	const prior = runQueues.get(session) ?? Promise.resolve();
	const run = prior.then(fn, fn);
	runQueues.set(
		session,
		run.then(
			() => undefined,
			() => undefined
		)
	);
	return run;
}

export async function runDraftQ(
	session: ort.InferenceSession,
	charEncsRow: Float32Array,
	playerCharStates: Int32Array,
	turnToken: number,
	nChars: number,
	h: number
): Promise<Float32Array> {
	const feeds = {
		char_encs: new ort.Tensor('float32', charEncsRow, [nChars, h]),
		player_char_states: new ort.Tensor('int32', playerCharStates, [nChars]),
		turn_token: new ort.Tensor('int32', Int32Array.of(turnToken), [1])
	};
	const out = await serialized(session, () => session.run(feeds));
	return out.q_values.data as Float32Array;
}

export async function runTerminalPick6(
	session: ort.InferenceSession,
	eventIdx: number,
	modeIdx: number,
	teamAChars: Int32Array,
	teamAMeta: Int32Array,
	teamBChars: Int32Array,
	teamBMeta: Int32Array
): Promise<number> {
	const feeds = {
		event_idx: new ort.Tensor('int32', Int32Array.of(eventIdx), [1]),
		mode_idx: new ort.Tensor('int32', Int32Array.of(modeIdx), [1]),
		team_a_chars: new ort.Tensor('int32', teamAChars, [3]),
		team_a_meta: new ort.Tensor('int32', teamAMeta, [3, 3]),
		team_b_chars: new ort.Tensor('int32', teamBChars, [3]),
		team_b_meta: new ort.Tensor('int32', teamBMeta, [3, 3])
	};
	const out = await serialized(session, () => session.run(feeds));
	return (out.logit.data as Float32Array)[0];
}
