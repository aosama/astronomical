import assert from "node:assert/strict";
import { test } from "node:test";
import { DAEMON_BASE_URL } from "./support/journey";
import { SpeechClient } from "../src/speech/SpeechClient";
import type { SpeechFetchFailure } from "../src/speech/SpeechFetchFailure";
import { SpeechFetchFailureKind } from "../src/speech/SpeechFetchFailureKind";

/**
 * Runs one test body with the global fetch free to be swapped, and puts back
 * the fetch it found. No `afterEach`: the test files are bundled into one
 * module, so a top-level hook here would run after every test of the whole
 * suite and take the fake daemon away from pages earlier journeys left
 * polling, which then call a real engine address and keep the run alive.
 */
function withScopedFetch(testBody: () => Promise<void>): () => Promise<void> {
  return async () => {
    const fetchBeforeTest = globalThis.fetch;
    try {
      await testBody();
    } finally {
      globalThis.fetch = fetchBeforeTest;
    }
  };
}

function answerWith(status: number, body: unknown): void {
  globalThis.fetch = (async () => new Response(typeof body === "string" ? body : JSON.stringify(body), { status })) as typeof fetch;
}

async function failureOf(client: SpeechClient): Promise<SpeechFetchFailure> {
  try {
    await client.fetchSpokenPiece({ text: "Hello.", seed: 1, model: null, signal: new AbortController().signal });
  } catch (fetchError) {
    return fetchError as SpeechFetchFailure;
  }
  throw new Error("the fetch was expected to fail");
}

const client = new SpeechClient(DAEMON_BASE_URL);

test("should_send_the_text_seed_and_wav_format_and_no_model_when_none_is_named", withScopedFetch(async () => {
  let sentBody: Record<string, unknown> = {};
  let sentHeaders: Record<string, string> = {};
  globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
    sentBody = JSON.parse(String(init?.body));
    sentHeaders = init?.headers as Record<string, string>;
    return new Response(new ArrayBuffer(8));
  }) as typeof fetch;
  const audioBytes = await client.fetchSpokenPiece({ text: "Hello.", seed: 7, model: null, signal: new AbortController().signal });
  assert.equal(audioBytes.byteLength, 8);
  assert.deepEqual(sentBody, { input: "Hello.", seed: 7, response_format: "wav" });
  assert.deepEqual(Object.keys(sentHeaders), ["content-type"], "the canvas may send only content-type");
}));

test("should_name_the_model_when_asked_to", withScopedFetch(async () => {
  let sentBody: Record<string, unknown> = {};
  globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
    sentBody = JSON.parse(String(init?.body));
    return new Response(new ArrayBuffer(1));
  }) as typeof fetch;
  await client.fetchSpokenPiece({ text: "Hi.", seed: 1, model: "Kokoro-82M-bf16", signal: new AbortController().signal });
  assert.equal(sentBody["model"], "Kokoro-82M-bf16");
}));

test("should_classify_each_failure_the_button_must_tell_apart", withScopedFetch(async () => {
  answerWith(400, { error: { code: "no_speech_model" } });
  assert.equal((await failureOf(client)).kind, SpeechFetchFailureKind.NO_VOICE);
  answerWith(400, { error: { code: "speech_model_ambiguous" } });
  assert.equal((await failureOf(client)).kind, SpeechFetchFailureKind.AMBIGUOUS_VOICE);
  answerWith(429, { error: { code: "capacity_unavailable" } });
  assert.equal((await failureOf(client)).kind, SpeechFetchFailureKind.BUSY);
  answerWith(503, "not json at all");
  assert.equal((await failureOf(client)).kind, SpeechFetchFailureKind.BUSY);
  answerWith(500, { error: { code: "speech_failed" } });
  assert.equal((await failureOf(client)).kind, SpeechFetchFailureKind.FAILED);
}));

test("should_list_only_models_that_speak", withScopedFetch(async () => {
  answerWith(200, { data: [
    { id: "Kokoro-82M-bf16", supported_endpoints: ["/v1/audio/speech"] },
    { id: "chat-model", supported_endpoints: ["/v1/chat/completions"] },
    { id: "no-endpoints" },
  ] });
  assert.deepEqual(await client.listVoiceModels(), ["Kokoro-82M-bf16"]);
  answerWith(500, "boom");
  assert.deepEqual(await client.listVoiceModels(), []);
}));

test("should_refuse_audio_that_declares_more_than_the_cap_before_reading_it", withScopedFetch(async () => {
  let bodyWasRead = false;
  globalThis.fetch = (async () => {
    const response = new Response(new ArrayBuffer(8), { headers: { "content-length": String(64 * 1024 * 1024) } });
    response.arrayBuffer = async () => { bodyWasRead = true; return new ArrayBuffer(8); };
    return response;
  }) as typeof fetch;
  const failure = await failureOf(client);
  assert.equal(failure.kind, SpeechFetchFailureKind.FAILED);
  assert.equal(bodyWasRead, false);
}));

test("should_let_a_cancelled_fetch_surface_as_an_abort_not_a_classified_failure", withScopedFetch(async () => {
  globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
    init?.signal?.throwIfAborted();
    return new Response(new ArrayBuffer(1));
  }) as typeof fetch;
  const abortController = new AbortController();
  abortController.abort();
  await assert.rejects(client.fetchSpokenPiece({ text: "Hi.", seed: 1, model: null, signal: abortController.signal }), (rejection: unknown) => {
    return !(rejection instanceof Error && "kind" in rejection);
  });
}));
