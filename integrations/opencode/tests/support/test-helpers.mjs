import { readFileSync } from "node:fs";

// Shared fetch fakes and a representative /v1/models payload for the tests.
// Free functions and plain-object fakes are used (rather than a helper class)
// because every neighbouring test in the repo is written that way.

// The shared /v1/models sample, read from the file a rest-contract test holds
// to what the app serializes. Using the app's own listing instead of a
// hand-written one means a field the app renames or retypes breaks these
// tests until the plugin follows. It carries two chat models first
// ("text-only-model", then "vision-model"), then one image, embedding, speech
// and decision model, which the chat-only filter must drop.
const MODEL_LIST_FIXTURE_URL = new URL(
  "../../../../crates/rest-contract/fixtures/openai_model_list.json",
  import.meta.url,
);

export const representativeModels = JSON.parse(
  readFileSync(MODEL_LIST_FIXTURE_URL, "utf8"),
).data;

// A fetch stub returning the given model list (or any status) for any URL.
export function modelListFetch(models, status = 200) {
  return async () =>
    new Response(JSON.stringify({ object: "list", data: models }), { status });
}

// A fetch stub that resolves per-base-URL from a map of handler functions. The
// "*" key is the fallback when no explicit base matches. A handler may return a
// Response or throw to simulate a transport failure.
export function perUrlFetch(handlers) {
  return async (url) => {
    for (const [base, handler] of Object.entries(handlers)) {
      if (base === "*") continue;
      if (url === base || url.startsWith(`${base}/`)) return handler(url);
    }
    const fallback = handlers["*"];
    if (!fallback) throw new Error(`no handler for URL: ${url}`);
    return fallback(url);
  };
}

export const STABLE = "http://127.0.0.1:6732/v1";
export const DEV = "http://127.0.0.1:6733/v1";

// Channel descriptors shaped like resolveAstronomicalChannels output, so tests
// exercise the model list fetch the same way the server plugin does.
export const STABLE_CHANNEL = { instance: "stable", baseUrl: STABLE };
export const DEV_CHANNEL = { instance: "development", baseUrl: DEV };

// The audit logger writes to the real state file by default; tests inject this
// no-op so a test run never appends to the user's audit trail.
export const noOpAudit = async () => {};
