// Shared fetch fakes and a representative /v1/models payload for the tests.
// Free functions and plain-object fakes are used (rather than a helper class)
// because every neighbouring test in the repo is written that way.

// One model of each class so tests can assert the chat-only filter and the
// capability mapping without inventing data:
//   - text-only reasoning/tool model
//   - vision reasoning/tool model
//   - image-generation model (no chat limits, no chat endpoint)
//   - embeddings model (no chat limits, no chat endpoint)
export const representativeModels = [
  {
    id: "text-only-model",
    object: "model",
    context_window: 262_144,
    max_output_tokens: 65_535,
    input_modalities: ["text"],
    output_modalities: ["text"],
    supports_reasoning: true,
    supports_tool_calls: true,
    supported_endpoints: ["/v1/chat/completions", "/v1/responses"],
  },
  {
    id: "vision-model",
    object: "model",
    context_window: 131_072,
    max_output_tokens: 65_535,
    input_modalities: ["text", "image"],
    output_modalities: ["text"],
    supports_reasoning: true,
    supports_tool_calls: true,
    supported_endpoints: ["/v1/chat/completions"],
  },
  {
    id: "image-gen-model",
    object: "model",
    input_modalities: ["text"],
    output_modalities: ["image"],
    supports_reasoning: false,
    supports_tool_calls: false,
    supported_endpoints: ["/v1/images/generations"],
  },
  {
    id: "embedding-model",
    object: "model",
    input_modalities: ["text"],
    output_modalities: ["embedding"],
    supports_reasoning: false,
    supports_tool_calls: false,
    supported_endpoints: ["/v1/embeddings"],
  },
];

// A fetch stub returning the given catalogue (or any status) for any URL.
export function catalogueFetch(models, status = 200) {
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
// exercise the catalogue the same way the server plugin does.
export const STABLE_CHANNEL = { instance: "stable", baseUrl: STABLE };
export const DEV_CHANNEL = { instance: "dev", baseUrl: DEV };

// The audit logger writes to the real state file by default; tests inject this
// no-op so a test run never appends to the user's audit trail.
export const noOpAudit = async () => {};
