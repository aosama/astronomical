// Provider identity, network location, and instance tuning for the local
// Astronomical model runner. These are the only values that would change if
// Astronomical rebrands or moves its ports; everything else is derived from the
// daemon's live /v1/models catalogue.

// The provider id keys the source registered through OpenCode V2's provider
// transform. It is the same id `astronomical launch opencode`, the website
// sample and Observatory's Connect page use, so a model chosen as
// "astronomical/<model id>" resolves however the provider was set up.
export const ASTRONOMICAL_PROVIDER_ID = "astronomical";

// The server plugin's id, which is also what a user writes as
// "-astronomical" in OpenCode's plugins list to switch the plugin off.
export const ASTRONOMICAL_PLUGIN_ID = "astronomical";

// The Development daemon registers under its own provider id so both channels
// can serve models side by side in one picker. Without a distinct id the two
// catalogues would collide: both daemons list overlapping model ids (same
// weights on both channels), and one provider definition can only point at
// one baseURL, so a merged list would route half the requests to the wrong
// daemon.
export const ASTRONOMICAL_DEV_PROVIDER_ID = "astronomical-development";

// Display name shown in the model picker and terminal command feedback.
export const ASTRONOMICAL_PROVIDER_NAME = "Astronomical";

// Display name for the Development channel's provider entry in the same
// picker, matching the instance name the app itself shows.
export const ASTRONOMICAL_DEV_PROVIDER_NAME = "Astronomical Development";

// The V2 terminal plugin has a separate lifecycle from the server plugin.
export const ASTRONOMICAL_TUI_PLUGIN_ID = "astronomical.cli";

// Terminal command ids, namespaced under the plugin id so they cannot collide
// with another plugin's commands in the palette.
export const ASTRONOMICAL_STATUS_COMMAND_ID = "astronomical.status";
export const ASTRONOMICAL_TOGGLE_COMMAND_ID = "astronomical.toggle";

// Legacy V1 terminal settings are consulted only when the new shared state file
// has not been written yet.
export const ASTRONOMICAL_LEGACY_TUI_STATE_KEY = "plugin_enabled";

// Astronomical speaks OpenAI-compatible requests through OpenCode V2's runtime.
export const ASTRONOMICAL_OPENAI_COMPATIBLE_PACKAGE =
  "@opencode/ai/providers/openai-compatible";

// Astronomical runs one daemon per channel on a fixed localhost TCP port. Both
// channels expose the same OpenAI-compatible surface; neither is up all the
// time, so the plugin probes every channel and serves each one that answers
// rather than the first that did — stable and dev coexist in one picker.
export const ASTRONOMICAL_STABLE_PORT = 6732;
export const ASTRONOMICAL_DEV_PORT = 6733;

// Every endpoint is served under this base path, which the adapter and the
// catalogue fetch append routes to (e.g. ${base}/models, ${base}/status).
export const ASTRONOMICAL_API_BASE_PATH = "/v1";

// Channel identifiers for the ASTRONOMICAL_INSTANCE override.
export const ASTRONOMICAL_INSTANCE_STABLE = "stable";
export const ASTRONOMICAL_INSTANCE_DEV = "dev";

// Every channel the plugin can serve, labeled by daemon installation. "custom"
// is not a daemon choice but the label for an ASTRONOMICAL_BASE_URL override,
// which targets exactly one endpoint the user pointed at; it shares the stable
// provider identity because it replaces it as "the" Astronomical endpoint.
export type AstronomicalChannelInstance = "stable" | "dev" | "custom";

// Provider identity per channel instance. Stable takes the plain
// "astronomical" id that every other setup path uses, because it is the only
// channel people who install the published app have; Development gets its own
// entry.
export const ASTRONOMICAL_CHANNEL_PROVIDER_IDS: Readonly<
  Record<AstronomicalChannelInstance, string>
> = {
  stable: ASTRONOMICAL_PROVIDER_ID,
  dev: ASTRONOMICAL_DEV_PROVIDER_ID,
  custom: ASTRONOMICAL_PROVIDER_ID,
};

// Picker-facing display name per channel instance. Kept beside the ids so both
// halves of a channel's identity always change together.
export const ASTRONOMICAL_CHANNEL_PROVIDER_NAMES: Readonly<
  Record<AstronomicalChannelInstance, string>
> = {
  stable: ASTRONOMICAL_PROVIDER_NAME,
  dev: ASTRONOMICAL_DEV_PROVIDER_NAME,
  custom: ASTRONOMICAL_PROVIDER_NAME,
};

// Env overrides. The base URL override is not a secret, so a custom host or
// proxy stays an environment concern rather than a stored credential.
export const ASTRONOMICAL_BASE_URL_ENV = "ASTRONOMICAL_BASE_URL";
export const ASTRONOMICAL_INSTANCE_ENV = "ASTRONOMICAL_INSTANCE";

// The live catalogue is fetched on every start so models the runner adds,
// retires, or hot-swaps show up with no config edits. A short ceiling keeps
// startup from hanging on a stopped channel; on timeout the provider simply
// exposes zero models.
export const ASTRONOMICAL_CATALOGUE_TIMEOUT_MS = 5_000;

// Refreshing at this cadence notices model changes and terminal toggle changes
// without making a stopped local daemon part of the normal request path.
export const ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS = 30_000;

// The route name Astronomical publishes in each model row's supported_endpoints
// for chat. opencode's openai-compatible adapter can only serve the
// chat-completions route, so the catalogue keeps exactly the models that list
// it. This is a wire value, not a URL path, and this filter is what drops
// image-generation and embeddings models that opencode's chat picker cannot
// present.
export const ASTRONOMICAL_CHAT_ENDPOINT = "/v1/chat/completions";

// A cold model load can delay the first streamed response; V2's default header
// timeout is shorter than the local runner's expected load window.
export const ASTRONOMICAL_INFERENCE_TIMEOUT_MS = 600_000;
