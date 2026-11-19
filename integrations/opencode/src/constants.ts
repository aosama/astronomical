// Fixed values the plugin is built on, grouped by concern. Everything about
// the models themselves comes from the app's live /v1/models list instead.

import type { AstronomicalChannelName } from "./plugin-options.ts";

// ── Identity: how the plugin, its providers and its commands are named ──

// The provider id keys the source registered through OpenCode's provider
// transform. It is the same id `astronomical launch opencode`, the website
// sample and Observatory's Connect page use, so a model chosen as
// "astronomical/<model id>" resolves however the provider was set up.
export const ASTRONOMICAL_PROVIDER_ID = "astronomical";

// The server plugin's id, which is also what a user writes as
// "-astronomical" in OpenCode's plugins list to switch the plugin off.
export const ASTRONOMICAL_PLUGIN_ID = "astronomical";

// The Development daemon registers under its own provider id so both channels
// can serve models side by side in one picker. Without a distinct id the two
// model lists would collide: both daemons list overlapping model ids (same
// weights on both channels), and one provider definition can only point at
// one baseURL, so a merged list would route half the requests to the wrong
// daemon.
export const ASTRONOMICAL_DEVELOPMENT_PROVIDER_ID = "astronomical-development";

// Display name shown in the model picker and terminal command feedback.
export const ASTRONOMICAL_PROVIDER_NAME = "Astronomical";

// Display name for the Development channel's provider entry in the same
// picker, matching the instance name the app itself shows.
export const ASTRONOMICAL_DEVELOPMENT_PROVIDER_NAME =
  "Astronomical Development";

// The terminal plugin has a separate lifecycle from the server plugin.
export const ASTRONOMICAL_TUI_PLUGIN_ID = "astronomical.cli";

// The status command's id, namespaced under the plugin id so it cannot
// collide with another plugin's commands in the palette.
export const ASTRONOMICAL_STATUS_COMMAND_ID = "astronomical.status";

// ── Channels: where each Astronomical instance listens ──

// Astronomical runs one daemon per channel on a fixed loopback port. Both
// channels expose the same OpenAI-compatible surface; neither is up all the
// time, so the plugin probes every channel the user named and serves each one
// that answers, so Stable and Development coexist.
export const ASTRONOMICAL_CHANNEL_PORTS: Readonly<
  Record<AstronomicalChannelName, number>
> = {
  stable: 6732,
  development: 6733,
};

// Every endpoint is served under this base path, which the runtime and the
// model list fetch append routes to (e.g. ${base}/models, ${base}/status).
export const ASTRONOMICAL_API_BASE_PATH = "/v1";

// Every channel the plugin can serve, labeled by daemon installation. "custom"
// is not a daemon choice but the label for a "baseUrl" option, which targets
// exactly one endpoint the user pointed at; it shares the Stable
// provider identity because it replaces it as "the" Astronomical endpoint.
export type AstronomicalChannelInstance = "stable" | "development" | "custom";

// Provider identity per channel instance. Stable takes the plain
// "astronomical" id that every other setup path uses, because it is the only
// channel people who install the published app have; Development gets its own
// entry.
export const ASTRONOMICAL_CHANNEL_PROVIDER_IDS: Readonly<
  Record<AstronomicalChannelInstance, string>
> = {
  stable: ASTRONOMICAL_PROVIDER_ID,
  development: ASTRONOMICAL_DEVELOPMENT_PROVIDER_ID,
  custom: ASTRONOMICAL_PROVIDER_ID,
};

// Picker-facing display name per channel instance. Kept beside the ids so both
// halves of a channel's identity always change together.
export const ASTRONOMICAL_CHANNEL_PROVIDER_NAMES: Readonly<
  Record<AstronomicalChannelInstance, string>
> = {
  stable: ASTRONOMICAL_PROVIDER_NAME,
  development: ASTRONOMICAL_DEVELOPMENT_PROVIDER_NAME,
  custom: ASTRONOMICAL_PROVIDER_NAME,
};

// ── Timing ──

// The longest wait for one probe of a channel, its model list or its status.
// The model list is fetched on every refresh so models the app adds, retires
// or swaps show up with no configuration edits; a short ceiling keeps startup
// from hanging on a stopped channel, which is then left out.
export const ASTRONOMICAL_PROBE_TIMEOUT_MS = 5_000;

// Refreshing at this cadence notices models added or removed in the app
// without making a stopped local daemon part of the normal request path.
export const ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS = 30_000;

// A cold model load can delay the first streamed response; OpenCode's default
// header timeout is shorter than the local runner's expected load window.
export const ASTRONOMICAL_INFERENCE_TIMEOUT_MS = 600_000;

// ── Wire values ──

// Astronomical speaks OpenAI-compatible requests through OpenCode's runtime.
export const ASTRONOMICAL_OPENAI_COMPATIBLE_PACKAGE =
  "@opencode/ai/providers/openai-compatible";

// The route name Astronomical publishes in each model row's
// supported_endpoints for chat. OpenCode's OpenAI-compatible runtime serves
// only this route, so the plugin offers exactly the models that list it. This
// filter is what drops image, embedding, speech and decision models.
export const ASTRONOMICAL_CHAT_ENDPOINT = "/v1/chat/completions";
