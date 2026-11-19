// Turn the plugin's options into the addresses to probe. Stable and
// Development are independent installations with independent model sets, so
// every named channel is probed and served; one is not a failover for the
// other. Neither is up all the time, so an unreachable channel is simply left
// out by the model list fetch rather than blocking the other.

import {
  ASTRONOMICAL_API_BASE_PATH,
  ASTRONOMICAL_CHANNEL_PORTS,
  type AstronomicalChannelInstance,
} from "./constants.ts";
import type { AstronomicalPluginOptions } from "./plugin-options.ts";

// A channel is one probe target plus the label that identifies which daemon
// installation answered. The label is what downstream code keys provider
// identity and audit summaries on; the URL alone cannot tell the two apart.
export interface AstronomicalChannel {
  readonly instance: AstronomicalChannelInstance;
  readonly baseUrl: string;
}

// A base URL is the OpenAI-compatible prefix requests are built against, so it
// must end in the /v1 base path; the app serves /models, /status and the chat
// route under it. A user's address is normalized the same way, so a missing
// /v1 or a trailing slash still resolves.
function normalizeBaseUrl(rawBaseUrl: string): string {
  const withoutTrailingSlash = rawBaseUrl.replace(/\/+$/u, "");
  return withoutTrailingSlash.endsWith(ASTRONOMICAL_API_BASE_PATH)
    ? withoutTrailingSlash
    : `${withoutTrailingSlash}${ASTRONOMICAL_API_BASE_PATH}`;
}

export function resolveAstronomicalChannels(
  options: AstronomicalPluginOptions,
): readonly AstronomicalChannel[] {
  // An explicit address replaces channel probing entirely: the user pointed at
  // one specific endpoint, so probing the default ports beside it would list
  // models from an instance they did not ask for.
  if (options.baseUrl !== undefined) {
    return [{ instance: "custom", baseUrl: normalizeBaseUrl(options.baseUrl) }];
  }
  const channels: AstronomicalChannel[] = [];
  for (const channelName of options.channelNames) {
    const port = ASTRONOMICAL_CHANNEL_PORTS[channelName];
    channels.push({
      instance: channelName,
      baseUrl: `http://127.0.0.1:${port}${ASTRONOMICAL_API_BASE_PATH}`,
    });
  }
  return channels;
}
