// The plugin's settings, from its entry in OpenCode's `plugins` list:
//
//   { "package": "opencode-astronomical",
//     "options": { "channels": ["stable", "development"] } }
//
// Settings live in the user's OpenCode configuration beside everything else,
// readable by anyone who opens it, instead of in environment variables that
// leave no trace in any file. A wrong value stops the plugin with a message
// naming the setting, because silently falling back would hide the mistake
// behind models that never appear.

import { isRecord } from "./wire-values.ts";

/** The Astronomical instances a user can name in `channels`. */
export type AstronomicalChannelName = "stable" | "development";

const CHANNEL_NAMES: readonly AstronomicalChannelName[] = [
  "stable",
  "development",
];

// The published app is the Stable instance; Development exists only for
// people building Astronomical from source, so it is opt-in.
const DEFAULT_CHANNEL_NAMES: readonly AstronomicalChannelName[] = ["stable"];

export interface AstronomicalPluginOptions {
  /** The instances to list models from, in picker order. */
  readonly channelNames: readonly AstronomicalChannelName[];
  /** One address to use instead of the instances, such as a proxy. */
  readonly baseUrl: string | undefined;
}

export class AstronomicalOptionsError extends Error {
  constructor(problem: string) {
    super(`opencode-astronomical: ${problem}`);
    this.name = "AstronomicalOptionsError";
  }
}

function readChannelNames(
  rawChannels: unknown,
): readonly AstronomicalChannelName[] {
  if (rawChannels === undefined) return DEFAULT_CHANNEL_NAMES;
  const expected = `"channels" must be a list of ${CHANNEL_NAMES.map((name) => `"${name}"`).join(" or ")}`;
  if (!Array.isArray(rawChannels) || rawChannels.length === 0) {
    throw new AstronomicalOptionsError(`${expected}, such as ["stable"]`);
  }
  const channelNames: AstronomicalChannelName[] = [];
  for (const rawChannel of rawChannels) {
    if (!CHANNEL_NAMES.includes(rawChannel as AstronomicalChannelName)) {
      throw new AstronomicalOptionsError(
        `${expected}; ${JSON.stringify(rawChannel)} is not one`,
      );
    }
    if (!channelNames.includes(rawChannel as AstronomicalChannelName)) {
      channelNames.push(rawChannel as AstronomicalChannelName);
    }
  }
  return channelNames;
}

// Only http and https can reach the app. URL parsing alone is not enough:
// "127.0.0.1:6732" parses with "127.0.0.1:" as its scheme and would fail at
// every refresh instead of here, where the message can name the setting.
function isHttpAddress(candidate: string): boolean {
  if (!URL.canParse(candidate)) return false;
  const protocol = new URL(candidate).protocol;
  return protocol === "http:" || protocol === "https:";
}

function readBaseUrl(rawBaseUrl: unknown): string | undefined {
  if (rawBaseUrl === undefined) return undefined;
  if (typeof rawBaseUrl !== "string" || !isHttpAddress(rawBaseUrl.trim())) {
    throw new AstronomicalOptionsError(
      `"baseUrl" must be an http address such as "http://127.0.0.1:6732/v1"`,
    );
  }
  return rawBaseUrl.trim();
}

/** Validates the options OpenCode passes; unknown keys are rejected so a typo is caught. */
export function parseAstronomicalPluginOptions(
  rawOptions: unknown,
): AstronomicalPluginOptions {
  if (rawOptions === undefined || rawOptions === null) {
    return { channelNames: DEFAULT_CHANNEL_NAMES, baseUrl: undefined };
  }
  if (!isRecord(rawOptions)) {
    throw new AstronomicalOptionsError(`"options" must be an object`);
  }
  for (const optionName of Object.keys(rawOptions)) {
    if (optionName !== "channels" && optionName !== "baseUrl") {
      throw new AstronomicalOptionsError(
        `unknown option "${optionName}"; the options are "channels" and "baseUrl"`,
      );
    }
  }
  return {
    channelNames: readChannelNames(rawOptions["channels"]),
    baseUrl: readBaseUrl(rawOptions["baseUrl"]),
  };
}
