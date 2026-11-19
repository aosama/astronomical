// Resolve which Astronomical channel(s) to probe from the environment, with the
// default being both channels (stable first). The two daemons are independent
// installations with independent model sets, so every resolved channel is
// probed and served — one is not a failover mirror of the other. Neither
// channel is up all the time, so an unreachable channel is simply skipped by
// the catalogue fetch rather than blocking the other.

import {
  ASTRONOMICAL_API_BASE_PATH,
  ASTRONOMICAL_BASE_URL_ENV,
  ASTRONOMICAL_DEV_PORT,
  ASTRONOMICAL_INSTANCE_DEV,
  ASTRONOMICAL_INSTANCE_ENV,
  ASTRONOMICAL_INSTANCE_STABLE,
  ASTRONOMICAL_STABLE_PORT,
  type AstronomicalChannelInstance,
} from "./constants.js";

const STABLE_BASE_URL = `http://127.0.0.1:${ASTRONOMICAL_STABLE_PORT}${ASTRONOMICAL_API_BASE_PATH}`;
const DEV_BASE_URL = `http://127.0.0.1:${ASTRONOMICAL_DEV_PORT}${ASTRONOMICAL_API_BASE_PATH}`;

// A channel is one probe target plus the label that identifies which daemon
// installation answered. The label is what downstream code keys provider
// identity and audit summaries on; the URL alone cannot tell stable from dev.
export interface AstronomicalChannel {
  readonly instance: AstronomicalChannelInstance;
  readonly baseUrl: string;
}

// A base URL is the OpenAI-compatible prefix requests are built against, so it
// must end in the /v1 base path; the daemon appends /models, /status, and the
// route names under it. A caller-supplied override is normalized the same way
// so a missing or trailing-slash variant still resolves.
function normalizeBaseUrl(raw: string): string {
  const withoutTrailingSlash = raw.replace(/\/+$/u, "");
  return withoutTrailingSlash.endsWith(ASTRONOMICAL_API_BASE_PATH)
    ? withoutTrailingSlash
    : `${withoutTrailingSlash}${ASTRONOMICAL_API_BASE_PATH}`;
}

export function resolveAstronomicalChannels(
  env: NodeJS.ProcessEnv = process.env,
): readonly AstronomicalChannel[] {
  // An explicit endpoint override replaces channel probing entirely: the user
  // pointed at one specific Astronomical-compatible endpoint, so probing the
  // default ports beside it would register models from a daemon they did not
  // ask for.
  const override = env[ASTRONOMICAL_BASE_URL_ENV]?.trim();
  if (override) {
    return [{ instance: "custom", baseUrl: normalizeBaseUrl(override) }];
  }

  const instance = env[ASTRONOMICAL_INSTANCE_ENV]?.trim().toLowerCase();
  if (instance === ASTRONOMICAL_INSTANCE_STABLE) {
    return [{ instance: "stable", baseUrl: STABLE_BASE_URL }];
  }
  if (instance === ASTRONOMICAL_INSTANCE_DEV) {
    return [{ instance: "dev", baseUrl: DEV_BASE_URL }];
  }

  // Default: both channels, stable first. Order only fixes display order; the
  // catalogue fetch serves every channel that answers.
  return [
    { instance: "stable", baseUrl: STABLE_BASE_URL },
    { instance: "dev", baseUrl: DEV_BASE_URL },
  ];
}
