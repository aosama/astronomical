// Shared terminal/server state for the provider toggle. The CLI plugin writes
// this file and the server plugin reads it, since the V2 server and terminal
// plugin contexts do not share a plugin-enabled setting.

import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import {
  ASTRONOMICAL_LEGACY_TUI_STATE_KEY,
  ASTRONOMICAL_PROVIDER_NAME,
} from "./constants.js";

export interface AstronomicalPluginStateOptions {
  readonly stateFilePath?: string;
  readonly legacyStateFilePath?: string;
  readonly environment?: NodeJS.ProcessEnv;
  readonly homeDirectory?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Resolves the state file shared by the terminal and server plugin processes. */
export function resolveAstronomicalPluginStateFilePath(
  environment: NodeJS.ProcessEnv = process.env,
  homeDirectory: string = homedir(),
): string {
  const stateHome =
    environment.XDG_STATE_HOME?.trim() ||
    join(homeDirectory, ".local", "state");
  return join(stateHome, "opencode", "astronomical-plugin-state.json");
}

function resolveLegacyTuiStateFilePath(
  options: AstronomicalPluginStateOptions,
): string {
  if (options.legacyStateFilePath !== undefined) {
    return options.legacyStateFilePath;
  }

  const environment: NodeJS.ProcessEnv = options.environment ?? process.env;
  const homeDirectory: string = options.homeDirectory ?? homedir();
  const stateHome =
    environment.XDG_STATE_HOME?.trim() ||
    join(homeDirectory, ".local", "state");
  return join(stateHome, "opencode", "kv.json");
}

async function readLegacyTuiPreference(
  legacyStateFilePath: string,
): Promise<boolean> {
  try {
    const stateText: string = await readFile(legacyStateFilePath, "utf8");
    const state: unknown = JSON.parse(stateText);
    if (!isRecord(state)) return true;

    const pluginStates: unknown = state[ASTRONOMICAL_LEGACY_TUI_STATE_KEY];
    if (!isRecord(pluginStates)) return true;

    const enabled: unknown = pluginStates[ASTRONOMICAL_PROVIDER_NAME];
    return typeof enabled === "boolean" ? enabled : true;
  } catch {
    // Missing or unreadable legacy state means the user has not disabled it.
    return true;
  }
}

/** Reads the V2 toggle, falling back to the old terminal preference on upgrade. */
export async function readAstronomicalPluginEnabled(
  options: AstronomicalPluginStateOptions = {},
): Promise<boolean> {
  const environment: NodeJS.ProcessEnv = options.environment ?? process.env;
  const homeDirectory: string = options.homeDirectory ?? homedir();
  const stateFilePath: string =
    options.stateFilePath ??
    resolveAstronomicalPluginStateFilePath(environment, homeDirectory);

  try {
    const stateText: string = await readFile(stateFilePath, "utf8");
    const state: unknown = JSON.parse(stateText);
    if (isRecord(state) && typeof state["enabled"] === "boolean") {
      return state["enabled"];
    }
  } catch {
    // First run or invalid state falls back to the previous terminal preference.
  }

  return readLegacyTuiPreference(resolveLegacyTuiStateFilePath(options));
}

/**
 * Persists a provider toggle atomically so the server never reads partial JSON.
 */
export async function writeAstronomicalPluginEnabled(
  isEnabled: boolean,
  options: AstronomicalPluginStateOptions = {},
): Promise<void> {
  const environment: NodeJS.ProcessEnv = options.environment ?? process.env;
  const homeDirectory: string = options.homeDirectory ?? homedir();
  const stateFilePath: string =
    options.stateFilePath ??
    resolveAstronomicalPluginStateFilePath(environment, homeDirectory);
  const temporaryFilePath: string = `${stateFilePath}.${process.pid}.${randomUUID()}.tmp`;

  await mkdir(dirname(stateFilePath), { recursive: true });

  try {
    await writeFile(
      temporaryFilePath,
      `${JSON.stringify({ enabled: isEnabled })}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    await rename(temporaryFilePath, stateFilePath);
  } catch (cause: unknown) {
    try {
      await rm(temporaryFilePath, { force: true });
    } catch {
      // Preserve the write or rename error; cleanup is best-effort only.
    }
    throw cause;
  }
}
