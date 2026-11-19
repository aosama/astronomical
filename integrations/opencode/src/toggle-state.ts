// Shared terminal/server state for the provider toggle. The terminal plugin
// writes this file and the server plugin reads it, since OpenCode's server and
// terminal plugin contexts do not share a plugin-enabled setting.

import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import {
  ASTRONOMICAL_LEGACY_TUI_STATE_KEY,
  ASTRONOMICAL_PROVIDER_NAME,
} from "./constants.js";
import { resolveOpenCodeStateDirectory } from "./opencode-state-directory.js";
import { isRecord } from "./wire-values.js";

export interface AstronomicalPluginStateOptions {
  readonly stateFilePath?: string;
  readonly legacyStateFilePath?: string;
  readonly environment?: NodeJS.ProcessEnv;
  readonly homeDirectory?: string;
}

/** Resolves the state file shared by the terminal and server plugin processes. */
export function resolveAstronomicalPluginStateFilePath(
  environment: NodeJS.ProcessEnv = process.env,
  homeDirectory: string = homedir(),
): string {
  return join(
    resolveOpenCodeStateDirectory(environment, homeDirectory),
    "astronomical-plugin-state.json",
  );
}

function resolveLegacyTuiStateFilePath(
  options: AstronomicalPluginStateOptions,
): string {
  return (
    options.legacyStateFilePath ??
    join(
      resolveOpenCodeStateDirectory(options.environment, options.homeDirectory),
      "kv.json",
    )
  );
}

async function readLegacyTuiPreference(
  legacyStateFilePath: string,
): Promise<boolean> {
  try {
    const legacyState: unknown = JSON.parse(
      await readFile(legacyStateFilePath, "utf8"),
    );
    if (!isRecord(legacyState)) return true;
    const pluginStates: unknown =
      legacyState[ASTRONOMICAL_LEGACY_TUI_STATE_KEY];
    if (!isRecord(pluginStates)) return true;
    const isEnabled: unknown = pluginStates[ASTRONOMICAL_PROVIDER_NAME];
    return typeof isEnabled === "boolean" ? isEnabled : true;
  } catch {
    // Missing or unreadable legacy state means the user has not disabled it.
    return true;
  }
}

/** Reads the toggle, falling back to the old terminal preference on upgrade. */
export async function readAstronomicalPluginEnabled(
  options: AstronomicalPluginStateOptions = {},
): Promise<boolean> {
  const stateFilePath: string =
    options.stateFilePath ??
    resolveAstronomicalPluginStateFilePath(
      options.environment,
      options.homeDirectory,
    );
  try {
    const toggleState: unknown = JSON.parse(
      await readFile(stateFilePath, "utf8"),
    );
    if (isRecord(toggleState) && typeof toggleState["enabled"] === "boolean") {
      return toggleState["enabled"];
    }
  } catch {
    // First run or invalid state falls back to the previous terminal preference.
  }
  return readLegacyTuiPreference(resolveLegacyTuiStateFilePath(options));
}

/** Persists the toggle atomically so the server never reads partial JSON. */
export async function writeAstronomicalPluginEnabled(
  isEnabled: boolean,
  options: AstronomicalPluginStateOptions = {},
): Promise<void> {
  const stateFilePath: string =
    options.stateFilePath ??
    resolveAstronomicalPluginStateFilePath(
      options.environment,
      options.homeDirectory,
    );
  const temporaryFilePath = `${stateFilePath}.${process.pid}.${randomUUID()}.tmp`;
  await mkdir(dirname(stateFilePath), { recursive: true });
  try {
    await writeFile(
      temporaryFilePath,
      `${JSON.stringify({ enabled: isEnabled })}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    await rename(temporaryFilePath, stateFilePath);
  } catch (cause: unknown) {
    // Cleanup is best-effort; the write or rename error is the one to report.
    await rm(temporaryFilePath, { force: true }).catch((): void => {});
    throw cause;
  }
}
