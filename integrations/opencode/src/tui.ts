// The terminal part of the plugin: adds /astronomical-status and
// /astronomical-toggle to OpenCode's terminal. Each command lives in its own
// module; this file only registers them.

import { Plugin } from "@opencode/plugin/tui";
import type {
  Context as TuiPluginContext,
  KeymapLayer,
} from "@opencode/plugin/tui/context";
import {
  createAstronomicalAuditLogger,
  recordAuditEvent,
  type AstronomicalAuditLogger,
} from "./audit-log.js";
import {
  ASTRONOMICAL_PROVIDER_NAME,
  ASTRONOMICAL_STATUS_COMMAND_ID,
  ASTRONOMICAL_TOGGLE_COMMAND_ID,
  ASTRONOMICAL_TUI_PLUGIN_ID,
} from "./constants.js";
import {
  showAstronomicalStatus,
  type StatusCommandInput,
} from "./status-command.js";
import {
  showAstronomicalToggle,
  type ToggleCommandInput,
} from "./toggle-command.js";
import {
  readAstronomicalPluginEnabled,
  writeAstronomicalPluginEnabled,
} from "./toggle-state.js";

export interface AstronomicalTuiPluginDependencies {
  readonly env?: NodeJS.ProcessEnv;
  readonly fetchImpl?: typeof fetch;
  readonly readEnabled?: () => Promise<boolean>;
  readonly writeEnabled?: (isEnabled: boolean) => Promise<void>;
  readonly audit?: AstronomicalAuditLogger;
}

function commandLayer(
  context: TuiPluginContext,
  statusInput: StatusCommandInput,
  toggleInput: ToggleCommandInput,
): KeymapLayer {
  return {
    mode: "global",
    commands: [
      {
        id: ASTRONOMICAL_STATUS_COMMAND_ID,
        title: `${ASTRONOMICAL_PROVIDER_NAME}: Status`,
        description: "Check the active channel and available chat models",
        group: ASTRONOMICAL_PROVIDER_NAME,
        palette: true,
        slash: { name: "astronomical-status" },
        run: (): Promise<void> => showAstronomicalStatus(context, statusInput),
      },
      {
        id: ASTRONOMICAL_TOGGLE_COMMAND_ID,
        title: `${ASTRONOMICAL_PROVIDER_NAME}: Toggle`,
        description: "Enable or disable the local provider",
        group: ASTRONOMICAL_PROVIDER_NAME,
        palette: true,
        slash: { name: "astronomical-toggle" },
        run: (): Promise<void> => showAstronomicalToggle(context, toggleInput),
      },
    ],
  };
}

/** Creates the terminal plugin with injectable status and toggle collaborators. */
export function createAstronomicalTuiPlugin(
  dependencies: AstronomicalTuiPluginDependencies = {},
): Plugin.Definition {
  const environment: NodeJS.ProcessEnv = dependencies.env ?? process.env;
  const audit = dependencies.audit ?? createAstronomicalAuditLogger();
  const statusInput: StatusCommandInput = {
    environment,
    fetchImpl: dependencies.fetchImpl ?? fetch,
    audit,
  };
  const toggleInput: ToggleCommandInput = {
    readEnabled:
      dependencies.readEnabled ??
      ((): Promise<boolean> => readAstronomicalPluginEnabled({ environment })),
    writeEnabled:
      dependencies.writeEnabled ??
      ((isEnabled: boolean): Promise<void> =>
        writeAstronomicalPluginEnabled(isEnabled, { environment })),
    audit,
  };
  return Plugin.define({
    id: ASTRONOMICAL_TUI_PLUGIN_ID,
    setup: async (context: TuiPluginContext): Promise<void> => {
      await recordAuditEvent(audit, "setup-start", { entry: "terminal" });
      context.keymap.layer((): KeymapLayer =>
        commandLayer(context, statusInput, toggleInput),
      );
    },
  });
}

export const AstronomicalTuiPlugin: Plugin.Definition =
  createAstronomicalTuiPlugin();

export default AstronomicalTuiPlugin;
