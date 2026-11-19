// The terminal part of the plugin: adds /astronomical-status to OpenCode's
// terminal. The command lives in its own module; this file only registers it.
//
// There is no on/off command: OpenCode switches any plugin off with
// "-astronomical" in the plugins list, which also stops the server part.

import { Plugin } from "@opencode/plugin/tui";
import type {
  Context as TuiPluginContext,
  KeymapLayer,
} from "@opencode/plugin/tui/context";
import {
  createAstronomicalAuditLogger,
  recordAuditEvent,
  type AstronomicalAuditLogger,
} from "./audit-log.ts";
import {
  ASTRONOMICAL_PROVIDER_NAME,
  ASTRONOMICAL_STATUS_COMMAND_ID,
  ASTRONOMICAL_TUI_PLUGIN_ID,
} from "./constants.ts";
import { parseAstronomicalPluginOptions } from "./plugin-options.ts";
import {
  showAstronomicalStatus,
  type StatusCommandInput,
} from "./status-command.ts";

export interface AstronomicalTuiPluginDependencies {
  readonly fetchImpl?: typeof fetch;
  readonly audit?: AstronomicalAuditLogger;
}

function commandLayer(
  context: TuiPluginContext,
  statusInput: StatusCommandInput,
): KeymapLayer {
  return {
    mode: "global",
    commands: [
      {
        id: ASTRONOMICAL_STATUS_COMMAND_ID,
        title: `${ASTRONOMICAL_PROVIDER_NAME}: Status`,
        description:
          "Check which Astronomical instances answer and their chat models",
        group: ASTRONOMICAL_PROVIDER_NAME,
        palette: true,
        slash: { name: "astronomical-status" },
        run: (): Promise<void> => showAstronomicalStatus(context, statusInput),
      },
    ],
  };
}

/** Creates the terminal plugin with injectable network and audit collaborators. */
export function createAstronomicalTuiPlugin(
  dependencies: AstronomicalTuiPluginDependencies = {},
): Plugin.Definition {
  const audit = dependencies.audit ?? createAstronomicalAuditLogger();
  return Plugin.define({
    id: ASTRONOMICAL_TUI_PLUGIN_ID,
    setup: async (context: TuiPluginContext): Promise<void> => {
      const statusInput: StatusCommandInput = {
        options: parseAstronomicalPluginOptions(context.options),
        fetchImpl: dependencies.fetchImpl ?? fetch,
        audit,
      };
      await recordAuditEvent(audit, "setup-start", { entry: "terminal" });
      context.keymap.layer((): KeymapLayer =>
        commandLayer(context, statusInput),
      );
    },
  });
}

export const AstronomicalTuiPlugin: Plugin.Definition =
  createAstronomicalTuiPlugin();

export default AstronomicalTuiPlugin;
