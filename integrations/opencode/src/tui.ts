// V2 terminal companion: report channel health and persist the shared provider
// toggle used by the server-side provider transform.

import { Plugin } from "@opencode/plugin/tui";
import type {
  Context as TuiPluginContext,
  KeymapLayer,
} from "@opencode/plugin/tui/context";
import {
  createAstronomicalAuditLogger,
  type AstronomicalAuditLogger,
} from "./astronomer-audit-log.js";
import { resolveAstronomicalChannels } from "./astronomer-base-url.js";
import {
  fetchAstronomicalCatalogue,
  type AstronomicalChannelCatalogue,
  type FetchAstronomicalCatalogueResult,
} from "./astronomer-catalogue.js";
import {
  readAstronomicalPluginEnabled,
  writeAstronomicalPluginEnabled,
} from "./astronomer-plugin-state.js";
import {
  ASTRONOMICAL_CATALOGUE_TIMEOUT_MS,
  ASTRONOMICAL_CHANNEL_PROVIDER_NAMES,
  ASTRONOMICAL_PROVIDER_NAME,
  ASTRONOMICAL_STATUS_COMMAND_ID,
  ASTRONOMICAL_TOGGLE_COMMAND_ID,
  ASTRONOMICAL_TUI_PLUGIN_ID,
} from "./constants.js";

export interface AstronomicalTuiPluginDependencies {
  readonly env?: NodeJS.ProcessEnv;
  readonly fetchImpl?: typeof fetch;
  readonly readEnabled?: () => Promise<boolean>;
  readonly writeEnabled?: (isEnabled: boolean) => Promise<void>;
  readonly audit?: AstronomicalAuditLogger;
}

type AstronomicalToggleChoice = "enabled" | "disabled";

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

async function safelyAudit(
  audit: AstronomicalAuditLogger,
  event: Parameters<AstronomicalAuditLogger>[0],
  detail?: Record<string, unknown>,
): Promise<void> {
  try {
    await audit(event, detail);
  } catch {
    // A failed diagnostic write must not interrupt a terminal command.
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readAstronomicalVersion(statusPayload: unknown): string | undefined {
  if (!isRecord(statusPayload)) return undefined;

  const application: unknown = statusPayload["application"];
  if (!isRecord(application)) return undefined;

  const version: unknown = application["version"];
  return typeof version === "string" ? version : undefined;
}

// One status line per channel, e.g. "Astronomical 0.2.77: 13 chat models". A
// channel whose /status probe fails still reports its model count, because the
// catalogue already proved the daemon is up; only the version is optional.
async function describeChannel(
  catalogue: AstronomicalChannelCatalogue,
  fetchImpl: typeof fetch,
): Promise<string> {
  const channelName = ASTRONOMICAL_CHANNEL_PROVIDER_NAMES[catalogue.instance];
  let version: string | undefined;

  try {
    const statusResponse: Response = await fetchImpl(
      `${catalogue.baseUrl}/status`,
      { signal: AbortSignal.timeout(ASTRONOMICAL_CATALOGUE_TIMEOUT_MS) },
    );
    if (statusResponse.ok) {
      version = readAstronomicalVersion(await statusResponse.json());
    }
  } catch {
    // A failed status read downgrades to "reachable, version unknown" rather
    // than failing the whole status command.
  }

  const labeledName =
    version === undefined ? channelName : `${channelName} ${version}`;
  return `${labeledName}: ${catalogue.models.length} chat models`;
}

async function showStatus(
  context: TuiPluginContext,
  dependencies: AstronomicalTuiPluginDependencies,
  audit: AstronomicalAuditLogger,
): Promise<void> {
  await safelyAudit(audit, "status-command");
  const environment: NodeJS.ProcessEnv = dependencies.env ?? process.env;
  let diagnosticMessage: string | undefined;
  const catalogue: FetchAstronomicalCatalogueResult =
    await fetchAstronomicalCatalogue({
      channels: resolveAstronomicalChannels(environment),
      fetchImpl: dependencies.fetchImpl,
      onDiagnostic: (message: string): void => {
        diagnosticMessage = message;
      },
    });

  if (catalogue.kind === "failure") {
    await safelyAudit(audit, "diagnostic", {
      stage: "status-command",
      reason: diagnosticMessage ?? "No Astronomical channel is reachable",
    });
    context.ui.toast.show({
      variant: "warning",
      title: ASTRONOMICAL_PROVIDER_NAME,
      message:
        "No Astronomical channel is reachable. Start Astronomical to use its models.",
    });
    return;
  }

  // Every answering channel gets a line even when it serves zero chat models:
  // "Astronomical 0.2.77: 0 chat models" tells the user the daemon is up and
  // merely has nothing chat-capable loaded, which a silent omission cannot.
  const fetchImpl: typeof fetch = dependencies.fetchImpl ?? fetch;
  // Channels are described in resolution order (stable before dev) so the
  // toast reads the same way the picker lists the providers.
  const channelLines: string[] = [];
  for (const channelCatalogue of catalogue.channels) {
    channelLines.push(await describeChannel(channelCatalogue, fetchImpl));
  }

  context.ui.toast.show({
    variant: "success",
    title: ASTRONOMICAL_PROVIDER_NAME,
    message: channelLines.join("; "),
  });
}

async function showToggle(
  context: TuiPluginContext,
  readEnabled: () => Promise<boolean>,
  writeEnabled: (isEnabled: boolean) => Promise<void>,
  audit: AstronomicalAuditLogger,
): Promise<void> {
  try {
    const isCurrentlyEnabled: boolean = await readEnabled();
    const choice: AstronomicalToggleChoice | undefined =
      await context.ui.dialog.select<AstronomicalToggleChoice>({
        title: "Astronomical provider",
        current: isCurrentlyEnabled ? "enabled" : "disabled",
        options: [
          {
            title: "Enabled",
            value: "enabled",
            description: "Discover and offer the live chat models.",
          },
          {
            title: "Disabled",
            value: "disabled",
            description: "Remove Astronomical models from the picker.",
          },
        ],
      });

    if (choice === undefined) return;

    const isEnabled: boolean = choice === "enabled";
    await writeEnabled(isEnabled);
    await safelyAudit(audit, "toggle-command", { enabled: isEnabled });
    context.ui.toast.show({
      variant: "success",
      title: ASTRONOMICAL_PROVIDER_NAME,
      message: isEnabled
        ? "Astronomical is enabled. The server will refresh its models shortly."
        : "Astronomical is disabled. The server will remove its models shortly.",
    });
  } catch (cause: unknown) {
    const reason: string = errorMessage(cause);
    await safelyAudit(audit, "diagnostic", {
      stage: "toggle-command",
      reason,
    });
    context.ui.toast.show({
      variant: "error",
      title: ASTRONOMICAL_PROVIDER_NAME,
      message: `Could not update the provider toggle: ${reason}`,
    });
  }
}

/** Creates the V2 terminal plugin with injectable status and toggle collaborators. */
export function createAstronomicalTuiPlugin(
  dependencies: AstronomicalTuiPluginDependencies = {},
): Plugin.Definition {
  const environment: NodeJS.ProcessEnv = dependencies.env ?? process.env;
  const readEnabled: () => Promise<boolean> =
    dependencies.readEnabled ??
    ((): Promise<boolean> => readAstronomicalPluginEnabled({ environment }));
  const writeEnabled: (isEnabled: boolean) => Promise<void> =
    dependencies.writeEnabled ??
    ((isEnabled: boolean): Promise<void> =>
      writeAstronomicalPluginEnabled(isEnabled, { environment }));
  const audit: AstronomicalAuditLogger =
    dependencies.audit ?? createAstronomicalAuditLogger();

  return Plugin.define({
    id: ASTRONOMICAL_TUI_PLUGIN_ID,
    setup: async (context: TuiPluginContext): Promise<void> => {
      await safelyAudit(audit, "setup-start", { entry: "terminal" });
      context.keymap.layer((): KeymapLayer => ({
        mode: "global",
        commands: [
          {
            id: ASTRONOMICAL_STATUS_COMMAND_ID,
            title: `${ASTRONOMICAL_PROVIDER_NAME}: Status`,
            description: "Check the active channel and available chat models",
            group: ASTRONOMICAL_PROVIDER_NAME,
            palette: true,
            slash: { name: "astronomical-status" },
            run: async (): Promise<void> =>
              showStatus(context, dependencies, audit),
          },
          {
            id: ASTRONOMICAL_TOGGLE_COMMAND_ID,
            title: `${ASTRONOMICAL_PROVIDER_NAME}: Toggle`,
            description: "Enable or disable the local provider",
            group: ASTRONOMICAL_PROVIDER_NAME,
            palette: true,
            slash: { name: "astronomical-toggle" },
            run: async (): Promise<void> =>
              showToggle(context, readEnabled, writeEnabled, audit),
          },
        ],
      }));
    },
  });
}

export const AstronomicalTuiPlugin: Plugin.Definition =
  createAstronomicalTuiPlugin();

export default AstronomicalTuiPlugin;
