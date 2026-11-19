// Register Astronomical as a live OpenCode V2 provider, one provider entry per
// running channel (stable and dev are independent daemons with independent
// model catalogues). V1 config hooks returned a hook map; V2 provider
// definitions must be registered through a transform.

import { Plugin } from "@opencode/plugin";
import type { ProviderEditor } from "@opencode/plugin/promise/provider";
import {
  createAstronomicalAuditLogger,
  type AstronomicalAuditLogger,
} from "./astronomer-audit-log.js";
import { resolveAstronomicalChannels } from "./astronomer-base-url.js";
import {
  fetchAstronomicalCatalogue,
  type AstronomicalChannelCatalogue,
} from "./astronomer-catalogue.js";
import {
  buildAstronomicalProviderDefinition,
  type AstronomicalProviderDefinition,
} from "./astronomer-provider-config.js";
import { readAstronomicalPluginEnabled } from "./astronomer-plugin-state.js";
import {
  ASTRONOMICAL_PLUGIN_ID,
  ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS,
} from "./constants.js";

export type AstronomicalRefreshScheduler = (
  refresh: () => Promise<void>,
  intervalMs: number,
) => () => void;

export interface AstronomicalPluginDependencies {
  readonly env?: NodeJS.ProcessEnv;
  readonly fetchImpl?: typeof fetch;
  readonly readEnabled?: () => Promise<boolean>;
  readonly onDiagnostic?: (message: string) => void;
  readonly audit?: AstronomicalAuditLogger;
  readonly scheduleRefresh?: AstronomicalRefreshScheduler;
}

// One row per channel in the audit trail, so "which daemon answered with how
// many models" is answerable from the log alone.
export interface AstronomicalChannelSummary {
  readonly instance: string;
  readonly modelCount: number;
}

type ProviderRefreshState =
  | { readonly kind: "disabled" }
  | { readonly kind: "unavailable"; readonly reason: string }
  | { readonly kind: "empty"; readonly reason: string }
  | {
      readonly kind: "ready";
      readonly definitions: readonly AstronomicalProviderDefinition[];
      readonly channels: readonly AstronomicalChannelSummary[];
    }
  | { readonly kind: "failed"; readonly reason: string };

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function definitionFingerprint(
  definitions: readonly AstronomicalProviderDefinition[],
): string {
  const serializedDefinitions: string | undefined = JSON.stringify(definitions);
  if (serializedDefinitions === undefined) {
    throw new Error("Could not compare the Astronomical provider definitions");
  }
  return serializedDefinitions;
}

// A channel that answers but serves no chat models is excluded from
// registration, because an OpenCode provider entry with zero models is
// dead weight in the picker; the reason is surfaced instead so the silence is
// explainable.
function buildChannelDefinitions(
  catalogues: readonly AstronomicalChannelCatalogue[],
  onDiagnostic: (message: string) => void,
): {
  readonly definitions: readonly AstronomicalProviderDefinition[];
  readonly channels: readonly AstronomicalChannelSummary[];
  readonly skipped: readonly string[];
} {
  const definitions: AstronomicalProviderDefinition[] = [];
  const channelSummaries: AstronomicalChannelSummary[] = [];
  const skipped: string[] = [];

  for (const catalogue of catalogues) {
    if (catalogue.models.length === 0) {
      const reason = `No chat-capable models were returned by ${catalogue.baseUrl}`;
      onDiagnostic(reason);
      skipped.push(reason);
      continue;
    }

    definitions.push(
      buildAstronomicalProviderDefinition({
        instance: catalogue.instance,
        baseUrl: catalogue.baseUrl,
        models: catalogue.models,
      }),
    );
    channelSummaries.push({
      instance: catalogue.instance,
      modelCount: catalogue.models.length,
    });
  }

  return { definitions, channels: channelSummaries, skipped };
}

async function safelyAudit(
  audit: AstronomicalAuditLogger,
  event: Parameters<AstronomicalAuditLogger>[0],
  detail?: Record<string, unknown>,
): Promise<void> {
  try {
    await audit(event, detail);
  } catch {
    // Diagnostics must not prevent provider registration or removal.
  }
}

function scheduleProviderRefresh(
  refresh: () => Promise<void>,
  intervalMs: number,
): () => void {
  const interval: NodeJS.Timeout = setInterval((): void => {
    void refresh();
  }, intervalMs);

  return (): void => clearInterval(interval);
}

async function discoverProviderState(input: {
  readonly environment: NodeJS.ProcessEnv;
  readonly fetchImpl?: typeof fetch;
  readonly readEnabled: () => Promise<boolean>;
  readonly onDiagnostic?: (message: string) => void;
}): Promise<ProviderRefreshState> {
  let diagnosticMessage: string | undefined;
  const onDiagnostic = (message: string): void => {
    diagnosticMessage = message;
    try {
      input.onDiagnostic?.(message);
    } catch {
      // An optional diagnostic observer cannot be allowed to block discovery.
    }
  };

  try {
    if (!(await input.readEnabled())) return { kind: "disabled" };

    const channels = resolveAstronomicalChannels(input.environment);
    const catalogue = await fetchAstronomicalCatalogue({
      channels,
      fetchImpl: input.fetchImpl,
      onDiagnostic,
    });

    if (catalogue.kind === "failure") {
      return {
        kind: "unavailable",
        reason: diagnosticMessage ?? "No Astronomical channel is reachable",
      };
    }

    const built = buildChannelDefinitions(catalogue.channels, onDiagnostic);
    if (built.definitions.length === 0) {
      return {
        kind: "empty",
        reason: built.skipped.join("; "),
      };
    }

    return {
      kind: "ready",
      definitions: built.definitions,
      channels: built.channels,
    };
  } catch (cause: unknown) {
    return { kind: "failed", reason: errorMessage(cause) };
  }
}

function describeSkippedState(state: ProviderRefreshState): string {
  switch (state.kind) {
    case "disabled":
      return "terminal toggle is off";
    case "unavailable":
    case "empty":
    case "failed":
      return state.reason;
    case "ready":
      return "provider is available";
  }
}

/** Creates the V2 server plugin with injectable network, state, and audit collaborators. */
export function createAstronomicalPlugin(
  dependencies: AstronomicalPluginDependencies = {},
): Plugin.Plugin {
  const environment: NodeJS.ProcessEnv = dependencies.env ?? process.env;
  const readEnabled: () => Promise<boolean> =
    dependencies.readEnabled ??
    ((): Promise<boolean> => readAstronomicalPluginEnabled({ environment }));
  const audit: AstronomicalAuditLogger =
    dependencies.audit ?? createAstronomicalAuditLogger();
  const refreshScheduler: AstronomicalRefreshScheduler =
    dependencies.scheduleRefresh ?? scheduleProviderRefresh;

  return Plugin.define({
    id: ASTRONOMICAL_PLUGIN_ID,
    setup: async (context: Plugin.Context): Promise<() => void> => {
      await safelyAudit(audit, "setup-start", { entry: "server" });

      let currentDefinitions: readonly AstronomicalProviderDefinition[] = [];
      let currentDefinitionFingerprint: string | undefined;
      let registeredDefinitionFingerprint: string | undefined;
      let currentState: ProviderRefreshState | undefined;
      let isTransformRegistered: boolean = false;
      let isRefreshInProgress: boolean = false;
      let isDisposed: boolean = false;

      const refreshProvider = async (): Promise<void> => {
        if (isRefreshInProgress || isDisposed) return;
        isRefreshInProgress = true;

        try {
          const nextState: ProviderRefreshState = await discoverProviderState({
            environment,
            fetchImpl: dependencies.fetchImpl,
            readEnabled,
            onDiagnostic: dependencies.onDiagnostic,
          });
          if (isDisposed) return;

          const nextDefinitions: readonly AstronomicalProviderDefinition[] =
            nextState.kind === "ready" ? nextState.definitions : [];
          const nextFingerprint: string | undefined =
            nextDefinitions.length > 0
              ? definitionFingerprint(nextDefinitions)
              : undefined;
          const previousState: ProviderRefreshState | undefined = currentState;
          const previousRegisteredFingerprint: string | undefined =
            registeredDefinitionFingerprint;

          currentDefinitions = nextDefinitions;
          currentDefinitionFingerprint = nextFingerprint;

          if (
            isTransformRegistered &&
            currentDefinitionFingerprint !== registeredDefinitionFingerprint
          ) {
            await context.provider.reload();
            registeredDefinitionFingerprint = currentDefinitionFingerprint;

            if (nextState.kind === "ready") {
              await safelyAudit(
                audit,
                previousRegisteredFingerprint === undefined
                  ? "provider-registered"
                  : "provider-refreshed",
                {
                  channels: nextState.channels,
                },
              );
            } else if (previousRegisteredFingerprint !== undefined) {
              await safelyAudit(audit, "provider-removed", {
                reason: describeSkippedState(nextState),
              });
            }
          }

          if (
            nextDefinitions.length === 0 &&
            previousState?.kind !== nextState.kind
          ) {
            if (previousState === undefined) {
              await safelyAudit(audit, "setup-skipped", {
                reason: describeSkippedState(nextState),
              });
            } else if (nextState.kind !== "disabled") {
              await safelyAudit(audit, "diagnostic", {
                reason: describeSkippedState(nextState),
              });
            }
          }

          currentState = nextState;
        } catch (cause: unknown) {
          await safelyAudit(audit, "diagnostic", {
            stage: "provider-refresh",
            reason: errorMessage(cause),
          });
        } finally {
          isRefreshInProgress = false;
        }
      };

      await refreshProvider();
      await context.provider.transform((editor: ProviderEditor): void => {
        // One editor.add per channel: each definition carries its own provider
        // id and baseURL, so stable and dev coexist in the picker.
        for (const definition of currentDefinitions) {
          editor.add(definition);
        }
      });
      isTransformRegistered = true;
      registeredDefinitionFingerprint = currentDefinitionFingerprint;

      if (currentState?.kind === "ready") {
        await safelyAudit(audit, "provider-registered", {
          channels: currentState.channels,
        });
      }

      const stopRefresh: () => void = refreshScheduler(
        refreshProvider,
        ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS,
      );

      return (): void => {
        isDisposed = true;
        stopRefresh();
      };
    },
  });
}

export const AstronomicalPlugin: Plugin.Plugin = createAstronomicalPlugin();

export default AstronomicalPlugin;
