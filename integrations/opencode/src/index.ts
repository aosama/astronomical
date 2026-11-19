// The server part of the plugin: registers Astronomical as a live OpenCode
// provider, one provider entry per running channel, and refreshes the list on
// a timer. Discovery, the provider definitions and the refresh decisions each
// live in their own module; this file only wires them to OpenCode's lifecycle.

import { Plugin } from "@opencode/plugin";
import {
  createAstronomicalAuditLogger,
  recordAuditEvent,
  type AstronomicalAuditLogger,
} from "./audit-log.js";
import {
  ASTRONOMICAL_PLUGIN_ID,
  ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS,
} from "./constants.js";
import { discoverProviderState } from "./provider-discovery.js";
import { ProviderRefreshController } from "./provider-refresh-controller.js";
import { readAstronomicalPluginEnabled } from "./toggle-state.js";

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

function scheduleProviderRefresh(
  refresh: () => Promise<void>,
  intervalMs: number,
): () => void {
  const interval: NodeJS.Timeout = setInterval((): void => {
    void refresh();
  }, intervalMs);
  return (): void => clearInterval(interval);
}

/** Creates the server plugin with injectable network, state and audit collaborators. */
export function createAstronomicalPlugin(
  dependencies: AstronomicalPluginDependencies = {},
): Plugin.Plugin {
  const environment: NodeJS.ProcessEnv = dependencies.env ?? process.env;
  const readEnabled =
    dependencies.readEnabled ??
    ((): Promise<boolean> => readAstronomicalPluginEnabled({ environment }));
  const audit = dependencies.audit ?? createAstronomicalAuditLogger();
  const scheduleRefresh =
    dependencies.scheduleRefresh ?? scheduleProviderRefresh;

  return Plugin.define({
    id: ASTRONOMICAL_PLUGIN_ID,
    setup: async (context: Plugin.Context): Promise<() => void> => {
      await recordAuditEvent(audit, "setup-start", { entry: "server" });
      const controller = new ProviderRefreshController(
        context,
        () =>
          discoverProviderState({
            environment,
            fetchImpl: dependencies.fetchImpl,
            readEnabled,
            onDiagnostic: dependencies.onDiagnostic,
          }),
        audit,
      );
      await controller.refresh();
      await controller.registerTransform();
      const stopRefresh = scheduleRefresh(
        () => controller.refresh(),
        ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS,
      );
      return (): void => {
        controller.dispose();
        stopRefresh();
      };
    },
  });
}

export const AstronomicalPlugin: Plugin.Plugin = createAstronomicalPlugin();

export default AstronomicalPlugin;
