// The server part of the plugin: registers Astronomical as a live OpenCode
// provider, one provider entry per running channel, and refreshes the list on
// a timer. Discovery, the provider definitions and the refresh decisions each
// live in their own module; this file only wires them to OpenCode's lifecycle.

import { Plugin } from "@opencode/plugin";
import {
  createAstronomicalAuditLogger,
  recordAuditEvent,
  type AstronomicalAuditLogger,
} from "./audit-log.ts";
import {
  ASTRONOMICAL_PLUGIN_ID,
  ASTRONOMICAL_PROVIDER_REFRESH_INTERVAL_MS,
} from "./constants.ts";
import { parseAstronomicalPluginOptions } from "./plugin-options.ts";
import { discoverProviderState } from "./provider-discovery.ts";
import { ProviderRefreshController } from "./provider-refresh-controller.ts";

export type AstronomicalRefreshScheduler = (
  refresh: () => Promise<void>,
  intervalMs: number,
) => () => void;

export interface AstronomicalPluginDependencies {
  readonly fetchImpl?: typeof fetch;
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

/** Creates the server plugin with injectable network, audit and timer collaborators. */
export function createAstronomicalPlugin(
  dependencies: AstronomicalPluginDependencies = {},
): Plugin.Plugin {
  const audit = dependencies.audit ?? createAstronomicalAuditLogger();
  const scheduleRefresh =
    dependencies.scheduleRefresh ?? scheduleProviderRefresh;

  return Plugin.define({
    id: ASTRONOMICAL_PLUGIN_ID,
    setup: async (context: Plugin.Context): Promise<() => void> => {
      // A wrong option stops setup here, with a message naming the option.
      const options = parseAstronomicalPluginOptions(context.options);
      await recordAuditEvent(audit, "setup-start", { entry: "server" });
      const controller = new ProviderRefreshController(
        context,
        () =>
          discoverProviderState({
            options,
            fetchImpl: dependencies.fetchImpl,
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
