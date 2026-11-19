// One discovery pass: which channels answer, and which provider definitions
// they produce. The result is a plain state
// value; deciding what to reload or log about it is provider-refresh.ts's job.

import { resolveAstronomicalChannels } from "./channels.ts";
import {
  fetchChannelChatModels,
  type ChannelChatModels,
} from "./channel-chat-models.ts";
import { describeError } from "./error-message.ts";
import type { AstronomicalPluginOptions } from "./plugin-options.ts";
import {
  buildAstronomicalProviderDefinition,
  type AstronomicalProviderDefinition,
} from "./provider-definition.ts";

// One row per channel in the audit trail, so "which daemon answered with how
// many models" is answerable from the log alone.
export interface AstronomicalChannelSummary {
  readonly instance: string;
  readonly modelCount: number;
}

export type ProviderRefreshState =
  | { readonly kind: "unavailable"; readonly reason: string }
  | { readonly kind: "empty"; readonly reason: string }
  | {
      readonly kind: "ready";
      readonly definitions: readonly AstronomicalProviderDefinition[];
      readonly channels: readonly AstronomicalChannelSummary[];
    }
  | { readonly kind: "failed"; readonly reason: string };

export interface ProviderDiscoveryInput {
  readonly options: AstronomicalPluginOptions;
  readonly fetchImpl?: typeof fetch;
  readonly onDiagnostic?: (message: string) => void;
}

/** A sentence for the log saying why no provider is registered. */
export function describeProviderState(state: ProviderRefreshState): string {
  switch (state.kind) {
    case "unavailable":
    case "empty":
    case "failed":
      return state.reason;
    case "ready":
      return "provider is available";
  }
}

// A channel that answers with no chat models is left out, because a provider
// with zero models is dead weight in the picker; the reason is kept instead,
// so the silence is explainable.
function toReadyOrEmptyState(
  channels: readonly ChannelChatModels[],
  reportDiagnostic: (message: string) => void,
): ProviderRefreshState {
  const definitions: AstronomicalProviderDefinition[] = [];
  const channelSummaries: AstronomicalChannelSummary[] = [];
  const emptyChannelReasons: string[] = [];
  for (const channel of channels) {
    if (channel.models.length === 0) {
      const reason = `No chat-capable models were returned by ${channel.baseUrl}`;
      reportDiagnostic(reason);
      emptyChannelReasons.push(reason);
      continue;
    }
    definitions.push(buildAstronomicalProviderDefinition(channel));
    channelSummaries.push({
      instance: channel.instance,
      modelCount: channel.models.length,
    });
  }
  if (definitions.length === 0) {
    return { kind: "empty", reason: emptyChannelReasons.join("; ") };
  }
  return { kind: "ready", definitions, channels: channelSummaries };
}

/** Runs one discovery pass; never throws. */
export async function discoverProviderState(
  input: ProviderDiscoveryInput,
): Promise<ProviderRefreshState> {
  let latestDiagnostic: string | undefined;
  const reportDiagnostic = (message: string): void => {
    latestDiagnostic = message;
    try {
      input.onDiagnostic?.(message);
    } catch {
      // An optional diagnostic observer cannot be allowed to block discovery.
    }
  };
  try {
    const channelChatModels = await fetchChannelChatModels({
      channels: resolveAstronomicalChannels(input.options),
      fetchImpl: input.fetchImpl,
      onDiagnostic: reportDiagnostic,
    });
    if (channelChatModels.kind === "failure") {
      const reason = latestDiagnostic ?? "No Astronomical channel is reachable";
      return { kind: "unavailable", reason };
    }
    return toReadyOrEmptyState(channelChatModels.channels, reportDiagnostic);
  } catch (cause: unknown) {
    return { kind: "failed", reason: describeError(cause) };
  }
}
