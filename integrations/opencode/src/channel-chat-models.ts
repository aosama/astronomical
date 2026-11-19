// The chat models of every reachable Astronomical channel.
//
// Every channel is probed and every answer is kept: the Stable and
// Development daemons are separate installations serving separate model sets,
// so one answering must not hide the other. A channel that fails is left out;
// when every channel fails the result is a failure rather than an empty or
// stale list, so OpenCode simply omits the providers until one answers.

import type { AstronomicalChannel } from "./channels.ts";
import { selectChatModels } from "./chat-models.ts";
import {
  ASTRONOMICAL_PROBE_TIMEOUT_MS,
  type AstronomicalChannelInstance,
} from "./constants.ts";
import type { AstronomicalModelRow } from "./model-row.ts";
import {
  fetchModelListPayload,
  type ModelListPayloadResult,
} from "./model-list-transport.ts";

export type FetchChannelChatModelsOptions = {
  readonly channels: readonly AstronomicalChannel[];
  readonly fetchImpl?: typeof fetch;
  readonly timeoutMs?: number;
  readonly onDiagnostic?: (message: string) => void;
};

// One reachable channel. A channel that answered appears here even with zero
// chat models, so "daemon up, nothing chat-capable" stays distinguishable from
// "daemon down".
export interface ChannelChatModels {
  readonly instance: AstronomicalChannelInstance;
  readonly baseUrl: string;
  readonly models: readonly AstronomicalModelRow[];
}

export type FetchChannelChatModelsResult =
  | {
      readonly kind: "success";
      readonly channels: readonly ChannelChatModels[];
    }
  | { readonly kind: "failure" };

/**
 * Probes all channels at once: probing one after another let a stopped
 * channel's five-second timeout delay every answering one behind it.
 */
export async function fetchChannelChatModels(
  options: FetchChannelChatModelsOptions,
): Promise<FetchChannelChatModelsResult> {
  const fetchImpl: typeof fetch = options.fetchImpl ?? fetch;
  const timeoutMs: number = options.timeoutMs ?? ASTRONOMICAL_PROBE_TIMEOUT_MS;
  const pendingProbes: Promise<ModelListPayloadResult>[] = [];
  for (const channel of options.channels) {
    pendingProbes.push(
      fetchModelListPayload({ baseUrl: channel.baseUrl, fetchImpl, timeoutMs }),
    );
  }
  const probeResults = await Promise.all(pendingProbes);

  const reachableChannels: ChannelChatModels[] = [];
  const unreachableChannels: string[] = [];
  for (const [channelIndex, channel] of options.channels.entries()) {
    const probeResult = probeResults[channelIndex]!;
    if (probeResult.kind === "success") {
      reachableChannels.push(chatModelsOf(channel, probeResult.payload));
    } else {
      unreachableChannels.push(`${channel.baseUrl} (${probeResult.reason})`);
    }
  }
  return summarizeProbes(options, reachableChannels, unreachableChannels);
}

function chatModelsOf(
  channel: AstronomicalChannel,
  payload: unknown,
): ChannelChatModels {
  return {
    instance: channel.instance,
    baseUrl: channel.baseUrl,
    models: selectChatModels(payload),
  };
}

// One diagnostic per refresh, not one per channel, so the log stays readable.
function summarizeProbes(
  options: FetchChannelChatModelsOptions,
  reachableChannels: readonly ChannelChatModels[],
  unreachableChannels: readonly string[],
): FetchChannelChatModelsResult {
  if (reachableChannels.length === 0 && unreachableChannels.length > 0) {
    options.onDiagnostic?.(
      `no Astronomical channel is reachable — ${unreachableChannels.join("; ")}; start Astronomical to register its models`,
    );
    return { kind: "failure" };
  }
  if (unreachableChannels.length > 0) {
    options.onDiagnostic?.(
      `skipped unreachable Astronomical channel(s) — ${unreachableChannels.join("; ")}`,
    );
  }
  return { kind: "success", channels: reachableChannels };
}
