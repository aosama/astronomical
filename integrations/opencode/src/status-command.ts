// /astronomical-status: one line per reachable channel, such as
// "Astronomical 0.2.118: 8 chat models", shown as a terminal toast.

import type { Context as TuiPluginContext } from "@opencode/plugin/tui/context";
import { recordAuditEvent, type AstronomicalAuditLogger } from "./audit-log.js";
import { resolveAstronomicalChannels } from "./channels.js";
import {
  fetchChannelChatModels,
  type ChannelChatModels,
} from "./channel-chat-models.js";
import { fetchApplicationVersion } from "./application-status.js";
import {
  ASTRONOMICAL_CHANNEL_PROVIDER_NAMES,
  ASTRONOMICAL_PROVIDER_NAME,
} from "./constants.js";

export interface StatusCommandInput {
  readonly environment: NodeJS.ProcessEnv;
  readonly fetchImpl: typeof fetch;
  readonly audit: AstronomicalAuditLogger;
}

// A channel whose status read fails still reports its model count, because
// the model list already proved the daemon is up; only the version is optional.
async function describeChannel(
  channel: ChannelChatModels,
  fetchImpl: typeof fetch,
): Promise<string> {
  const channelName = ASTRONOMICAL_CHANNEL_PROVIDER_NAMES[channel.instance];
  const version = await fetchApplicationVersion(channel.baseUrl, fetchImpl);
  const labeledName =
    version === undefined ? channelName : `${channelName} ${version}`;
  return `${labeledName}: ${channel.models.length} chat models`;
}

/** Shows every reachable channel's version and chat model count. */
export async function showAstronomicalStatus(
  context: TuiPluginContext,
  input: StatusCommandInput,
): Promise<void> {
  await recordAuditEvent(input.audit, "status-command");
  let latestDiagnostic: string | undefined;
  const channelChatModels = await fetchChannelChatModels({
    channels: resolveAstronomicalChannels(input.environment),
    fetchImpl: input.fetchImpl,
    onDiagnostic: (message: string): void => {
      latestDiagnostic = message;
    },
  });
  if (channelChatModels.kind === "failure") {
    await recordAuditEvent(input.audit, "diagnostic", {
      stage: "status-command",
      reason: latestDiagnostic ?? "No Astronomical channel is reachable",
    });
    context.ui.toast.show({
      variant: "warning",
      title: ASTRONOMICAL_PROVIDER_NAME,
      message:
        "No Astronomical channel is reachable. Start Astronomical to use its models.",
    });
    return;
  }
  // Every answering channel gets a line, even with zero chat models, in the
  // order the picker lists the providers (Stable before Development).
  const channelLines: string[] = [];
  for (const channel of channelChatModels.channels) {
    channelLines.push(await describeChannel(channel, input.fetchImpl));
  }
  context.ui.toast.show({
    variant: "success",
    title: ASTRONOMICAL_PROVIDER_NAME,
    message: channelLines.join("; "),
  });
}
