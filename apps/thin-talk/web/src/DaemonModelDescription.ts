import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";

const DAEMON_PROVIDER_ID = "astronomical";
const DAEMON_API_ID = "astronomical-daemon";

/** The daemon model as Pi describes one. The agent loop reads only the
 * identity fields; limits stay zero until turns are sized from the daemon's
 * model catalog. */
export function daemonModel(modelID: string): Model<Api> {
  return {
    id: modelID,
    name: modelID,
    api: DAEMON_API_ID,
    provider: DAEMON_PROVIDER_ID,
    baseUrl: "",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 0,
    maxTokens: 0,
  };
}

/** An assistant message from this model with no content yet. The daemon's
 * usage figures are not tracked, so usage stays zero. */
export function emptyAssistantMessage(model: Model<Api>): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };
}
