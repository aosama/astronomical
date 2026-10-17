import { AppConfig } from "./AppConfig";
import { ChatMessage } from "./ChatMessage";
import { ChatModel } from "./ChatModel";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureClassifier } from "./ChatFailureClassifier";
import { ChatFailureKind } from "./ChatFailureKind";
import { ModelCatalogEntry } from "./ModelCatalogEntry";
import { DaemonStatus } from "./DaemonStatus";
import { ThinkingEffort } from "./ThinkingEffort";

/** How long a readiness/model request may take before the page gives up. */
const GET_TIMEOUT_MILLISECONDS = 2_000;
/** The largest status/models body the page will read, matching the Swift client. */
const MAXIMUM_RESPONSE_BYTE_COUNT = 1_048_576;

/** One streaming chat completion request body. `thinking_budget` uses the
 * daemon's canonical numeric spelling: the REST contract rejects
 * unrecognised fields outright, so this key name is part of the wire agreement. */
interface ChatCompletionRequestBody {
  model: string;
  stream: boolean;
  messages: { role: string; content: string }[];
  thinking_budget: number;
}

/**
 * Talks to a single daemon instance over its public REST endpoints: the
 * readiness handshake, the model list, and streaming chat completion. All
 * network and decoding logic is isolated here so the controller never touches
 * the wire.
 */
export class DaemonClient {
  public readonly stallTimeoutMilliseconds = 60_000;
  private readonly config: AppConfig;

  public constructor(config: AppConfig) {
    this.config = config;
  }

  /** Verifies this window is talking to the matching daemon instance,
   * returning its status text. Throws when the endpoint answers with a
   * different runtime channel or state directory. */
  public async handshake(): Promise<string> {
    const bodyText = await this.requestText("/v1/status");
    let status: DaemonStatus;
    try {
      status = DaemonStatus.fromPlain(JSON.parse(bodyText));
    } catch {
      throw new Error("The server returned an unexpected response.");
    }
    const matchesChannel = status.channel === this.config.expectedChannel;
    const matchesStateDirectory = status.stateDirectory === this.config.expectedStateDirectory;
    if (!matchesChannel || !matchesStateDirectory) {
      throw new Error("This app must connect to its own runtime channel and state directory.");
    }
    const reportedStatus = status.status;
    return reportedStatus != null ? reportedStatus : "unknown";
  }

  /** Returns the chat-capable models discovered on this channel. */
  public async fetchModels(): Promise<ChatModel[]> {
    const bodyText = await this.requestText("/v1/models");
    let document: unknown;
    try {
      document = JSON.parse(bodyText);
    } catch {
      throw new Error("The server returned an unexpected response.");
    }
    const record = typeof document === "object" && document !== null ? (document as Record<string, unknown>) : {};
    const rawModels = Array.isArray(record["data"]) ? record["data"] : [];
    return rawModels
      .map((rawModel) => ModelCatalogEntry.fromPlain(rawModel))
      .filter((entry): entry is ModelCatalogEntry => entry !== null)
      .filter((entry) => entry.supportsChat)
      .map((entry) => new ChatModel({ id: entry.id, name: entry.id, inputModalities: entry.inputModalities }));
  }

  /** Consumes one streaming chat completion, invoking the handlers per delta.
   * A user stop (abort signal) ends the stream normally; every other failure
   * throws a classified ChatFailure. */
  public async streamChat(init: {
    modelID: string;
    messages: ChatMessage[];
    thinkingEffort: ThinkingEffort;
    onText: (delta: string) => void;
    onReasoning: (delta: string) => void;
    signal: AbortSignal;
  }): Promise<void> {
    const requestBody: ChatCompletionRequestBody = {
      model: init.modelID,
      stream: true,
      messages: init.messages.map((message) => ({ role: message.role, content: message.content })),
      thinking_budget: ThinkingEffort.thinkingBudgetTokens(init.thinkingEffort),
    };
    let response: Response;
    try {
      response = await fetch(`${this.config.daemonBaseURL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: init.signal,
      });
    } catch (error) {
      if (init.signal.aborted) {
        return;
      }
      // Pre-stream failure (e.g. a model that will not load): the daemon's
      // refusal never arrived as a response, so classify as unavailable.
      throw new ChatFailure(ChatFailureKind.WORKER_UNAVAILABLE, DaemonClient.describeNetworkError(error));
    }
    if (!response.ok) {
      const bodyText = await response.text().catch(() => "");
      const reason = ChatFailureClassifier.reasonFromErrorDocument(bodyText);
      const searchReason: string = reason != null ? reason : "";
      throw new ChatFailure(ChatFailureClassifier.classify(response.status, searchReason), reason);
    }
    await this.consumeStream(response, init);
  }

  private async consumeStream(
    response: Response,
    init: { onText: (delta: string) => void; onReasoning: (delta: string) => void; signal: AbortSignal },
  ): Promise<void> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the stream carried no body");
    }
    const decoder = new TextDecoder();
    let buffered = "";
    try {
      for (;;) {
        const read = await reader.read();
        if (read.done) {
          break;
        }
        buffered += decoder.decode(read.value, { stream: true });
        const lines = buffered.split("\n");
        const remainder: string | undefined = lines.pop();
        buffered = remainder != null ? remainder : "";
        for (const line of lines) {
          this.consumeLine(line, init);
        }
      }
      if (buffered.length > 0) {
        this.consumeLine(buffered, init);
      }
    } catch (error) {
      if (init.signal.aborted) {
        return;
      }
      throw error;
    }
  }

  private consumeLine(
    line: string,
    handlers: { onText: (delta: string) => void; onReasoning: (delta: string) => void },
  ): void {
    const trimmedLine = line.trim();
    if (!trimmedLine.startsWith("data:")) {
      return;
    }
    const payload = trimmedLine.slice("data:".length).trim();
    if (payload.length === 0 || payload === "[DONE]") {
      return;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return;
    }
    if (typeof chunk !== "object" || chunk === null) {
      return;
    }
    const record = chunk as Record<string, unknown>;
    const errorField = record["error"];
    if (typeof errorField === "object" && errorField !== null) {
      const message = (errorField as Record<string, unknown>)["message"];
      const reason = typeof message === "string" ? message : null;
      const classifiedReason: string = reason != null ? reason : "";
      throw new ChatFailure(ChatFailureClassifier.classifyReason(classifiedReason), reason);
    }
    const choices = Array.isArray(record["choices"]) ? record["choices"] : [];
    for (const rawChoice of choices) {
      if (typeof rawChoice !== "object" || rawChoice === null) {
        continue;
      }
      const delta = (rawChoice as Record<string, unknown>)["delta"];
      if (typeof delta !== "object" || delta === null) {
        continue;
      }
      const deltaRecord = delta as Record<string, unknown>;
      const content = deltaRecord["content"];
      if (typeof content === "string" && content.length > 0) {
        handlers.onText(content);
      }
      const reasoning = deltaRecord["reasoning_content"];
      if (typeof reasoning === "string" && reasoning.length > 0) {
        handlers.onReasoning(reasoning);
      }
    }
  }

  private async requestText(path: string): Promise<string> {
    let response: Response;
    try {
      response = await fetch(`${this.config.daemonBaseURL}${path}`, {
        signal: AbortSignal.timeout(GET_TIMEOUT_MILLISECONDS),
      });
    } catch (error) {
      throw new ChatFailure(ChatFailureKind.WORKER_UNAVAILABLE, DaemonClient.describeNetworkError(error));
    }
    const bodyText = await response.text();
    if (bodyText.length > MAXIMUM_RESPONSE_BYTE_COUNT) {
      throw new Error("The server response was too large.");
    }
    if (!response.ok) {
      const reason = ChatFailureClassifier.reasonFromErrorDocument(bodyText);
      const searchReason: string = reason != null ? reason : "";
      throw new ChatFailure(ChatFailureClassifier.classify(response.status, searchReason), reason);
    }
    return bodyText;
  }

  private static describeNetworkError(error: unknown): string | null {
    return error instanceof Error ? error.message : null;
  }
}
