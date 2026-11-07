import { AppConfig } from "./AppConfig";
import { ChatCompletionChunk } from "./ChatCompletionChunk";
import type { ChatCompletionRequestBody } from "./ChatCompletionRequestBody";
import { ChatModel } from "./ChatModel";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureClassifier } from "./ChatFailureClassifier";
import { ChatFailureKind } from "./ChatFailureKind";
import { ModelCatalogEntry } from "./ModelCatalogEntry";
import { DaemonStatus } from "./DaemonStatus";

/** How long a readiness/model request may take before the page gives up. */
const GET_TIMEOUT_MILLISECONDS = 2_000;
/** The largest status/models body the page will read, matching the Swift client. */
const MAXIMUM_RESPONSE_BYTE_COUNT = 1_048_576;

/**
 * Talks to a single daemon instance over its public REST endpoints: the
 * readiness handshake, the model list, and streaming chat completion. The
 * network calls and the server-sent-events line framing live here, so the
 * controller never touches the wire; decoding one line is ChatCompletionChunk's.
 */
export class DaemonClient {
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

  /** Sends one streaming chat completion and hands every decoded chunk to
   * `onChunk` in arrival order. A user stop (the abort signal) ends the stream
   * normally, so the caller decides what a stopped turn means; every other
   * failure throws a classified ChatFailure.
   *
   * Only `content-type` is sent: the daemon's CORS rule for the canvas origin
   * allows no other request header, and WebKit refuses the whole request at
   * the preflight when one is added. */
  public async streamChatCompletion(
    requestBody: ChatCompletionRequestBody,
    signal: AbortSignal,
    onChunk: (chunk: ChatCompletionChunk) => void,
  ): Promise<void> {
    let response: Response;
    try {
      response = await fetch(`${this.config.daemonBaseURL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal,
      });
    } catch (error) {
      if (signal.aborted) {
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
    await DaemonClient.consumeServerSentEvents(response, signal, onChunk);
  }

  private static async consumeServerSentEvents(
    response: Response,
    signal: AbortSignal,
    onChunk: (chunk: ChatCompletionChunk) => void,
  ): Promise<void> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the stream carried no body");
    }
    const decoder = new TextDecoder();
    let unterminatedLine = "";
    let isStreamFinished = false;
    try {
      for (;;) {
        const read = await reader.read();
        if (read.done) {
          isStreamFinished = true;
          break;
        }
        unterminatedLine += decoder.decode(read.value, { stream: true });
        const completeLines = unterminatedLine.split("\n");
        const trailingPartialLine: string | undefined = completeLines.pop();
        unterminatedLine = trailingPartialLine != null ? trailingPartialLine : "";
        DaemonClient.deliverLines(completeLines, onChunk);
      }
      if (unterminatedLine.length > 0) {
        DaemonClient.deliverLines([unterminatedLine], onChunk);
      }
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      throw error;
    } finally {
      // A chunk handler that throws (an error chunk, a failing listener)
      // leaves the body unread; cancelling it closes the connection so the
      // daemon stops generating for a reader that has gone.
      if (!isStreamFinished) {
        await reader.cancel().catch(() => undefined);
      }
    }
  }

  private static deliverLines(serverSentEventLines: string[], onChunk: (chunk: ChatCompletionChunk) => void): void {
    for (const serverSentEventLine of serverSentEventLines) {
      const chunk = ChatCompletionChunk.fromServerSentEventLine(serverSentEventLine);
      if (chunk != null) {
        onChunk(chunk);
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
