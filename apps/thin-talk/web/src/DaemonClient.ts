import { AppConfig } from "./AppConfig";
import { ChatCompletionChunk } from "./ChatCompletionChunk";
import type { ChatCompletionRequestBody } from "./ChatCompletionRequestBody";
import type { ImageGenerationRequestBody } from "./ImageGenerationRequestBody";
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
/** The largest image response the page will read: the host keeps images of
 * up to 20 MB, which is about 27 MB of base64, plus a margin for the JSON
 * around it. */
const MAXIMUM_IMAGE_RESPONSE_CHARACTER_COUNT = Math.ceil((20 * 1024 * 1024 * 4) / 3) + 64 * 1024;

/** One question about one image for a vision model. */
export interface ImageDescriptionRequest {
  readonly modelID: string;
  readonly question: string;
  readonly mimeType: string;
  readonly base64: string;
}

/** A description needs little deliberation, and every thinking token delays
 * the answer the conversation is waiting on. */
const IMAGE_DESCRIPTION_THINKING_BUDGET_TOKENS = 256;
/** A description is a tool result, capped at 16 KB of text anyway; this keeps
 * a model that will not stop from running on behind the tool's timeout. */
const IMAGE_DESCRIPTION_MAXIMUM_OUTPUT_TOKENS = 1_024;

/**
 * Talks to a single daemon instance over its public REST endpoints: the
 * readiness handshake, the model list, streaming chat completion and image
 * generation. The network calls and the server-sent-events line framing live
 * here, so the controller never touches the wire; decoding one line is
 * ChatCompletionChunk's.
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
    const catalogEntries = await this.fetchCatalogEntries();
    return catalogEntries
      .filter((entry) => entry.supportsChat)
      .map(
        (entry) =>
          new ChatModel({
            id: entry.id,
            name: entry.id,
            inputModalities: entry.inputModalities,
            supportsToolCalls: entry.supportsToolCalls,
          }),
      );
  }

  /** The ids of the installed models that generate images, in catalog order. */
  public async fetchImageModelIDs(signal?: AbortSignal): Promise<string[]> {
    const catalogEntries = await this.fetchCatalogEntries(signal);
    return catalogEntries.filter((entry) => entry.generatesImages).map((entry) => entry.id);
  }

  /** The ids of the installed chat models that can look at images. */
  public async fetchVisionModelIDs(signal?: AbortSignal): Promise<string[]> {
    const catalogEntries = await this.fetchCatalogEntries(signal);
    return catalogEntries.filter((entry) => entry.acceptsImages).map((entry) => entry.id);
  }

  /** Asks a vision model one question about one image and returns its answer.
   * A single, non-streamed request: the answer becomes a tool result, so
   * nothing is shown while it is written. */
  public async describeImage(request: ImageDescriptionRequest, signal: AbortSignal): Promise<string> {
    const requestBody = {
      model: request.modelID,
      stream: false,
      thinking_budget: IMAGE_DESCRIPTION_THINKING_BUDGET_TOKENS,
      max_tokens: IMAGE_DESCRIPTION_MAXIMUM_OUTPUT_TOKENS,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: request.question },
          { type: "image_url", image_url: { url: `data:${request.mimeType};base64,${request.base64}` } },
        ],
      }],
    };
    const bodyText = await this.postJSON("/v1/chat/completions", requestBody, signal, MAXIMUM_RESPONSE_BYTE_COUNT);
    return DaemonClient.firstAnswerText(bodyText);
  }

  /** Posts one JSON request and returns the response text, refusing a body
   * past `maximumCharacterCount`. Network failures and refusals become
   * classified ChatFailures, as for every other daemon call. */
  private async postJSON(path: string, requestBody: unknown, signal: AbortSignal, maximumCharacterCount: number): Promise<string> {
    let response: Response;
    try {
      response = await fetch(`${this.config.daemonBaseURL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal,
      });
    } catch (error) {
      throw new ChatFailure(ChatFailureKind.WORKER_UNAVAILABLE, DaemonClient.describeNetworkError(error));
    }
    const bodyText = await response.text();
    if (bodyText.length > maximumCharacterCount) {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the response was too large");
    }
    if (!response.ok) {
      const reason = ChatFailureClassifier.reasonFromErrorDocument(bodyText);
      throw new ChatFailure(ChatFailureClassifier.classify(response.status, reason != null ? reason : ""), reason);
    }
    return bodyText;
  }

  private static firstAnswerText(bodyText: string): string {
    let document: unknown;
    try {
      document = JSON.parse(bodyText);
    } catch {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the description was not JSON");
    }
    const choices = typeof document === "object" && document !== null ? (document as Record<string, unknown>)["choices"] : null;
    const firstChoice = Array.isArray(choices) ? choices[0] : null;
    const message = typeof firstChoice === "object" && firstChoice !== null ? (firstChoice as Record<string, unknown>)["message"] : null;
    const content = typeof message === "object" && message !== null ? (message as Record<string, unknown>)["content"] : null;
    if (typeof content !== "string" || content.trim().length === 0) {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the vision model returned no description");
    }
    return content.trim();
  }

  private async fetchCatalogEntries(signal?: AbortSignal): Promise<ModelCatalogEntry[]> {
    const bodyText = await this.requestText("/v1/models", signal);
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
      .filter((entry): entry is ModelCatalogEntry => entry !== null);
  }

  /** Generates one image and returns it as base64. Image generation loads an
   * image model and can take tens of seconds, so only the caller's signal
   * bounds it. Failures throw a classified ChatFailure. */
  public async generateImage(requestBody: ImageGenerationRequestBody, signal: AbortSignal): Promise<string> {
    // An image response is far larger than the status and model documents
    // requestText caps at a megabyte, so it has its own cap: the largest image
    // the host will keep, in base64, plus room for the response's other fields.
    const bodyText = await this.postJSON("/v1/images/generations", requestBody, signal, MAXIMUM_IMAGE_RESPONSE_CHARACTER_COUNT);
    return DaemonClient.firstImageBase64(bodyText);
  }

  private static firstImageBase64(bodyText: string): string {
    let document: unknown;
    try {
      document = JSON.parse(bodyText);
    } catch {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the image response was not JSON");
    }
    const record = typeof document === "object" && document !== null ? (document as Record<string, unknown>) : {};
    const generatedImages = Array.isArray(record["data"]) ? record["data"] : [];
    const firstImage = generatedImages[0];
    const encodedImage =
      typeof firstImage === "object" && firstImage !== null ? (firstImage as Record<string, unknown>)["b64_json"] : null;
    if (typeof encodedImage !== "string" || encodedImage.length === 0) {
      throw new ChatFailure(ChatFailureKind.MALFORMED_MODEL_OUTPUT, "the image response carried no image");
    }
    return encodedImage;
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

  private async requestText(path: string, callerSignal?: AbortSignal): Promise<string> {
    const timeoutSignal = AbortSignal.timeout(GET_TIMEOUT_MILLISECONDS);
    let response: Response;
    try {
      response = await fetch(`${this.config.daemonBaseURL}${path}`, {
        signal: callerSignal != null ? AbortSignal.any([callerSignal, timeoutSignal]) : timeoutSignal,
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
