/**
 * A global fetch that answers chat completions from a script, for the turn
 * engine's own tests (the page journeys use FakeDaemon instead). Each request
 * takes the next scripted reply in order; a reply can stay open after its
 * lines until the request is aborted, which is how a user stop mid-answer is
 * exercised. Request bodies, header names and whether each request was
 * cancelled are recorded for assertions.
 */
export interface ScriptedChatReply {
  readonly status?: number;
  readonly lines?: string[];
  readonly errorBody?: string;
  /** Keep the stream open after the lines until the request is aborted. */
  readonly holdOpenUntilAborted?: boolean;
  /** Fail the fetch itself, as when no daemon is listening. */
  readonly networkFailure?: boolean;
  /** Deliver the stream in pieces of this many bytes, so one line can arrive
   * split across reads. */
  readonly bytesPerRead?: number;
  /** Leave the final line without its terminating newline. */
  readonly omitFinalNewline?: boolean;
}

export interface RecordedChatRequest {
  readonly body: Record<string, unknown>;
  readonly headerNames: string[];
  /** Becomes true when the client aborts the request or cancels its body. */
  wasCancelled: boolean;
}

export const DAEMON_BASE_URL = "http://127.0.0.1:6733";

export class ScriptedChatFetch {
  public readonly requests: RecordedChatRequest[] = [];
  private readonly replies: ScriptedChatReply[];

  public constructor(replies: ScriptedChatReply[]) {
    this.replies = [...replies];
  }

  public install(): void {
    globalThis.fetch = this.handle.bind(this) as typeof fetch;
  }

  private async handle(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const bodyText = typeof init?.body === "string" ? init.body : "{}";
    const recordedRequest: RecordedChatRequest = {
      body: JSON.parse(bodyText) as Record<string, unknown>,
      headerNames: [...new Headers(init?.headers).keys()],
      wasCancelled: false,
    };
    this.requests.push(recordedRequest);
    init?.signal?.addEventListener("abort", () => { recordedRequest.wasCancelled = true; });
    const reply = this.replies.shift();
    if (reply == null) {
      throw new Error(`no scripted reply left for ${String(input)}`);
    }
    if (reply.networkFailure) {
      throw new TypeError("Load failed");
    }
    if (reply.status != null && reply.status !== 200) {
      return new Response(reply.errorBody != null ? reply.errorBody : "", { status: reply.status });
    }
    return new Response(ScriptedChatFetch.eventStream(reply, init?.signal, recordedRequest), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  }

  private static eventStream(
    reply: ScriptedChatReply,
    signal: AbortSignal | null | undefined,
    recordedRequest: RecordedChatRequest,
  ): ReadableStream<Uint8Array> {
    const streamText = (reply.lines != null ? reply.lines : []).map((line) => `${line}\n\n`).join("");
    const encodedStream = new TextEncoder().encode(reply.omitFinalNewline ? streamText.replace(/\n+$/, "") : streamText);
    const bytesPerRead = reply.bytesPerRead != null ? reply.bytesPerRead : encodedStream.byteLength;
    return new ReadableStream<Uint8Array>({
      start(streamController) {
        for (let readOffset = 0; readOffset < encodedStream.byteLength; readOffset += bytesPerRead) {
          streamController.enqueue(encodedStream.slice(readOffset, readOffset + bytesPerRead));
        }
        if (!reply.holdOpenUntilAborted) {
          streamController.close();
          return;
        }
        signal?.addEventListener("abort", () => streamController.error(new DOMException("Aborted", "AbortError")));
      },
      cancel() {
        recordedRequest.wasCancelled = true;
      },
    });
  }
}

/** One streamed chunk line in the daemon's wire shape. */
export function chunkLine(delta: { content?: string; reasoning_content?: string }, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finishReason }] })}`;
}

/** Waits for a condition the event loop will make true, without a fixed sleep. */
export async function waitUntil(condition: () => boolean, description: string): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`timed out waiting for ${description}`);
}

/** One streamed tool call in the daemon's wire shape: the daemon sends the
 * whole call in a single delta. */
export function toolCallLine(toolCallID: string, toolName: string, argumentsJSON: string, toolCallIndex = 0): string {
  return `data: ${JSON.stringify({
    choices: [{
      index: 0,
      delta: { tool_calls: [{ index: toolCallIndex, id: toolCallID, type: "function", function: { name: toolName, arguments: argumentsJSON } }] },
      finish_reason: null,
    }],
  })}`;
}
