/**
 * A scripted stand-in for the daemon's HTTP surface. It answers the status
 * handshake, the model catalog, and a chat-completions request whose reply is a
 * real SSE byte stream, so DaemonClient's streaming parser runs against
 * genuine Response/ReadableStream objects.
 */
export interface RecordedDaemonRequest {
  readonly url: string;
  readonly method: string;
  readonly body: string | null;
}

export interface FakeDaemonOptions {
  readonly daemonBaseURL: string;
  readonly channel: string;
  readonly stateDirectory: string;
  readonly models?: unknown[];
  /** SSE lines streamed for chat completions; defaults to a short reply. */
  readonly chatStreamLines?: string[];
  /** Makes the status handshake fail, to exercise the failed load path. */
  readonly failHandshake?: boolean;
}

const DEFAULT_CHAT_LINES = [
  'data: {"choices":[{"delta":{"content":"Hello from the fake daemon."}}]}',
  'data: {"choices":[{"delta":{"content":" Second sentence."}}]}',
  "data: [DONE]",
];

export class FakeDaemon {
  public readonly requests: RecordedDaemonRequest[] = [];
  private readonly options: FakeDaemonOptions;

  public constructor(options: FakeDaemonOptions) {
    this.options = options;
  }

  /** Replaces the global fetch the DaemonClient resolves at call time. */
  public install(): void {
    const handler = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> =>
      this.handle(input, init);
    globalThis.fetch = handler as typeof fetch;
  }

  private async handle(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = String(input);
    this.requests.push({ url, method: init?.method ?? "GET", body: typeof init?.body === "string" ? init.body : null });
    if (url.endsWith("/v1/status")) {
      if (this.options.failHandshake) {
        return new Response("daemon is pretending to be down", { status: 503 });
      }
      return Response.json({
        status: "ready",
        application: {
          channel: this.options.channel,
          state_directory: this.options.stateDirectory,
        },
      });
    }
    if (url.endsWith("/v1/models")) {
      return Response.json({ data: this.options.models ?? [defaultFakeModel()] });
    }
    if (url.endsWith("/v1/chat/completions")) {
      return sseResponse(this.options.chatStreamLines ?? DEFAULT_CHAT_LINES);
    }
    return new Response("no such route on the fake daemon", { status: 404 });
  }

  public chatRequestBody(): Record<string, unknown> | null {
    const chatRequest = this.requests.find((request) => request.url.endsWith("/v1/chat/completions"));
    if (!chatRequest || !chatRequest.body) {
      return null;
    }
    return JSON.parse(chatRequest.body) as Record<string, unknown>;
  }
}

export function defaultFakeModel(): unknown {
  return { id: "fake-chat-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] };
}

function sseResponse(lines: string[]): Response {
  const encoder = new TextEncoder();
  const body = lines.map((line) => `${line}\n\n`).join("");
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(body));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
}
