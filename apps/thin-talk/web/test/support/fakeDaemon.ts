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
  /** SSE lines for successive chat completions, one entry per request, for a
   * turn that calls a tool and then answers. The last entry repeats. */
  readonly chatReplySequence?: string[][];
  /** Makes the status handshake fail, to exercise the failed load path. */
  readonly failHandshake?: boolean;
  /** The Library catalog document; without one the endpoint is not found,
   * as on a daemon whose Library cannot be read. */
  readonly libraryCatalog?: unknown;
  /** Successive answers to `GET /v1/library/download` once a download was
   * requested; the last repeats. Before any request the Library is idle. */
  readonly libraryDownloadSequence?: unknown[];
  /** Refuses `POST /v1/library/download` with this status and error code. */
  readonly libraryDownloadRefusal?: { status: number; code: string };
  /** What lands when a requested download finishes: the model joins
   * /v1/models and its catalog entry turns ready, as on the real daemon. */
  readonly installsWhenDownloadFinishes?: { huggingfaceID: string; model: unknown };
  /** Refuses `POST /v1/audio/speech` with this status and error code, as a
   * daemon with no voice model installed does. */
  readonly speechRefusal?: { status: number; code: string };
}

const DEFAULT_CHAT_LINES = [
  'data: {"choices":[{"delta":{"content":"Hello from the fake daemon."}}]}',
  'data: {"choices":[{"delta":{"content":" Second sentence."}}]}',
  // The daemon closes every streamed choice with a finish reason before
  // [DONE]; Thin Talk's AssistantEventAssembler treats a stream without one as
  // cut off.
  'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
  "data: [DONE]",
];

export class FakeDaemon {
  /** What the status reports as loading, as the engine does while it loads a
   * model: `{ model_id }`, or null when nothing is loading. */
  public modelLoadInFlight: { model_id: string | null } | null = null;

  public readonly requests: RecordedDaemonRequest[] = [];
  private readonly options: FakeDaemonOptions;
  /** The models /v1/models lists; a finished download adds to it. */
  private readonly installedModels: unknown[];
  /** Hugging Face ids whose downloads were requested, in order. */
  public readonly requestedLibraryDownloads: string[] = [];
  /** While true, model reads fail as a network error would when the engine
   * is not running, so a page can be tested against a silent engine. */
  public modelsUnreachable = false;
  private libraryDownloadPollCount = 0;
  private readonly readyLibraryModelIDs = new Set<string>();

  /** A chat model arrives without this window asking for it, as when the
   * person downloads one on the Models page in their browser. */
  public installChatModelFromElsewhere(): void {
    this.installedModels.push(defaultFakeModel());
  }

  public constructor(options: FakeDaemonOptions) {
    this.options = options;
    const configuredModels: unknown[] | undefined = options.models;
    this.installedModels = configuredModels != null ? [...configuredModels] : [defaultFakeModel()];
  }

  /** Replaces the global fetch the DaemonClient resolves at call time. */
  public install(): void {
    const handler = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> =>
      this.handle(input, init);
    globalThis.fetch = handler as typeof fetch;
  }

  private async handle(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = String(input);
    const requestMethod: string = init?.method != null ? init.method : "GET";
    this.requests.push({ url, method: requestMethod, body: typeof init?.body === "string" ? init.body : null });
    if (url.endsWith("/v1/status")) {
      if (this.options.failHandshake) {
        return new Response("daemon is pretending to be down", { status: 503 });
      }
      return Response.json({
        status: "ready",
        model_load_in_flight: this.modelLoadInFlight,
        application: {
          channel: this.options.channel,
          state_directory: this.options.stateDirectory,
        },
      });
    }
    if (url.endsWith("/v1/models")) {
      if (this.modelsUnreachable) {
        throw new TypeError("fetch failed: the fake engine is not running");
      }
      return Response.json({ data: this.installedModels });
    }
    if (url.endsWith("/v1/library/catalog") && this.options.libraryCatalog !== undefined) {
      return Response.json(this.catalogWithReadyModels());
    }
    if (url.endsWith("/v1/library/download")) {
      return requestMethod === "POST" ? this.startLibraryDownload(init) : this.nextLibraryDownloadStatus();
    }
    if (url.endsWith("/v1/audio/speech")) {
      const speechRefusal = this.options.speechRefusal;
      if (speechRefusal !== undefined) {
        return Response.json({ error: { code: speechRefusal.code, message: "refused by the fake daemon" } }, { status: speechRefusal.status });
      }
      return new Response(new ArrayBuffer(8), { headers: { "content-type": "audio/wav" } });
    }
    if (url.endsWith("/v1/images/generations")) {
      return Response.json({ data: [{ b64_json: FAKE_GENERATED_IMAGE_BASE64, mime_type: "image/png" }] });
    }
    if (url.endsWith("/v1/chat/completions")) {
      const scriptedReply = this.nextScriptedReply();
      if (scriptedReply != null) {
        return sseResponse(scriptedReply);
      }
      const configuredChatStreamLines: string[] | undefined = this.options.chatStreamLines;
      const chatStreamLines: string[] =
        configuredChatStreamLines != null ? configuredChatStreamLines : DEFAULT_CHAT_LINES;
      return sseResponse(chatStreamLines);
    }
    return new Response("no such route on the fake daemon", { status: 404 });
  }

  private chatRepliesServed = 0;

  private startLibraryDownload(init?: RequestInit): Response {
    const requestBody = typeof init?.body === "string" ? (JSON.parse(init.body) as Record<string, unknown>) : {};
    const huggingfaceID = String(requestBody["huggingface_id"]);
    this.requestedLibraryDownloads.push(huggingfaceID);
    const refusal = this.options.libraryDownloadRefusal;
    if (refusal !== undefined) {
      return Response.json({ error: { code: refusal.code, message: "refused by the fake daemon" } }, { status: refusal.status });
    }
    return Response.json({ state: "checking_disk", huggingface_id: huggingfaceID, bytes_completed: 0, bytes_total: 0, error_code: null });
  }

  /** Idle until a download is requested; then the scripted sequence, and the
   * install lands when the sequence reaches its last answer. */
  private nextLibraryDownloadStatus(): Response {
    const downloadSequence = this.options.libraryDownloadSequence;
    if (this.requestedLibraryDownloads.length === 0 || downloadSequence === undefined || downloadSequence.length === 0) {
      return Response.json({ state: "idle", huggingface_id: null, bytes_completed: 0, bytes_total: 0, error_code: null });
    }
    const sequencePosition = Math.min(this.libraryDownloadPollCount, downloadSequence.length - 1);
    this.libraryDownloadPollCount += 1;
    if (sequencePosition === downloadSequence.length - 1) {
      this.finishInstall();
    }
    return Response.json(downloadSequence[sequencePosition]);
  }

  private finishInstall(): void {
    const install = this.options.installsWhenDownloadFinishes;
    if (install === undefined || this.readyLibraryModelIDs.has(install.huggingfaceID)) {
      return;
    }
    this.readyLibraryModelIDs.add(install.huggingfaceID);
    this.installedModels.push(install.model);
  }

  private catalogWithReadyModels(): unknown {
    const catalog = this.options.libraryCatalog as { entries?: Record<string, unknown>[] };
    if (!Array.isArray(catalog.entries)) {
      return catalog;
    }
    const entries = catalog.entries.map((entry) =>
      this.readyLibraryModelIDs.has(String(entry["huggingface_id"])) ? { ...entry, ready_on_this_mac: true } : entry,
    );
    return { ...catalog, entries };
  }

  private nextScriptedReply(): string[] | null {
    const replySequence: string[][] | undefined = this.options.chatReplySequence;
    if (replySequence == null || replySequence.length === 0) {
      return null;
    }
    const replyIndex = Math.min(this.chatRepliesServed, replySequence.length - 1);
    this.chatRepliesServed += 1;
    const scriptedReply = replySequence[replyIndex];
    return scriptedReply != null ? scriptedReply : null;
  }

  /** Every chat completion request body, in the order they were sent. */
  public chatRequestBodies(): Record<string, unknown>[] {
    return this.requests
      .filter((request) => request.url.endsWith("/v1/chat/completions") && request.body != null)
      .map((request) => JSON.parse(request.body as string) as Record<string, unknown>);
  }

  public chatRequestBody(): Record<string, unknown> | null {
    const chatRequest = this.requests.find((request) => request.url.endsWith("/v1/chat/completions"));
    if (!chatRequest || !chatRequest.body) {
      return null;
    }
    return JSON.parse(chatRequest.body) as Record<string, unknown>;
  }
}

/** The image every fake image request returns: the first bytes of a PNG. */
export const FAKE_GENERATED_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUg==";

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
