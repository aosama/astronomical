import type { SessionBridgeRequest } from "../../src/SessionBridgeRequest";

/** The name the fake host gives every stored generated image. */
export const FAKE_STORED_IMAGE_NAME = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png";

/**
 * An in-memory stand-in for the Swift session host. It answers `sessionCall`
 * envelopes exactly the way SessionFileStore does — same plain shapes, same
 * newest-first ordering, same null-for-missing semantics — so a test boot
 * exercises the real bridge, store, and controller against a faithful double.
 */
export class FakeSessionHost {
  private readonly files = new Map<string, { title: string; updatedAt: string; messages: unknown[]; skill?: string }>();
  private preferences: unknown = null;
  /** Astro's saved profile, or null before the first save, like astro-profile.json. */
  public astroProfile: unknown = null;
  private nextIdentifier = 1;
  public readonly receivedRequests: SessionBridgeRequest[] = [];
  /** Every `action` envelope the page posted (copy, openExternal), in order. */
  public readonly receivedActions: Record<string, unknown>[] = [];
  /** Generated images the page asked the host to delete, by stored name. */
  public readonly deletedGeneratedImageNames: string[] = [];
  /** How the fake approval sheet answers. */
  public toolApprovalAnswer = true;
  /** What the fake open panels return; null is a cancelled panel. */
  /** What the right-click menu answers, as the native menu would after the
   * user picks (a "delete" counts as already confirmed). */
  /** Images the host can read back by name, as the app's image store would. */
  public readonly storedImagesByName = new Map<string, { base64: string; mimeType: string }>();
  public conversationMenuChoice: "rename" | "delete" | "dismissed" = "dismissed";
  public chosenTextFile: { fileName: string; text: string } | null = null;
  public chosenImageFile: { fileName: string; base64: string; mimeType: string } | null = null;

  public constructor(
    private readonly resolveCall: (callId: string, payload: unknown) => void,
    private readonly rejectCall: (callId: string, message: string) => void,
  ) {}

  /** Installs the fake WKWebView message handler onto the live window. */
  public install(): void {
    window.webkit = {
      messageHandlers: {
        thintalk: {
          postMessage: (payload: unknown) => this.receive(payload),
        },
      },
    };
  }

  private receive(payload: unknown): void {
    if (typeof payload !== "object" || payload === null) {
      return;
    }
    const record = payload as Record<string, unknown>;
    if (record["kind"] === "action") {
      this.receivedActions.push(record);
      return;
    }
    if (record["kind"] !== "sessionCall") {
      return;
    }
    const request = record as unknown as SessionBridgeRequest;
    this.receivedRequests.push(request);
    try {
      const reply = this.perform(request.op, request.payload);
      queueMicrotask(() => this.resolveCall(request.callId, reply));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      queueMicrotask(() => this.rejectCall(request.callId, message));
    }
  }

  private perform(op: string, payload: unknown): unknown {
    switch (op) {
      case "list":
        return this.list();
      case "load":
        return this.load(this.stringField(payload, "id"));
      case "save":
        this.save(payload);
        return null;
      case "delete":
        this.files.delete(this.stringField(payload, "id"));
        return null;
      case "rename":
        this.rename(this.stringField(payload, "id"), this.stringField(payload, "title"));
        return null;
      case "loadPrefs":
        return this.preferences;
      case "savePrefs":
        this.preferences = payload;
        return null;
      case "loadAstroProfile":
        return this.astroProfile;
      case "saveAstroProfile":
        this.astroProfile = payload;
        return null;
      case "storeGeneratedImage": {
        // Kept so a later turn can read the image back, as the app's store does.
        const storedBase64 = typeof payload === "object" && payload !== null ? (payload as Record<string, unknown>)["base64"] : null;
        if (typeof storedBase64 === "string") {
          this.storedImagesByName.set(FAKE_STORED_IMAGE_NAME, { base64: storedBase64, mimeType: "image/png" });
        }
        return { imageName: FAKE_STORED_IMAGE_NAME };
      }
      case "loadGeneratedImage":
        return this.storedImagesByName.get(this.stringField(payload, "imageName")) ?? null;
      case "useHostTool":
        return this.useHostTool(payload);
      case "cancelHostToolCall":
        return null;
      case "showConversationMenu":
        return { choice: this.conversationMenuChoice };
      case "deleteGeneratedImages": {
        const imageNames = typeof payload === "object" && payload !== null ? (payload as Record<string, unknown>)["imageNames"] : null;
        this.deletedGeneratedImageNames.push(...(Array.isArray(imageNames) ? imageNames.filter((name): name is string => typeof name === "string") : []));
        return null;
      }
      default:
        throw new Error(`fake host does not know op "${op}"`);
    }
  }

  /** Answers like ThinTalkToolHost: the sheet's answer, then the panel's. */
  private useHostTool(payload: unknown): unknown {
    if (!this.toolApprovalAnswer) {
      return { outcome: "refused" };
    }
    const toolName = typeof payload === "object" && payload !== null ? (payload as Record<string, unknown>)["toolName"] : null;
    if (toolName === "search_conversations") {
      return { outcome: "approved" };
    }
    const chosenFile = toolName === "describe_image" ? this.chosenImageFile : this.chosenTextFile;
    return chosenFile != null ? { outcome: "chosen", ...chosenFile } : { outcome: "nothingChosen" };
  }

  private list(): unknown[] {
    const summaries = [...this.files.entries()].map(([id, file]) => ({
      id,
      title: file.title,
      updatedAt: file.updatedAt,
      ...(file.skill !== undefined ? { skill: file.skill } : {}),
    }));
    summaries.sort((first, second) => second.updatedAt.localeCompare(first.updatedAt));
    return summaries;
  }

  private load(id: string): unknown {
    const file = this.files.get(id);
    if (!file) {
      return null;
    }
    return { id, title: file.title, updatedAt: file.updatedAt, messages: file.messages };
  }

  private save(payload: unknown): void {
    if (typeof payload !== "object" || payload === null) {
      throw new Error("save payload must be an object");
    }
    const record = payload as Record<string, unknown>;
    const id = this.stringField(payload, "id");
    this.files.set(id, {
      title: typeof record["title"] === "string" ? record["title"] : "",
      updatedAt: typeof record["updatedAt"] === "string" ? record["updatedAt"] : "",
      messages: Array.isArray(record["messages"]) ? record["messages"] : [],
      ...(typeof record["skill"] === "string" ? { skill: record["skill"] } : {}),
    });
  }

  private rename(id: string, title: string): void {
    const file = this.files.get(id);
    if (file) {
      file.title = title;
    }
  }

  private stringField(payload: unknown, key: string): string {
    if (typeof payload !== "object" || payload === null) {
      throw new Error(`payload for "${key}" must be an object`);
    }
    const value = (payload as Record<string, unknown>)[key];
    if (typeof value !== "string") {
      throw new Error(`payload field "${key}" must be a string`);
    }
    return value;
  }

  /** Seeds a session file directly, as if a previous run had written it. */
  public seedSession(id: string, title: string, updatedAt: string, messages: unknown[]): void {
    this.files.set(id, { title, updatedAt, messages });
  }

  public seedPreferences(preferences: unknown): void {
    this.preferences = preferences;
  }

  public fileCount(): number {
    return this.files.size;
  }

  public nextSessionIdentifier(): string {
    return `seeded-${this.nextIdentifier++}`;
  }
}
