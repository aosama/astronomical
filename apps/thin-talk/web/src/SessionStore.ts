import { SessionBridge } from "./SessionBridge";
import { SessionDocument } from "./SessionDocument";
import { SessionSummary } from "./SessionSummary";
import { ChatMessage } from "./ChatMessage";

const AUTOSAVE_DEBOUNCE_MILLISECONDS = 500;

/**
 * Owns the session lifecycle on top of the file bridge: the sidebar's session
 * list, the active document, and autosaving.
 *
 * Saves are debounced while a stream is growing, and forced immediately at every
 * terminal point (turn end, failure, stop, session switch), so a crash loses at
 * most the debounce window and a finished turn is never lost.
 */
export class SessionStore {
  private readonly bridge: SessionBridge;
  private sessions: SessionSummary[] = [];
  private activeDocument: SessionDocument | null = null;
  private saveTimer: number | null = null;
  private saveInFlight: Promise<void> = Promise.resolve();

  public constructor(bridge: SessionBridge) {
    this.bridge = bridge;
  }

  public listSessions(): readonly SessionSummary[] {
    return this.sessions;
  }

  public activeSessionID(): string | null {
    return this.activeDocument?.id ?? null;
  }

  public activeTitle(): string {
    return this.activeDocument?.title ?? "";
  }

  public async refreshList(): Promise<void> {
    this.sessions = await this.bridge.list();
  }

  /** Loads one session's document, or null when the file is missing or unreadable. */
  public async open(id: string): Promise<SessionDocument | null> {
    await this.flushPendingSave();
    try {
      this.activeDocument = await this.bridge.load(id);
    } catch {
      this.activeDocument = null;
    }
    return this.activeDocument;
  }

  /** Starts a fresh, unsaved conversation. The document is written only when
   * the first turn gives it content, so abandoned new chats leave no files. */
  public startNew(): SessionDocument {
    const now = new Date().toISOString();
    this.activeDocument = new SessionDocument({
      id: SessionStore.newIdentifier(),
      title: "",
      createdAt: now,
      updatedAt: now,
      messages: [],
    });
    return this.activeDocument;
  }

  /** Schedules a debounced save of the active document. */
  public scheduleSave(): void {
    if (!this.activeDocument) {
      return;
    }
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer);
    }
    this.saveTimer = window.setTimeout(() => {
      this.saveTimer = null;
      void this.saveNow();
    }, AUTOSAVE_DEBOUNCE_MILLISECONDS);
  }

  /** Copies the live conversation into the active document ahead of a save. The
   * copy matters: the save path rewrites message states for persistence, and it
   * must never mutate the log the streaming view is reading. */
  public stageMessages(messages: readonly ChatMessage[]): void {
    if (!this.activeDocument) {
      return;
    }
    this.activeDocument.messages = messages.map(
      (message) =>
        new ChatMessage({
          id: message.id,
          role: message.role,
          content: message.content,
          reasoning: message.reasoning,
          state: message.state,
        }),
    );
  }

  /** Saves the active document immediately: stamps the update time, derives the
   * title from the first user message when the session is still untitled, and
   * refreshes the sidebar list. */
  public async saveNow(): Promise<void> {
    const document = this.activeDocument;
    if (!document) {
      return;
    }
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    document.updatedAt = new Date().toISOString();
    document.messages = SessionDocument.persistableMessages(document.messages);
    if (document.title.length === 0) {
      document.title = SessionDocument.titleFromFirstUserMessage(document.messages);
    }
    const write = this.bridge.save(document).then(() => this.refreshList());
    this.saveInFlight = write.catch(() => undefined);
    await write;
  }

  public async flushPendingSave(): Promise<void> {
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer);
      this.saveTimer = null;
      await this.saveNow();
      return;
    }
    await this.saveInFlight;
  }

  public async delete(id: string): Promise<void> {
    await this.bridge.delete(id);
    if (this.activeDocument?.id === id) {
      this.activeDocument = null;
    }
    await this.refreshList();
  }

  private static newIdentifier(): string {
    return crypto.randomUUID();
  }
}
