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
 * most the debounce window and a finished turn is never lost. Only staged
 * changes are written: a conversation that was merely opened is never saved.
 */
export class SessionStore {
  private readonly bridge: SessionBridge;
  private sessions: SessionSummary[] = [];
  private activeDocument: SessionDocument | null = null;
  private saveTimer: number | null = null;
  private saveInFlight: Promise<void> = Promise.resolve();
  /** True when messages were staged since the last successful write. Saving
   * stamps the update time, and the sidebar is ordered by it, so a save with
   * nothing new would move a conversation to the top just for being viewed. */
  private hasUnsavedChanges = false;

  public constructor(bridge: SessionBridge) {
    this.bridge = bridge;
  }

  public listSessions(): readonly SessionSummary[] {
    return this.sessions;
  }

  public activeSessionID(): string | null {
    const activeDocument = this.activeDocument;
    return activeDocument != null ? activeDocument.id : null;
  }

  public activeTitle(): string {
    const activeDocument = this.activeDocument;
    return activeDocument != null ? activeDocument.title : "";
  }

  public async refreshList(): Promise<void> {
    this.sessions = await this.bridge.list();
  }

  /** Opens a saved conversation. A missing or unreadable file returns null
   * and leaves the current conversation active: the controller keeps it on
   * screen in that case, so the store must keep saving it too. */
  public async open(id: string): Promise<SessionDocument | null> {
    // A failed write of the conversation being left must not block opening
    // another one; the failure was already reported where the save started.
    await this.flushPendingSave().catch(() => undefined);
    const openedDocument = await this.bridge.load(id).catch(() => null);
    if (openedDocument != null) {
      this.activeDocument = openedDocument;
      this.hasUnsavedChanges = false;
    }
    return openedDocument;
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
    this.hasUnsavedChanges = false;
    return this.activeDocument;
  }

  /** Schedules a debounced save of the active document; when the timer fires,
   * only messages staged by then are written. */
  public scheduleSave(): void {
    if (!this.activeDocument) {
      return;
    }
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer);
    }
    this.saveTimer = window.setTimeout(() => {
      this.saveTimer = null;
      // Nobody awaits a debounced save; a failure stays marked and is
      // retried at the next save, so it must not surface as unhandled.
      void this.saveNow().catch(() => undefined);
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
          // Steps are immutable values, so a copy of the list is enough to keep
          // the save path from mutating the log.
          toolSteps: [...message.toolSteps],
        }),
    );
    this.hasUnsavedChanges = true;
  }

  /** Saves the active document immediately when it has changes: stamps the
   * update time, derives the title from the first user message when the
   * session is still untitled, and refreshes the sidebar list. A conversation
   * that was only opened and read is left untouched. */
  public async saveNow(): Promise<void> {
    const document = this.activeDocument;
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    if (!document || !this.hasUnsavedChanges) {
      // Callers rely on any earlier write having finished when this returns,
      // as it did when every call wrote: a switch must not race the last save.
      await this.saveInFlight;
      return;
    }
    // Cleared before the write so changes staged while it is in flight mark
    // the conversation again. A failed write marks it again only while it is
    // still the open conversation; marking a newer one would write a chat
    // that was never typed in.
    this.hasUnsavedChanges = false;
    document.updatedAt = new Date().toISOString();
    document.messages = SessionDocument.persistableMessages(document.messages);
    if (document.title.length === 0) {
      document.title = SessionDocument.titleFromFirstUserMessage(document.messages);
    }
    const write = this.bridge.save(document).then(
      () => this.refreshList(),
      (saveFailure: unknown) => {
        if (this.activeDocument === document) {
          this.hasUnsavedChanges = true;
        }
        throw saveFailure;
      },
    );
    this.saveInFlight = write.catch(() => undefined);
    await write;
  }

  /** Writes staged changes, scheduled or not, and waits for any write in
   * flight; keyed on unsaved changes rather than the timer so a staged change
   * without a scheduled save is never dropped by a switch. */
  public async flushPendingSave(): Promise<void> {
    await this.saveNow();
  }

  /** Renames a saved conversation. The open conversation's title is updated in
   * memory too, and its pending save lands first: either would otherwise
   * write the old title back over the new one. The host keeps the update
   * time, so a rename does not move the conversation in the sidebar. */
  public async rename(id: string, title: string): Promise<void> {
    const renamedDocument = this.activeDocument?.id === id ? this.activeDocument : null;
    const previousTitle = renamedDocument?.title ?? null;
    if (renamedDocument != null) {
      await this.flushPendingSave().catch(() => undefined);
      renamedDocument.title = title;
    }
    try {
      await this.bridge.rename(id, title);
    } catch (renameFailure) {
      // The file kept its old name, so the open conversation must too, or
      // its next save would quietly write the rename that just failed.
      if (renamedDocument != null && previousTitle != null) {
        renamedDocument.title = previousTitle;
      }
      throw renameFailure;
    }
    await this.refreshList();
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
