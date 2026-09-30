import { SupervisorClient } from "./SupervisorClient";
import { SessionBridge } from "./SessionBridge";
import { SessionStore } from "./SessionStore";
import { SessionPreferences } from "./SessionPreferences";
import { ConversationLog } from "./ConversationLog";
import { ChatSurfaceState } from "./ChatSurfaceState";
import { TranscriptMessage } from "./TranscriptMessage";
import type { TranscriptSnapshot } from "./TranscriptView";
import { ComposerState } from "./ComposerState";
import { ChatMessage } from "./ChatMessage";
import { ChatMessageState } from "./ChatMessageState";
import { ChatModel } from "./ChatModel";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { ThinkingEffort } from "./ThinkingEffort";
import { StallWatchdog } from "./StallWatchdog";

/** How the chat surface discovered its models at startup. */
export type LoadPhase = "idle" | "loading" | "ready" | "empty" | "failed";

const COMPOSER_FONT_SIZE_STEP = 1;

/**
 * Owns the conversation: drives the streaming client, keeps the message history
 * and the session lifecycle, and preserves the user's ask so a failure always
 * offers a recoverable next action. It only publishes state; all wire logic
 * lives in the client and all drawing lives in the views.
 *
 * The surface is a pure function of the published ChatSurfaceState, so this
 * class decides what the conversation looks like and the views decide only how
 * it is drawn.
 */
export class ChatController {
  private readonly client: SupervisorClient;
  private readonly bridge: SessionBridge;
  private readonly store: SessionStore;
  private readonly log = new ConversationLog();
  private readonly listeners = new Set<(state: ChatSurfaceState) => void>();
  private readonly watchdog: StallWatchdog;
  private preferences: SessionPreferences = SessionPreferences.default();
  private activeStream: AbortController | null = null;
  private activeStreamPromise: Promise<void> | null = null;
  private stopRequested = false;
  private stallFailure: ChatFailure | null = null;
  private loadPhase: LoadPhase = "idle";
  private loadFailureMessage = "";
  private failure: ChatFailure | null = null;
  private models: ChatModel[] = [];
  private selectedModelID: string | null = null;
  private isStreaming = false;

  public constructor(init: { client: SupervisorClient; bridge: SessionBridge; store: SessionStore }) {
    this.client = init.client;
    this.bridge = init.bridge;
    this.store = init.store;
    this.watchdog = new StallWatchdog(this.client.stallTimeoutMilliseconds, () => this.handleStall());
  }

  public subscribe(listener: (state: ChatSurfaceState) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  // MARK: - Startup

  public async load(): Promise<void> {
    this.loadPhase = "loading";
    this.publish();
    try {
      this.preferences = SessionPreferences.fromPlain(await this.bridge.loadPrefs());
    } catch {
      this.preferences = SessionPreferences.default();
    }
    try {
      await this.client.handshake();
      this.models = await this.client.fetchModels();
      const firstModel = this.models[0];
      this.selectedModelID = firstModel ? firstModel.id : null;
      this.loadPhase = this.models.length === 0 ? "empty" : "ready";
    } catch (error) {
      this.loadPhase = "failed";
      this.loadFailureMessage = error instanceof Error ? error.message : String(error);
      this.publish();
      return;
    }
    try {
      await this.restoreMostRecentSession();
    } catch {
      this.store.startNew();
    }
    this.publish();
  }

  private async restoreMostRecentSession(): Promise<void> {
    await this.store.refreshList();
    const sessions = this.store.listSessions().slice();
    sessions.sort((first, second) => second.updatedAt.localeCompare(first.updatedAt));
    const mostRecent = sessions[0];
    if (!mostRecent) {
      this.store.startNew();
      return;
    }
    const document = await this.store.open(mostRecent.id);
    if (document) {
      this.log.reset();
      this.log.messages = document.messages;
    } else {
      this.store.startNew();
    }
  }

  // MARK: - Sending

  public send(draft: string): void {
    const trimmed = draft.trim();
    if (trimmed.length === 0 || !this.selectedModelID || this.isStreaming || this.loadPhase !== "ready") {
      return;
    }
    this.failure = null;
    this.log.appendUserTurn(trimmed);
    this.beginStreaming();
  }

  /** Retries after a failure, keeping the exact ask the user typed. */
  public retry(): void {
    if (this.isStreaming || !this.selectedModelID || !this.log.lastUserTurn()) {
      return;
    }
    this.failure = null;
    this.beginStreaming();
  }

  public async stop(): Promise<void> {
    if (!this.isStreaming || !this.activeStream) {
      return;
    }
    this.stopRequested = true;
    this.activeStream.abort();
    await this.activeStreamPromise?.catch(() => undefined);
  }

  // MARK: - Preferences

  public setEffort(effort: ThinkingEffort): void {
    if (effort === this.preferences.thinkingEffort) {
      return;
    }
    this.preferences.thinkingEffort = effort;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
    this.publish();
  }

  public adjustFontSize(delta: number): void {
    const adjusted = this.preferences.clampComposerFontSize(
      this.preferences.composerFontSize + delta * COMPOSER_FONT_SIZE_STEP,
    );
    if (adjusted === this.preferences.composerFontSize) {
      return;
    }
    this.preferences.composerFontSize = adjusted;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
    this.publish();
  }

  // MARK: - Sessions

  public async newChat(): Promise<void> {
    await this.settleActiveStream();
    await this.store.saveNow().catch(() => undefined);
    this.store.startNew();
    this.log.reset();
    this.failure = null;
    this.publish();
  }

  public async openSession(id: string): Promise<void> {
    if (id === this.store.activeSessionID()) {
      return;
    }
    await this.settleActiveStream();
    await this.store.saveNow().catch(() => undefined);
    const document = await this.store.open(id);
    if (!document) {
      // The session file is gone or unreadable; keep the current conversation
      // and let the refreshed list reflect reality.
      await this.store.refreshList().catch(() => undefined);
      this.publish();
      return;
    }
    this.log.reset();
    this.log.messages = document.messages;
    this.failure = null;
    this.publish();
  }

  public async deleteSession(id: string): Promise<void> {
    await this.store.delete(id);
    if (id === this.store.activeSessionID()) {
      this.store.startNew();
      this.log.reset();
      this.failure = null;
    }
    this.publish();
  }

  // MARK: - Streaming

  private beginStreaming(): void {
    if (!this.selectedModelID) {
      return;
    }
    // Open the assistant turn up-front: a visible placeholder prevents stacked
    // user turns from looking unanswered.
    this.log.openAssistantTurn();
    this.isStreaming = true;
    this.stopRequested = false;
    this.stallFailure = null;
    this.autosave();
    this.publish();
    this.watchdog.start();
    const stream = new AbortController();
    this.activeStream = stream;
    const modelID = this.selectedModelID;
    const effort = this.preferences.thinkingEffort;
    this.activeStreamPromise = this.runStream(stream, modelID, effort);
  }

  private async runStream(stream: AbortController, modelID: string, effort: ThinkingEffort): Promise<void> {
    try {
      await this.client.streamChat({
        modelID,
        messages: this.log.requestMessages(),
        thinkingEffort: effort,
        onText: (delta) => this.ingestDelta(delta, (text) => this.log.applyTextDelta(text)),
        onReasoning: (delta) => this.ingestDelta(delta, (text) => this.log.applyReasoningDelta(text)),
        signal: stream.signal,
      });
      this.watchdog.stop();
      if (this.stallFailure) {
        this.failure = this.stallFailure;
        this.stallFailure = null;
        this.log.markAssistantTurn(ChatMessageState.FAILED);
      } else if (this.stopRequested) {
        this.log.markAssistantTurn(ChatMessageState.STOPPED);
      } else {
        this.log.markAssistantTurn(ChatMessageState.COMPLETE);
      }
    } catch (error) {
      this.watchdog.stop();
      this.failure =
        error instanceof ChatFailure
          ? error
          : new ChatFailure(ChatFailureKind.UNKNOWN, error instanceof Error ? error.message : String(error));
      this.log.markAssistantTurn(ChatMessageState.FAILED);
    } finally {
      this.log.removeEmptyAssistantTurn();
      this.isStreaming = false;
      this.log.assistantMessageID = null;
      this.activeStream = null;
      this.activeStreamPromise = null;
      this.autosave();
      void this.store
        .saveNow()
        .catch(() => undefined)
        .then(() => this.publish());
      this.publish();
    }
  }

  private ingestDelta(delta: string, apply: (text: string) => void): void {
    this.watchdog.activity();
    apply(delta);
    this.autosave();
    this.publish();
  }

  /** A stall has no backend marker, so the watchdog aborts the stream and the
   * run records the stall as the turn's failure. */
  private handleStall(): void {
    if (!this.isStreaming || !this.activeStream) {
      return;
    }
    this.stallFailure = new ChatFailure(ChatFailureKind.STALL);
    this.activeStream.abort();
  }

  /** Aborts an in-flight stream and waits for its bookkeeping to finish, so a
   * session switch never races the stream's final save. */
  private async settleActiveStream(): Promise<void> {
    if (!this.activeStream) {
      return;
    }
    this.stopRequested = true;
    this.activeStream.abort();
    await this.activeStreamPromise?.catch(() => undefined);
  }

  private autosave(): void {
    this.store.stageMessages(this.log.messages);
    this.store.scheduleSave();
  }

  // MARK: - Surface state

  private publish(): void {
    const state = new ChatSurfaceState({
      transcript: this.transcriptSnapshot(),
      composer: this.composerState(),
      sessions: this.store.listSessions(),
      activeSessionID: this.store.activeSessionID(),
      composerFontSize: this.preferences.composerFontSize,
    });
    this.listeners.forEach((listener) => listener(state));
  }

  private transcriptSnapshot(): TranscriptSnapshot {
    const messages = this.log.messages.map(
      (message) =>
        new TranscriptMessage({
          id: message.id,
          role: message.role,
          markdown: message.content,
          reasoning: message.reasoning,
          state: this.viewState(message),
        }),
    );
    const snapshot: TranscriptSnapshot = { messages, notice: this.noticeText() };
    return snapshot;
  }

  private viewState(message: ChatMessage): ChatMessageState {
    if (this.isStreaming && message.id === this.log.assistantMessageID) {
      return ChatMessageState.STREAMING;
    }
    return message.state;
  }

  private composerState(): ComposerState {
    return new ComposerState({
      isStreaming: this.isStreaming,
      acceptsInput: this.loadPhase !== "loading",
      isReady: this.loadPhase === "ready",
      effort: this.preferences.thinkingEffort,
      failure: this.failure
        ? { message: this.failure.message, nextAction: this.failure.nextAction }
        : null,
    });
  }

  private noticeText(): string | null {
    switch (this.loadPhase) {
      case "empty":
        return "No chat model is available yet. Add one from the Library.";
      case "idle":
      case "loading":
        return "Looking for a chat model on this Mac.";
      case "failed":
        return this.loadFailureMessage;
      case "ready":
        return null;
    }
  }
}
