import { DaemonClient } from "./DaemonClient";
import type { ChatStreamer } from "./ChatStreamer";
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
import { ToolStep, ToolStepState } from "./ToolStep";

/** How the chat surface discovered its models at startup. */
export type LoadPhase = "idle" | "loading" | "ready" | "empty" | "failed";

const COMPOSER_FONT_SIZE_STEP = 1;
/** Extra time before a stall is reported once tools finish: the next model
 * request may first reload the chat model, which an image tool swaps out, and
 * a large model can take longer than the stall timeout to load. */
const MODEL_RELOAD_GRACE_MILLISECONDS = 120_000;

/**
 * Owns the conversation: asks the turn streamer for each answer, keeps the
 * message history and the session lifecycle, and preserves the user's ask so a
 * failure always offers a recoverable next action. It only publishes state; the
 * daemon handshake and model list come from the client, each turn's wire and
 * agent loop live behind the streamer, and all drawing lives in the views.
 *
 * The surface is a pure function of the published ChatSurfaceState, so this
 * class decides what the conversation looks like and the views decide only how
 * it is drawn.
 */
export class ChatController {
  private readonly client: DaemonClient;
  private readonly streamer: ChatStreamer;
  private readonly bridge: SessionBridge;
  private readonly store: SessionStore;
  private readonly log = new ConversationLog();
  private readonly listeners = new Set<(state: ChatSurfaceState) => void>();
  private readonly watchdog: StallWatchdog;
  private preferences: SessionPreferences = SessionPreferences.default();
  private activeStreamAbort: AbortController | null = null;
  private activeStreamPromise: Promise<void> | null = null;
  private stopRequested = false;
  private stallFailure: ChatFailure | null = null;
  private loadPhase: LoadPhase = "idle";
  private loadFailureMessage = "";
  private failure: ChatFailure | null = null;
  private models: ChatModel[] = [];
  private selectedModelID: string | null = null;
  private isStreaming = false;
  /** Tool calls of the current turn that have started and not yet finished. */
  private readonly runningToolCallIDs = new Set<string>();

  public constructor(init: { client: DaemonClient; streamer: ChatStreamer; bridge: SessionBridge; store: SessionStore }) {
    this.client = init.client;
    this.streamer = init.streamer;
    this.bridge = init.bridge;
    this.store = init.store;
    this.watchdog = new StallWatchdog(this.streamer.stallTimeoutMilliseconds, () => this.handleStall());
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
      this.selectedModelID = this.initialModelID();
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
    if (!this.isStreaming || !this.activeStreamAbort) {
      return;
    }
    this.stopRequested = true;
    this.activeStreamAbort.abort();
    await this.activeStreamPromise?.catch(() => undefined);
  }

  // MARK: - Preferences

  /** The model to start with: the user's saved choice when the catalog still
   * carries it, otherwise the catalog's first model. */
  private initialModelID(): string | null {
    const preferredModelID: string | null = this.preferences.modelID;
    if (preferredModelID != null && this.models.some((model) => model.id === preferredModelID)) {
      return preferredModelID;
    }
    const firstModel: ChatModel | undefined = this.models[0];
    return firstModel != null ? firstModel.id : null;
  }

  /** Switches the model future turns go to. An unknown or unchanged id is
   * ignored so a stale menu can never desync the surface from the catalog. */
  public selectModel(modelID: string): void {
    if (modelID === this.selectedModelID) {
      return;
    }
    if (!this.models.some((model) => model.id === modelID)) {
      return;
    }
    this.selectedModelID = modelID;
    this.preferences.modelID = modelID;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
    this.publish();
  }

  public setEffort(effort: ThinkingEffort): void {
    if (effort === this.preferences.thinkingEffort) {
      return;
    }
    this.preferences.thinkingEffort = effort;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
    this.publish();
  }

  /** Turns tool use on or off for future turns. */
  public setToolsEnabled(toolsEnabled: boolean): void {
    if (toolsEnabled === this.preferences.toolsEnabled) {
      return;
    }
    this.preferences.toolsEnabled = toolsEnabled;
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

  /** Deletes one conversation; deleting the open one also clears the screen
   * and starts a fresh conversation, so later turns save to a new file. */
  public async deleteSession(id: string): Promise<void> {
    // Decided before the delete: the store forgets the active document as part
    // of deleting it, so asking afterwards always answered "not active" and the
    // deleted messages stayed on screen with nothing to save new turns into.
    const isDeletingActiveSession = id === this.store.activeSessionID();
    if (isDeletingActiveSession) {
      // As in newChat: a reply still streaming or a debounced save in flight
      // would otherwise write the conversation back after it was deleted.
      await this.settleActiveStream();
      await this.store.flushPendingSave().catch(() => undefined);
    }
    const generatedImageNames = await this.generatedImageNamesOf(id, isDeletingActiveSession);
    await this.store.delete(id);
    if (generatedImageNames.length > 0) {
      // The conversation is already gone; an image left behind by a failed
      // delete costs disk space but must not undo or block the delete.
      await this.bridge.deleteGeneratedImages(generatedImageNames).catch(() => undefined);
    }
    if (isDeletingActiveSession) {
      this.store.startNew();
      this.log.reset();
      this.failure = null;
    }
    this.publish();
  }

  /** The images a conversation's tool steps generated, so they are deleted
   * with it. The page reads them because it owns the conversation format; a
   * conversation that is not open is loaded to find them. */
  private async generatedImageNamesOf(id: string, isActiveSession: boolean): Promise<string[]> {
    const messages = isActiveSession
      ? this.log.messages
      : await this.bridge.load(id).then((document) => (document != null ? document.messages : [])).catch(() => []);
    return messages.flatMap((message) =>
      message.toolSteps.flatMap((toolStep) => (toolStep.generatedImageName != null ? [toolStep.generatedImageName] : [])),
    );
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
    this.runningToolCallIDs.clear();
    this.watchdog.start();
    const streamAbort = new AbortController();
    this.activeStreamAbort = streamAbort;
    const modelID = this.selectedModelID;
    const effort = this.preferences.thinkingEffort;
    this.activeStreamPromise = this.runStream(streamAbort, modelID, effort);
  }

  private async runStream(
    streamAbort: AbortController,
    modelID: string,
    effort: ThinkingEffort,
  ): Promise<void> {
    try {
      await this.streamer.streamChat({
        modelID,
        messages: this.log.requestMessages(),
        thinkingEffort: effort,
        toolsEnabled: this.preferences.toolsEnabled,
        modelSupportsToolCalls: this.selectedModelSupportsToolCalls(),
        onText: (delta) => this.ingestDelta(delta, (text) => this.log.applyTextDelta(text)),
        onReasoning: (delta) => this.ingestDelta(delta, (text) => this.log.applyReasoningDelta(text)),
        onToolStep: (toolStep) => this.ingestToolStep(toolStep),
        signal: streamAbort.signal,
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
      this.activeStreamAbort = null;
      this.activeStreamPromise = null;
      this.autosave();
      void this.store
        .saveNow()
        .catch(() => undefined)
        .then(() => this.publish());
      this.publish();
    }
  }

  /** A tool step counts as activity, and stall detection pauses while any
   * tool runs (image generation can take longer than the stall timeout); the
   * row shows the moment the call starts. */
  private ingestToolStep(toolStep: ToolStep): void {
    this.watchdog.activity();
    if (toolStep.state === ToolStepState.RUNNING) {
      this.runningToolCallIDs.add(toolStep.toolCallID);
      this.watchdog.pause();
    } else if (this.runningToolCallIDs.delete(toolStep.toolCallID) && this.runningToolCallIDs.size === 0) {
      this.watchdog.resume(MODEL_RELOAD_GRACE_MILLISECONDS);
    }
    this.log.applyToolStep(toolStep);
    this.autosave();
    this.publish();
  }

  private selectedModelSupportsToolCalls(): boolean {
    const selectedModel = this.models.find((model) => model.id === this.selectedModelID);
    return selectedModel != null && selectedModel.supportsToolCalls;
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
    if (!this.isStreaming || !this.activeStreamAbort) {
      return;
    }
    this.stallFailure = new ChatFailure(ChatFailureKind.STALL);
    this.activeStreamAbort.abort();
  }

  /** Aborts an in-flight stream and waits for its bookkeeping to finish, so a
   * session switch never races the stream's final save. */
  private async settleActiveStream(): Promise<void> {
    if (!this.activeStreamAbort) {
      return;
    }
    this.stopRequested = true;
    this.activeStreamAbort.abort();
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
          toolSteps: message.toolSteps,
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
      models: this.models,
      selectedModelID: this.selectedModelID,
      toolsEnabled: this.preferences.toolsEnabled,
      toolsAvailable: this.selectedModelSupportsToolCalls(),
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
