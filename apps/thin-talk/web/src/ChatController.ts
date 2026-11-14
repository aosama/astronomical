import type { ExportChoices } from "./export/ExportOptions";
import { ImageModelDirectory } from "./ImageModelDirectory";
import { ImageModelSelection } from "./ImageModelSelection";
import { ModelDescriptionDirectory } from "./ModelDescriptionDirectory";
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
import { StallWatchdog } from "./StallWatchdog";
import { ToolStep, ToolStepState } from "./ToolStep";
import { describeLoadFailure, loadNoticeAsksForAction, loadNoticeText } from "./LoadNotice";
import { OtherCopyAnsweredError } from "./OtherCopyAnsweredError";
import type { LoadPhase } from "./LoadNotice";

/** Extra time before a stall is reported once tools finish: the next model
 * request may first reload the chat model, which an image tool swaps out, and
 * a large model can take longer than the stall timeout to load. */
const MODEL_RELOAD_GRACE_MILLISECONDS = 120_000;

/**
 * How the window waits for an engine that was not answering when it opened.
 * The engine normally starts within a few seconds, so the first retries are
 * quick; after that a stopped engine is asked rarely, which still brings the
 * window back by itself once the engine returns. Tests shorten these.
 */
export const ENGINE_START_PATIENCE = {
  quickRetryMilliseconds: 1_000,
  quickRetryCount: 30,
  slowRetryMilliseconds: 5_000,
};

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
  private loadAttemptInFlight: Promise<void> | null = null;
  private loadRetryTimer: ReturnType<typeof setTimeout> | null = null;
  private failedLoadAttemptCount = 0;
  private failure: ChatFailure | null = null;
  private models: ChatModel[] = [];
  private selectedModelID: string | null = null;
  private readonly imageModelDirectory: ImageModelDirectory;
  private readonly imageModelSelection: ImageModelSelection;
  /** Model details for the model panel, read after the model list. */
  private readonly modelDescriptionDirectory: ModelDescriptionDirectory;
  private isStreaming = false;
  /** Tool calls of the current turn that have started and not yet finished. */
  private readonly runningToolCallIDs = new Set<string>();

  /** `imageModelSelection` is shared with the image tool: choosing here is what
   * the next picture is drawn with, and the tool's fresh list of installed
   * models redraws the composer's model panel through it. */
  public constructor(init: {
    client: DaemonClient;
    streamer: ChatStreamer;
    bridge: SessionBridge;
    store: SessionStore;
    imageModelDirectory: ImageModelDirectory;
    imageModelSelection: ImageModelSelection;
  }) {
    this.client = init.client;
    this.modelDescriptionDirectory = new ModelDescriptionDirectory(init.client, () => this.publish());
    this.imageModelDirectory = init.imageModelDirectory;
    this.imageModelSelection = init.imageModelSelection;
    this.imageModelSelection.onChange(() => this.publish());
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
    await this.attemptLoad();
  }

  /**
   * One start-up attempt. The window opens with the app, at the same moment
   * the menu-bar agent starts the engine, so the first attempt can reach a
   * port nothing listens on yet. A failed attempt is therefore repeated until
   * the engine answers: without that the window kept "Load failed" and an
   * empty model list until the app was quit and opened again.
   *
   * Attempts made at the same moment share one run, so the retry timer and a
   * caller such as a model landing never start the session twice.
   */
  private attemptLoad(): Promise<void> {
    if (this.loadAttemptInFlight === null) {
      this.clearLoadRetry();
      this.loadAttemptInFlight = this.runLoadAttempt().finally(() => {
        this.loadAttemptInFlight = null;
      });
    }
    return this.loadAttemptInFlight;
  }

  private async runLoadAttempt(): Promise<void> {
    try {
      this.preferences = SessionPreferences.fromPlain(await this.bridge.loadPrefs());
    } catch {
      this.preferences = SessionPreferences.default();
    }
    this.imageModelSelection.prefer(this.preferences.imageModelID);
    try {
      await this.client.handshake();
      this.models = await this.client.fetchModels();
      this.selectedModelID = this.initialModelID();
      this.loadPhase = this.models.length === 0 ? "empty" : "ready";
    } catch (error) {
      this.loadPhase = "failed";
      this.loadFailureMessage = describeLoadFailure(error);
      this.publish();
      // Only what can change by itself is asked again: what the network or
      // the engine reported, and another copy holding the address, which the
      // person can quit. An engine whose answer makes no sense will answer
      // the same way every time, so asking again helps nobody.
      if (error instanceof ChatFailure || error instanceof OtherCopyAnsweredError) {
        this.scheduleLoadRetry();
      }
      return;
    }
    this.failedLoadAttemptCount = 0;
    try {
      await this.restoreMostRecentSession();
    } catch {
      this.store.startNew();
    }
    this.publish();
    // Image models and plain model names only decorate the composer, so they
    // are read after chat is ready rather than delaying it.
    void this.refreshImageModels();
  }

  private scheduleLoadRetry(): void {
    this.failedLoadAttemptCount += 1;
    const retryDelayMilliseconds = this.failedLoadAttemptCount <= ENGINE_START_PATIENCE.quickRetryCount
      ? ENGINE_START_PATIENCE.quickRetryMilliseconds
      : ENGINE_START_PATIENCE.slowRetryMilliseconds;
    this.loadRetryTimer = setTimeout(() => {
      this.loadRetryTimer = null;
      void this.attemptLoad();
    }, retryDelayMilliseconds);
  }

  private clearLoadRetry(): void {
    if (this.loadRetryTimer !== null) {
      clearTimeout(this.loadRetryTimer);
      this.loadRetryTimer = null;
    }
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

  /** Sends a message. While an answer is still coming, the message is
   * queued instead: the running turn takes it when it can, and anything it
   * did not take is sent as soon as the turn is over. */
  public send(draft: string): void {
    const trimmed = draft.trim();
    if (trimmed.length === 0 || !this.selectedModelID || this.loadPhase !== "ready") {
      return;
    }
    if (this.isStreaming) {
      const queuedMessage = this.log.appendQueuedUserTurn(trimmed);
      this.streamer.queueUserMessage(queuedMessage.id, trimmed);
      this.autosave();
      this.publish();
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

  /**
   * Re-reads the chat models after one was installed while the window was
   * open (a chat model Astro downloaded landing). Keeps the current choice when it is still
   * installed; a window that had no model becomes ready to chat. A failed read
   * keeps the previous list, as the picture list does.
   */
  public async refreshChatModels(): Promise<void> {
    if (this.loadPhase === "failed") {
      // The window never reached the daemon; a model landing means it can
      // now, so the whole start-up runs again rather than half of it.
      await this.attemptLoad();
      return;
    }
    if (this.loadPhase !== "ready" && this.loadPhase !== "empty") {
      return;
    }
    try {
      this.models = await this.client.fetchModels();
    } catch {
      return;
    }
    void this.modelDescriptionDirectory.refresh();
    const selectedModelID = this.selectedModelID;
    if (selectedModelID === null || !this.models.some((model) => model.id === selectedModelID)) {
      this.selectedModelID = this.initialModelID();
    }
    this.loadPhase = this.models.length === 0 ? "empty" : "ready";
    this.publish();
  }

  /** Lists the installed image models again, for the model panel. A failed
   * read keeps the last list: the image tool lists them again when it runs. */
  public async refreshImageModels(): Promise<void> {
    void this.modelDescriptionDirectory.refresh();
    try {
      this.imageModelSelection.updateInstalled(await this.imageModelDirectory.load());
    } catch {
      // Keep the previous list; the panel is a convenience, not a requirement.
    }
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

  /** Switches the image model the next picture is drawn with. An unknown or
   * unchanged id is ignored, as for the chat model. */
  public selectImageModel(imageModelID: string): void {
    if (imageModelID === this.selectedImageModelID()) {
      return;
    }
    if (!this.imageModelSelection.installed.some((imageModel) => imageModel.id === imageModelID)) {
      return;
    }
    this.preferences.imageModelID = imageModelID;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
    // Publishes through the selection's change listener.
    this.imageModelSelection.prefer(imageModelID);
  }

  private selectedImageModelID(): string | null {
    const selectedImageModel = this.imageModelSelection.current();
    return selectedImageModel != null ? selectedImageModel.id : null;
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

  /** Keeps the width the user dragged the sidebar to; null goes back to the
   * default width. Saved so the sidebar opens at that width next launch. */
  public setSidebarWidth(widthPixels: number | null): void {
    const sidebarWidthPixels = widthPixels != null ? SessionPreferences.clampSidebarWidth(widthPixels) : null;
    if (sidebarWidthPixels === this.preferences.sidebarWidthPixels) {
      return;
    }
    this.preferences.sidebarWidthPixels = sidebarWidthPixels;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
    this.publish();
  }

  /** The choices of the last export that was saved, or null before the first. */
  public get exportChoices(): ExportChoices | null {
    return this.preferences.exportChoices;
  }

  /** Keeps an export's choices so the sheet opens on them next time, in
   * this launch and the next. */
  public setExportChoices(exportChoices: ExportChoices): void {
    this.preferences.exportChoices = exportChoices;
    void this.bridge.savePrefs(this.preferences).catch(() => undefined);
  }

  /** Renames one conversation. A blank or unchanged title keeps the old one,
   * so an accidental Enter on an emptied field never leaves a nameless chat. */
  public async renameSession(id: string, requestedTitle: string): Promise<void> {
    const newTitle = requestedTitle.trim();
    const currentTitle = this.store.listSessions().find((session) => session.id === id)?.title;
    if (newTitle.length === 0 || newTitle === currentTitle) {
      this.publish();
      return;
    }
    await this.store.rename(id, newTitle).catch(() => this.store.refreshList().catch(() => undefined));
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
    this.startTurn();
  }

  /** Answers the oldest message still queued once a turn ends normally; after
   * a stop or a failure, the queued messages are marked not sent instead, so
   * nothing goes out that the user may no longer want. */
  private continueWithQueuedMessages(turnEndedNormally: boolean): void {
    const queuedMessage = this.log.firstQueuedUserTurn();
    if (queuedMessage == null) {
      return;
    }
    if (!turnEndedNormally || !this.selectedModelID) {
      this.log.markQueuedTurnsNotSent();
      return;
    }
    this.log.beginAnsweringQueuedTurn(queuedMessage.id);
    this.startTurn();
    // The rest go to the new turn at once, oldest first, so they are answered
    // in the order they were sent rather than behind anything sent later.
    for (const laterQueuedMessage of this.log.messages.filter((message) => message.state === ChatMessageState.QUEUED)) {
      this.streamer.queueUserMessage(laterQueuedMessage.id, laterQueuedMessage.content);
    }
  }

  /** Starts streaming into the assistant turn the log has already opened. */
  private startTurn(): void {
    if (!this.selectedModelID) {
      return;
    }
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
    this.activeStreamPromise = this.runStream(streamAbort, modelID);
  }

  private async runStream(
    streamAbort: AbortController,
    modelID: string,
  ): Promise<void> {
    try {
      await this.streamer.streamChat({
        modelID,
        conversationID: this.store.activeSessionID(),
        messages: this.log.requestMessages(),
        toolsEnabled: this.preferences.toolsEnabled,
        modelSupportsToolCalls: this.selectedModelSupportsToolCalls(),
        modelAcceptsImages: this.selectedModelAcceptsImages(),
        contextWindowTokens: this.selectedModelContextWindowTokens(),
        onText: (delta) => this.ingestDelta(delta, (text) => this.log.applyTextDelta(text)),
        onReasoning: (delta) => this.ingestDelta(delta, (text) => this.log.applyReasoningDelta(text)),
        onToolStep: (toolStep) => this.ingestToolStep(toolStep),
        onQueuedMessageAnswered: (queuedMessageID) => this.ingestQueuedMessageAnswer(queuedMessageID),
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
      const turnEndedNormally = !this.stopRequested && this.failure == null;
      this.log.removeEmptyAssistantTurn();
      this.isStreaming = false;
      this.log.assistantMessageID = null;
      this.activeStreamAbort = null;
      this.activeStreamPromise = null;
      this.continueWithQueuedMessages(turnEndedNormally);
      this.autosave();
      void this.store
        .saveNow()
        .catch(() => undefined)
        .then(() => this.publish());
      this.publish();
    }
  }

  /** The running turn started on a queued message: the current answer is
   * finished and a new one opens under that message. */
  private ingestQueuedMessageAnswer(queuedMessageID: string): void {
    this.watchdog.activity();
    // The runner only reports ids it was handed for messages still queued,
    // so this always opens a new answer.
    this.log.beginAnsweringQueuedTurn(queuedMessageID);
    this.autosave();
    this.publish();
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

  private selectedModelContextWindowTokens(): number | null {
    const selectedModel = this.models.find((model) => model.id === this.selectedModelID);
    return selectedModel != null ? selectedModel.contextWindowTokens : null;
  }

  private selectedModelAcceptsImages(): boolean {
    const selectedModel = this.models.find((model) => model.id === this.selectedModelID);
    return selectedModel != null && selectedModel.supportsVision;
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
      sidebarWidthPixels: this.preferences.sidebarWidthPixels,
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
    const noticeAsksForAction = loadNoticeAsksForAction(this.loadPhase, this.loadFailureMessage);
    const snapshot: TranscriptSnapshot = { messages, notice: loadNoticeText(this.loadPhase, this.loadFailureMessage), noticeAsksForAction };
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
      models: this.models,
      selectedModelID: this.selectedModelID,
      toolsEnabled: this.preferences.toolsEnabled,
      toolsAvailable: this.selectedModelSupportsToolCalls(),
      imageModels: this.imageModelSelection.installed,
      selectedImageModelID: this.selectedImageModelID(),
      modelDescriptions: this.modelDescriptionDirectory.descriptions,
      failure: this.failure
        ? {
            message: this.failure.headline,
            nextAction: this.failure.nextAction,
            detail: this.failure.reason !== this.failure.headline ? this.failure.reason : null,
          }
        : null,
    });
  }
}
