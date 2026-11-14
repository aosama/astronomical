import { CODE_BLOCK_TOOLS_CLASS, COPY_CODE_ACTION, COPY_IMAGE_ACTION, EXPORT_ANSWER_ACTION, Renderer, SAVE_IMAGE_ACTION, SPEAK_ANSWER_ACTION } from "./Renderer";
import { TranscriptMessage } from "./TranscriptMessage";
import { TranscriptAction } from "./TranscriptAction";
import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";
import { reasoningDisclosureIdentifier } from "./ReasoningDisclosure";
import type { TranscriptPersona } from "./TranscriptPersona";
import { SpeakPhase } from "./speech/SpeakPhase";
import type { SpeechState } from "./speech/SpeechState";
import morphdom from "morphdom";

/** Long enough to read, short enough that a second copy is not held up. */
const COPIED_CONFIRMATION_MILLISECONDS = 1500;

/** The transcript half of one surface refresh. */
export interface TranscriptSnapshot {
  messages: readonly TranscriptMessage[];
  notice: string | null;
  /** True when the notice tells the person to do something before the
   * window can work (quit another copy), so it is drawn to be read. Other
   * notices describe a wait that ends by itself and stay quiet. */
  noticeAsksForAction?: boolean;
}

/**
 * Conversation transcript layer.
 *
 * This class turns each message into a DOM article, reports interactions back,
 * and owns nothing else. The rendering rules themselves live in Renderer.
 *
 * Two properties matter for a streaming conversation:
 *
 * 1. Only the message that changed is patched, so a growing answer never
 *    re-renders the conversation and finished blocks keep their DOM identity.
 * 2. Failures are reported rather than swallowed, because a canvas that renders
 *    nothing while throwing nothing is indistinguishable from a broken model.
 */
export class TranscriptView {
  private readonly transcript: HTMLElement;
  private readonly emptyState: HTMLElement;
  private readonly emptyStateNotice: HTMLElement | null;
  private readonly renderer: Renderer;
  private readonly onAction: (action: TranscriptAction) => void;
  private readonly onError: (detail: string) => void;
  private readonly inner: HTMLElement;
  private readonly articlesByMessageID = new Map<string, HTMLElement>();
  // Reader intent per message: an explicit expand/collapse wins over the
  // renderer's default for that message while it streams.
  private readonly reasoningOpenByMessageID = new Map<string, boolean>();
  // Messages whose one-time terminal collapse has been applied. Completion must
  // fold the reasoning away exactly once; after that the reader may reopen it
  // and the choice must stick through later re-renders.
  private readonly terminalCollapseAppliedByMessageID = new Set<string>();
  // Disclosure elements already carrying the toggle listener; morphdom keeps a
  // patched element's identity, so one install per element is enough, and the
  // set simply misses nodes a patch has replaced.
  private readonly disclosuresWithToggleListener = new WeakSet<HTMLDetailsElement>();
  // True while applyMessage is mutating disclosures programmatically: morphdom
  // attribute sync and the reconciliation itself fire `toggle` events that must
  // never be mistaken for the reader's own expand/collapse choice.
  private suppressReasoningRecording = false;
  // Last-rendered content signature per message: unchanged bubbles skip the
  // markdown parse and DOM diff entirely.
  private readonly renderedSignatures = new Map<string, string>();
  private pendingMessages: TranscriptMessage[] = [];
  private flushRequested = false;
  private followScroll = true;
  // The last snapshot drawn, so a persona change can redraw the conversation
  // without waiting for the controller to publish again.
  private latestSnapshot: TranscriptSnapshot | null = null;
  // The newest answer: only its Astro animates, so a long chat stays calm.
  private latestAssistantMessageID: string | null = null;
  private latestSpeechState: SpeechState | null = null;
  private latestSpeechBlocked = false;

  public constructor(init: {
    transcript: HTMLElement;
    emptyState: HTMLElement;
    renderer: Renderer;
    onAction: (action: TranscriptAction) => void;
    onError: (detail: string) => void;
  }) {
    this.transcript = init.transcript;
    this.emptyState = init.emptyState;
    this.emptyStateNotice = init.emptyState.querySelector("#empty-state-notice");
    this.renderer = init.renderer;
    this.onAction = init.onAction;
    this.onError = init.onError;
    this.inner = document.createElement("div");
    this.inner.className = "transcript__inner";
    this.transcript.appendChild(this.inner);
    this.transcript.addEventListener("scroll", () => {
      this.followScroll = this.shouldFollowScroll();
    });
    this.inner.addEventListener("click", (event) => this.handleClick(event));
  }

  /**
   * The one handler per disclosure element (attached when reconciled): `toggle`
   * does not bubble, and delegating from an ancestor container is unreliable
   * across engines, so the listener lives on the element itself. Recording the
   * reader's choice here — rather than a click handler on the summary — also
   * covers keyboard activation.
   */
  private handleDisclosureToggle = (event: Event): void => {
    if (this.suppressReasoningRecording) {
      return;
    }
    const target = event.target;
    if (!(target instanceof HTMLDetailsElement)) {
      return;
    }
    const article = target.closest("[data-message-id]");
    if (!(article instanceof HTMLElement) || !article.dataset.messageId) {
      return;
    }
    const messageID: string = article.dataset.messageId;
    this.reasoningOpenByMessageID.set(messageID, target.open);
  };

  /** Renders one snapshot: removes articles the snapshot no longer carries,
   * patches the rest, and updates the empty state. */
  public render(snapshot: TranscriptSnapshot): void {
    this.latestSnapshot = snapshot;
    this.markLatestAssistantMessage(snapshot);
    const incomingIDs = new Set(snapshot.messages.map((message) => message.id));
    for (const [messageID, article] of this.articlesByMessageID) {
      if (!incomingIDs.has(messageID)) {
        this.inner.removeChild(article);
        this.articlesByMessageID.delete(messageID);
        this.renderedSignatures.delete(messageID);
      }
    }
    for (const message of snapshot.messages) {
      if (this.messageNeedsRender(message)) {
        this.pendingMessages.push(message);
      }
    }
    this.emptyState.hidden = snapshot.messages.length > 0;
    if (this.emptyStateNotice) {
      if (snapshot.messages.length > 0) {
        this.emptyStateNotice.textContent = "";
      } else {
        const configuredNotice: string | null = snapshot.notice;
        this.emptyStateNotice.textContent = configuredNotice != null ? configuredNotice : "";
      }
      const asksForAction = snapshot.messages.length === 0 && snapshot.notice != null && snapshot.noticeAsksForAction === true;
      this.emptyStateNotice.classList.toggle("empty-state__notice--asks", asksForAction);
    }
    this.scheduleFlush();
  }

  /** Switches between classic Thin Talk and Astro. The persona is part of every
   * message's signature, so the redraw below repaints each message once. */
  public setPersona(persona: TranscriptPersona): void {
    this.renderer.setPersona(persona);
    if (this.latestSnapshot !== null) {
      this.render(this.latestSnapshot);
    }
  }

  private markLatestAssistantMessage(snapshot: TranscriptSnapshot): void {
    const assistantMessages = snapshot.messages.filter((message) => message.role === ChatRole.ASSISTANT);
    const newestAssistantMessage = assistantMessages[assistantMessages.length - 1];
    const newestAssistantMessageID = newestAssistantMessage !== undefined ? newestAssistantMessage.id : null;
    if (newestAssistantMessageID === this.latestAssistantMessageID) {
      return;
    }
    const previousLatestID = this.latestAssistantMessageID;
    if (previousLatestID !== null) {
      this.articlesByMessageID.get(previousLatestID)?.classList.remove("message--latest");
    }
    this.latestAssistantMessageID = newestAssistantMessageID;
    if (newestAssistantMessageID !== null) {
      this.articlesByMessageID.get(newestAssistantMessageID)?.classList.add("message--latest");
    }
  }

  /**
   * A message re-renders only when something the renderer reads — the
   * persona (classic or Astro), role, content, reasoning, state, attachments,
   * or tool steps — changed since it was last drawn. Every value Renderer.messageHtml consumes must appear here: a
   * missing field would let a changed message keep its stale node.
   *
   * Without this gate every published snapshot re-ran the markdown parse and
   * the DOM diff for every earlier message in the conversation, so one
   * streamed delta cost work proportional to the whole transcript's length.
   * The signature comparison is the cheap equality that keeps an unchanged
   * bubble on its already-rendered node.
   */
  private messageNeedsRender(message: TranscriptMessage): boolean {
    const signature = this.messageSignature(message);
    if (this.renderedSignatures.get(message.id) === signature) {
      return false;
    }
    this.renderedSignatures.set(message.id, signature);
    return true;
  }

  /** JSON encoding keeps the signature unambiguous for any content: a plain
   * delimiter could collide with message text containing that delimiter and
   * silently skip a needed re-render. */
  private messageSignature(message: TranscriptMessage): string {
    return JSON.stringify([
      this.renderer.currentPersona.signature(),
      message.role,
      message.markdown,
      message.reasoning,
      message.state,
      message.attachments.map((attachment) => [attachment.id, attachment.assetURL, attachment.label]),
      message.toolSteps.map((toolStep) => [toolStep.toolCallID, toolStep.label, toolStep.argumentsJSON, toolStep.resultText, toolStep.state, toolStep.generatedImageName, toolStep.generatedImageModelName]),
    ]);
  }

  /** Who the conversation is drawn as, so an export can name the assistant
   * the way the window does. */
  public get persona(): TranscriptPersona {
    return this.renderer.currentPersona;
  }

  /** The messages last drawn, in conversation order. */
  public latestMessages(): readonly TranscriptMessage[] {
    return this.latestSnapshot?.messages ?? [];
  }

  /** A detached copy of one message's rendered answer body, with its code
   * highlighted, its maths drawn and its diagrams in place, for an export to
   * reshape without touching the transcript. Null for a message not drawn. */
  public renderedBodyClone(messageID: string): HTMLElement | null {
    const answerBody = this.articlesByMessageID.get(messageID)?.querySelector(".markdown-body");
    return answerBody instanceof HTMLElement ? (answerBody.cloneNode(true) as HTMLElement) : null;
  }

  /** Draws any messages still waiting for the next frame, so a reader that
   * runs right after a conversation opens finds them drawn. */
  public flushPendingRender(): void {
    this.flushPending();
  }

  /** The drawn article for a message, for the message picker to mark. */
  public messageArticle(messageID: string): HTMLElement | null {
    return this.articlesByMessageID.get(messageID) ?? null;
  }

  public messageCount(): number {
    return this.articlesByMessageID.size;
  }

  public renderedText(messageID: string): string {
    const article = this.articlesByMessageID.get(messageID);
    if (article == null) {
      return "";
    }
    // Copy carries the rendered answer body only. The bubble also holds the
    // role label, the state label, the collapsible reasoning, and the action
    // buttons; reading the whole article's text would paste that chrome into
    // the clipboard alongside the reply.
    const answerBody = article.querySelector(".markdown-body");
    if (answerBody == null) {
      return "";
    }
    // A code block's language and Copy button sit inside the body, so they are
    // taken out of a copy of it: reading the body as it stands pasted
    // "pythonCopy" after every script.
    const copiedBody = answerBody.cloneNode(true) as Element;
    copiedBody.querySelectorAll(`.${CODE_BLOCK_TOOLS_CLASS}`).forEach((codeBlockTools) => codeBlockTools.remove());
    const answerText = copiedBody.textContent;
    return answerText != null ? answerText : "";
  }

  public diagnostics(): Record<string, unknown> {
    return {
      readyState: document.readyState,
      messages: this.articlesByMessageID.size,
      transcriptChildren: this.inner.children.length,
      pendingMessages: this.pendingMessages.length,
    };
  }

  // MARK: - Scheduling

  /**
   * Batches pending messages into one flush.
   *
   * A frame is the natural unit of visible work, but a frame is not guaranteed: an
   * occluded window and an offscreen web view both pause `requestAnimationFrame`,
   * which would leave a streamed answer unrendered until the window came back. The
   * timer is the floor that keeps progress moving, and whichever wins performs the
   * single flush.
   */
  private scheduleFlush(): void {
    if (this.flushRequested) {
      return;
    }
    this.flushRequested = true;
    window.requestAnimationFrame(() => this.flushPending());
    window.setTimeout(() => this.flushPending(), 16);
  }

  private flushPending(): void {
    if (!this.flushRequested) {
      return;
    }
    this.flushRequested = false;
    const batch = this.pendingMessages;
    this.pendingMessages = [];
    const wasFollowing = this.followScroll;
    batch.forEach((message) => {
      try {
        this.applyMessage(message);
      } catch (error) {
        this.onError(`render: ${error instanceof Error ? error.message : String(error)}`);
        this.renderPlainTextFallback(message);
      }
    });
    // A turn the local user just sent always pulls the transcript back to the
    // newest content. The decision is made here, from the batch itself, rather
    // than from followScroll, because the reader's scroll event from being
    // dragged up can land between the snapshot and this flush and reset it.
    // A redrawn reply brings its button back to resting; only the replies of
    // this batch need their buttons set again.
    for (const redrawnMessage of batch) {
      const redrawnArticle = this.articlesByMessageID.get(redrawnMessage.id);
      if (redrawnArticle !== undefined) {
        this.applySpeechToButtons(redrawnArticle);
      }
    }
    const followNow = wasFollowing || batch.some((message) => message.role === ChatRole.USER);
    if (followNow) {
      this.scrollToBottom();
    }
  }

  /** Shows each reply's Speak button the way the speech controller says.
   * Applied by attribute and text after every draw too, because a redraw of a
   * reply's markup brings its button back to resting. `isGenerating` greys the
   * buttons while an answer is being written: the engine holds one model, so
   * it cannot speak until the answer is finished. */
  public showSpeech(speechState: SpeechState, isGenerating: boolean): void {
    // The controller publishes on every streamed token; the buttons change
    // only when the speech state or the generating flag does.
    if (this.latestSpeechState === speechState && this.latestSpeechBlocked === isGenerating) {
      return;
    }
    this.latestSpeechState = speechState;
    this.latestSpeechBlocked = isGenerating;
    this.applySpeechToButtons(this.inner);
  }

  private applySpeechToButtons(scope: ParentNode): void {
    const speechState = this.latestSpeechState;
    for (const speakButton of scope.querySelectorAll<HTMLButtonElement>(`button[data-action="${SPEAK_ANSWER_ACTION}"]`)) {
      const buttonMessageID = speakButton.closest<HTMLElement>("[data-message-id]")?.dataset.messageId;
      const phase = speechState !== null && speechState.messageID === buttonMessageID ? speechState.phase : SpeakPhase.RESTING;
      speakButton.dataset.speakPhase = phase;
      speakButton.textContent = TranscriptView.speakLabels[phase];
      speakButton.disabled = this.latestSpeechBlocked && phase === SpeakPhase.RESTING;
    }
  }

  private static readonly speakLabels: Record<SpeakPhase, string> = {
    [SpeakPhase.RESTING]: "Speak",
    [SpeakPhase.PREPARING]: "Preparing…",
    [SpeakPhase.PLAYING]: "Stop",
    [SpeakPhase.NO_VOICE]: "Get a voice…",
    [SpeakPhase.BUSY]: "Busy, try again",
    [SpeakPhase.FAILED]: "Couldn't speak",
  };

  private shouldFollowScroll(): boolean {
    return this.transcript.scrollHeight - this.transcript.scrollTop - this.transcript.clientHeight < 120;
  }

  private scrollToBottom(): void {
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  // MARK: - Messages

  /**
   * The stable identifier for one bubble: `message:<role>:<message id>`, keyed
   * on the message's own id, never on its position, so an automation agent
   * locates the same bubble identically across snapshots and streaming
   * re-renders. The id attribute mirrors the data-testid so the accessibility
   * tree (which exposes the DOM id) and selector-based agents agree.
   */
  private static bubbleStableIdentifier(message: TranscriptMessage): string {
    const roleToken = message.role === ChatRole.USER ? "user" : "assistant";
    return `message:${roleToken}:${message.id}`;
  }

  private markBubbleStableIdentifier(article: HTMLElement, message: TranscriptMessage): void {
    const bubbleIdentifier = TranscriptView.bubbleStableIdentifier(message);
    article.dataset.testid = bubbleIdentifier;
    article.id = bubbleIdentifier;
  }

  private applyMessage(message: TranscriptMessage): void {
    if (!message.id) {
      return;
    }
    // A turn the local user just sent always pulls the transcript back to the
    // newest content; only model output respects the reader's scroll position.
    if (message.role === ChatRole.USER) {
      this.followScroll = true;
    }
    let article = this.articlesByMessageID.get(message.id);
    if (!article) {
      article = document.createElement("article");
      article.dataset.messageId = message.id;
      this.markBubbleStableIdentifier(article, message);
      this.inner.appendChild(article);
      this.articlesByMessageID.set(message.id, article);
    }
    article.className = `message message--${message.role === ChatRole.USER ? "user" : "assistant"}`;
    article.classList.toggle("message--latest", message.id === this.latestAssistantMessageID);
    const probe = document.createElement("div");
    probe.className = "message__probe";
    probe.innerHTML = this.renderer.messageHtml(message);
    this.renderer.labelCodeBlocks(probe);
    this.renderer.stampExpensiveKeys(probe);
    // Every DOM mutation below — morphdom attribute sync included — can fire
    // `toggle`, and none of those mutations carry reader intent.
    this.suppressReasoningRecording = true;
    try {
      morphdom(article, probe, {
        childrenOnly: true,
        getNodeKey: (node) => this.renderer.nodeKey(node),
        onBeforeElUpdated: (fromElement: Element, toElement: Element) => {
          TranscriptView.keepToolStepOpenState(fromElement, toElement);
          return this.renderer.beforeElUpdated(fromElement, toElement);
        },
      });
      this.reconcileReasoningDisclosure(article, message);
    } finally {
      this.suppressReasoningRecording = false;
    }
    this.renderer.highlightCodeBlocks(article);
    // Diagrams render once for finished messages; streamed ones show code until
    // the fence closes, exactly like code highlighting does.
    if (message.state === ChatMessageState.COMPLETE || message.state === ChatMessageState.STOPPED) {
      this.renderer.renderDiagrams(article);
    }
  }

  /** A tool-step row arrives collapsed from the template on every patch; the
   * reader's expand or collapse is carried over so it survives streaming. */
  private static keepToolStepOpenState(fromElement: Element, toElement: Element): void {
    if (fromElement instanceof HTMLDetailsElement && toElement instanceof HTMLDetailsElement && fromElement.classList.contains("tool-step")) {
      toElement.open = fromElement.open;
    }
  }

  /**
   * Re-applies the disclosure state morphdom would otherwise undo: the fresh
   * template always arrives with its own default (`open` while streaming,
   * collapsed afterwards), so without this step every streamed patch resets the
   * element and a manual expansion survives exactly one frame. Rules, in order:
   *
   * 1. While streaming, the reader's recorded choice wins; absent one, the
   *    disclosure starts open so the reasoning is visible as it arrives.
   * 2. The first render after streaming folds the reasoning away exactly once —
   *    the answer is ready, the scratch work is not needed — and clears any
   *    recorded streaming-time choice.
   * 3. After that one-time collapse the reader may reopen the reasoning and the
   *    choice sticks through every later re-render of the finished message.
   */
  private reconcileReasoningDisclosure(article: HTMLElement, message: TranscriptMessage): void {
    const disclosure = article.querySelector(
      `details.reasoning#${CSS.escape(reasoningDisclosureIdentifier(message.id))}`,
    );
    if (!(disclosure instanceof HTMLDetailsElement)) {
      return;
    }
    if (!this.disclosuresWithToggleListener.has(disclosure)) {
      disclosure.addEventListener("toggle", this.handleDisclosureToggle);
      this.disclosuresWithToggleListener.add(disclosure);
    }
    const isStreaming = message.state === ChatMessageState.STREAMING;
    if (!isStreaming && !this.terminalCollapseAppliedByMessageID.has(message.id)) {
      this.terminalCollapseAppliedByMessageID.add(message.id);
      this.reasoningOpenByMessageID.delete(message.id);
      if (disclosure.open) {
        disclosure.open = false;
      }
      return;
    }
    const recordedPreference: boolean | undefined = this.reasoningOpenByMessageID.get(message.id);
    const desiredOpen: boolean = recordedPreference != null ? recordedPreference : isStreaming;
    if (disclosure.open !== desiredOpen) {
      disclosure.open = desiredOpen;
    }
  }

  /**
   * The one thing this layer can always do without the rendering pipeline: show
   * the answer as plain text so a reader is never left with a blank pane.
   */
  private renderPlainTextFallback(message: TranscriptMessage): void {
    if (!message.id || this.articlesByMessageID.has(message.id)) {
      return;
    }
    const article = document.createElement("article");
    article.className = `message message--${message.role === ChatRole.USER ? "user" : "assistant"}`;
    article.dataset.messageId = message.id;
    this.markBubbleStableIdentifier(article, message);
    const body = document.createElement("div");
    body.className = "message__answer markdown-body";
    body.textContent = message.markdown;
    article.appendChild(body);
    this.inner.appendChild(article);
    this.articlesByMessageID.set(message.id, article);
  }

  // MARK: - Interaction

  /** Says "Copied" on the button for a moment. The pasteboard is the host's,
   * so nothing else on the page shows that the click did anything. */
  private static confirmCopied(copyButton: Element): void {
    const restingLabel = copyButton.getAttribute("data-resting-label") ?? copyButton.textContent ?? "Copy";
    copyButton.setAttribute("data-resting-label", restingLabel);
    copyButton.textContent = "Copied";
    setTimeout(() => {
      copyButton.textContent = restingLabel;
    }, COPIED_CONFIRMATION_MILLISECONDS);
  }

  private handleClick(event: MouseEvent): void {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    const actionButton = target.closest("button[data-action]");
    if (actionButton) {
      const article = actionButton.closest("[data-message-id]");
      const kind = actionButton.getAttribute("data-action");
      if (kind === "copy") {
        let messageID = "";
        if (article instanceof HTMLElement) {
          const storedMessageID: string | undefined = article.dataset.messageId;
          messageID = storedMessageID != null ? storedMessageID : "";
        }
        this.onAction(new TranscriptAction({ kind: "copy", messageID, detail: "" }));
        TranscriptView.confirmCopied(actionButton);
      }
      if (kind === SAVE_IMAGE_ACTION || kind === COPY_IMAGE_ACTION) {
        const storedImageName = actionButton.getAttribute("data-image-name") ?? "";
        this.onAction(
          new TranscriptAction({ kind: kind === SAVE_IMAGE_ACTION ? "saveImage" : "copyImage", messageID: "", detail: storedImageName }),
        );
        if (kind === COPY_IMAGE_ACTION) {
          TranscriptView.confirmCopied(actionButton);
        }
      }
      if (kind === EXPORT_ANSWER_ACTION) {
        let answerMessageID = "";
        if (article instanceof HTMLElement && article.dataset.messageId != null) {
          answerMessageID = article.dataset.messageId;
        }
        this.onAction(new TranscriptAction({ kind: "exportAnswer", messageID: answerMessageID, detail: "" }));
      }
      if (kind === SPEAK_ANSWER_ACTION) {
        const speakingMessageID = article instanceof HTMLElement ? article.dataset.messageId : undefined;
        this.onAction(new TranscriptAction({ kind: "speakAnswer", messageID: speakingMessageID ?? "", detail: "" }));
      }
      if (kind === COPY_CODE_ACTION) {
        const copiedCode = actionButton.closest(".code-block")?.querySelector("pre > code")?.textContent ?? "";
        this.onAction(new TranscriptAction({ kind: "copy", messageID: "", detail: copiedCode }));
        TranscriptView.confirmCopied(actionButton);
      }
      return;
    }
    const anchor = target.closest("a[href]");
    if (anchor) {
      event.preventDefault();
      const storedHref: string | null = anchor.getAttribute("href");
      const hrefDetail: string = storedHref != null ? storedHref : "";
      this.onAction(
        new TranscriptAction({ kind: "openExternal", messageID: "", detail: hrefDetail }),
      );
    }
  }
}
