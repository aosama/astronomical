import { Renderer } from "./Renderer";
import { TranscriptMessage } from "./TranscriptMessage";
import { TranscriptAction } from "./TranscriptAction";
import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";
import morphdom from "morphdom";

/** The transcript half of one surface refresh. */
export interface TranscriptSnapshot {
  messages: readonly TranscriptMessage[];
  notice: string | null;
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
  private pendingMessages: TranscriptMessage[] = [];
  private flushRequested = false;
  private followScroll = true;

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

  /** Renders one snapshot: removes articles the snapshot no longer carries,
   * patches the rest, and updates the empty state. */
  public render(snapshot: TranscriptSnapshot): void {
    const incomingIDs = new Set(snapshot.messages.map((message) => message.id));
    for (const [messageID, article] of this.articlesByMessageID) {
      if (!incomingIDs.has(messageID)) {
        this.inner.removeChild(article);
        this.articlesByMessageID.delete(messageID);
      }
    }
    this.pendingMessages.push(...snapshot.messages);
    this.emptyState.hidden = snapshot.messages.length > 0;
    if (this.emptyStateNotice) {
      this.emptyStateNotice.textContent = snapshot.messages.length > 0 ? "" : snapshot.notice ?? "";
    }
    this.scheduleFlush();
  }

  public messageCount(): number {
    return this.articlesByMessageID.size;
  }

  public renderedText(messageID: string): string {
    return this.articlesByMessageID.get(messageID)?.textContent ?? "";
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
    const followNow = wasFollowing || batch.some((message) => message.role === ChatRole.USER);
    if (followNow) {
      this.scrollToBottom();
    }
  }

  private shouldFollowScroll(): boolean {
    return this.transcript.scrollHeight - this.transcript.scrollTop - this.transcript.clientHeight < 120;
  }

  private scrollToBottom(): void {
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  // MARK: - Messages

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
      this.inner.appendChild(article);
      this.articlesByMessageID.set(message.id, article);
    }
    article.className = `message message--${message.role === ChatRole.USER ? "user" : "assistant"}`;
    const probe = document.createElement("div");
    probe.className = "message__probe";
    probe.innerHTML = this.renderer.messageHtml(message);
    this.renderer.labelCodeBlocks(probe);
    this.renderer.stampExpensiveKeys(probe);
    morphdom(article, probe, {
      childrenOnly: true,
      getNodeKey: (node) => this.renderer.nodeKey(node),
      onBeforeElUpdated: (fromElement: Element, toElement: Element) =>
        this.renderer.beforeElUpdated(fromElement, toElement),
    });
    this.renderer.highlightCodeBlocks(article);
    // Diagrams render once for finished messages; streamed ones show code until
    // the fence closes, exactly like code highlighting does.
    if (message.state === ChatMessageState.COMPLETE || message.state === ChatMessageState.STOPPED) {
      this.renderer.renderDiagrams(article);
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
    const body = document.createElement("div");
    body.className = "message__answer markdown-body";
    body.textContent = message.markdown;
    article.appendChild(body);
    this.inner.appendChild(article);
    this.articlesByMessageID.set(message.id, article);
  }

  // MARK: - Interaction

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
        const messageID =
          article instanceof HTMLElement ? (article.dataset.messageId ?? "") : "";
        this.onAction(new TranscriptAction({ kind: "copy", messageID, detail: "" }));
      }
      return;
    }
    const anchor = target.closest("a[href]");
    if (anchor) {
      event.preventDefault();
      this.onAction(
        new TranscriptAction({ kind: "openExternal", messageID: "", detail: anchor.getAttribute("href") ?? "" }),
      );
    }
  }
}
