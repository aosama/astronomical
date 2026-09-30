import { marked } from "marked";
import hljs from "highlight.js/lib/common";
import { TrustBoundary } from "./TrustBoundary";
import { MathPipeline } from "./MathPipeline";
import type { MathSpan } from "./MathPipeline";
import { DiagramPipeline } from "./DiagramPipeline";
import { ContentKey } from "./ContentKey";
import { TranscriptMessage } from "./TranscriptMessage";
import type { TranscriptAttachment } from "./TranscriptAttachment";
import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";

const VISIBLE_STATE_LABELS: Record<string, string> = {
  streaming: "generating",
  stopped: "stopped",
  failed: "failed",
};

/**
 * Answer composition for the conversation canvas.
 *
 * This class answers one question: given a message payload, what HTML may the
 * transcript insert? It composes that HTML from four collaborators — the trust
 * boundary, the maths pipeline, the diagram pipeline, and the content keys that
 * let finished blocks survive diffing — and owns the expensive-block rules for
 * code. It never parses conversation state or talks to the Swift bridge.
 */
export class Renderer {
  private readonly trust: TrustBoundary;
  private readonly math: MathPipeline;
  private readonly diagrams: DiagramPipeline;

  public constructor(init: {
    trust: TrustBoundary;
    math: MathPipeline;
    diagrams: DiagramPipeline;
  }) {
    this.trust = init.trust;
    this.math = init.math;
    this.diagrams = init.diagrams;
    marked.setOptions({ gfm: true, breaks: false });
  }

  public messageHtml(message: TranscriptMessage): string {
    const reasoning = message.reasoning
      ? `<details class="reasoning"><summary>Reasoning</summary><div class="reasoning__body">${this.trust.escapeHtml(message.reasoning)}</div></details>`
      : "";
    const answerBody = this.renderMarkdown(message.markdown);
    const attachments = this.attachmentStripHtml(message.attachments);
    if (message.role === ChatRole.USER) {
      // The sender label lives inside the user bubble so stacked user turns
      // read as sent messages instead of extra compose fields.
      return (
        `<div class="message__answer message__answer--user">` +
        `<span class="message__role">You</span>` +
        `<div class="markdown-body">${answerBody}</div>` +
        attachments +
        `</div>` +
        this.actionsHtml(message)
      );
    }
    return (
      `<div class="message__meta">` +
      `<span class="message__role">Assistant</span>` +
      `<span class="message__state">${this.stateLabel(message.state)}</span>` +
      `</div>` +
      `<div class="message__answer">` +
      reasoning +
      `<div class="markdown-body">${answerBody}</div>` +
      attachments +
      `</div>` +
      this.actionsHtml(message)
    );
  }

  private stateLabel(state: ChatMessageState): string {
    if (state === ChatMessageState.STREAMING) {
      return 'generating<span class="message__streaming-caret"></span>';
    }
    return VISIBLE_STATE_LABELS[state] ?? "";
  }

  private attachmentStripHtml(attachments: readonly TranscriptAttachment[]): string {
    if (attachments.length === 0) {
      return "";
    }
    const items = attachments
      .map(
        (attachment) =>
          `<span class="attachment-strip__item" title="${this.trust.escapeHtml(attachment.label)}">` +
          `<img alt="${this.trust.escapeHtml(attachment.label || "attachment")}" src="${this.trust.escapeHtml(attachment.assetURL)}">` +
          `</span>`,
      )
      .join("");
    const badge =
      attachments.length > 1
        ? `<span class="attachment-strip__badge" data-testid="attachment-badge">${attachments.length} images</span>`
        : "";
    return `<div class="attachment-strip" data-testid="attachment-strip">${items}${badge}</div>`;
  }

  private actionsHtml(message: TranscriptMessage): string {
    if (message.state !== ChatMessageState.COMPLETE && message.state !== ChatMessageState.STOPPED) {
      return "";
    }
    return `<div class="message__actions"><button class="message__action" data-testid="action-copy" data-action="copy">Copy</button></div>`;
  }

  /**
   * One rule set for streaming and snapshots alike: maths is carved out before
   * marked sees the text, the surviving prose is sanitised, and the rendered maths
   * is grafted back into that sanitised result.
   */
  private renderMarkdown(markdown: string): string {
    const source = typeof markdown === "string" ? markdown : "";
    if (!source) {
      return "";
    }
    const collectedMathSpans: MathSpan[] = [];
    try {
      const tokenedSource = this.math.tokenize(source, collectedMathSpans);
      const parsed = marked.parse(tokenedSource);
      if (typeof parsed !== "string") {
        throw new Error("marked returned an asynchronous parse");
      }
      const sanitisedHtml = this.trust.sanitizeHtml(parsed);
      return this.math.graft(sanitisedHtml, collectedMathSpans);
    } catch {
      // A partial answer can defeat the parser mid-stream. Showing the raw text
      // keeps the conversation readable instead of dropping the message body.
      return this.trust.sanitizeHtml(`<p>${this.trust.escapeHtml(source)}</p>`);
    }
  }

  public highlightCodeBlocks(root: ParentNode): void {
    root.querySelectorAll("pre > code").forEach((codeBlock) => {
      const codeElement = codeBlock as HTMLElement;
      if (codeElement.dataset.highlighted === "true" || codeElement.dataset.highlighted === "skipped") {
        return;
      }
      const language = (codeElement.className.match(/language-([\w+#-]+)/) ?? [])[1];
      if (!language || !hljs.getLanguage(language)) {
        codeElement.dataset.highlighted = "skipped";
        return;
      }
      codeElement.innerHTML = hljs.highlight(codeElement.textContent ?? "", { language }).value;
      codeElement.dataset.highlighted = "true";
      codeElement.classList.add("hljs");
    });
  }

  public labelCodeBlocks(root: ParentNode): void {
    root.querySelectorAll("pre > code").forEach((codeBlock) => {
      const pre = codeBlock.parentElement;
      if (
        !pre ||
        !pre.parentElement ||
        pre.parentElement.classList.contains("code-block") ||
        pre.parentElement.dataset.mdKey
      ) {
        return;
      }
      const language = ((codeBlock as HTMLElement).className.match(/language-([\w+#-]+)/) ?? [])[1];
      if (!language) {
        return;
      }
      const codeBlockWrapper = document.createElement("div");
      codeBlockWrapper.className = "code-block";
      const languageLabel = document.createElement("span");
      languageLabel.className = "code-block__label";
      languageLabel.textContent = language;
      pre.parentElement.insertBefore(codeBlockWrapper, pre);
      codeBlockWrapper.appendChild(pre);
      codeBlockWrapper.appendChild(languageLabel);
    });
  }

  /**
   * Expensive blocks are stamped with a stable key derived from their content, so
   * the transcript's diffing keeps the existing node when nothing about that block
   * changed. That is what lets a streamed answer stay rendered: highlighting,
   * expanded details, and installed diagrams are not recomputed while unrelated
   * prose updates around them.
   */
  public stampExpensiveKeys(root: ParentNode): void {
    root.querySelectorAll("div.code-block").forEach((codeBlockWrapper) => {
      const wrapper = codeBlockWrapper as HTMLElement;
      const codeBlock = wrapper.querySelector("code");
      if (codeBlock && (codeBlock as HTMLElement).dataset.mdKey) {
        wrapper.dataset.mdKey = (codeBlock as HTMLElement).dataset.mdKey;
        return;
      }
      wrapper.dataset.mdKey = `code-${ContentKey.hash(codeBlock ? codeBlock.textContent : "")}`;
    });
  }

  public nodeKey(node: Node): string {
    if (!(node instanceof Element)) {
      return "";
    }
    return node.getAttribute("data-md-key") || node.id || "";
  }

  /**
   * Reuses the existing subtree when a keyed block is textually identical, which
   * preserves its rendered form instead of re-highlighting or collapsing it.
   */
  public beforeElUpdated(fromElement: Element, toElement: Element): boolean {
    const fromKey = this.nodeKey(fromElement);
    const toKey = this.nodeKey(toElement);
    if (fromKey && fromKey === toKey && fromElement.textContent === toElement.textContent) {
      return false;
    }
    return true;
  }

  /** Diagrams render once per finished message; the pipeline owns the policy. */
  public renderDiagrams(root: ParentNode): void {
    this.diagrams.render(root);
  }
}
