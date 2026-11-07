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
import { reasoningDisclosureIdentifier } from "./ReasoningDisclosure";
import { ToolStep, ToolStepState } from "./ToolStep";

/** The outcome shown after a tool step's label. */
const TOOL_STEP_OUTCOME_LABELS: Record<ToolStepState, string> = {
  [ToolStepState.RUNNING]: "running",
  [ToolStepState.SUCCEEDED]: "done",
  [ToolStepState.FAILED]: "failed",
};

/** The longest prompt excerpt used in an image's text alternative; a screen
 * reader reads the whole attribute, and prompts can run to paragraphs. */
const MAXIMUM_IMAGE_DESCRIPTION_PROMPT_CHARACTERS = 250;

/** The image's text alternative: it says the image was generated and what was
 * asked for, since nothing describes what was actually drawn. */
function generatedImageDescription(argumentsJSON: string): string {
  let prompt: unknown = null;
  try {
    const decodedArguments: unknown = JSON.parse(argumentsJSON);
    prompt = typeof decodedArguments === "object" && decodedArguments !== null ? (decodedArguments as Record<string, unknown>)["prompt"] : null;
  } catch {
    prompt = null;
  }
  if (typeof prompt !== "string" || prompt.trim().length === 0) {
    return "Generated image";
  }
  const trimmedPrompt = prompt.trim();
  const promptExcerpt =
    trimmedPrompt.length > MAXIMUM_IMAGE_DESCRIPTION_PROMPT_CHARACTERS
      ? `${trimmedPrompt.slice(0, MAXIMUM_IMAGE_DESCRIPTION_PROMPT_CHARACTERS)}…`
      : trimmedPrompt;
  return `Generated image: ${promptExcerpt}`;
}

/** The stable id of one tool-step row, keyed on the message and the call so
 * the transcript and automation agents find the same row across patches. */
export function toolStepIdentifier(messageID: string, toolCallID: string): string {
  return `tool-step:${messageID}:${toolCallID}`;
}

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
      ? this.reasoningHtml(message)
      : "";
    const answerBody = this.renderMarkdown(message.markdown);
    const attachments = this.attachmentStripHtml(message.attachments);
    const toolSteps = this.toolStepsHtml(message) + this.generatedImagesHtml(message);
    if (message.role === ChatRole.USER) {
      // The sender label lives inside the user bubble so stacked user turns
      // read as sent messages instead of extra compose fields.
      return (
        `<div class="message__answer message__answer--user">` +
        `<span class="message__role">You</span>` +
        this.userStateHtml(message.state) +
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
      toolSteps +
      `<div class="markdown-body">${answerBody}</div>` +
      attachments +
      `</div>` +
      this.actionsHtml(message)
    );
  }

  /** A user message shows its state only when it was not simply sent: still
   * queued behind an answer, or never sent because the turn stopped. */
  private userStateHtml(state: ChatMessageState): string {
    if (state === ChatMessageState.QUEUED) {
      return `<span class="message__state" data-testid="message-state-queued">queued</span>`;
    }
    if (state === ChatMessageState.FAILED) {
      return `<span class="message__state" data-testid="message-state-not-sent">not sent</span>`;
    }
    return "";
  }

  private stateLabel(state: ChatMessageState): string {
    if (state === ChatMessageState.STREAMING) {
      return 'generating<span class="message__streaming-caret"></span>';
    }
    const configuredLabel: string | undefined = VISIBLE_STATE_LABELS[state];
    return configuredLabel != null ? configuredLabel : "";
  }

  /**
   * The reasoning disclosure carries a stable id plus a default-open state while
   * the message is streaming. The id is what lets TranscriptView re-apply the
   * reader's expand/collapse preference after each streamed patch; without it
   * morphdom would re-apply this template's collapsed form and clobber every
   * manual expansion the moment the next delta arrives.
   */
  private reasoningHtml(message: TranscriptMessage): string {
    const disclosureIdentifier = reasoningDisclosureIdentifier(message.id);
    const openAttribute = message.state === ChatMessageState.STREAMING ? " open" : "";
    return (
      `<details class="reasoning" id="${disclosureIdentifier}" data-testid="${disclosureIdentifier}"${openAttribute}>` +
      `<summary>Reasoning</summary>` +
      `<div class="reasoning__body">${this.trust.escapeHtml(message.reasoning ?? "")}</div>` +
      `</details>`
    );
  }

  /** One collapsed row per tool call: the label and outcome in the summary,
   * the arguments and result inside. Every value is escaped, since both come
   * from the model or from a tool, never from trusted markup. */
  private toolStepsHtml(message: TranscriptMessage): string {
    if (message.toolSteps.length === 0) {
      return "";
    }
    const toolStepRows = message.toolSteps.map((toolStep) => this.toolStepHtml(message.id, toolStep)).join("");
    return `<div class="tool-steps" data-testid="tool-steps:${this.trust.escapeHtml(message.id)}">${toolStepRows}</div>`;
  }

  /** The images a message's tool steps generated, in call order. */
  private generatedImagesHtml(message: TranscriptMessage): string {
    return message.toolSteps
      .filter((toolStep) => toolStep.generatedImageName != null)
      .map((toolStep) => this.generatedImageHtml(toolStep))
      .join("");
  }

  /** A generated image is part of the answer, so it is shown outside the
   * collapsed rows. Its source is the app's own `generated` host; the name was
   * checked against the stored-name pattern when the step was built. */
  private generatedImageHtml(toolStep: ToolStep): string {
    const imageSource = `thintalk-asset://generated/${toolStep.generatedImageName ?? ""}`;
    if (!this.trust.isAllowedImageSource(imageSource)) {
      return "";
    }
    const imageDescription = this.trust.escapeHtml(generatedImageDescription(toolStep.argumentsJSON));
    return (
      `<figure class="generated-image" data-testid="generated-image:${this.trust.escapeHtml(toolStep.toolCallID)}">` +
      `<img src="${this.trust.escapeHtml(imageSource)}" alt="${imageDescription}" loading="lazy">` +
      `</figure>`
    );
  }

  private toolStepHtml(messageID: string, toolStep: ToolStep): string {
    const stepIdentifier = this.trust.escapeHtml(toolStepIdentifier(messageID, toolStep.toolCallID));
    const argumentsText = toolStep.argumentsJSON.length === 0 || toolStep.argumentsJSON === "{}" ? "None" : toolStep.argumentsJSON;
    const resultText = toolStep.state === ToolStepState.RUNNING ? "Waiting for the result." : toolStep.resultText;
    return (
      `<details class="tool-step" id="${stepIdentifier}" data-testid="${stepIdentifier}" data-state="${toolStep.state}">` +
      `<summary>${this.trust.escapeHtml(toolStep.label)}: ${TOOL_STEP_OUTCOME_LABELS[toolStep.state]}</summary>` +
      `<div class="tool-step__body">` +
      `<span>Arguments</span><pre>${this.trust.escapeHtml(argumentsText)}</pre>` +
      `<span>Result</span><pre>${this.trust.escapeHtml(resultText)}</pre>` +
      `</div>` +
      `</details>`
    );
  }

  private attachmentStripHtml(attachments: readonly TranscriptAttachment[]): string {
    if (attachments.length === 0) {
      return "";
    }
    const items = attachments.map((attachment) => this.attachmentItemHtml(attachment)).join("");
    const badge =
      attachments.length > 1
        ? `<span class="attachment-strip__badge" data-testid="attachment-badge">${attachments.length} images</span>`
        : "";
    return `<div class="attachment-strip" data-testid="attachment-strip">${items}${badge}</div>`;
  }

  /** One attachment chip. The thumbnail only renders when its source passes
   * the trust boundary's single image-scheme rule; a disallowed source falls
   * back to its escaped label so conversation text can never nominate a
   * remote or otherwise disallowed image for loading. */
  private attachmentItemHtml(attachment: TranscriptAttachment): string {
    const label = this.trust.escapeHtml(attachment.label || "attachment");
    if (!this.trust.isAllowedImageSource(attachment.assetURL)) {
      return `<span class="attachment-strip__item" title="${label}">${label}</span>`;
    }
    return (
      `<span class="attachment-strip__item" title="${label}">` +
      `<img alt="${label}" src="${this.trust.escapeHtml(attachment.assetURL)}">` +
      `</span>`
    );
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
      const languageMatch: RegExpMatchArray | null = codeElement.className.match(/language-([\w+#-]+)/);
      const language = languageMatch != null ? languageMatch[1] : undefined;
      if (!language || !hljs.getLanguage(language)) {
        codeElement.dataset.highlighted = "skipped";
        return;
      }
      const codeText: string | null = codeElement.textContent;
      const highlightSource: string = codeText != null ? codeText : "";
      codeElement.innerHTML = hljs.highlight(highlightSource, { language }).value;
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
      const languageMatch: RegExpMatchArray | null =
        (codeBlock as HTMLElement).className.match(/language-([\w+#-]+)/);
      const language = languageMatch != null ? languageMatch[1] : undefined;
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
