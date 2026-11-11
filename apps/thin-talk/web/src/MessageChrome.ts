import { ChatMessageState } from "./ChatMessageState";
import type { TranscriptMessage } from "./TranscriptMessage";
import { ToolStepState } from "./ToolStep";
import type { ToolStep } from "./ToolStep";

/** Escapes text that came from a model, a tool or the person before it is
 * placed in markup. */
export type EscapeHtml = (text: string) => string;

/**
 * The parts of a message that name who is talking and what happened, around
 * the answer itself: the sender labels, the assistant's header, the reasoning
 * disclosure's summary and each tool step's summary. The Renderer asks one
 * chrome for all of them, picked once when the persona changes, so classic
 * Thin Talk and Astro never mix inside one transcript.
 */
export interface MessageChrome {
  userRoleHtml(): string;
  assistantHeaderHtml(message: TranscriptMessage): string;
  reasoningSummary(): string;
  /** Extra attributes for a tool step's <details>, with a leading space, or "". */
  toolStepAttributes(toolStep: ToolStep): string;
  /** `imageModelSuffix` is already escaped. */
  toolStepSummaryHtml(toolStep: ToolStep, imageModelSuffix: string): string;
}

/** The outcome shown after a tool step's label. */
const TOOL_STEP_OUTCOME_LABELS: Record<ToolStepState, string> = {
  [ToolStepState.RUNNING]: "running",
  [ToolStepState.SUCCEEDED]: "done",
  [ToolStepState.FAILED]: "failed",
};

const VISIBLE_STATE_LABELS: Record<string, string> = {
  streaming: "generating",
  stopped: "stopped",
  failed: "failed",
};

/**
 * An answer's status words, with the words to show instead while the engine
 * loads a model, when there are any. Both are in the markup and the
 * stylesheet shows one: the load starts and ends on the engine's clock, and
 * swapping by a class on the transcript changes the words without drawing
 * the conversation again. Both arguments are fixed text, never model output.
 */
export function stateWordsHtml(stateWords: string, modelLoadingWords: string | null): string {
  if (modelLoadingWords === null) {
    return stateWords;
  }
  return `<span class="message__state-words">${stateWords}</span><span class="message__loading-words">${modelLoadingWords}</span>`;
}

/** Plain Thin Talk: "You" and "Assistant", and tool steps by their labels. */
export class ClassicMessageChrome implements MessageChrome {
  private readonly escapeHtml: EscapeHtml;

  public constructor(escapeHtml: EscapeHtml) {
    this.escapeHtml = escapeHtml;
  }

  public userRoleHtml(): string {
    return `<span class="message__role">You</span>`;
  }

  public assistantHeaderHtml(message: TranscriptMessage): string {
    return (
      `<div class="message__meta">` +
      `<span class="message__role">Assistant</span>` +
      `<span class="message__state">${ClassicMessageChrome.stateLabel(message)}</span>` +
      `</div>`
    );
  }

  public reasoningSummary(): string {
    return "Reasoning";
  }

  public toolStepAttributes(): string {
    return "";
  }

  public toolStepSummaryHtml(toolStep: ToolStep, imageModelSuffix: string): string {
    return `<summary>${this.escapeHtml(toolStep.label)}: ${TOOL_STEP_OUTCOME_LABELS[toolStep.state]}${imageModelSuffix}</summary>`;
  }

  private static stateLabel(message: TranscriptMessage): string {
    if (message.state === ChatMessageState.STREAMING) {
      const hasShownNothing = message.markdown.length === 0;
      return `${stateWordsHtml("generating", hasShownNothing ? "loading the model…" : null)}<span class="message__streaming-caret"></span>`;
    }
    const configuredLabel: string | undefined = VISIBLE_STATE_LABELS[message.state];
    return configuredLabel != null ? configuredLabel : "";
  }
}
