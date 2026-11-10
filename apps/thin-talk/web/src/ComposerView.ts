import { ComposerState } from "./ComposerState";
import { ComposerAction } from "./ComposerAction";
import { ModelPanelView } from "./ModelPanelView";
import type { ComposerFriendlyNames } from "./ComposerFriendlyNames";

/** The field's hint while an answer streams: a message sent now is read as
 * soon as the model can take it, after the current reply or its tool calls. */
const QUEUEING_PLACEHOLDER = "Add a message; Thin Talk reads it next";
const MAX_FIELD_HEIGHT_PX = 160;
const FIELD_HEIGHT_ROUNDING_ALLOWANCE_PX = 2;

/**
 * Composer layer: the ask field, the brain control and its panel, and the
 * failure banner.
 *
 * This class renders the state the controller pushes and reports every intent
 * back; it owns no chat behaviour and holds no chat state beyond the field's
 * in-progress text, the same way a native field owns its caret.
 */
export class ComposerView {
  private readonly field: HTMLTextAreaElement;
  private readonly sendButton: HTMLButtonElement;
  private readonly stopButton: HTMLButtonElement;
  private readonly retryButton: HTMLButtonElement;
  private readonly modelPanel: ModelPanelView;
  /** Plain words for the brain control, or null for the models' own names. */
  private friendlyNames: ComposerFriendlyNames | null = null;
  /** The hint shown in the empty field while no answer is streaming: the
   * markup's wording, or what setIdlePlaceholder asked for (Astro's). */
  private idlePlaceholder: string;
  /** The markup's own wording, which the Focus switch restores. */
  private readonly markupPlaceholder: string;
  private readonly banner: HTMLElement;
  private readonly bannerMessage: HTMLElement;
  private readonly bannerNextAction: HTMLElement;
  private readonly onAction: (action: ComposerAction) => void;
  private state: ComposerState = new ComposerState({
    isStreaming: false,
    acceptsInput: true,
    isReady: false,
    models: [],
    selectedModelID: null,
    failure: null,
  });

  public constructor(init: {
    field: HTMLTextAreaElement;
    sendButton: HTMLButtonElement;
    stopButton: HTMLButtonElement;
    retryButton: HTMLButtonElement;
    brainControl: HTMLButtonElement;
    brainLabel: HTMLElement;
    brainPanel: HTMLElement;
    banner: HTMLElement;
    bannerMessage: HTMLElement;
    bannerNextAction: HTMLElement;
    onAction: (action: ComposerAction) => void;
  }) {
    this.field = init.field;
    this.idlePlaceholder = init.field.placeholder;
    this.markupPlaceholder = init.field.placeholder;
    this.sendButton = init.sendButton;
    this.stopButton = init.stopButton;
    this.retryButton = init.retryButton;
    this.modelPanel = new ModelPanelView({
      trigger: init.brainControl,
      label: init.brainLabel,
      panel: init.brainPanel,
      onAction: (action) => this.onAction(action),
    });
    this.banner = init.banner;
    this.bannerMessage = init.bannerMessage;
    this.bannerNextAction = init.bannerNextAction;
    this.onAction = init.onAction;

    this.field.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey) {
        return;
      }
      event.preventDefault();
      this.reportSend();
    });
    this.field.addEventListener("input", () => this.refreshControls());
    this.sendButton.addEventListener("click", () => this.reportSend());
    this.stopButton.addEventListener("click", () => this.onAction(new ComposerAction({ kind: "stop" })));
    this.retryButton.addEventListener("click", () => this.onAction(new ComposerAction({ kind: "retry" })));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        this.modelPanel.close();
      }
    });
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (this.modelPanel.isOpen && target instanceof Element && !target.closest(".composer__menu-anchor")) {
        this.modelPanel.close();
      }
    });
  }

  /** Switches the brain control between friendly words (Astro's Explorer
   * style) and the models' own names, and redraws it at once. */
  public setFriendlyNames(friendlyNames: ComposerFriendlyNames | null): void {
    this.friendlyNames = friendlyNames;
    this.render(this.state);
  }

  public render(state: ComposerState): void {
    this.state = state;
    if (state.failure) {
      this.bannerMessage.textContent = state.failure.message;
      this.bannerNextAction.textContent = state.failure.nextAction;
      this.banner.hidden = false;
    } else {
      this.banner.hidden = true;
    }
    this.modelPanel.render(state, this.friendlyNames !== null ? this.friendlyNames.brain : null);
    this.refreshControls();
  }

  public draft(): string {
    return this.field.value.trim();
  }

  // MARK: - Send rules

  /** A message can be sent while an answer is still coming; it is queued
   * and answered next. */
  private canSend(): boolean {
    return this.state.isReady && this.draft().length > 0;
  }

  /** Changes what the empty field says when idle: Astro's home asks for "Ask
   * Astro anything", and the Focus switch restores the original. The queueing
   * placeholder still wins while an answer streams. */
  public setIdlePlaceholder(idlePlaceholder: string): void {
    this.idlePlaceholder = idlePlaceholder;
    this.refreshControls();
  }

  /** The placeholder the page's markup gives the field. */
  public get markupIdlePlaceholder(): string {
    return this.markupPlaceholder;
  }

  private refreshControls(): void {
    const streaming = this.state.isStreaming;
    this.sendButton.hidden = streaming;
    this.stopButton.hidden = !streaming;
    this.stopButton.disabled = !streaming;
    this.field.disabled = !this.state.acceptsInput;
    // While an answer streams, Return queues the message; the placeholder says
    // so, since the send button gives way to Stop.
    this.field.placeholder = streaming ? QUEUEING_PLACEHOLDER : this.idlePlaceholder;
    // The placeholder keeps the field discoverable while disabled; the send rule
    // needs the draft, so it is evaluated from the live value.
    this.sendButton.disabled = !this.canSend();
    this.growField();
  }

  private growField(): void {
    this.field.style.height = "auto";
    // scrollHeight is rounded to whole pixels while the line height is not
    // (18px × 1.62), so a field sized to it exactly can shave the bottom of
    // the last line; the spare pixels keep descenders whole.
    const grownHeight = Math.min(this.field.scrollHeight + FIELD_HEIGHT_ROUNDING_ALLOWANCE_PX, MAX_FIELD_HEIGHT_PX);
    this.field.style.height = `${grownHeight}px`;
    // The field scrolls only once the growth ceiling clamps it; a field that
    // still fits must never paint a scrollbar over the ask.
    this.field.style.overflowY = this.field.scrollHeight > grownHeight ? "auto" : "hidden";
  }

  private reportSend(): void {
    if (!this.canSend()) {
      return;
    }
    const ask = this.draft();
    this.field.value = "";
    this.refreshControls();
    this.onAction(new ComposerAction({ kind: "send", detail: ask }));
  }
}
