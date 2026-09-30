import { ComposerState } from "./ComposerState";
import { ComposerAction } from "./ComposerAction";
import { ThinkingEffort } from "./ThinkingEffort";

const MAX_FIELD_HEIGHT_PX = 160;

/**
 * Composer layer: the ask field, the thinking-effort pill, and the failure
 * banner.
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
  private readonly effortPill: HTMLButtonElement;
  private readonly effortLabel: HTMLElement;
  private readonly effortMenu: HTMLElement;
  private readonly banner: HTMLElement;
  private readonly bannerMessage: HTMLElement;
  private readonly bannerNextAction: HTMLElement;
  private readonly onAction: (action: ComposerAction) => void;
  private state: ComposerState = new ComposerState({
    isStreaming: false,
    acceptsInput: true,
    isReady: false,
    effort: ThinkingEffort.QUICK,
    failure: null,
  });

  public constructor(init: {
    field: HTMLTextAreaElement;
    sendButton: HTMLButtonElement;
    stopButton: HTMLButtonElement;
    retryButton: HTMLButtonElement;
    effortPill: HTMLButtonElement;
    effortLabel: HTMLElement;
    effortMenu: HTMLElement;
    banner: HTMLElement;
    bannerMessage: HTMLElement;
    bannerNextAction: HTMLElement;
    onAction: (action: ComposerAction) => void;
  }) {
    this.field = init.field;
    this.sendButton = init.sendButton;
    this.stopButton = init.stopButton;
    this.retryButton = init.retryButton;
    this.effortPill = init.effortPill;
    this.effortLabel = init.effortLabel;
    this.effortMenu = init.effortMenu;
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
    this.effortPill.addEventListener("click", () => this.toggleEffortMenu());
    this.effortMenu.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const choice = target.closest("[data-effort-value]");
      if (!choice) {
        return;
      }
      this.closeEffortMenu();
      this.onAction(
        new ComposerAction({ kind: "setEffort", detail: (choice as HTMLElement).dataset.effortValue ?? "" }),
      );
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        this.closeEffortMenu();
      }
    });
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!this.effortMenu.hidden && target instanceof Element && !target.closest(".composer__menu-anchor")) {
        this.closeEffortMenu();
      }
    });
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
    this.effortLabel.textContent = ThinkingEffort.displayName(state.effort);
    this.effortPill.title = ThinkingEffort.budgetSummary(state.effort);
    this.buildEffortMenu();
    if (!state.isReady) {
      this.closeEffortMenu();
    }
    this.refreshControls();
  }

  public draft(): string {
    return this.field.value.trim();
  }

  // MARK: - Send rules

  private canSend(): boolean {
    return this.state.isReady && !this.state.isStreaming && this.draft().length > 0;
  }

  private refreshControls(): void {
    const streaming = this.state.isStreaming;
    this.sendButton.hidden = streaming;
    this.stopButton.hidden = !streaming;
    this.stopButton.disabled = !streaming;
    this.field.disabled = !this.state.acceptsInput;
    // The placeholder keeps the field discoverable while disabled; the send rule
    // needs the draft, so it is evaluated from the live value.
    this.sendButton.disabled = !this.canSend();
    this.growField();
  }

  private growField(): void {
    this.field.style.height = "auto";
    const grownHeight = Math.min(this.field.scrollHeight, MAX_FIELD_HEIGHT_PX);
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

  // MARK: - Effort menu

  private closeEffortMenu(): void {
    this.effortMenu.hidden = true;
    this.effortPill.setAttribute("aria-expanded", "false");
  }

  private toggleEffortMenu(): void {
    const opening = this.effortMenu.hidden;
    this.effortMenu.hidden = !opening;
    this.effortPill.setAttribute("aria-expanded", opening ? "true" : "false");
  }

  private buildEffortMenu(): void {
    this.effortMenu.textContent = "";
    ThinkingEffort.ALL.forEach((option) => {
      const item = document.createElement("li");
      item.setAttribute("role", "none");
      const choice = document.createElement("button");
      choice.type = "button";
      choice.setAttribute("role", "menuitemradio");
      choice.setAttribute("aria-checked", option === this.state.effort ? "true" : "false");
      choice.dataset.testid = `composer-effort-${option}`;
      choice.dataset.effortValue = option;
      choice.textContent = ThinkingEffort.budgetSummary(option);
      item.appendChild(choice);
      this.effortMenu.appendChild(item);
    });
  }
}
