import { ComposerState } from "./ComposerState";
import { ComposerAction } from "./ComposerAction";
import { ImageModelPillView } from "./ImageModelPillView";
import type { ComposerFriendlyNames } from "./ComposerFriendlyNames";

/** The field's hint while an answer streams: a message sent now is read as
 * soon as the model can take it, after the current reply or its tool calls. */
const QUEUEING_PLACEHOLDER = "Add a message; Thin Talk reads it next";
const MAX_FIELD_HEIGHT_PX = 160;
const FIELD_HEIGHT_ROUNDING_ALLOWANCE_PX = 2;

/**
 * Composer layer: the ask field, the model and picture pills, and the
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
  private readonly modelPill: HTMLButtonElement;
  private readonly modelLabel: HTMLElement;
  private readonly modelMenu: HTMLElement;
  private readonly imageModelPill: ImageModelPillView;
  /** Plain words for the pills, or null for the models' own names. */
  private friendlyNames: ComposerFriendlyNames | null = null;
  /** The hint shown in the empty field while no answer is streaming: the
   * markup's wording, or what setIdlePlaceholder asked for (Astro's). */
  private idlePlaceholder: string;
  /** The markup's own wording, which the Focus switch restores. */
  private readonly markupPlaceholder: string;
  /** What the model menu last drew, so an unchanged menu is not rebuilt. */
  private renderedModelMenuSignature = "";
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
    modelPill: HTMLButtonElement;
    modelLabel: HTMLElement;
    modelMenu: HTMLElement;
    imageModelPill: HTMLButtonElement;
    imageModelLabel: HTMLElement;
    imageModelMenu: HTMLElement;
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
    this.modelPill = init.modelPill;
    this.modelLabel = init.modelLabel;
    this.modelMenu = init.modelMenu;
    this.imageModelPill = new ImageModelPillView({
      pill: init.imageModelPill,
      label: init.imageModelLabel,
      menu: init.imageModelMenu,
      onOpen: () => {
        this.closeModelMenu();
        // A model installed since the list was read appears once it is re-read.
        this.onAction(new ComposerAction({ kind: "openImageModels" }));
      },
      onSelect: (imageModelID) => this.onAction(new ComposerAction({ kind: "selectImageModel", detail: imageModelID })),
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
    this.modelPill.addEventListener("click", () => this.toggleModelMenu());
    this.modelMenu.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const toolsSwitch = target.closest("[data-tools-next-state]");
      if (toolsSwitch instanceof HTMLButtonElement && !toolsSwitch.disabled) {
        this.closeModelMenu();
        const nextToolsState: string | undefined = toolsSwitch.dataset.toolsNextState;
        this.onAction(new ComposerAction({ kind: "setTools", detail: nextToolsState != null ? nextToolsState : "on" }));
        return;
      }
      const choice = target.closest("[data-model-value]");
      if (!choice) {
        return;
      }
      this.closeModelMenu();
      const storedModelValue: string | undefined = (choice as HTMLElement).dataset.modelValue;
      const modelDetail: string = storedModelValue != null ? storedModelValue : "";
      this.onAction(
        new ComposerAction({ kind: "selectModel", detail: modelDetail }),
      );
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        this.closeModelMenu();
        this.imageModelPill.close();
      }
    });
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!this.modelMenu.hidden && target instanceof Element && !target.closest(".composer__menu-anchor")) {
        this.closeModelMenu();
      }
      if (this.imageModelPill.isOpen && target instanceof Element && !target.closest(".composer__menu-anchor")) {
        this.imageModelPill.close();
      }
    });
  }

  /** Switches the pills between friendly words (Astro's Explorer style) and
   * the models' own names, and redraws them at once. */
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
    this.renderModelControls(state);
    this.imageModelPill.render(state.imageModels, state.selectedImageModelID, state.toolsEnabled && state.toolsAvailable, this.friendlyNames?.sketchbook ?? null);
    if (!state.isReady) {
      this.closeModelMenu();
      this.imageModelPill.close();
    }
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

  // MARK: - Model menu

  private closeModelMenu(): void {
    this.modelMenu.hidden = true;
    this.modelPill.setAttribute("aria-expanded", "false");
  }

  private toggleModelMenu(): void {
    this.imageModelPill.close();
    const opening = this.modelMenu.hidden;
    this.modelMenu.hidden = !opening;
    this.modelPill.setAttribute("aria-expanded", opening ? "true" : "false");
  }

  private renderModelControls(state: ComposerState): void {
    const hasModels = state.models.length > 0;
    this.modelPill.hidden = !hasModels;
    if (!hasModels) {
      this.modelMenu.hidden = true;
      this.modelPill.setAttribute("aria-expanded", "false");
    }
    const selectedModel = state.models.find((model) => model.id === state.selectedModelID);
    if (selectedModel) {
      this.modelLabel.textContent = this.friendlyNames !== null ? this.friendlyNames.brain : selectedModel.name;
      this.modelPill.title = selectedModel.id;
    } else {
      this.modelLabel.textContent = "Model";
      this.modelPill.title = "";
    }
    this.buildModelMenu(state);
  }

  /**
   * Rebuilds the menu only when what it shows has changed. The composer is
   * rendered on every published state, including every streamed delta, and
   * replacing the buttons that often could drop a click whose button was
   * swapped out between mouse-down and mouse-up.
   */
  private buildModelMenu(state: ComposerState): void {
    const menuSignature = JSON.stringify([
      state.models.map((model) => [model.id, model.name]),
      state.selectedModelID,
      state.toolsEnabled,
      state.toolsAvailable,
    ]);
    if (menuSignature === this.renderedModelMenuSignature) {
      return;
    }
    this.renderedModelMenuSignature = menuSignature;
    this.modelMenu.textContent = "";
    state.models.forEach((model) => {
      const item = document.createElement("li");
      item.setAttribute("role", "none");
      const choice = document.createElement("button");
      choice.type = "button";
      choice.setAttribute("role", "menuitemradio");
      choice.setAttribute("aria-checked", model.id === state.selectedModelID ? "true" : "false");
      choice.dataset.testid = `composer-model-${model.id}`;
      choice.id = `composer-model-${model.id}`;
      choice.dataset.modelValue = model.id;
      choice.textContent = model.name;
      item.appendChild(choice);
      this.modelMenu.appendChild(item);
    });
    if (state.models.length > 0) {
      this.appendToolsSwitch(state);
    }
  }

  /** The Tools switch sits under the model list because whether tools can
   * run depends on the chosen model. A model that cannot call tools shows the
   * switch disabled and unchecked, with the reason in the label. */
  private appendToolsSwitch(state: ComposerState): void {
    const separator = document.createElement("li");
    separator.setAttribute("role", "separator");
    separator.className = "composer__menu-separator";
    this.modelMenu.appendChild(separator);
    const toolsSwitchListItem = document.createElement("li");
    toolsSwitchListItem.setAttribute("role", "none");
    const toolsSwitch = document.createElement("button");
    toolsSwitch.type = "button";
    toolsSwitch.setAttribute("role", "menuitemcheckbox");
    const isToolsSwitchChecked = state.toolsEnabled && state.toolsAvailable;
    toolsSwitch.setAttribute("aria-checked", isToolsSwitchChecked ? "true" : "false");
    toolsSwitch.disabled = !state.toolsAvailable;
    toolsSwitch.dataset.testid = "composer-tools";
    toolsSwitch.id = "composer-tools";
    toolsSwitch.dataset.toolsNextState = state.toolsEnabled ? "off" : "on";
    toolsSwitch.textContent = state.toolsAvailable ? "Use tools" : "Use tools (this model cannot)";
    toolsSwitchListItem.appendChild(toolsSwitch);
    this.modelMenu.appendChild(toolsSwitchListItem);
  }
}
