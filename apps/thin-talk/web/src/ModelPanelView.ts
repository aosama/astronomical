import { ComposerAction } from "./ComposerAction";
import type { ComposerState } from "./ComposerState";

/** A picture choice only helps when there is something to pick: with one
 * image model installed the Pictures section stays out and that model is used. */
const MINIMUM_IMAGE_MODELS_TO_OFFER_A_CHOICE = 2;

/** One row of the panel: a model's one name, its quieter details, and the
 * data attribute that says what choosing it means. */
interface PanelRow {
  readonly testID: string;
  readonly choiceAttribute: "modelValue" | "imageModelValue";
  readonly modelID: string;
  readonly name: string;
  readonly detail: string;
  readonly isChosen: boolean;
}

/**
 * The composer's brain control: one small button beside Send that names the
 * chat model in use and opens one panel listing every installed brain, chat
 * models first, then picture models, then the Tools switch and a way to the
 * Models page. It replaced two wide pills that showed folder names; the
 * folder name now lives only in the button's tooltip.
 *
 * It draws the state it is given and reports intents; the controller owns
 * every choice.
 */
export class ModelPanelView {
  private readonly trigger: HTMLButtonElement;
  private readonly label: HTMLElement;
  private readonly panel: HTMLElement;
  private readonly onAction: (action: ComposerAction) => void;
  /** What the panel last drew, so an unchanged panel is not rebuilt. The
   * composer renders on every streamed delta, and swapping a row out between
   * mouse-down and mouse-up would drop the click. */
  private renderedPanelSignature = "";

  public constructor(init: { trigger: HTMLButtonElement; label: HTMLElement; panel: HTMLElement; onAction: (action: ComposerAction) => void }) {
    this.trigger = init.trigger;
    this.label = init.label;
    this.panel = init.panel;
    this.onAction = init.onAction;
    this.trigger.addEventListener("click", () => this.toggle());
    this.panel.addEventListener("click", (clickEvent) => this.reportChoice(clickEvent));
  }

  public get isOpen(): boolean {
    return !this.panel.hidden;
  }

  public close(): void {
    this.panel.hidden = true;
    this.trigger.setAttribute("aria-expanded", "false");
  }

  public render(state: ComposerState): void {
    const hasModels = state.models.length > 0;
    this.trigger.hidden = !hasModels;
    if (!hasModels || !state.isReady) {
      this.close();
    }
    const selectedModel = state.models.find((model) => model.id === state.selectedModelID);
    const selectedName = selectedModel != null ? selectedModel.id : "Model";
    this.label.textContent = selectedName;
    this.trigger.title = selectedModel != null ? selectedModel.id : "";
    this.trigger.setAttribute("aria-label", selectedModel != null ? `Model: ${selectedName}` : "Model");
    this.rebuildPanelIfChanged(state);
  }

  private toggle(): void {
    const opening = this.panel.hidden;
    if (opening) {
      // A model installed since the list was read appears once it is re-read.
      this.onAction(new ComposerAction({ kind: "openBrainPanel" }));
    }
    this.panel.hidden = !opening;
    this.trigger.setAttribute("aria-expanded", opening ? "true" : "false");
  }

  private rebuildPanelIfChanged(state: ComposerState): void {
    const chatRows = ModelPanelView.chatRows(state);
    const pictureRows = ModelPanelView.pictureRows(state);
    const panelSignature = JSON.stringify([chatRows, pictureRows, state.toolsEnabled, state.toolsAvailable]);
    if (panelSignature === this.renderedPanelSignature) {
      return;
    }
    this.renderedPanelSignature = panelSignature;
    this.panel.textContent = "";
    if (chatRows.length === 0) {
      return;
    }
    this.appendSection("Chat", chatRows);
    if (pictureRows.length > 0) {
      this.appendSeparator();
      this.appendSection("Pictures", pictureRows);
    }
    this.appendSeparator();
    this.appendToolsSwitch(state);
    this.appendModelsPageLink();
  }

  private static chatRows(state: ComposerState): PanelRow[] {
    return state.models.map((model) => ({
      testID: `composer-model-${model.id}`,
      choiceAttribute: "modelValue",
      modelID: model.id,
      name: model.id,
      detail: state.modelDescriptions.get(model.id)?.detailLine() ?? "",
      isChosen: model.id === state.selectedModelID,
    }));
  }

  /** Picture models are offered only when the chat model may draw at all
   * (tools on, and a model that can call them), as the composer never
   * promises a tool that will not run. */
  private static pictureRows(state: ComposerState): PanelRow[] {
    const canDraw = state.toolsEnabled && state.toolsAvailable;
    if (!canDraw || state.imageModels.length < MINIMUM_IMAGE_MODELS_TO_OFFER_A_CHOICE) {
      return [];
    }
    return state.imageModels.map((imageModel) => ({
      testID: `composer-image-model-${imageModel.id}`,
      choiceAttribute: "imageModelValue",
      modelID: imageModel.id,
      name: imageModel.id,
      detail: state.modelDescriptions.get(imageModel.id)?.detailLine() ?? "",
      isChosen: imageModel.id === state.selectedImageModelID,
    }));
  }

  private appendSection(heading: string, rows: readonly PanelRow[]): void {
    const headingItem = document.createElement("li");
    headingItem.setAttribute("role", "presentation");
    headingItem.className = "composer__menu-heading";
    headingItem.textContent = heading;
    this.panel.appendChild(headingItem);
    for (const row of rows) {
      this.panel.appendChild(ModelPanelView.rowItem(row));
    }
  }

  private static rowItem(row: PanelRow): HTMLLIElement {
    const rowItem = document.createElement("li");
    rowItem.setAttribute("role", "none");
    const choice = document.createElement("button");
    choice.type = "button";
    choice.className = "composer__model-choice";
    choice.setAttribute("role", "menuitemradio");
    choice.setAttribute("aria-checked", row.isChosen ? "true" : "false");
    choice.dataset.testid = row.testID;
    choice.id = row.testID;
    choice.dataset[row.choiceAttribute] = row.modelID;
    choice.title = row.modelID;
    const nameLine = document.createElement("span");
    nameLine.className = "composer__model-name";
    nameLine.textContent = row.name;
    choice.appendChild(nameLine);
    if (row.detail.length > 0) {
      const detailLine = document.createElement("span");
      detailLine.className = "composer__model-detail";
      detailLine.textContent = row.detail;
      choice.appendChild(detailLine);
    }
    rowItem.appendChild(choice);
    return rowItem;
  }

  private appendSeparator(): void {
    const separator = document.createElement("li");
    separator.setAttribute("role", "separator");
    separator.className = "composer__menu-separator";
    this.panel.appendChild(separator);
  }

  /** The Tools switch sits in the panel because whether tools can run depends
   * on the chosen model. A model that cannot call tools shows the switch
   * disabled and unchecked, with the reason in the label. */
  private appendToolsSwitch(state: ComposerState): void {
    const toolsSwitchItem = document.createElement("li");
    toolsSwitchItem.setAttribute("role", "none");
    const toolsSwitch = document.createElement("button");
    toolsSwitch.type = "button";
    toolsSwitch.setAttribute("role", "menuitemcheckbox");
    toolsSwitch.setAttribute("aria-checked", state.toolsEnabled && state.toolsAvailable ? "true" : "false");
    toolsSwitch.disabled = !state.toolsAvailable;
    toolsSwitch.dataset.testid = "composer-tools";
    toolsSwitch.id = "composer-tools";
    toolsSwitch.dataset.toolsNextState = state.toolsEnabled ? "off" : "on";
    toolsSwitch.textContent = state.toolsAvailable ? "Use tools" : "Use tools (this model cannot)";
    toolsSwitchItem.appendChild(toolsSwitch);
    this.panel.appendChild(toolsSwitchItem);
  }

  private appendModelsPageLink(): void {
    const linkItem = document.createElement("li");
    linkItem.setAttribute("role", "none");
    const modelsPageLink = document.createElement("button");
    modelsPageLink.type = "button";
    modelsPageLink.className = "composer__models-page";
    modelsPageLink.setAttribute("role", "menuitem");
    modelsPageLink.dataset.testid = "composer-models-page";
    modelsPageLink.id = "composer-models-page";
    modelsPageLink.dataset.opensModelsPage = "true";
    modelsPageLink.textContent = "Get more models…";
    linkItem.appendChild(modelsPageLink);
    this.panel.appendChild(linkItem);
  }

  private reportChoice(clickEvent: Event): void {
    const clickedElement = clickEvent.target instanceof Element ? clickEvent.target : null;
    const choice = clickedElement?.closest<HTMLButtonElement>("button");
    if (choice == null || choice.disabled) {
      return;
    }
    const chosenAction = ModelPanelView.actionFor(choice);
    if (chosenAction === null) {
      return;
    }
    this.close();
    this.onAction(chosenAction);
  }

  private static actionFor(choice: HTMLButtonElement): ComposerAction | null {
    if (choice.dataset.modelValue != null) {
      return new ComposerAction({ kind: "selectModel", detail: choice.dataset.modelValue });
    }
    if (choice.dataset.imageModelValue != null) {
      return new ComposerAction({ kind: "selectImageModel", detail: choice.dataset.imageModelValue });
    }
    if (choice.dataset.toolsNextState != null) {
      return new ComposerAction({ kind: "setTools", detail: choice.dataset.toolsNextState });
    }
    if (choice.dataset.opensModelsPage != null) {
      return new ComposerAction({ kind: "openModelsPage" });
    }
    return null;
  }
}
