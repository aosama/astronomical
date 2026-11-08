import { ImageModel } from "./ImageModel";

/** A picker only helps when there is something to pick: with one image model
 * installed the pill stays hidden and that model is used. */
const MINIMUM_IMAGE_MODELS_TO_OFFER_A_CHOICE = 2;

/**
 * The composer's picture pill: which installed image model draws the next
 * picture. It draws the choices it is given and reports when it opens and
 * which choice was picked; the controller owns the choice and its persistence.
 */
export class ImageModelPillView {
  private readonly pill: HTMLButtonElement;
  private readonly label: HTMLElement;
  private readonly menu: HTMLElement;
  private readonly onOpen: () => void;
  private readonly onSelect: (imageModelID: string) => void;
  /** What the menu last drew, so an unchanged menu is not rebuilt while a
   * click on one of its buttons is still in flight. */
  private renderedMenuSignature = "";

  /** `onOpen` runs before the menu opens, so the composer can close its
   * other menus and only one is ever open. */
  public constructor(init: {
    pill: HTMLButtonElement;
    label: HTMLElement;
    menu: HTMLElement;
    onOpen: () => void;
    onSelect: (imageModelID: string) => void;
  }) {
    this.pill = init.pill;
    this.label = init.label;
    this.menu = init.menu;
    this.onOpen = init.onOpen;
    this.onSelect = init.onSelect;
    this.pill.addEventListener("click", () => this.toggle());
    this.menu.addEventListener("click", (event) => this.reportChoice(event));
  }

  public get isOpen(): boolean {
    return !this.menu.hidden;
  }

  private toggle(): void {
    const opening = this.menu.hidden;
    if (opening) {
      this.onOpen();
    }
    this.menu.hidden = !opening;
    this.pill.setAttribute("aria-expanded", opening ? "true" : "false");
  }

  public close(): void {
    this.menu.hidden = true;
    this.pill.setAttribute("aria-expanded", "false");
  }

  /** `canDraw` is whether the chat model may use the image tool at all (tools
   * on, and a model that can call them); a pill that could not take effect is
   * not shown, as the composer never promises tools that will not run. */
  public render(imageModels: readonly ImageModel[], selectedImageModelID: string | null, canDraw: boolean): void {
    const offersChoice = canDraw && imageModels.length >= MINIMUM_IMAGE_MODELS_TO_OFFER_A_CHOICE;
    this.pill.hidden = !offersChoice;
    if (!offersChoice) {
      this.close();
    }
    const selectedImageModel = imageModels.find((imageModel) => imageModel.id === selectedImageModelID);
    this.label.textContent = selectedImageModel != null ? selectedImageModel.name : "Pictures";
    this.pill.title = selectedImageModel != null ? `Pictures are drawn with ${selectedImageModel.name}` : "";
    // The accessible name carries the choice, as the visible label does.
    this.pill.setAttribute("aria-label", selectedImageModel != null ? `Image model: ${selectedImageModel.name}` : "Image model");
    this.buildMenu(imageModels, selectedImageModelID);
  }

  private buildMenu(imageModels: readonly ImageModel[], selectedImageModelID: string | null): void {
    const menuSignature = JSON.stringify([imageModels.map((imageModel) => [imageModel.id, imageModel.name]), selectedImageModelID]);
    if (menuSignature === this.renderedMenuSignature) {
      return;
    }
    this.renderedMenuSignature = menuSignature;
    this.menu.textContent = "";
    for (const imageModel of imageModels) {
      const menuItem = document.createElement("li");
      menuItem.setAttribute("role", "none");
      const choice = document.createElement("button");
      choice.type = "button";
      choice.setAttribute("role", "menuitemradio");
      choice.setAttribute("aria-checked", imageModel.id === selectedImageModelID ? "true" : "false");
      choice.dataset.testid = `composer-image-model-${imageModel.id}`;
      choice.id = `composer-image-model-${imageModel.id}`;
      choice.dataset.imageModelValue = imageModel.id;
      choice.textContent = imageModel.name;
      menuItem.appendChild(choice);
      this.menu.appendChild(menuItem);
    }
  }

  private reportChoice(event: Event): void {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    const choice = target.closest("[data-image-model-value]");
    if (!(choice instanceof HTMLElement)) {
      return;
    }
    this.close();
    const imageModelID = choice.dataset.imageModelValue;
    if (imageModelID != null) {
      this.onSelect(imageModelID);
    }
  }
}
