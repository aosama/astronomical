import type { LibraryStarterPick } from "../LibraryStarterPicks";
import { makeButton, makeElement } from "./AstroDom";
import type { AstroLibrary } from "./AstroLibrary";
import { DownloadMeter } from "./DownloadMeter";
import { ModelDownloadPhase } from "./ModelDownload";
import type { ModelDownloadProgress } from "./ModelDownload";

/** What the offer's own status line says at each point of the download. */
const STATUS_BY_PHASE: Partial<Record<ModelDownloadPhase, string>> = {
  [ModelDownloadPhase.READY]: "The image model is ready.",
  [ModelDownloadPhase.FAILED]: "The download stopped. You can try again.",
  [ModelDownloadPhase.BUSY]: "Another download is running. Try again when it finishes.",
};

/**
 * Offers the Library's image model when a person asks Astro to draw before
 * one is installed. Nothing downloads until they press "Download image
 * model", and the size is shown first, because 4 to 5 GB is a real ask of
 * someone's disk.
 */
export class ImageModelOffer {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly onDeclined: () => void;
  private readonly onReady: () => void;
  private readonly status: HTMLElement;
  private readonly idPrefix: string;
  private stopListening: (() => void) | null = null;

  /** `onReady` runs once, when a download started from this offer lands. */
  public constructor(document: Document, idPrefix: string, library: AstroLibrary, onDeclined: () => void, onReady: () => void = () => {}) {
    this.document = document;
    this.idPrefix = idPrefix;
    this.library = library;
    this.onDeclined = onDeclined;
    this.onReady = onReady;
    this.element = makeElement(document, "div", "image-model-offer");
    this.status = makeElement(document, "p", "image-model-offer__status", "");
    this.status.setAttribute("role", "status");
    this.element.append(this.status);
  }

  public async loadOffer(): Promise<void> {
    const starterPicks = await this.library.starterPicks();
    const imageModelPick = starterPicks !== null ? starterPicks.imageModel : null;
    if (imageModelPick === null) {
      this.status.textContent = "No image model is available to download right now.";
      return;
    }
    this.showChoice(imageModelPick);
  }

  /** Stops listening when the offer leaves the screen; the download and
   * AstroLibrary's install notice (which refreshes the model rows) carry on. */
  public dispose(): void {
    this.stopListening?.();
    this.stopListening = null;
  }

  private showChoice(imageModelPick: LibraryStarterPick): void {
    const meter = new DownloadMeter(this.document, this.idPrefix, imageModelPick.approximateSizeBytes);
    const getButton = makeButton(this.document, `${this.idPrefix}-get-image-model`, "Download image model", false);
    const notNowButton = makeButton(this.document, `${this.idPrefix}-not-now`, "Not now", true);
    getButton.addEventListener("click", () => {
      getButton.disabled = true;
      notNowButton.hidden = true;
      this.status.textContent = "";
      this.dispose();
      this.stopListening = this.library.followDownload(imageModelPick.huggingfaceID, (progress) => this.showProgress(progress, meter, getButton));
    });
    notNowButton.addEventListener("click", () => this.onDeclined());
    const actionRow = makeElement(this.document, "div", "astro-screen__actions");
    actionRow.append(getButton, notNowButton);
    this.element.append(meter.element, actionRow);
  }

  private showProgress(progress: ModelDownloadProgress, meter: DownloadMeter, getButton: HTMLButtonElement): void {
    meter.show(progress);
    const statusText = STATUS_BY_PHASE[progress.phase];
    if (statusText !== undefined) {
      this.status.textContent = statusText;
    }
    if (progress.phase === ModelDownloadPhase.FAILED || progress.phase === ModelDownloadPhase.BUSY) {
      getButton.disabled = false;
      getButton.textContent = "Try again";
    }
    if (progress.phase === ModelDownloadPhase.READY) {
      getButton.hidden = true;
      this.dispose();
      this.onReady();
    }
  }
}
