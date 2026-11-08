import type { LibraryStarterPick } from "../LibraryStarterPicks";
import { makeButton, makeElement } from "./AstroDom";
import type { AstroLibrary } from "./AstroLibrary";
import { DownloadMeter } from "./DownloadMeter";
import { ModelDownloadPhase } from "./ModelDownload";
import type { ModelDownloadProgress } from "./ModelDownload";

/** What the offer's own status line says at each point of the download. */
const STATUS_BY_PHASE: Partial<Record<ModelDownloadPhase, string>> = {
  [ModelDownloadPhase.READY]: "My sketchbook is ready. Let's draw!",
  [ModelDownloadPhase.FAILED]: "My sketchbook didn't arrive. You can try again.",
  [ModelDownloadPhase.BUSY]: "Something else is downloading. Try again when it's done.",
};

/**
 * Offers the Library's picture model when a person asks Astro to draw before
 * one is installed. Nothing downloads until they press "Get my sketchbook",
 * and the size is shown on the button first, because 4 to 5 GB is a real ask
 * on a family Mac.
 */
export class SketchbookOffer {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly onDeclined: () => void;
  private readonly status: HTMLElement;
  private readonly idPrefix: string;
  private stopListening: (() => void) | null = null;

  public constructor(document: Document, idPrefix: string, library: AstroLibrary, onDeclined: () => void) {
    this.document = document;
    this.idPrefix = idPrefix;
    this.library = library;
    this.onDeclined = onDeclined;
    this.element = makeElement(document, "div", "sketchbook-offer");
    this.status = makeElement(document, "p", "sketchbook-offer__status", "");
    this.status.setAttribute("role", "status");
    this.element.append(this.status);
  }

  public async loadOffer(): Promise<void> {
    const starterPicks = await this.library.starterPicks();
    const sketchbookPick = starterPicks !== null ? starterPicks.sketchbook : null;
    if (sketchbookPick === null) {
      this.status.textContent = "My sketchbook isn't available right now.";
      return;
    }
    this.showChoice(sketchbookPick);
  }

  /** Stops listening when the offer leaves the screen; the download and
   * AstroLibrary's install notice (which refreshes the picture pill) carry on. */
  public dispose(): void {
    this.stopListening?.();
    this.stopListening = null;
  }

  private showChoice(sketchbookPick: LibraryStarterPick): void {
    const meter = new DownloadMeter(this.document, this.idPrefix, sketchbookPick.approximateSizeBytes);
    const getButton = makeButton(this.document, `${this.idPrefix}-get-sketchbook`, "Get my sketchbook", false);
    const notNowButton = makeButton(this.document, `${this.idPrefix}-not-now`, "Not now", true);
    getButton.addEventListener("click", () => {
      getButton.disabled = true;
      notNowButton.hidden = true;
      this.status.textContent = "";
      this.dispose();
      this.stopListening = this.library.followDownload(sketchbookPick.huggingfaceID, (progress) => this.showProgress(progress, meter, getButton));
    });
    notNowButton.addEventListener("click", () => this.onDeclined());
    const actionRow = makeElement(this.document, "div", "first-flight__actions");
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
    }
  }
}
