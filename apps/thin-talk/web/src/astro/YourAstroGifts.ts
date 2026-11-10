import { makeElement } from "./AstroDom";
import type { AstroFigure } from "./AstroFigure";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment } from "./AstroLines";
import { SketchbookOffer } from "./SketchbookOffer";
import { WakeUpPanel } from "./WakeUpPanel";
import { makeRow, makeSection, rowButton } from "./YourAstroRows";

/** Prefix for the download panels' own controls, kept apart from the page's
 * "your-astro" rows so a panel's "Get my sketchbook" never shares an id with
 * the row button that opened it. */
const GIFT_PANEL_ID_PREFIX = "your-astro-gift";

export interface YourAstroGiftsHost {
  readonly figure: AstroFigure;
  say(moment: AstroMoment): void;
  /** Puts back the page's own line once the group has nothing more to say,
   * so a stale "engine isn't answering" or sketchbook line never lingers. */
  sayPageLine(): void;
  /** Whether the page is still on show; a download landing later must not
   * talk over another screen. */
  isShowing(): boolean;
}

/**
 * The page's "Astro's gifts" group: whether Astro has a brain (a chat model)
 * and a sketchbook (an image model), with the download offer for whichever is
 * missing.
 *
 * Both rows come from one reading of the engine. AstroLibrary answers "no
 * picture model" when the engine cannot be reached at all, so asking it alone
 * would tell a child to download a sketchbook they already have; when the
 * brain check cannot reach the engine, the group says so once instead.
 */
export class YourAstroGifts {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly host: YourAstroGiftsHost;
  private readonly followUp: HTMLElement;
  private openPanel: { dispose(): void } | null = null;
  /** Each refresh counts up; an answer for an older one is dropped, so a slow
   * check can never draw over a newer page. */
  private refreshCount = 0;
  /** Whether the bubble currently carries the engine-silent line. */
  private isShowingEngineSilent = false;

  public constructor(document: Document, library: AstroLibrary, host: YourAstroGiftsHost) {
    this.document = document;
    this.library = library;
    this.host = host;
    this.element = makeSection(document, "Astro's gifts", "your-astro-gifts");
    this.followUp = makeElement(document, "div", "your-astro__follow-up");
  }

  public async refresh(): Promise<void> {
    this.refreshCount += 1;
    const thisRefresh = this.refreshCount;
    this.showRows(makeRow(this.document, null, makeElement(this.document, "span", "your-astro__row-value--muted", "Checking…"), null));
    const hasBrain = await this.library.hasChatModel();
    if (thisRefresh !== this.refreshCount) {
      return;
    }
    if (hasBrain === null) {
      this.showEngineSilent();
      return;
    }
    const hasSketchbook = await this.library.hasImageModel();
    if (thisRefresh !== this.refreshCount) {
      return;
    }
    this.showRows(this.brainRow(hasBrain), this.sketchbookRow(hasSketchbook));
    if (this.isShowingEngineSilent) {
      this.isShowingEngineSilent = false;
      this.host.sayPageLine();
    }
    if (!hasBrain) {
      this.offerBrain();
    }
  }

  /** Stops the panels listening; the daemon keeps downloading. */
  public dispose(): void {
    this.refreshCount += 1;
    this.replacePanel(null);
  }

  private showRows(...rows: HTMLElement[]): void {
    this.replacePanel(null);
    const heading = this.element.querySelector(".your-astro__section-title");
    this.element.replaceChildren(...(heading !== null ? [heading] : []), ...rows, this.followUp);
  }

  private showEngineSilent(): void {
    const sentence = "Astro's engine isn't answering, so Astro can't check its brain or sketchbook right now.";
    this.showRows(makeRow(this.document, null, sentence, rowButton(this.document, "your-astro-check-again", "Check again", () => void this.refresh())));
    if (this.host.isShowing()) {
      this.isShowingEngineSilent = true;
      this.host.say(AstroMoment.ENGINE_SILENT);
    }
  }

  private brainRow(hasBrain: boolean): HTMLElement {
    const brainValue = hasBrain
      ? makeElement(this.document, "span", "your-astro__row-value--ready", "Ready")
      : makeElement(this.document, "span", "your-astro__row-value--wanted", "Not yet. Astro can't talk without one.");
    brainValue.dataset.testid = "your-astro-brain-shown";
    return makeRow(this.document, "Brain", brainValue, null);
  }

  private sketchbookRow(hasSketchbook: boolean): HTMLElement {
    if (hasSketchbook) {
      return makeRow(this.document, "Sketchbook", makeElement(this.document, "span", "your-astro__row-value--ready", "Ready"), null);
    }
    const offerButton = rowButton(this.document, "your-astro-sketchbook", "Get the sketchbook", () => this.offerSketchbook());
    return makeRow(this.document, "Sketchbook", makeElement(this.document, "span", "your-astro__row-value--muted", "Not yet. Astro can draw once it has one."), offerButton);
  }

  /** A missing brain is offered at once, not behind a button: without one
   * Astro cannot answer anything, so it is the one gift every Mac needs. */
  private offerBrain(): void {
    const wakeUpPanel = new WakeUpPanel(this.document, GIFT_PANEL_ID_PREFIX, this.library, {
      figure: this.host.figure,
      trailChoice: null,
      say: (moment) => this.host.say(moment),
      isShowing: () => this.host.isShowing(),
      // The library's install notice reaches AstroCompanion, which asks this
      // group to check again, so the arrival itself needs nothing here.
      onBrainArrived: () => undefined,
      onContinue: () => void this.refresh(),
    });
    this.replacePanel(wakeUpPanel, wakeUpPanel.element);
    if (this.host.isShowing()) {
      this.host.say(AstroMoment.NEEDS_BRAIN);
    }
    void wakeUpPanel.loadOffer();
  }

  private offerSketchbook(): void {
    const sketchbookOffer = new SketchbookOffer(this.document, GIFT_PANEL_ID_PREFIX, this.library, () => this.declineSketchbook());
    this.replacePanel(sketchbookOffer, sketchbookOffer.element);
    this.host.say(AstroMoment.NEEDS_SKETCHBOOK);
    void sketchbookOffer.loadOffer();
  }

  private declineSketchbook(): void {
    this.replacePanel(null);
    this.host.sayPageLine();
  }

  private replacePanel(panel: { dispose(): void } | null, ...content: HTMLElement[]): void {
    this.openPanel?.dispose();
    this.openPanel = panel;
    this.followUp.replaceChildren(...content);
  }
}
