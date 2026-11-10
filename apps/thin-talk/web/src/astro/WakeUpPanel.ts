import type { LibraryStarterPick } from "../LibraryStarterPicks";
import { makeButton, makeElement } from "./AstroDom";
import { AstroMoment } from "./AstroLines";
import type { AstroFigure } from "./AstroFigure";
import type { AstroLibrary } from "./AstroLibrary";
import { buildTrailChips } from "./AstroTrail";
import type { AstroTrail } from "./AstroTrail";
import { DownloadMeter } from "./DownloadMeter";
import { ModelDownloadPhase } from "./ModelDownload";
import type { ModelDownloadProgress } from "./ModelDownload";

/** The trail chips shown while the brain downloads: a small, harmless choice
 * that makes the wait feel like the person's own. */
export interface WakeUpTrailChoice {
  /** The saved trail, shown pressed among the chips. */
  readonly trail: AstroTrail;
  /** The host saves the choice. */
  onTrailChosen(trail: AstroTrail): void;
}

export interface WakeUpPanelHost {
  readonly figure: AstroFigure;
  /** Null where the trail has a control of its own (Your Astro). */
  readonly trailChoice: WakeUpTrailChoice | null;
  say(moment: AstroMoment): void;
  /** Whether the wake-up screen is still on show; a download that lands after
   * the person moved on must not talk over the next screen. */
  isShowing(): boolean;
  onBrainArrived(): void;
  onContinue(): void;
}

/**
 * "Give Astro a brain": the first chat model's download, told as Astro
 * charging up. The person can keep going while it runs; the daemon carries on
 * downloading, and the host still hears when the brain lands.
 */
export class WakeUpPanel {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly host: WakeUpPanelHost;
  private readonly idPrefix: string;
  /** Stops this panel listening to the brain's download; the download and
   * AstroLibrary's install notice carry on without it. */
  private stopListening: (() => void) | null = null;
  private hasBrain = false;

  public constructor(document: Document, idPrefix: string, library: AstroLibrary, host: WakeUpPanelHost) {
    this.document = document;
    this.idPrefix = idPrefix;
    this.library = library;
    this.host = host;
    this.element = makeElement(document, "div", "wake-up");
  }

  /** Reads the Library's recommended brain and shows the offer; a download of
   * it already under way (started from another screen) is shown at once. */
  public async loadOffer(): Promise<void> {
    const starterPicks = await this.library.starterPicks();
    const brainPick = starterPicks !== null ? starterPicks.brain : null;
    if (brainPick === null) {
      // No catalog (offline daemon or a catalog without a pick): say so kindly
      // and let the person carry on rather than stranding them here.
      this.host.say(AstroMoment.DOWNLOAD_FAILED);
      this.element.append(this.actionRow(this.continueButton()));
      return;
    }
    const { meter, wakeButton } = this.showDownloadControls(brainPick);
    if (await this.library.isDownloading(brainPick.huggingfaceID)) {
      this.startBrainDownload(brainPick, meter, wakeButton);
    }
  }

  /** Stops listening when the panel leaves the screen. */
  public dispose(): void {
    this.stopListening?.();
    this.stopListening = null;
  }

  private showDownloadControls(brainPick: LibraryStarterPick): { meter: DownloadMeter; wakeButton: HTMLButtonElement } {
    const meter = new DownloadMeter(this.document, this.idPrefix, brainPick.approximateSizeBytes);
    const wakeButton = makeButton(this.document, `${this.idPrefix}-wake`, "Give Astro a brain", false);
    const keepGoingButton = this.continueButton();
    keepGoingButton.hidden = true;
    wakeButton.addEventListener("click", () => {
      if (this.hasBrain) {
        this.host.onContinue();
        return;
      }
      this.startBrainDownload(brainPick, meter, wakeButton);
    });
    this.element.append(meter.element);
    const trailChoice = this.host.trailChoice;
    if (trailChoice !== null) {
      this.element.append(this.trailChips(trailChoice));
    }
    this.element.append(this.actionRow(wakeButton, keepGoingButton));
    return { meter, wakeButton };
  }

  private startBrainDownload(brainPick: LibraryStarterPick, meter: DownloadMeter, wakeButton: HTMLButtonElement): void {
    wakeButton.disabled = true;
    wakeButton.textContent = "Charging up…";
    this.host.figure.setCharging(true);
    this.host.say(AstroMoment.CHARGING);
    const keepGoingButton = this.element.querySelector<HTMLButtonElement>(`#${this.idPrefix}-keep-going`);
    if (keepGoingButton !== null) {
      keepGoingButton.hidden = false;
    }
    this.dispose();
    this.stopListening = this.library.followDownload(brainPick.huggingfaceID, (progress) => {
      meter.show(progress);
      this.react(progress, wakeButton);
    });
  }

  private react(progress: ModelDownloadProgress, wakeButton: HTMLButtonElement): void {
    if (progress.phase === ModelDownloadPhase.READY) {
      this.hasBrain = true;
      this.host.figure.setCharging(false);
      this.host.onBrainArrived();
      this.speakIfShowing(AstroMoment.AWAKE);
      wakeButton.disabled = false;
      wakeButton.textContent = "Let's go!";
      return;
    }
    if (progress.phase === ModelDownloadPhase.FAILED || progress.phase === ModelDownloadPhase.BUSY) {
      this.host.figure.setCharging(false);
      this.speakIfShowing(progress.phase === ModelDownloadPhase.BUSY ? AstroMoment.LIBRARY_BUSY : AstroMoment.DOWNLOAD_FAILED);
      wakeButton.disabled = false;
      wakeButton.textContent = "Try again";
    }
  }

  private speakIfShowing(moment: AstroMoment): void {
    if (this.host.isShowing()) {
      this.host.say(moment);
      this.host.figure.hop();
    }
  }

  private continueButton(): HTMLButtonElement {
    const continueButton = makeButton(this.document, `${this.idPrefix}-keep-going`, "Keep going", true);
    continueButton.addEventListener("click", () => this.host.onContinue());
    return continueButton;
  }

  private actionRow(...buttons: HTMLButtonElement[]): HTMLElement {
    const actionRow = makeElement(this.document, "div", "first-flight__actions");
    actionRow.append(...buttons);
    return actionRow;
  }

  private trailChips(trailChoice: WakeUpTrailChoice): HTMLElement {
    const trailRow = makeElement(this.document, "div", "wake-up__trails");
    trailRow.append(
      makeElement(this.document, "span", "wake-up__trails-label", "While you wait, pick a trail:"),
      buildTrailChips(this.document, this.idPrefix, trailChoice.trail, (trail) => this.chooseTrail(trailChoice, trail)),
    );
    return trailRow;
  }

  private chooseTrail(trailChoice: WakeUpTrailChoice, trail: AstroTrail): void {
    trailChoice.onTrailChosen(trail);
    this.host.figure.hop();
  }
}
