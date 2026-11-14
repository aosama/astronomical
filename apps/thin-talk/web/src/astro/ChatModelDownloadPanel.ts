import type { LibraryStarterPick } from "../LibraryStarterPicks";
import { makeButton, makeElement } from "./AstroDom";
import { AstroMoment } from "./AstroLines";
import type { AstroLibrary } from "./AstroLibrary";
import { DownloadMeter } from "./DownloadMeter";
import { modelNameFromHuggingfaceID } from "./HuggingfaceModelName";
import { ModelDownloadPhase } from "./ModelDownload";
import type { ModelDownloadProgress } from "./ModelDownload";

/** What a newcomer can do once the model lands, shown while they wait so the
 * wait says what it is for. */
const WHAT_COMES_NEXT: readonly string[] = ["Ask a question", "Attach a picture and ask about it", "Attach a file to summarise or explain"];

export interface ChatModelDownloadPanelHost {
  /** True on the get-started journey, where "Skip for now" is offered before
   * the download starts. Elsewhere the panel sits inside a page the person
   * can simply leave. */
  readonly canSkip: boolean;
  /** Opens the Models page for someone who wants a different model; null
   * where the page offers no such link. */
  readonly openModelsPage: (() => void) | null;
  say(moment: AstroMoment): void;
  /** Whether the panel is still on show; a download that lands after the
   * person moved on must not talk over the next screen. */
  isShowing(): boolean;
  onChatModelArrived(): void;
  onContinue(): void;
}

/**
 * The first chat model's download: the Library's recommended model by name,
 * its size, and one Download button. The person can keep going while it
 * runs; the daemon carries on downloading, and the host still hears when the
 * model lands.
 */
export class ChatModelDownloadPanel {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly host: ChatModelDownloadPanelHost;
  private readonly idPrefix: string;
  /** Stops this panel listening to the download; the download and
   * AstroLibrary's install notice carry on without it. */
  private stopListening: (() => void) | null = null;
  private hasChatModel = false;

  public constructor(document: Document, idPrefix: string, library: AstroLibrary, host: ChatModelDownloadPanelHost) {
    this.document = document;
    this.idPrefix = idPrefix;
    this.library = library;
    this.host = host;
    this.element = makeElement(document, "div", "model-download");
  }

  /** Reads the Library's recommended chat model and shows the offer; a
   * download of it already under way (started from another screen) is shown
   * at once. */
  public async loadOffer(): Promise<void> {
    const starterPicks = await this.library.starterPicks();
    const chatModelPick = starterPicks !== null ? starterPicks.chatModel : null;
    if (chatModelPick === null) {
      // No catalog (offline daemon or a catalog without a pick): say so and
      // let the person carry on rather than stranding them here.
      this.host.say(AstroMoment.DOWNLOAD_FAILED);
      this.element.append(this.actionRow(this.continueButton("Continue")));
      return;
    }
    const { meter, downloadButton } = this.showDownloadControls(chatModelPick);
    if (await this.library.isDownloading(chatModelPick.huggingfaceID)) {
      this.startDownload(chatModelPick, meter, downloadButton);
    }
  }

  /** True from the moment the person pressed Download here (or a download
   * of the offered model was found already running). The panel then tells
   * the story to its end itself, and nothing else should end it early. */
  public get isFollowingDownload(): boolean {
    return this.stopListening !== null || this.hasChatModel;
  }

  /** Stops listening when the panel leaves the screen. */
  public dispose(): void {
    this.stopListening?.();
    this.stopListening = null;
  }

  private showDownloadControls(chatModelPick: LibraryStarterPick): { meter: DownloadMeter; downloadButton: HTMLButtonElement } {
    const modelName = makeElement(this.document, "p", "model-download__name", modelNameFromHuggingfaceID(chatModelPick.huggingfaceID));
    modelName.dataset.testid = `${this.idPrefix}-download-model`;
    const meter = new DownloadMeter(this.document, this.idPrefix, chatModelPick.approximateSizeBytes);
    const downloadButton = makeButton(this.document, `${this.idPrefix}-download`, "Download", false);
    downloadButton.addEventListener("click", () => {
      if (this.hasChatModel) {
        this.host.onContinue();
        return;
      }
      this.startDownload(chatModelPick, meter, downloadButton);
    });
    const continueButton = this.continueButton("Skip for now");
    continueButton.hidden = !this.host.canSkip;
    this.element.append(modelName, meter.element, this.actionRow(downloadButton, ...this.modelsPageButtons(), continueButton), this.whatComesNext());
    return { meter, downloadButton };
  }

  private modelsPageButtons(): HTMLButtonElement[] {
    const openModelsPage = this.host.openModelsPage;
    if (openModelsPage === null) {
      return [];
    }
    const otherModelButton = makeButton(this.document, `${this.idPrefix}-other-model`, "Choose a different model", true);
    otherModelButton.addEventListener("click", () => openModelsPage());
    return [otherModelButton];
  }

  /** Hidden until the download starts: before that it would read as steps
   * the person has to do. */
  private whatComesNext(): HTMLElement {
    const whatComesNext = makeElement(this.document, "div", "model-download__next");
    whatComesNext.dataset.testid = `${this.idPrefix}-download-next`;
    whatComesNext.hidden = true;
    const nextList = makeElement(this.document, "ul", "");
    for (const nextThing of WHAT_COMES_NEXT) {
      nextList.append(makeElement(this.document, "li", "", nextThing));
    }
    whatComesNext.append(makeElement(this.document, "p", "model-download__next-lead", "When it finishes you can:"), nextList);
    return whatComesNext;
  }

  private startDownload(chatModelPick: LibraryStarterPick, meter: DownloadMeter, downloadButton: HTMLButtonElement): void {
    downloadButton.disabled = true;
    downloadButton.textContent = "Downloading…";
    this.host.say(AstroMoment.DOWNLOADING);
    this.setContinueButton({ isHidden: false, label: "Continue in the background" });
    const whatComesNext = this.element.querySelector<HTMLElement>(".model-download__next");
    if (whatComesNext !== null) {
      whatComesNext.hidden = false;
    }
    this.dispose();
    this.stopListening = this.library.followDownload(chatModelPick.huggingfaceID, (progress) => {
      meter.show(progress);
      this.react(progress, downloadButton);
    });
  }

  private react(progress: ModelDownloadProgress, downloadButton: HTMLButtonElement): void {
    if (progress.phase === ModelDownloadPhase.READY) {
      this.hasChatModel = true;
      this.host.onChatModelArrived();
      this.speakIfShowing(AstroMoment.MODEL_READY);
      downloadButton.disabled = false;
      downloadButton.textContent = "Start chatting";
      // The main button now does what "Continue in the background" did.
      this.setContinueButton({ isHidden: true });
      this.showArrival();
      return;
    }
    if (progress.phase === ModelDownloadPhase.FAILED || progress.phase === ModelDownloadPhase.BUSY) {
      this.speakIfShowing(progress.phase === ModelDownloadPhase.BUSY ? AstroMoment.LIBRARY_BUSY : AstroMoment.DOWNLOAD_FAILED);
      downloadButton.disabled = false;
      downloadButton.textContent = "Try again";
    }
  }

  /** Once the model is here the screen stops talking about a download still
   * to come: the list reads as what the person can do now, and the offer of a
   * different model goes, since choosing one would start a second download
   * just as this one landed. */
  private showArrival(): void {
    const nextLead = this.element.querySelector<HTMLElement>(".model-download__next-lead");
    if (nextLead !== null) {
      nextLead.textContent = "You can now:";
    }
    const otherModelButton = this.element.querySelector<HTMLButtonElement>(`#${this.idPrefix}-other-model`);
    if (otherModelButton !== null) {
      otherModelButton.hidden = true;
    }
  }

  private speakIfShowing(moment: AstroMoment): void {
    if (this.host.isShowing()) {
      this.host.say(moment);
    }
  }

  private setContinueButton(change: { isHidden: boolean; label?: string }): void {
    const continueButton = this.element.querySelector<HTMLButtonElement>(`#${this.idPrefix}-continue`);
    if (continueButton === null) {
      return;
    }
    continueButton.hidden = change.isHidden;
    if (change.label !== undefined) {
      continueButton.textContent = change.label;
    }
  }

  private continueButton(label: string): HTMLButtonElement {
    const continueButton = makeButton(this.document, `${this.idPrefix}-continue`, label, true);
    continueButton.addEventListener("click", () => this.host.onContinue());
    return continueButton;
  }

  private actionRow(...buttons: HTMLButtonElement[]): HTMLElement {
    const actionRow = makeElement(this.document, "div", "astro-screen__actions");
    actionRow.append(...buttons);
    return actionRow;
  }
}
