import { makeElement } from "./AstroDom";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment } from "./AstroLines";
import { ChatModelDownloadPanel } from "./ChatModelDownloadPanel";
import { ImageModelOffer } from "./ImageModelOffer";
import { makeRow, makeSection, rowButton } from "./SettingsRows";

/** Prefix for the download panels' own controls, kept apart from the page's
 * "settings" rows so a panel's "Download image model" never shares an id
 * with the row button that opened it. */
const MODEL_PANEL_ID_PREFIX = "settings-model";

export interface SettingsModelsHost {
  say(moment: AstroMoment): void;
  /** Puts back the page's own line once the group has nothing more to say,
   * so a stale "engine isn't answering" or download line never lingers. */
  sayPageLine(): void;
  /** Whether the page is still on show; a download landing later must not
   * talk over another screen. */
  isShowing(): boolean;
}

/**
 * The page's "Models" group: whether a chat model and an image model are
 * installed, with the download offer for whichever is missing.
 *
 * Both rows come from one reading of the engine. AstroLibrary answers "no
 * image model" when the engine cannot be reached at all, so asking it alone
 * would offer a download of a model the person already has; when the chat
 * model check cannot reach the engine, the group says so once instead.
 */
export class SettingsModels {
  public readonly element: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly host: SettingsModelsHost;
  private readonly followUp: HTMLElement;
  private openPanel: { dispose(): void } | null = null;
  /** Each refresh counts up; an answer for an older one is dropped, so a slow
   * check can never draw over a newer page. */
  private refreshCount = 0;
  /** Whether the page's line currently carries the engine-silent sentence. */
  private isShowingEngineSilent = false;

  public constructor(document: Document, library: AstroLibrary, host: SettingsModelsHost) {
    this.document = document;
    this.library = library;
    this.host = host;
    this.element = makeSection(document, "Models", "settings-models");
    this.followUp = makeElement(document, "div", "settings__follow-up");
  }

  public async refresh(): Promise<void> {
    this.refreshCount += 1;
    const thisRefresh = this.refreshCount;
    this.showRows(makeRow(this.document, null, makeElement(this.document, "span", "settings__row-value--muted", "Checking…"), null));
    const hasChatModel = await this.library.hasChatModel();
    if (thisRefresh !== this.refreshCount) {
      return;
    }
    if (hasChatModel === null) {
      this.showEngineSilent();
      return;
    }
    const hasImageModel = await this.library.hasImageModel();
    if (thisRefresh !== this.refreshCount) {
      return;
    }
    this.showRows(this.chatModelRow(hasChatModel), this.imageModelRow(hasImageModel));
    if (this.isShowingEngineSilent) {
      this.isShowingEngineSilent = false;
      this.host.sayPageLine();
    }
    if (!hasChatModel) {
      this.offerChatModel();
    }
  }

  /** Stops the panels listening; the daemon keeps downloading. */
  public dispose(): void {
    this.refreshCount += 1;
    this.replacePanel(null);
  }

  private showRows(...rows: HTMLElement[]): void {
    this.replacePanel(null);
    const heading = this.element.querySelector(".settings__section-title");
    this.element.replaceChildren(...(heading !== null ? [heading] : []), ...rows, this.followUp);
  }

  private showEngineSilent(): void {
    const sentence = "The engine isn't answering, so the installed models can't be checked right now.";
    this.showRows(makeRow(this.document, null, sentence, rowButton(this.document, "settings-check-again", "Check again", () => void this.refresh())));
    if (this.host.isShowing()) {
      this.isShowingEngineSilent = true;
      this.host.say(AstroMoment.ENGINE_SILENT);
    }
  }

  private chatModelRow(hasChatModel: boolean): HTMLElement {
    const chatModelValue = hasChatModel
      ? makeElement(this.document, "span", "settings__row-value--ready", "Installed")
      : makeElement(this.document, "span", "settings__row-value--wanted", "Not installed. Astro can't answer without one.");
    chatModelValue.dataset.testid = "settings-chat-model-shown";
    return makeRow(this.document, "Chat model", chatModelValue, null);
  }

  private imageModelRow(hasImageModel: boolean): HTMLElement {
    if (hasImageModel) {
      return makeRow(this.document, "Image model", makeElement(this.document, "span", "settings__row-value--ready", "Installed"), null);
    }
    const offerButton = rowButton(this.document, "settings-image-model", "Download", () => this.offerImageModel());
    return makeRow(this.document, "Image model", makeElement(this.document, "span", "settings__row-value--muted", "Not installed. Needed to draw pictures."), offerButton);
  }

  /** A missing chat model is offered at once, not behind a button: without
   * one nothing can be answered, so it is the one download every Mac needs. */
  private offerChatModel(): void {
    const downloadPanel = new ChatModelDownloadPanel(this.document, MODEL_PANEL_ID_PREFIX, this.library, {
      canSkip: false,
      openModelsPage: null,
      say: (moment) => this.host.say(moment),
      isShowing: () => this.host.isShowing(),
      // The library's install notice reaches AstroCompanion, which asks this
      // group to check again, so the arrival itself needs nothing here.
      onChatModelArrived: () => undefined,
      onContinue: () => void this.refresh(),
    });
    this.replacePanel(downloadPanel, downloadPanel.element);
    if (this.host.isShowing()) {
      this.host.say(AstroMoment.NEEDS_CHAT_MODEL);
    }
    void downloadPanel.loadOffer();
  }

  private offerImageModel(): void {
    const imageModelOffer = new ImageModelOffer(this.document, MODEL_PANEL_ID_PREFIX, this.library, () => this.declineImageModel());
    this.replacePanel(imageModelOffer, imageModelOffer.element);
    this.host.say(AstroMoment.NEEDS_IMAGE_MODEL);
    void imageModelOffer.loadOffer();
  }

  private declineImageModel(): void {
    this.replacePanel(null);
    this.host.sayPageLine();
  }

  private replacePanel(panel: { dispose(): void } | null, ...content: HTMLElement[]): void {
    this.openPanel?.dispose();
    this.openPanel = panel;
    this.followUp.replaceChildren(...content);
  }
}
