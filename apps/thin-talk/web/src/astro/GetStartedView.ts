import { makeButton, makeElement, setSiblingsInert } from "./AstroDom";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment, astroLine } from "./AstroLines";
import { ChatModelDownloadPanel } from "./ChatModelDownloadPanel";
import { ENGINE_PATIENCE } from "./EngineWatch";
import type { EngineWatch } from "./EngineWatch";
import { GetStartedJourney, GetStartedStep } from "./GetStartedJourney";

/** Prefix for the journey's control ids; Astro's home uses "astro-home". */
const ID_PREFIX = "astro";

/** Something on screen that must stop listening when its step leaves. */
interface DisposablePanel {
  dispose(): void;
}

/**
 * The get-started journey: a full-window welcome for a newcomer, one action
 * per screen. It draws one step at a time and asks GetStartedJourney where
 * to go next; the download runs through AstroLibrary.
 */
export class GetStartedView {
  private readonly overlay: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly onFinished: () => void;
  private readonly openModelsPage: () => void;
  private readonly engineWatch: EngineWatch;
  /** Asks the host to restart the engine; null where the host cannot. */
  private readonly restartEngine: (() => void) | null;
  /** The download step's panel while that step is on screen. */
  private downloadPanel: ChatModelDownloadPanel | null = null;
  /** Stops listening for the engine's first answer. */
  private stopAwaitingEngine: (() => void) | null = null;
  private readonly headline: HTMLElement;
  private readonly line: HTMLElement;
  private readonly controls: HTMLElement;
  private journey = new GetStartedJourney({ hasChatModel: null });
  private openPanels: DisposablePanel[] = [];

  public constructor(init: {
    overlay: HTMLElement;
    library: AstroLibrary;
    engineWatch: EngineWatch;
    restartEngine: (() => void) | null;
    /** Opens the Models page, for someone who wants a different model. */
    openModelsPage: () => void;
    onFinished: () => void;
  }) {
    this.overlay = init.overlay;
    this.document = init.overlay.ownerDocument;
    this.library = init.library;
    this.onFinished = init.onFinished;
    this.openModelsPage = init.openModelsPage;
    this.engineWatch = init.engineWatch;
    this.restartEngine = init.restartEngine;
    this.headline = makeElement(this.document, "h1", "get-started__headline");
    this.line = makeElement(this.document, "p", "astro-screen__line");
    this.line.setAttribute("role", "status");
    this.line.dataset.testid = "astro-says";
    this.controls = makeElement(this.document, "div", "astro-screen__controls");
    const stage = makeElement(this.document, "div", "astro-screen__stage");
    stage.append(this.headline, this.line);
    this.overlay.replaceChildren(stage, this.controls);
    this.overlay.addEventListener("keydown", (keyEvent) => {
      if (keyEvent.key === "Escape") {
        this.skip();
      }
    });
  }

  public get isOpen(): boolean {
    return !this.overlay.hidden;
  }

  /** `hasChatModel` is null while the engine has not answered. The journey
   * starts anyway, because the welcome needs no engine; the answer is handed
   * to the journey whenever it arrives. */
  public open(hasChatModel: boolean | null): void {
    this.journey = new GetStartedJourney({ hasChatModel });
    this.stopAwaitingEngine?.();
    this.stopAwaitingEngine = hasChatModel === null ? this.engineWatch.onceAnswered((answeredHasChatModel) => this.engineAnswered(answeredHasChatModel)) : null;
    this.overlay.hidden = false;
    setSiblingsInert(this.overlay, true);
    this.render();
  }

  /** Each step replaces the controls, which drops focus on the floor; the
   * step's first control takes it, so keyboard and VoiceOver users follow. */
  private focusFirstControl(): void {
    this.controls.querySelector<HTMLElement>("button:not([disabled])")?.focus();
  }

  private disposeOpenPanels(): void {
    for (const openPanel of this.openPanels) {
      openPanel.dispose();
    }
    this.openPanels = [];
    this.downloadPanel = null;
  }

  private render(): void {
    this.overlay.dataset.step = this.journey.step;
    this.disposeOpenPanels();
    this.controls.replaceChildren();
    switch (this.journey.step) {
      case GetStartedStep.WELCOME:
        this.renderWelcome();
        break;
      case GetStartedStep.ENGINE_STARTING:
        this.renderEngineStarting();
        break;
      case GetStartedStep.ENGINE_SILENT:
        this.renderEngineSilent();
        break;
      case GetStartedStep.DOWNLOAD:
        this.renderDownload();
        break;
      case GetStartedStep.FINISHED:
        this.close();
        return;
    }
    this.focusFirstControl();
  }

  private say(headline: string, moment: AstroMoment): void {
    this.headline.textContent = headline;
    this.line.textContent = astroLine(moment, "");
  }

  private renderWelcome(): void {
    this.say("Welcome to Astronomical", AstroMoment.WELCOME);
    const beginButton = makeButton(this.document, "astro-get-started", "Get started", false);
    beginButton.addEventListener("click", () => this.advance(() => this.journey.begin()));
    this.controls.append(this.actionRow(beginButton));
  }

  /** Redraws only when the answer moved a waiting journey on; an answer that
   * arrives during the welcome changes nothing on screen. */
  private engineAnswered(hasChatModel: boolean): void {
    const stepBeforeAnswer = this.journey.step;
    this.journey.engineAnswered(hasChatModel);
    if (this.isOpen && this.journey.step !== stepBeforeAnswer) {
      this.render();
    }
  }

  private renderEngineStarting(): void {
    this.say("Starting up", AstroMoment.ENGINE_STARTING);
    const waitingDots = makeElement(this.document, "div", "get-started__waiting");
    waitingDots.setAttribute("aria-hidden", "true");
    waitingDots.append(makeElement(this.document, "span", ""), makeElement(this.document, "span", ""), makeElement(this.document, "span", ""));
    const note = makeElement(this.document, "p", "astro-screen__note", "The engine is starting on this Mac. The first time takes a few seconds.");
    this.controls.append(waitingDots, note);
    const silenceTimer = setTimeout(() => this.advance(() => this.journey.engineStayedSilent()), ENGINE_PATIENCE.silentAfterMilliseconds);
    this.openPanels.push({ dispose: () => clearTimeout(silenceTimer) });
    void this.engineWatch.ask();
  }

  private renderEngineSilent(): void {
    this.say("The engine didn't start", AstroMoment.ENGINE_WONT_START);
    const restartEngine = this.restartEngine;
    const retryButton = makeButton(this.document, "astro-engine-retry", restartEngine !== null ? "Restart the engine" : "Try again", false);
    retryButton.addEventListener("click", () => {
      restartEngine?.();
      this.advance(() => this.journey.retryEngine());
    });
    const skipButton = makeButton(this.document, "astro-skip", "Skip for now", true);
    skipButton.addEventListener("click", () => this.skip());
    const statusNote = makeElement(this.document, "p", "astro-screen__note", "The local server on this Mac didn't answer. The Astronomical icon in the menu bar shows its status.");
    this.controls.append(this.actionRow(retryButton, skipButton), statusNote);
  }

  private renderDownload(): void {
    this.say("Download a model", AstroMoment.NEEDS_CHAT_MODEL);
    const downloadPanel = new ChatModelDownloadPanel(this.document, ID_PREFIX, this.library, {
      canSkip: true,
      openModelsPage: this.openModelsPage,
      say: (moment) => {
        this.line.textContent = astroLine(moment, "");
      },
      isShowing: () => this.isOpen && this.journey.step === GetStartedStep.DOWNLOAD,
      onChatModelArrived: () => this.chatModelArrived(),
      onContinue: () => this.advance(() => this.journey.leaveDownload()),
    });
    this.controls.append(downloadPanel.element);
    this.openPanels.push(downloadPanel);
    this.downloadPanel = downloadPanel;
    void this.focusDownloadWhenOffered(downloadPanel);
  }

  /**
   * Called when the person comes back to the window. Someone who pressed
   * "Choose a different model" downloads on the Models page in the browser;
   * on their return this screen would still offer the first download, with
   * nothing to say a model is already here. If a chat model has appeared and
   * no download was started from this screen, the journey ends in the chat.
   */
  public async noticeModelInstalledElsewhere(): Promise<void> {
    if (!this.isWaitingOnDownloadChoice()) {
      return;
    }
    if ((await this.library.hasChatModel()) !== true) {
      return;
    }
    // The answer took a moment; the person may have pressed a button meanwhile.
    if (this.isWaitingOnDownloadChoice()) {
      this.advance(() => this.journey.chatModelFoundElsewhere());
    }
  }

  private isWaitingOnDownloadChoice(): boolean {
    return this.isOpen && this.journey.step === GetStartedStep.DOWNLOAD && this.downloadPanel?.isFollowingDownload !== true;
  }

  /** A headline that still says "Download a model" over a finished download
   * reads as one more thing to do, so it changes while the screen is on show. */
  private chatModelArrived(): void {
    this.journey.chatModelArrived();
    if (this.isOpen && this.journey.step === GetStartedStep.DOWNLOAD) {
      this.headline.textContent = "Your model is ready";
    }
  }

  /** The panel draws its buttons only once it has read the catalog, after
   * render() looked for a control to focus and found none. Focus moves in
   * when they exist, unless the person has already left this step. */
  private async focusDownloadWhenOffered(downloadPanel: ChatModelDownloadPanel): Promise<void> {
    await downloadPanel.loadOffer();
    if (this.isOpen && this.journey.step === GetStartedStep.DOWNLOAD && this.openPanels.includes(downloadPanel)) {
      this.focusFirstControl();
    }
  }

  private actionRow(...buttons: HTMLButtonElement[]): HTMLElement {
    const actionRow = makeElement(this.document, "div", "astro-screen__actions");
    actionRow.append(...buttons);
    return actionRow;
  }

  private advance(move: () => void): void {
    move();
    this.render();
  }

  private skip(): void {
    this.journey.skip();
    this.render();
  }

  private close(): void {
    this.disposeOpenPanels();
    this.stopAwaitingEngine?.();
    this.stopAwaitingEngine = null;
    this.overlay.hidden = true;
    setSiblingsInert(this.overlay, false);
    this.onFinished();
  }
}
