import { makeButton, makeElement, setSiblingsInert } from "./AstroDom";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment, astroLine } from "./AstroLines";
import type { AstroProfile } from "./AstroProfile";
import { buildNameForm } from "./NameForm";
import { buildPrivacyNotePanel } from "./PrivacyNote";
import { YourAstroJourney, YourAstroStep } from "./YourAstroJourney";
import { YourAstroModels } from "./YourAstroModels";
import { buildAboutYouSection, buildAppearanceSection, buildPrivacySection } from "./YourAstroSections";
import type { YourAstroRowActions } from "./YourAstroSections";

/**
 * What Settings asks of its owner. AstroCompanion owns the saved profile:
 * the page only reads it through `currentProfile` and asks for changes, so a
 * change made elsewhere while the page is open (View > Show or Hide Astro) is
 * never overwritten by an older copy. Each save answers whether it reached
 * the disk, so a failed save is shown instead of looking saved.
 */
export interface YourAstroCallbacks {
  currentProfile(): AstroProfile;
  saveNickname(typedNickname: string): Promise<boolean>;
  toggleShown(): Promise<boolean>;
  /** Saves the profile as it is now, after an earlier save failed. */
  retrySave(): Promise<boolean>;
  /** Shows the get-started journey again; the page has already closed. */
  replay(): void;
  onClosed(): void;
}

/**
 * Settings: the chat window's few settings on one page, each changed on its
 * own. Opened from the app menu's Settings (⌘,). The name and the privacy
 * note open a screen of their own whose Save, Cancel or Back come back here.
 */
export class YourAstroView {
  private readonly overlay: HTMLElement;
  private readonly document: Document;
  private readonly callbacks: YourAstroCallbacks;
  private readonly line: HTMLElement;
  private readonly models: YourAstroModels;
  private readonly journey = new YourAstroJourney();
  private answers: HTMLElement | null = null;
  private saveAlert: HTMLElement | null = null;
  private hasFailedSave = false;

  public constructor(init: { overlay: HTMLElement; library: AstroLibrary; callbacks: YourAstroCallbacks }) {
    this.overlay = init.overlay;
    this.document = init.overlay.ownerDocument;
    this.callbacks = init.callbacks;
    this.line = makeElement(this.document, "p", "first-flight__line");
    this.line.setAttribute("role", "status");
    this.line.dataset.testid = "your-astro-says";
    this.models = new YourAstroModels(this.document, init.library, {
      say: (moment) => this.say(moment),
      sayPageLine: () => this.say(this.hasFailedSave ? AstroMoment.SAVE_FAILED : AstroMoment.SETTINGS),
      isShowing: () => this.journey.step === YourAstroStep.PAGE,
    });
    this.overlay.addEventListener("keydown", (keyEvent) => {
      if (keyEvent.key === "Escape") {
        this.escape();
      }
    });
  }

  public get isOpen(): boolean {
    return !this.overlay.hidden;
  }

  public open(): void {
    this.journey.open();
    this.hasFailedSave = false;
    this.overlay.hidden = false;
    setSiblingsInert(this.overlay, true);
    this.render();
  }

  /** The profile changed: redraw the answers in place, leaving the model
   * group's download panels running. */
  public refresh(): void {
    if (this.journey.step === YourAstroStep.PAGE) {
      this.redrawAnswers();
    }
  }

  /** A model landed: the models group checks again. */
  public refreshModels(): void {
    if (this.journey.step === YourAstroStep.PAGE) {
      void this.models.refresh();
    }
  }

  private profile(): AstroProfile {
    return this.callbacks.currentProfile();
  }

  private say(moment: AstroMoment): void {
    this.line.textContent = astroLine(moment, this.profile().nickname);
  }

  private render(): void {
    const step = this.journey.step;
    this.overlay.dataset.step = step;
    this.models.dispose();
    this.answers = this.saveAlert = null;
    switch (step) {
      case YourAstroStep.PAGE:
        this.renderPage();
        break;
      case YourAstroStep.NAME:
        this.renderName();
        break;
      case YourAstroStep.PRIVACY:
        this.renderPrivacy();
        break;
      case YourAstroStep.CLOSED:
        this.close();
        return;
    }
    this.overlay.querySelector<HTMLElement>("input, .your-astro__list button:not([disabled]), .your-astro__step button:not([disabled])")?.focus();
  }

  private renderPage(): void {
    this.say(this.hasFailedSave ? AstroMoment.SAVE_FAILED : AstroMoment.SETTINGS);
    const intro = makeElement(this.document, "div", "your-astro__intro");
    intro.append(makeElement(this.document, "h1", "your-astro__title", "Settings"), this.line);
    const doneButton = makeButton(this.document, "your-astro-done", "Done", false);
    doneButton.addEventListener("click", () => this.closeAndReturn());
    const topBar = makeElement(this.document, "div", "your-astro__top-bar");
    topBar.append(doneButton);
    const list = makeElement(this.document, "div", "your-astro__list");
    this.saveAlert = this.makeSaveAlert();
    this.answers = makeElement(this.document, "div", "your-astro__answers");
    list.append(topBar, this.saveAlert, this.answers);
    const page = makeElement(this.document, "div", "your-astro__page");
    page.append(intro, list);
    this.overlay.replaceChildren(page);
    this.redrawAnswers();
    void this.models.refresh();
  }

  /** Rebuilds the answer groups, keeping keyboard focus on the control that
   * had it, so flipping the switch never drops VoiceOver. */
  private redrawAnswers(): void {
    const answers = this.answers;
    if (answers === null) {
      return;
    }
    const focusedID = this.document.activeElement instanceof HTMLElement ? this.document.activeElement.id : "";
    const profile = this.profile();
    const rowActions = this.rowActions();
    answers.replaceChildren(buildAboutYouSection(this.document, profile, rowActions), this.models.element, buildAppearanceSection(this.document, profile, rowActions), buildPrivacySection(this.document, rowActions));
    if (focusedID.length > 0) {
      this.document.getElementById(focusedID)?.focus();
    }
  }

  private rowActions(): YourAstroRowActions {
    return {
      editName: () => this.advance(() => this.journey.editName()),
      toggleShown: () => void this.showSaveOutcome(this.callbacks.toggleShown()),
      readPrivacyNote: () => this.advance(() => this.journey.readPrivacyNote()),
      replay: () => {
        this.journey.close();
        this.hideOverlay();
        this.callbacks.replay();
      },
    };
  }

  private makeSaveAlert(): HTMLElement {
    const saveAlert = makeElement(this.document, "div", "your-astro__alert");
    saveAlert.setAttribute("role", "alert");
    saveAlert.dataset.testid = "your-astro-save-failed";
    const retryButton = makeButton(this.document, "your-astro-retry-save", "Try again", false);
    retryButton.addEventListener("click", () => void this.showSaveOutcome(this.callbacks.retrySave()));
    saveAlert.append(makeElement(this.document, "span", "", "Your last change wasn't saved on this Mac."), retryButton);
    saveAlert.hidden = !this.hasFailedSave;
    return saveAlert;
  }

  /** Shows whether a save reached the disk. The change stays on screen
   * either way; the alert says it is not kept yet and offers to try again. */
  private async showSaveOutcome(saving: Promise<boolean>): Promise<void> {
    const isSaved = await saving;
    this.hasFailedSave = !isSaved;
    if (this.journey.step !== YourAstroStep.PAGE) {
      return;
    }
    if (this.saveAlert !== null) {
      this.saveAlert.hidden = isSaved;
    }
    this.say(isSaved ? AstroMoment.SETTINGS : AstroMoment.SAVE_FAILED);
  }

  private renderName(): void {
    this.say(AstroMoment.ASK_NAME);
    const cancelButton = makeButton(this.document, "your-astro-name-cancel", "Cancel", true);
    cancelButton.addEventListener("click", () => this.advance(() => this.journey.backToPage()));
    const nameForm = buildNameForm(this.document, {
      fieldID: "your-astro-nickname",
      nickname: this.profile().nickname,
      confirmButton: makeButton(this.document, "your-astro-name-save", "Save", false),
      secondaryButton: cancelButton,
      onSubmit: (typedNickname) => this.saveAndReturn(this.callbacks.saveNickname(typedNickname)),
    });
    this.showStep(nameForm);
  }

  private renderPrivacy(): void {
    this.say(AstroMoment.PRIVACY);
    const backButton = makeButton(this.document, "your-astro-privacy-back", "Back to Settings", false);
    backButton.addEventListener("click", () => this.advance(() => this.journey.backToPage()));
    const actionRow = makeElement(this.document, "div", "first-flight__actions");
    actionRow.append(backButton);
    this.showStep(buildPrivacyNotePanel(this.document), actionRow);
  }

  /** A setting's own screen: the way back, the page's line, and the controls. */
  private showStep(...controls: HTMLElement[]): void {
    const backLink = makeButton(this.document, "your-astro-back", "‹ Settings", true);
    backLink.classList.add("your-astro__back");
    backLink.addEventListener("click", () => this.advance(() => this.journey.backToPage()));
    const stage = makeElement(this.document, "div", "first-flight__stage");
    stage.append(this.line);
    const controlArea = makeElement(this.document, "div", "first-flight__controls");
    controlArea.append(...controls);
    const step = makeElement(this.document, "div", "your-astro__step");
    step.append(stage, controlArea);
    this.overlay.replaceChildren(backLink, step);
  }

  private saveAndReturn(saving: Promise<boolean>): void {
    this.advance(() => this.journey.backToPage());
    void this.showSaveOutcome(saving);
  }

  private advance(move: () => void): void {
    move();
    this.render();
  }

  private escape(): void {
    this.journey.escape();
    this.render();
  }

  private closeAndReturn(): void {
    this.journey.close();
    this.render();
  }

  private close(): void {
    this.hideOverlay();
    this.callbacks.onClosed();
  }

  private hideOverlay(): void {
    this.models.dispose();
    this.overlay.hidden = true;
    setSiblingsInert(this.overlay, false);
  }
}
