import { makeButton, makeElement, setSiblingsInert } from "./AstroDom";
import { AstroFigure } from "./AstroFigure";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment, astroLine } from "./AstroLines";
import type { AstroProfile } from "./AstroProfile";
import { AstroStyle } from "./AstroStyle";
import type { AstroColours } from "./AstroColours";
import { buildGrownUpNotePanel } from "./GrownUpNote";
import { buildNameForm } from "./NameForm";
import { buildStyleChoiceRow } from "./StyleChoices";
import { YourAstroGifts } from "./YourAstroGifts";
import { YourAstroJourney, YourAstroStep } from "./YourAstroJourney";
import { buildAboutYouSection, buildGrownUpSection, buildLookSection, YOUR_ASTRO_ID_PREFIX } from "./YourAstroSections";
import type { YourAstroRowActions } from "./YourAstroSections";

/**
 * What Your Astro asks of its owner. AstroCompanion owns the saved profile:
 * the page only reads it through `currentProfile` and asks for changes, so a
 * change made elsewhere while the page is open (View > Show or Hide Astro) is
 * never overwritten by an older copy. Each save answers whether it reached
 * the disk, so a failed save is shown instead of looking saved.
 */
export interface YourAstroCallbacks {
  currentProfile(): AstroProfile;
  saveNickname(typedNickname: string): Promise<boolean>;
  saveStyle(style: AstroStyle): Promise<boolean>;
  saveColours(colours: AstroColours): Promise<boolean>;
  toggleShown(): Promise<boolean>;
  /** Saves the profile as it is now, after an earlier save failed. */
  retrySave(): Promise<boolean>;
  /** Plays the first flight again; the page has already closed. */
  replay(): void;
  onClosed(): void;
}

/**
 * Your Astro: every answer from the first flight on one page, each changed on
 * its own. Opened from the app menu's Settings (⌘,). Name and style open the
 * flight's own screens in an edit mode whose Save and Cancel come back here.
 */
export class YourAstroView {
  private readonly overlay: HTMLElement;
  private readonly document: Document;
  private readonly callbacks: YourAstroCallbacks;
  private readonly figure: AstroFigure;
  private readonly bubble: HTMLElement;
  private readonly gifts: YourAstroGifts;
  private readonly journey = new YourAstroJourney();
  private answers: HTMLElement | null = null;
  private saveAlert: HTMLElement | null = null;
  private hasFailedSave = false;
  /** The style card picked on the style screen, not saved until Save. */
  private previewStyle: AstroStyle = AstroStyle.EXPLORER;

  public constructor(init: { overlay: HTMLElement; library: AstroLibrary; callbacks: YourAstroCallbacks }) {
    this.overlay = init.overlay;
    this.document = init.overlay.ownerDocument;
    this.callbacks = init.callbacks;
    this.figure = new AstroFigure(this.document, "first-flight__astro");
    this.bubble = makeElement(this.document, "p", "first-flight__bubble");
    this.bubble.setAttribute("role", "status");
    this.bubble.dataset.testid = "your-astro-says";
    this.gifts = new YourAstroGifts(this.document, init.library, {
      figure: this.figure,
      say: (moment) => this.say(moment),
      sayPageLine: () => this.say(this.hasFailedSave ? AstroMoment.SAVE_FAILED : AstroMoment.YOUR_ASTRO),
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

  /** The profile changed: redraw the answers in place, leaving the gifts'
   * download panels running. */
  public refresh(): void {
    if (this.journey.step === YourAstroStep.PAGE) {
      this.redrawAnswers();
    }
  }

  /** A model landed: the gifts group checks again. */
  public refreshGifts(): void {
    if (this.journey.step === YourAstroStep.PAGE) {
      void this.gifts.refresh();
    }
  }

  private profile(): AstroProfile {
    return this.callbacks.currentProfile();
  }

  private say(moment: AstroMoment): void {
    const shownStyle = this.journey.step === YourAstroStep.STYLE ? this.previewStyle : this.profile().style;
    this.bubble.textContent = astroLine(moment, shownStyle, this.profile().nickname);
  }

  private render(): void {
    const step = this.journey.step;
    this.overlay.dataset.step = step;
    const shownStyle = step === YourAstroStep.STYLE ? this.previewStyle : this.profile().style;
    this.overlay.classList.toggle("your-astro--pilot", shownStyle === AstroStyle.PILOT);
    this.gifts.dispose();
    this.answers = this.saveAlert = null;
    switch (step) {
      case YourAstroStep.PAGE:
        this.renderPage();
        break;
      case YourAstroStep.NAME:
        this.renderName();
        break;
      case YourAstroStep.STYLE:
        this.renderStyle();
        break;
      case YourAstroStep.GROWN_UP:
        this.renderGrownUp();
        break;
      case YourAstroStep.CLOSED:
        this.close();
        return;
    }
    this.figure.hop();
    this.overlay.querySelector<HTMLElement>("input, .your-astro__list button:not([disabled]), .your-astro__step button:not([disabled])")?.focus();
  }

  private renderPage(): void {
    this.say(this.hasFailedSave ? AstroMoment.SAVE_FAILED : AstroMoment.YOUR_ASTRO);
    const intro = makeElement(this.document, "div", "your-astro__intro");
    intro.append(makeElement(this.document, "h1", "your-astro__title", "Your Astro"), this.figure.element, this.bubble, makeElement(this.document, "p", "first-flight__note", "All of this stays on this Mac."));
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
    this.overlay.replaceChildren(makeElement(this.document, "div", "first-flight__sky"), page);
    this.redrawAnswers();
    void this.gifts.refresh();
  }

  /** Rebuilds the answer groups, keeping keyboard focus on the control that
   * had it, so flipping the switch or a colour chip never drops VoiceOver. */
  private redrawAnswers(): void {
    const answers = this.answers;
    if (answers === null) {
      return;
    }
    const focusedID = this.document.activeElement instanceof HTMLElement ? this.document.activeElement.id : "";
    const profile = this.profile();
    const rowActions = this.rowActions();
    answers.replaceChildren(buildAboutYouSection(this.document, profile, rowActions), this.gifts.element, buildLookSection(this.document, profile, rowActions), buildGrownUpSection(this.document, rowActions));
    if (focusedID.length > 0) {
      this.document.getElementById(focusedID)?.focus();
    }
  }

  private rowActions(): YourAstroRowActions {
    return {
      editName: () => this.advance(() => this.journey.editName()),
      editStyle: () => {
        this.previewStyle = this.profile().style;
        this.advance(() => this.journey.editStyle());
      },
      chooseColours: (colours) => void this.showSaveOutcome(this.callbacks.saveColours(colours)),
      toggleShown: () => void this.showSaveOutcome(this.callbacks.toggleShown()),
      readGrownUpNote: () => this.advance(() => this.journey.readGrownUpNote()),
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
    this.say(isSaved ? AstroMoment.YOUR_ASTRO : AstroMoment.SAVE_FAILED);
  }

  private renderName(): void {
    this.say(AstroMoment.ASK_NEW_NAME);
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

  private renderStyle(): void {
    this.say(AstroMoment.STYLE_PREVIEW);
    const choiceRow = buildStyleChoiceRow(this.document, {
      idPrefix: YOUR_ASTRO_ID_PREFIX,
      includesGrownUpNote: false,
      pickedStyle: this.previewStyle,
      savedStyle: this.profile().style,
      onChoose: (style) => this.previewStyleChoice(style),
    });
    const saveButton = makeButton(this.document, "your-astro-style-save", "Save", false);
    saveButton.addEventListener("click", () => this.saveAndReturn(this.callbacks.saveStyle(this.previewStyle)));
    const cancelButton = makeButton(this.document, "your-astro-style-cancel", "Cancel", true);
    cancelButton.addEventListener("click", () => this.advance(() => this.journey.backToPage()));
    const actionRow = makeElement(this.document, "div", "first-flight__actions");
    actionRow.append(saveButton, cancelButton);
    this.showStep(choiceRow, makeElement(this.document, "p", "first-flight__note", "Astro tries the new voice right away, so you can hear it before saving."), actionRow);
  }

  /** A picked card is only a preview: Astro speaks in that voice and the
   * window takes its look, but nothing is saved until Save. */
  private previewStyleChoice(style: AstroStyle | null): void {
    if (style === null) {
      return;
    }
    this.previewStyle = style;
    const focusedID = this.document.activeElement instanceof HTMLElement ? this.document.activeElement.id : "";
    this.render();
    if (focusedID.length > 0) {
      this.document.getElementById(focusedID)?.focus();
    }
  }

  private renderGrownUp(): void {
    this.say(AstroMoment.GROWN_UP);
    const backButton = makeButton(this.document, "your-astro-grown-up-back", "Back to Your Astro", false);
    backButton.addEventListener("click", () => this.advance(() => this.journey.backToPage()));
    const actionRow = makeElement(this.document, "div", "first-flight__actions");
    actionRow.append(backButton);
    this.showStep(buildGrownUpNotePanel(this.document), actionRow);
  }

  /** An edit screen: the way back, Astro with its bubble, and the controls. */
  private showStep(...controls: HTMLElement[]): void {
    const backLink = makeButton(this.document, "your-astro-back", "‹ Your Astro", true);
    backLink.classList.add("your-astro__back");
    backLink.addEventListener("click", () => this.advance(() => this.journey.backToPage()));
    const stage = makeElement(this.document, "div", "first-flight__stage");
    stage.append(this.figure.element, this.bubble);
    const controlArea = makeElement(this.document, "div", "first-flight__controls");
    controlArea.append(...controls);
    const step = makeElement(this.document, "div", "your-astro__step");
    step.append(stage, controlArea);
    this.overlay.replaceChildren(makeElement(this.document, "div", "first-flight__sky"), backLink, step);
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
    this.gifts.dispose();
    this.overlay.hidden = true;
    setSiblingsInert(this.overlay, false);
    this.figure.setCharging(false);
  }
}
