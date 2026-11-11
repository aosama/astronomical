import { makeButton, makeElement, setSiblingsInert } from "./AstroDom";
import { AstroFigure } from "./AstroFigure";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment, astroLine } from "./AstroLines";
import { AstroProfile } from "./AstroProfile";
import { AstroStyle } from "./AstroStyle";
import type { AstroColours } from "./AstroColours";
import { ENGINE_PATIENCE } from "./EngineWatch";
import type { EngineWatch } from "./EngineWatch";
import { FirstFlightJourney, FirstFlightStep } from "./FirstFlightJourney";
import { buildGrownUpNotePanel } from "./GrownUpNote";
import { buildNameForm } from "./NameForm";
import { SketchbookOffer } from "./SketchbookOffer";
import { SkillCardKind, skillCard } from "./SkillCard";
import { buildSkillShelf } from "./SkillShelf";
import { buildStyleChoiceRow } from "./StyleChoices";
import { WakeUpPanel } from "./WakeUpPanel";

/** Prefix for the flight's control ids; Astro's home uses "astro-home". */
const ID_PREFIX = "astro";

/**
 * What the flight reports. It hands back only the answers it asked for, so
 * AstroCompanion stays the one owner of the saved profile: a change made
 * elsewhere while the flight is open (View > Show or Hide Astro) is never
 * overwritten by the flight's older copy.
 */
export interface FirstFlightCallbacks {
  /** Saved as soon as it is chosen, so quitting halfway keeps it. */
  onNicknameChosen(typedNickname: string): void;
  onStyleChosen(style: AstroStyle): void;
  onColoursChosen(colours: AstroColours): void;
  /** `chosenSkill` is the skill taught to Astro, whose starter prompt fills
   * the message field; null when the person skipped or chose nothing. */
  onFinished(chosenSkill: SkillCardKind | null): void;
}

/** A panel showing a download, which stops listening when it leaves the screen. */
interface DisposablePanel {
  dispose(): void;
}

/**
 * Astro's first flight: a full-window welcome that replaces Thin Talk's old
 * empty screen for a newcomer. It draws one step at a time and asks
 * FirstFlightJourney where to go next; downloads run through AstroLibrary.
 */
export class FirstFlightView {
  private readonly overlay: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly callbacks: FirstFlightCallbacks;
  private readonly engineWatch: EngineWatch;
  /** Asks the host to restart the engine; null where the host cannot. */
  private readonly restartEngine: (() => void) | null;
  /** Stops listening for the engine's first answer. */
  private stopAwaitingEngine: (() => void) | null = null;
  private readonly figure: AstroFigure;
  private readonly bubble: HTMLElement;
  private readonly controls: HTMLElement;
  private journey = new FirstFlightJourney({ hasBrain: null });
  /** The name and style as the flight shows them in Astro's words; the saved
   * profile belongs to AstroCompanion. */
  private shownProfile = AstroProfile.newcomer();
  private chosenSkill: SkillCardKind | null = null;
  private openPanels: DisposablePanel[] = [];

  public constructor(init: {
    overlay: HTMLElement;
    library: AstroLibrary;
    engineWatch: EngineWatch;
    restartEngine: (() => void) | null;
    callbacks: FirstFlightCallbacks;
  }) {
    this.overlay = init.overlay;
    this.document = init.overlay.ownerDocument;
    this.library = init.library;
    this.callbacks = init.callbacks;
    this.engineWatch = init.engineWatch;
    this.restartEngine = init.restartEngine;
    this.figure = new AstroFigure(this.document, "first-flight__astro");
    this.bubble = makeElement(this.document, "p", "first-flight__bubble");
    this.bubble.setAttribute("role", "status");
    this.bubble.dataset.testid = "astro-says";
    this.controls = makeElement(this.document, "div", "first-flight__controls");
    const skipButton = makeButton(this.document, "astro-skip", "I'll explore on my own", true);
    skipButton.classList.add("first-flight__skip");
    skipButton.addEventListener("click", () => this.skip());
    const stage = makeElement(this.document, "div", "first-flight__stage");
    stage.append(this.figure.element, this.bubble);
    this.overlay.replaceChildren(makeElement(this.document, "div", "first-flight__sky"), skipButton, stage, this.controls);
    this.overlay.addEventListener("keydown", (keyEvent) => {
      if (keyEvent.key === "Escape") {
        this.skip();
      }
    });
  }

  public get isOpen(): boolean {
    return !this.overlay.hidden;
  }

  /** `hasBrain` is null while the engine has not answered. The flight starts
   * anyway, because hello, name and style need no engine; the answer is
   * handed to the journey whenever it arrives. */
  public open(profile: AstroProfile, hasBrain: boolean | null): void {
    this.shownProfile = profile;
    this.journey = new FirstFlightJourney({ hasBrain });
    this.stopAwaitingEngine?.();
    this.stopAwaitingEngine = hasBrain === null ? this.engineWatch.onceAnswered((answeredHasBrain) => this.engineAnswered(answeredHasBrain)) : null;
    this.chosenSkill = null;
    this.overlay.hidden = false;
    setSiblingsInert(this.overlay, true);
    this.render();
  }

  /** Each step replaces the controls, which drops focus on the floor; the
   * step's first control takes it, so keyboard and VoiceOver users follow. */
  private focusFirstControl(): void {
    this.controls.querySelector<HTMLElement>("input, button:not([disabled])")?.focus();
  }

  private disposeOpenPanels(): void {
    for (const openPanel of this.openPanels) {
      openPanel.dispose();
    }
    this.openPanels = [];
  }

  private render(): void {
    this.overlay.dataset.step = this.journey.step;
    this.overlay.classList.toggle("first-flight--pilot", this.shownProfile.style === AstroStyle.PILOT);
    this.disposeOpenPanels();
    this.controls.replaceChildren();
    switch (this.journey.step) {
      case FirstFlightStep.HELLO:
        this.renderHello();
        break;
      case FirstFlightStep.NAME:
        this.renderName();
        break;
      case FirstFlightStep.STYLE:
        this.renderStyle();
        break;
      case FirstFlightStep.GROWN_UP:
        this.renderGrownUp();
        break;
      case FirstFlightStep.ENGINE_STARTING:
        this.renderEngineStarting();
        break;
      case FirstFlightStep.ENGINE_SILENT:
        this.renderEngineSilent();
        break;
      case FirstFlightStep.WAKE:
        this.renderWake();
        break;
      case FirstFlightStep.SKILLS:
        this.renderSkills();
        break;
      case FirstFlightStep.FINISHED:
        this.close();
        return;
    }
    this.figure.hop();
    this.focusFirstControl();
  }

  private say(moment: AstroMoment): void {
    this.bubble.textContent = astroLine(moment, this.shownProfile.style, this.shownProfile.nickname);
  }

  private renderHello(): void {
    this.say(AstroMoment.HELLO);
    const greetButton = makeButton(this.document, "astro-greet", "Hi Astro!", false);
    greetButton.addEventListener("click", () => this.advance(() => this.journey.greet()));
    this.controls.append(this.actionRow(greetButton));
  }

  private renderName(): void {
    this.say(AstroMoment.ASK_NAME);
    const skipNameButton = makeButton(this.document, "astro-name-skip", "Skip", true);
    skipNameButton.addEventListener("click", () => this.advance(() => this.journey.giveName()));
    const nameForm = buildNameForm(this.document, {
      fieldID: "astro-nickname",
      nickname: this.shownProfile.nickname,
      confirmButton: makeButton(this.document, "astro-name-confirm", "That's me", false),
      secondaryButton: skipNameButton,
      onSubmit: (typedNickname) => {
        this.shownProfile = this.shownProfile.withNickname(typedNickname);
        this.callbacks.onNicknameChosen(typedNickname);
        this.advance(() => this.journey.giveName());
      },
    });
    this.controls.append(nameForm);
  }

  private renderStyle(): void {
    this.say(AstroMoment.ASK_STYLE);
    this.controls.append(
      buildStyleChoiceRow(this.document, {
        idPrefix: ID_PREFIX,
        includesGrownUpNote: true,
        pickedStyle: null,
        savedStyle: null,
        onChoose: (style) => this.chooseStyle(style),
      }),
    );
  }

  private chooseStyle(style: AstroStyle | null): void {
    if (style === null) {
      this.advance(() => this.journey.visitGrownUpNote());
      return;
    }
    this.shownProfile = this.shownProfile.withStyle(style);
    this.callbacks.onStyleChosen(style);
    this.advance(() => this.journey.chooseStyle());
  }

  private renderGrownUp(): void {
    this.say(AstroMoment.GROWN_UP);
    const notePanel = buildGrownUpNotePanel(this.document);
    const backButton = makeButton(this.document, "astro-grown-up-back", "Back to choosing a style", false);
    backButton.addEventListener("click", () => this.advance(() => this.journey.leaveGrownUpNote()));
    this.controls.append(notePanel, this.actionRow(backButton));
  }

  /** Redraws only when the answer moved a waiting flight on; an answer that
   * arrives during hello, name or style changes nothing on screen. */
  private engineAnswered(hasBrain: boolean): void {
    const stepBeforeAnswer = this.journey.step;
    this.journey.engineAnswered(hasBrain);
    if (this.isOpen && this.journey.step !== stepBeforeAnswer) {
      this.render();
    }
  }

  private renderEngineStarting(): void {
    this.say(AstroMoment.ENGINE_STARTING);
    const waitingDots = makeElement(this.document, "div", "first-flight__waiting");
    waitingDots.setAttribute("aria-hidden", "true");
    waitingDots.append(makeElement(this.document, "span", ""), makeElement(this.document, "span", ""), makeElement(this.document, "span", ""));
    const note = makeElement(this.document, "p", "first-flight__note", "Astro's engine is starting on this Mac. The first time takes a few seconds.");
    this.controls.append(waitingDots, note);
    const silenceTimer = setTimeout(() => this.advance(() => this.journey.engineStayedSilent()), ENGINE_PATIENCE.silentAfterMilliseconds);
    this.openPanels.push({ dispose: () => clearTimeout(silenceTimer) });
    void this.engineWatch.ask();
  }

  private renderEngineSilent(): void {
    this.say(AstroMoment.ENGINE_WONT_WAKE);
    const restartEngine = this.restartEngine;
    const retryButton = makeButton(this.document, "astro-engine-retry", restartEngine !== null ? "Restart Astro's engine" : "Try again", false);
    retryButton.addEventListener("click", () => {
      restartEngine?.();
      this.advance(() => this.journey.retryEngine());
    });
    const grownUpNote = makeElement(this.document, "div", "first-flight__panel");
    grownUpNote.append(makeElement(this.document, "p", "", "For grown-ups: the local server on this Mac didn't answer. The Astronomical icon in the menu bar shows its status."));
    this.controls.append(this.actionRow(retryButton), grownUpNote);
  }

  private renderWake(): void {
    this.say(AstroMoment.NEEDS_BRAIN);
    const wakeUpPanel = new WakeUpPanel(this.document, ID_PREFIX, this.library, {
      figure: this.figure,
      coloursChoice: {
        colours: this.shownProfile.colours,
        onColoursChosen: (colours) => {
          this.shownProfile = this.shownProfile.withColours(colours);
          this.callbacks.onColoursChosen(colours);
        },
      },
      say: (moment) => this.say(moment),
      isShowing: () => this.isOpen && this.journey.step === FirstFlightStep.WAKE,
      onBrainArrived: () => this.journey.brainArrived(),
      onContinue: () => this.advance(() => this.journey.leaveWake()),
    });
    this.controls.append(wakeUpPanel.element);
    this.openPanels.push(wakeUpPanel);
    void wakeUpPanel.loadOffer();
  }

  private renderSkills(): void {
    this.say(AstroMoment.ASK_SKILL);
    const followUp = makeElement(this.document, "div", "first-flight__follow-up");
    const shelf = buildSkillShelf(this.document, ID_PREFIX, {
      onChoose: (kind) => void this.chooseSkill(kind, followUp),
      isDropTarget: (clientX, clientY) => this.figure.containsPoint(clientX, clientY),
      onHoverTarget: (isOverTarget) => this.overlay.classList.toggle("first-flight--drop-armed", isOverTarget),
    });
    const hint = makeElement(this.document, "p", "first-flight__note", "Drag a card onto Astro, or click one");
    this.controls.append(shelf, hint, followUp);
  }

  private async chooseSkill(kind: SkillCardKind, followUp: HTMLElement): Promise<void> {
    const card = skillCard(kind);
    this.figure.hop();
    if (!card.isAvailable) {
      this.say(card.reply);
      return;
    }
    if (kind === SkillCardKind.PICTURES && !(await this.library.hasImageModel())) {
      this.offerSketchbook(followUp);
      return;
    }
    this.chosenSkill = kind;
    this.say(card.reply);
    followUp.replaceChildren(this.homeButton());
  }

  private offerSketchbook(followUp: HTMLElement): void {
    this.say(AstroMoment.NEEDS_SKETCHBOOK);
    this.chosenSkill = SkillCardKind.PICTURES;
    const sketchbookOffer = new SketchbookOffer(this.document, ID_PREFIX, this.library, () => {
      this.chosenSkill = null;
      this.say(AstroMoment.ASK_SKILL);
      followUp.replaceChildren();
    });
    followUp.replaceChildren(sketchbookOffer.element, this.homeButton());
    this.openPanels.push(sketchbookOffer);
    void sketchbookOffer.loadOffer();
  }

  private homeButton(): HTMLButtonElement {
    const homeButton = makeButton(this.document, "astro-go-home", "Go to Astro's home", false);
    homeButton.addEventListener("click", () => this.advance(() => this.journey.finish()));
    return homeButton;
  }

  private actionRow(...buttons: HTMLButtonElement[]): HTMLElement {
    const actionRow = makeElement(this.document, "div", "first-flight__actions");
    actionRow.append(...buttons);
    return actionRow;
  }

  private advance(move: () => void): void {
    move();
    this.render();
  }

  private skip(): void {
    this.chosenSkill = null;
    this.journey.skip();
    this.render();
  }

  private close(): void {
    this.disposeOpenPanels();
    this.stopAwaitingEngine?.();
    this.stopAwaitingEngine = null;
    this.overlay.hidden = true;
    setSiblingsInert(this.overlay, false);
    this.figure.setCharging(false);
    this.callbacks.onFinished(this.chosenSkill);
  }
}
