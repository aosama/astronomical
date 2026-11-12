/** The screens of Settings. CLOSED means the page is not on show. */
export enum YourAstroStep {
  CLOSED = "closed",
  PAGE = "page",
  NAME = "name",
  PRIVACY = "privacy",
}

const STEPS_THAT_RETURN_TO_THE_PAGE: ReadonlySet<YourAstroStep> = new Set([YourAstroStep.NAME, YourAstroStep.PRIVACY]);

/**
 * Where Settings is, with no drawing: the page lists every setting, and each
 * one that needs a screen of its own opens it and comes straight back.
 *
 * As in FirstFlightJourney, a move is accepted only from the step it belongs
 * to, so a late second click never lands on the wrong screen.
 */
export class YourAstroJourney {
  private currentStep: YourAstroStep = YourAstroStep.CLOSED;

  public get step(): YourAstroStep {
    return this.currentStep;
  }

  /** Opening always starts on the page, whatever was on show last time. */
  public open(): void {
    this.currentStep = YourAstroStep.PAGE;
  }

  public editName(): void {
    this.moveFrom(YourAstroStep.PAGE, YourAstroStep.NAME);
  }

  public readPrivacyNote(): void {
    this.moveFrom(YourAstroStep.PAGE, YourAstroStep.PRIVACY);
  }

  /** Save, Cancel and the "Settings" link all come back here. */
  public backToPage(): void {
    if (STEPS_THAT_RETURN_TO_THE_PAGE.has(this.currentStep)) {
      this.currentStep = YourAstroStep.PAGE;
    }
  }

  public close(): void {
    this.currentStep = YourAstroStep.CLOSED;
  }

  /** Escape backs out one level: from a setting's screen to the page, from the page out. */
  public escape(): void {
    if (this.currentStep === YourAstroStep.PAGE) {
      this.close();
      return;
    }
    this.backToPage();
  }

  private moveFrom(expectedStep: YourAstroStep, nextStep: YourAstroStep): void {
    if (this.currentStep === expectedStep) {
      this.currentStep = nextStep;
    }
  }
}
