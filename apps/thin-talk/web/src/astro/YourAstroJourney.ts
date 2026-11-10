/** The screens of Your Astro. CLOSED means the page is not on show. */
export enum YourAstroStep {
  CLOSED = "closed",
  PAGE = "page",
  NAME = "name",
  STYLE = "style",
  GROWN_UP = "grownUp",
}

const STEPS_THAT_RETURN_TO_THE_PAGE: ReadonlySet<YourAstroStep> = new Set([YourAstroStep.NAME, YourAstroStep.STYLE, YourAstroStep.GROWN_UP]);

/**
 * Where Your Astro is, with no drawing: the page lists every answer, and each
 * answer opens on its own and comes straight back to the page instead of
 * carrying on through the rest of the first flight.
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

  public editStyle(): void {
    this.moveFrom(YourAstroStep.PAGE, YourAstroStep.STYLE);
  }

  public readGrownUpNote(): void {
    this.moveFrom(YourAstroStep.PAGE, YourAstroStep.GROWN_UP);
  }

  /** Save, Cancel and the "Your Astro" link all come back here. */
  public backToPage(): void {
    if (STEPS_THAT_RETURN_TO_THE_PAGE.has(this.currentStep)) {
      this.currentStep = YourAstroStep.PAGE;
    }
  }

  public close(): void {
    this.currentStep = YourAstroStep.CLOSED;
  }

  /** Escape backs out one level: from an answer to the page, from the page out. */
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
