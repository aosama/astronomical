/** The screens of Astro's first flight, in the order a newcomer meets them. */
export enum FirstFlightStep {
  HELLO = "hello",
  NAME = "name",
  STYLE = "style",
  /** A side trip from STYLE for a parent setting the Mac up. */
  GROWN_UP = "grownUp",
  /** Only when no chat model is installed: Astro needs a brain first. */
  WAKE = "wake",
  SKILLS = "skills",
  FINISHED = "finished",
}

/**
 * The order of the first flight, with no drawing and no network: the view asks
 * this object where to go next, so every path through the journey can be
 * tested without a page.
 *
 * Each move is accepted only from the step it belongs to; a stray click
 * arriving after the view has moved on is ignored instead of skipping a
 * screen.
 */
export class FirstFlightJourney {
  private currentStep: FirstFlightStep = FirstFlightStep.HELLO;
  private hasBrain: boolean;

  public constructor(init: { hasBrain: boolean }) {
    this.hasBrain = init.hasBrain;
  }

  public get step(): FirstFlightStep {
    return this.currentStep;
  }

  public greet(): void {
    this.moveFrom(FirstFlightStep.HELLO, FirstFlightStep.NAME);
  }

  public giveName(): void {
    this.moveFrom(FirstFlightStep.NAME, FirstFlightStep.STYLE);
  }

  /** After a style is picked the brain decides the route: a Mac that already
   * has a chat model goes straight to the skills. */
  public chooseStyle(): void {
    this.moveFrom(FirstFlightStep.STYLE, this.hasBrain ? FirstFlightStep.SKILLS : FirstFlightStep.WAKE);
  }

  public visitGrownUpNote(): void {
    this.moveFrom(FirstFlightStep.STYLE, FirstFlightStep.GROWN_UP);
  }

  public leaveGrownUpNote(): void {
    this.moveFrom(FirstFlightStep.GROWN_UP, FirstFlightStep.STYLE);
  }

  /** The brain finished downloading. The view stays on WAKE so Astro can
   * celebrate; the person moves on with `leaveWake`. */
  public brainArrived(): void {
    this.hasBrain = true;
  }

  /** Allowed while the download still runs: the daemon keeps downloading, and
   * a newcomer should not have to watch a progress bar to explore. */
  public leaveWake(): void {
    this.moveFrom(FirstFlightStep.WAKE, FirstFlightStep.SKILLS);
  }

  public finish(): void {
    this.moveFrom(FirstFlightStep.SKILLS, FirstFlightStep.FINISHED);
  }

  /** "I'll explore on my own" works from every screen. */
  public skip(): void {
    this.currentStep = FirstFlightStep.FINISHED;
  }

  private moveFrom(expectedStep: FirstFlightStep, nextStep: FirstFlightStep): void {
    if (this.currentStep === expectedStep) {
      this.currentStep = nextStep;
    }
  }
}
