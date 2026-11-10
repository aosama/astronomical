/** The screens of Astro's first flight, in the order a newcomer meets them. */
export enum FirstFlightStep {
  HELLO = "hello",
  NAME = "name",
  STYLE = "style",
  /** A side trip from STYLE for a parent setting the Mac up. */
  GROWN_UP = "grownUp",
  /** Only when the engine has not answered by the time the style is picked:
   * Astro waits for it, since the next screen depends on the answer. */
  ENGINE_STARTING = "engineStarting",
  /** The engine stayed silent too long: Astro says so and offers a restart. */
  ENGINE_SILENT = "engineSilent",
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
  /** Null while the engine has not said whether a chat model is installed. */
  private hasBrain: boolean | null;

  public constructor(init: { hasBrain: boolean | null }) {
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
   * has a chat model goes straight to the skills, one without goes to the
   * brain download, and one whose engine has not answered yet waits for it.
   * Hello, name and style need no engine, so on a first launch the engine's
   * start-up time is usually spent before anyone has to wait. */
  public chooseStyle(): void {
    this.moveFrom(FirstFlightStep.STYLE, this.hasBrain === null ? FirstFlightStep.ENGINE_STARTING : this.stepAfterEngineAnswer(this.hasBrain));
  }

  /** The engine answered. Recorded on any step; a flight waiting on the
   * answer moves on at once. */
  public engineAnswered(hasBrain: boolean): void {
    this.hasBrain = hasBrain;
    if (this.currentStep === FirstFlightStep.ENGINE_STARTING || this.currentStep === FirstFlightStep.ENGINE_SILENT) {
      this.currentStep = this.stepAfterEngineAnswer(hasBrain);
    }
  }

  public engineStayedSilent(): void {
    this.moveFrom(FirstFlightStep.ENGINE_STARTING, FirstFlightStep.ENGINE_SILENT);
  }

  /** After a restart Astro waits again. */
  public retryEngine(): void {
    this.moveFrom(FirstFlightStep.ENGINE_SILENT, FirstFlightStep.ENGINE_STARTING);
  }

  private stepAfterEngineAnswer(hasBrain: boolean): FirstFlightStep {
    return hasBrain ? FirstFlightStep.SKILLS : FirstFlightStep.WAKE;
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
