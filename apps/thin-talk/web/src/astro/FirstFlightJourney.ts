/** The screens of the get-started journey, in the order a newcomer meets them. */
export enum FirstFlightStep {
  WELCOME = "welcome",
  /** Only when the engine has not answered by the time "Get started" is
   * pressed: the next screen depends on whether a chat model is installed. */
  ENGINE_STARTING = "engineStarting",
  /** The engine stayed silent too long: the screen says so and offers a restart. */
  ENGINE_SILENT = "engineSilent",
  /** Only when no chat model is installed: the one thing a new Mac needs. */
  DOWNLOAD = "download",
  FINISHED = "finished",
}

/**
 * The order of the get-started journey, with no drawing and no network: the
 * view asks this object where to go next, so every path through the journey
 * can be tested without a page.
 *
 * The journey exists to get a chat model onto the Mac and nothing else: a
 * newcomer cannot get an answer without one, and every other question (a
 * name, a look) can wait in Settings. A Mac that already has a chat model
 * goes from the welcome straight to the chat.
 *
 * Each move is accepted only from the step it belongs to; a stray click
 * arriving after the view has moved on is ignored instead of skipping a
 * screen.
 */
export class FirstFlightJourney {
  private currentStep: FirstFlightStep = FirstFlightStep.WELCOME;
  /** Null while the engine has not said whether a chat model is installed. */
  private hasChatModel: boolean | null;

  public constructor(init: { hasChatModel: boolean | null }) {
    this.hasChatModel = init.hasChatModel;
  }

  public get step(): FirstFlightStep {
    return this.currentStep;
  }

  /** "Get started". The welcome needs no engine, so on a first launch the
   * engine's start-up time is usually spent before anyone has to wait. */
  public begin(): void {
    this.moveFrom(FirstFlightStep.WELCOME, this.hasChatModel === null ? FirstFlightStep.ENGINE_STARTING : FirstFlightJourney.stepAfterEngineAnswer(this.hasChatModel));
  }

  /** The engine answered. Recorded on any step; a journey waiting on the
   * answer moves on at once. */
  public engineAnswered(hasChatModel: boolean): void {
    this.hasChatModel = hasChatModel;
    if (this.currentStep === FirstFlightStep.ENGINE_STARTING || this.currentStep === FirstFlightStep.ENGINE_SILENT) {
      this.currentStep = FirstFlightJourney.stepAfterEngineAnswer(hasChatModel);
    }
  }

  public engineStayedSilent(): void {
    this.moveFrom(FirstFlightStep.ENGINE_STARTING, FirstFlightStep.ENGINE_SILENT);
  }

  /** After a restart the journey waits again. */
  public retryEngine(): void {
    this.moveFrom(FirstFlightStep.ENGINE_SILENT, FirstFlightStep.ENGINE_STARTING);
  }

  private static stepAfterEngineAnswer(hasChatModel: boolean): FirstFlightStep {
    return hasChatModel ? FirstFlightStep.FINISHED : FirstFlightStep.DOWNLOAD;
  }

  /** The chat model finished downloading. The view stays on DOWNLOAD so the
   * screen can say so; the person moves on with `leaveDownload`. */
  public chatModelArrived(): void {
    this.hasChatModel = true;
  }

  /** Allowed before and while the download runs: the daemon keeps
   * downloading, and a newcomer should not have to watch a progress bar. */
  public leaveDownload(): void {
    this.moveFrom(FirstFlightStep.DOWNLOAD, FirstFlightStep.FINISHED);
  }

  /** Escape and "Skip for now" work from every screen. */
  public skip(): void {
    this.currentStep = FirstFlightStep.FINISHED;
  }

  private moveFrom(expectedStep: FirstFlightStep, nextStep: FirstFlightStep): void {
    if (this.currentStep === expectedStep) {
      this.currentStep = nextStep;
    }
  }
}
