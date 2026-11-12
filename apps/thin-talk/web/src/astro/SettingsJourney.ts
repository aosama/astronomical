/** The screens of Settings. CLOSED means the page is not on show. */
export enum SettingsStep {
  CLOSED = "closed",
  PAGE = "page",
  NAME = "name",
  PRIVACY = "privacy",
}

const STEPS_THAT_RETURN_TO_THE_PAGE: ReadonlySet<SettingsStep> = new Set([SettingsStep.NAME, SettingsStep.PRIVACY]);

/**
 * Where Settings is, with no drawing: the page lists every setting, and each
 * one that needs a screen of its own opens it and comes straight back.
 *
 * As in GetStartedJourney, a move is accepted only from the step it belongs
 * to, so a late second click never lands on the wrong screen.
 */
export class SettingsJourney {
  private currentStep: SettingsStep = SettingsStep.CLOSED;

  public get step(): SettingsStep {
    return this.currentStep;
  }

  /** Opening always starts on the page, whatever was on show last time. */
  public open(): void {
    this.currentStep = SettingsStep.PAGE;
  }

  public editName(): void {
    this.moveFrom(SettingsStep.PAGE, SettingsStep.NAME);
  }

  public readPrivacyNote(): void {
    this.moveFrom(SettingsStep.PAGE, SettingsStep.PRIVACY);
  }

  /** Save, Cancel and the "Settings" link all come back here. */
  public backToPage(): void {
    if (STEPS_THAT_RETURN_TO_THE_PAGE.has(this.currentStep)) {
      this.currentStep = SettingsStep.PAGE;
    }
  }

  public close(): void {
    this.currentStep = SettingsStep.CLOSED;
  }

  /** Escape backs out one level: from a setting's screen to the page, from the page out. */
  public escape(): void {
    if (this.currentStep === SettingsStep.PAGE) {
      this.close();
      return;
    }
    this.backToPage();
  }

  private moveFrom(expectedStep: SettingsStep, nextStep: SettingsStep): void {
    if (this.currentStep === expectedStep) {
      this.currentStep = nextStep;
    }
  }
}
