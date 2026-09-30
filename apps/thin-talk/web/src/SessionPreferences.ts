import { ThinkingEffort } from "./ThinkingEffort";

/** The composer's default font size in points, matching the theme's composerInput. */
const DEFAULT_COMPOSER_FONT_SIZE = 14;
const MIN_COMPOSER_FONT_SIZE = 10;
const MAX_COMPOSER_FONT_SIZE = 24;

/**
 * The user's durable chat preferences, persisted through the session bridge so a
 * deliberate choice survives a relaunch while a fresh install still starts on
 * the defaults.
 */
export class SessionPreferences {
  public thinkingEffort: ThinkingEffort;
  public composerFontSize: number;

  public constructor(init: { thinkingEffort?: ThinkingEffort; composerFontSize?: number }) {
    this.thinkingEffort = init.thinkingEffort ?? ThinkingEffort.DEFAULT;
    this.composerFontSize = init.composerFontSize ?? DEFAULT_COMPOSER_FONT_SIZE;
  }

  public static default(): SessionPreferences {
    return new SessionPreferences({});
  }

  public clampComposerFontSize(size: number): number {
    return Math.max(MIN_COMPOSER_FONT_SIZE, Math.min(MAX_COMPOSER_FONT_SIZE, size));
  }

  public toPlain(): { thinkingEffort: string; composerFontSize: number } {
    return { thinkingEffort: this.thinkingEffort, composerFontSize: this.composerFontSize };
  }

  public static fromPlain(plain: unknown): SessionPreferences {
    if (typeof plain !== "object" || plain === null) {
      return SessionPreferences.default();
    }
    const record = plain as { thinkingEffort?: unknown; composerFontSize?: unknown };
    const effort = typeof record.thinkingEffort === "string" ? ThinkingEffort.parse(record.thinkingEffort) : ThinkingEffort.DEFAULT;
    const size = typeof record.composerFontSize === "number" ? record.composerFontSize : DEFAULT_COMPOSER_FONT_SIZE;
    return new SessionPreferences({ thinkingEffort: effort, composerFontSize: size });
  }
}
