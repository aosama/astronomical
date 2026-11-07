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
  /** The chat model the user chose last, or null before any choice. A stored id
   * that no longer appears in the daemon's catalog falls back to the first
   * model at load time. */
  public modelID: string | null;
  /** The Tools switch. On by default: tools are offered only to models that
   * declare tool support, and the tools offered so far read nothing personal
   * and change nothing. */
  public toolsEnabled: boolean;

  public constructor(init: {
    thinkingEffort?: ThinkingEffort;
    composerFontSize?: number;
    modelID?: string | null;
    toolsEnabled?: boolean;
  }) {
    const configuredEffort: ThinkingEffort | undefined = init.thinkingEffort;
    this.thinkingEffort = configuredEffort != null ? configuredEffort : ThinkingEffort.DEFAULT;
    const configuredFontSize: number | undefined = init.composerFontSize;
    this.composerFontSize =
      configuredFontSize != null ? configuredFontSize : DEFAULT_COMPOSER_FONT_SIZE;
    const configuredModelID: string | null | undefined = init.modelID;
    this.modelID = configuredModelID != null ? configuredModelID : null;
    this.toolsEnabled = init.toolsEnabled !== false;
  }

  public static default(): SessionPreferences {
    return new SessionPreferences({});
  }

  public clampComposerFontSize(size: number): number {
    return Math.max(MIN_COMPOSER_FONT_SIZE, Math.min(MAX_COMPOSER_FONT_SIZE, size));
  }

  public toPlain(): { thinkingEffort: string; composerFontSize: number; modelID: string | null; toolsEnabled: boolean } {
    return {
      thinkingEffort: this.thinkingEffort,
      composerFontSize: this.composerFontSize,
      modelID: this.modelID,
      toolsEnabled: this.toolsEnabled,
    };
  }

  public static fromPlain(plain: unknown): SessionPreferences {
    if (typeof plain !== "object" || plain === null) {
      return SessionPreferences.default();
    }
    const record = plain as { thinkingEffort?: unknown; composerFontSize?: unknown; modelID?: unknown; toolsEnabled?: unknown };
    const effort = typeof record.thinkingEffort === "string" ? ThinkingEffort.parse(record.thinkingEffort) : ThinkingEffort.DEFAULT;
    const size = typeof record.composerFontSize === "number" ? record.composerFontSize : DEFAULT_COMPOSER_FONT_SIZE;
    const modelID = typeof record.modelID === "string" ? record.modelID : null;
    // Only an explicit false turns tools off, so preferences saved before the
    // switch existed keep the default.
    const toolsEnabled = record.toolsEnabled !== false;
    return new SessionPreferences({ thinkingEffort: effort, composerFontSize: size, modelID, toolsEnabled });
  }
}
