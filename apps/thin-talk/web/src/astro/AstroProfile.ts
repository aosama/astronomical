/** Long enough for a first name or a nickname, short enough to fit the home
 * greeting on one line. */
export const MAXIMUM_NICKNAME_CHARACTERS = 20;

/** The saved file's key for "has finished the get-started journey". It keeps
 * the name it was first saved under: every profile already on disk carries
 * it, and reading a new key instead would show each of those people the
 * welcome again after an update. */
const FINISHED_GET_STARTED_STORED_KEY = "hasFinishedFirstFlight";

/**
 * What Astro remembers about the person, kept only on this Mac: the optional
 * name Astro greets them by, whether they have finished the get-started
 * journey, and whether Astro is shown at all (the Focus switch).
 *
 * Immutable: every change returns a new profile, so a view can never edit the
 * copy another view is drawing from.
 */
export class AstroProfile {
  public readonly nickname: string;
  public readonly hasFinishedGetStarted: boolean;
  public readonly isAstroShown: boolean;

  public constructor(init: { nickname?: string; hasFinishedGetStarted?: boolean; isAstroShown?: boolean }) {
    this.nickname = AstroProfile.cleanNickname(init.nickname != null ? init.nickname : "");
    this.hasFinishedGetStarted = init.hasFinishedGetStarted === true;
    this.isAstroShown = init.isAstroShown !== false;
  }

  /** A fresh install: no name, the get-started journey still ahead, Astro shown. */
  public static newcomer(): AstroProfile {
    return new AstroProfile({});
  }

  /**
   * Trims, folds runs of whitespace and drops control characters, then caps
   * the length by whole characters (never splitting an emoji), so whatever is
   * typed renders as one tidy line.
   */
  public static cleanNickname(typedNickname: string): string {
    // Whitespace folds first: newlines and tabs are control characters too,
    // and "Maya<newline>Rose" should read "Maya Rose", not "MayaRose".
    const printableNickname = Array.from(typedNickname.replace(/\s+/gu, " "))
      .filter((nicknameCharacter) => !/\p{Cc}/u.test(nicknameCharacter))
      .join("")
      .trim();
    return Array.from(printableNickname).slice(0, MAXIMUM_NICKNAME_CHARACTERS).join("").trim();
  }

  public withNickname(typedNickname: string): AstroProfile {
    return new AstroProfile({ ...this.fields(), nickname: typedNickname });
  }

  public withGetStartedFinished(): AstroProfile {
    return new AstroProfile({ ...this.fields(), hasFinishedGetStarted: true });
  }

  public withAstroShown(isAstroShown: boolean): AstroProfile {
    return new AstroProfile({ ...this.fields(), isAstroShown });
  }

  /** The profile as it is saved; see FINISHED_GET_STARTED_STORED_KEY. */
  public toPlain(): Record<string, string | boolean> {
    return { nickname: this.nickname, [FINISHED_GET_STARTED_STORED_KEY]: this.hasFinishedGetStarted, isAstroShown: this.isAstroShown };
  }

  /** Reads a saved profile. Null or anything that is not an object is a
   * newcomer, so a missing or damaged file starts the get-started journey
   * again rather than failing the window. Keys an older build saved (a style,
   * a colour choice) are not read. */
  public static fromPlain(storedProfile: unknown): AstroProfile {
    if (typeof storedProfile !== "object" || storedProfile === null) {
      return AstroProfile.newcomer();
    }
    const profileRecord = storedProfile as Record<string, unknown>;
    const storedNickname = profileRecord["nickname"];
    return new AstroProfile({
      nickname: typeof storedNickname === "string" ? storedNickname : "",
      hasFinishedGetStarted: profileRecord[FINISHED_GET_STARTED_STORED_KEY] === true,
      isAstroShown: profileRecord["isAstroShown"] !== false,
    });
  }

  private fields(): { nickname: string; hasFinishedGetStarted: boolean; isAstroShown: boolean } {
    return {
      nickname: this.nickname,
      hasFinishedGetStarted: this.hasFinishedGetStarted,
      isAstroShown: this.isAstroShown,
    };
  }
}
