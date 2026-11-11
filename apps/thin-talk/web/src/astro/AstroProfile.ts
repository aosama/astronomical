import { AstroStyle, parseAstroStyle } from "./AstroStyle";
import { AstroColours, parseAstroColours } from "./AstroColours";

/** Long enough for a first name or a nickname, short enough to fit Astro's
 * speech bubble and the home greeting on one line. */
export const MAXIMUM_NICKNAME_CHARACTERS = 20;

/**
 * What Astro remembers about the person, kept only on this Mac: the nickname
 * Astro greets them by, the style they picked, the colours of Astro's galaxy,
 * whether they have finished the first flight, and whether Astro is shown at
 * all (the Focus switch).
 *
 * Immutable: every change returns a new profile, so a view can never edit the
 * copy another view is drawing from.
 */
export class AstroProfile {
  public readonly nickname: string;
  public readonly style: AstroStyle;
  public readonly colours: AstroColours;
  public readonly hasFinishedFirstFlight: boolean;
  public readonly isAstroShown: boolean;

  public constructor(init: { nickname?: string; style?: AstroStyle; colours?: AstroColours; hasFinishedFirstFlight?: boolean; isAstroShown?: boolean }) {
    this.nickname = AstroProfile.cleanNickname(init.nickname != null ? init.nickname : "");
    this.style = init.style != null ? init.style : AstroStyle.EXPLORER;
    this.colours = init.colours != null ? init.colours : AstroColours.SILVER;
    this.hasFinishedFirstFlight = init.hasFinishedFirstFlight === true;
    this.isAstroShown = init.isAstroShown !== false;
  }

  /** A fresh install: no name yet, the first flight still ahead, Astro shown. */
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

  public withStyle(style: AstroStyle): AstroProfile {
    return new AstroProfile({ ...this.fields(), style });
  }

  public withColours(colours: AstroColours): AstroProfile {
    return new AstroProfile({ ...this.fields(), colours });
  }

  public withFirstFlightFinished(): AstroProfile {
    return new AstroProfile({ ...this.fields(), hasFinishedFirstFlight: true });
  }

  public withAstroShown(isAstroShown: boolean): AstroProfile {
    return new AstroProfile({ ...this.fields(), isAstroShown });
  }

  public toPlain(): { nickname: string; style: string; colours: string; hasFinishedFirstFlight: boolean; isAstroShown: boolean } {
    return this.fields();
  }

  /** Reads a saved profile. Null or anything that is not an object is a
   * newcomer, so a missing or damaged file starts the first flight again
   * rather than failing the window. */
  public static fromPlain(storedProfile: unknown): AstroProfile {
    if (typeof storedProfile !== "object" || storedProfile === null) {
      return AstroProfile.newcomer();
    }
    const profileRecord = storedProfile as Record<string, unknown>;
    const storedNickname = profileRecord["nickname"];
    return new AstroProfile({
      nickname: typeof storedNickname === "string" ? storedNickname : "",
      style: parseAstroStyle(profileRecord["style"]),
      colours: parseAstroColours(profileRecord["colours"]),
      hasFinishedFirstFlight: profileRecord["hasFinishedFirstFlight"] === true,
      isAstroShown: profileRecord["isAstroShown"] !== false,
    });
  }

  private fields(): { nickname: string; style: AstroStyle; colours: AstroColours; hasFinishedFirstFlight: boolean; isAstroShown: boolean } {
    return {
      nickname: this.nickname,
      style: this.style,
      colours: this.colours,
      hasFinishedFirstFlight: this.hasFinishedFirstFlight,
      isAstroShown: this.isAstroShown,
    };
  }
}
