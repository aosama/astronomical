import { AstroStyle } from "./astro/AstroStyle";

/**
 * Who the conversation is with, as the transcript draws it: plain Thin Talk
 * ("You" and "Assistant"), or Astro with the person's nickname and style.
 * The Focus switch moves between the two without touching the conversation.
 */
export class TranscriptPersona {
  public readonly isAstroShown: boolean;
  public readonly style: AstroStyle;
  public readonly nickname: string;

  public constructor(init: { isAstroShown: boolean; style: AstroStyle; nickname: string }) {
    this.isAstroShown = init.isAstroShown;
    this.style = init.style;
    this.nickname = init.nickname;
  }

  public static classic(): TranscriptPersona {
    return new TranscriptPersona({ isAstroShown: false, style: AstroStyle.EXPLORER, nickname: "" });
  }

  /** A stable string for render signatures: a changed persona redraws every
   * message, an unchanged one redraws none. */
  public signature(): string {
    return JSON.stringify([this.isAstroShown, this.style, this.nickname]);
  }
}
