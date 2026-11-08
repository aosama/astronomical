import { AstroMoment } from "./AstroLines";
import { AstroStyle } from "./AstroStyle";

/** The skills a person can teach Astro by dropping a card on it. */
export enum SkillCardKind {
  CHAT = "chat",
  PICTURES = "pictures",
  LOOK = "look",
  READ = "read",
  VOICE = "voice",
}

/**
 * One card on the shelf: its words, Astro's reply, and the prompt it starts.
 *
 * Every starter prompt only ever fills the message field; the person still
 * presses send, so a card can never start a download, open a file panel or
 * spend time on the GPU without a deliberate click.
 */
export interface SkillCard {
  readonly kind: SkillCardKind;
  readonly title: string;
  readonly caption: string;
  /** False for skills Astro does not have yet (voice): shown, but greyed. */
  readonly isAvailable: boolean;
  readonly reply: AstroMoment;
  readonly starterPrompt: { explorer: string; pilot: string } | null;
}

export const SKILL_CARDS: readonly SkillCard[] = [
  {
    kind: SkillCardKind.CHAT, title: "Chat", caption: "Talk and ask anything", isAvailable: true,
    reply: AstroMoment.SKILL_CHAT, starterPrompt: { explorer: "Why is the sky blue?", pilot: "Explain black holes in three sentences." },
  },
  {
    kind: SkillCardKind.PICTURES, title: "Pictures", caption: "Draw from an idea", isAvailable: true,
    reply: AstroMoment.SKILL_PICTURES,
    starterPrompt: { explorer: "Draw a cat astronaut floating past the moon.", pilot: "Draw a poster of a rocket launch at sunset." },
  },
  {
    kind: SkillCardKind.LOOK, title: "Look", caption: "Show Astro a photo", isAvailable: true,
    reply: AstroMoment.SKILL_LOOK,
    starterPrompt: { explorer: "Look at a photo I pick and tell me what you see.", pilot: "Describe a photo I choose." },
  },
  {
    kind: SkillCardKind.READ, title: "Read", caption: "Show Astro a file", isAvailable: true,
    reply: AstroMoment.SKILL_READ,
    starterPrompt: { explorer: "Read a file I pick and explain it simply.", pilot: "Summarise a file I choose." },
  },
  {
    kind: SkillCardKind.VOICE, title: "Voice", caption: "Coming soon", isAvailable: false,
    reply: AstroMoment.SKILL_VOICE, starterPrompt: null,
  },
];

export function skillCard(kind: SkillCardKind): SkillCard {
  const matchingCard = SKILL_CARDS.find((card) => card.kind === kind);
  if (matchingCard === undefined) {
    throw new Error(`there is no skill card for "${kind}"`);
  }
  return matchingCard;
}

export function starterPromptFor(card: SkillCard, style: AstroStyle): string | null {
  if (card.starterPrompt === null) {
    return null;
  }
  return style === AstroStyle.PILOT ? card.starterPrompt.pilot : card.starterPrompt.explorer;
}
