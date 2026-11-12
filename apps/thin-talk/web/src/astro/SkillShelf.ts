import { SKILL_CARDS, SkillCardKind } from "./SkillCard";
import type { SkillCard } from "./SkillCard";

/** The cards' icons, drawn inline: the page may load images only from its own
 * bundle, and inline SVG follows the card's text colour. */
const CARD_ICON_PATHS: Record<SkillCardKind, string> = {
  [SkillCardKind.CHAT]: "M4 5h16v10H9l-5 4z",
  [SkillCardKind.PICTURES]: "M3 4h18v16H3zM8 9.5a1.5 1.5 0 1 0 0 .01M3 18l6-5 4 3 3-2 5 4",
  [SkillCardKind.LOOK]: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
  [SkillCardKind.READ]: "M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h5",
  [SkillCardKind.VOICE]: "M12 4v10M8 8v6M16 8v6M4 11v2M20 11v2",
};

export interface SkillShelfCallbacks {
  /** A card was clicked or pressed with the keyboard. */
  onChoose(kind: SkillCardKind): void;
}

/** Builds the row of skill cards, each a real button so keyboard and
 * VoiceOver users can teach Astro as a pointer does. Cards are chosen by a
 * click alone: nothing is drawn for Astro on screen, so there is no place to
 * drop a dragged card. `idPrefix` starts every card's id ("astro-home" on
 * Astro's home). */
export function buildSkillShelf(document: Document, idPrefix: string, callbacks: SkillShelfCallbacks): HTMLElement {
  const shelf = document.createElement("div");
  shelf.className = "skill-shelf";
  shelf.setAttribute("role", "group");
  shelf.setAttribute("aria-label", "Skills to teach Astro");
  for (const card of SKILL_CARDS) {
    shelf.append(buildSkillCardButton(document, idPrefix, card, callbacks));
  }
  return shelf;
}

function buildSkillCardButton(document: Document, idPrefix: string, card: SkillCard, callbacks: SkillShelfCallbacks): HTMLButtonElement {
  const cardButton = document.createElement("button");
  cardButton.type = "button";
  cardButton.className = `skill-card skill-card--${card.kind}`;
  cardButton.dataset.skill = card.kind;
  cardButton.dataset.testid = `${idPrefix}-skill-${card.kind}`;
  cardButton.id = `${idPrefix}-skill-${card.kind}`;
  if (!card.isAvailable) {
    cardButton.setAttribute("aria-disabled", "true");
  }
  cardButton.innerHTML = `<span class="skill-card__icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="${CARD_ICON_PATHS[card.kind]}"/></svg></span>`;
  const cardTitle = document.createElement("strong");
  cardTitle.textContent = card.title;
  const cardCaption = document.createElement("small");
  cardCaption.textContent = card.caption;
  cardButton.append(cardTitle, cardCaption);
  cardButton.addEventListener("click", () => callbacks.onChoose(card.kind));
  return cardButton;
}
