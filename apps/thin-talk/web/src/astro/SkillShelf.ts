import { SKILL_CARDS, SkillCardKind } from "./SkillCard";
import type { SkillCard } from "./SkillCard";

/** How far a pointer must travel before a press counts as a drag, so a slightly
 * shaky click on a trackpad still reads as a click. */
const DRAG_THRESHOLD_PIXELS = 6;

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
  /** A card was clicked, pressed with the keyboard, or dropped on the target. */
  onChoose(kind: SkillCardKind): void;
  /** Whether a drag released at this point landed on Astro. */
  isDropTarget(clientX: number, clientY: number): boolean;
  /** The drag is over Astro (true) or has left it (false). */
  onHoverTarget(isOverTarget: boolean): void;
}

/** Builds the row of skill cards, each a real button so keyboard and
 * VoiceOver users can teach Astro without dragging. `idPrefix` keeps ids
 * unique when the first flight and Astro's home are both in the page
 * ("astro" for the flight, "astro-home" for home). */
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
  attachCardDragging(cardButton, card.kind, callbacks);
  return cardButton;
}

/**
 * Lets a card be dragged onto Astro. A press that never travels is a click,
 * and a drag released anywhere else springs the card back without choosing
 * it, so a dropped-by-accident card does nothing.
 */
function attachCardDragging(cardButton: HTMLButtonElement, kind: SkillCardKind, callbacks: SkillShelfCallbacks): void {
  let dragStart: { x: number; y: number } | null = null;
  let hasMoved = false;
  cardButton.addEventListener("pointerdown", (pointerEvent) => {
    dragStart = { x: pointerEvent.clientX, y: pointerEvent.clientY };
    hasMoved = false;
    cardButton.setPointerCapture?.(pointerEvent.pointerId);
  });
  cardButton.addEventListener("pointermove", (pointerEvent) => {
    if (dragStart === null) {
      return;
    }
    const travelX = pointerEvent.clientX - dragStart.x;
    const travelY = pointerEvent.clientY - dragStart.y;
    if (!hasMoved && Math.hypot(travelX, travelY) < DRAG_THRESHOLD_PIXELS) {
      return;
    }
    hasMoved = true;
    cardButton.classList.add("skill-card--dragging");
    cardButton.style.transform = `translate(${travelX}px, ${travelY}px) rotate(-3deg)`;
    callbacks.onHoverTarget(callbacks.isDropTarget(pointerEvent.clientX, pointerEvent.clientY));
  });
  cardButton.addEventListener("pointerup", (pointerEvent) => {
    const wasDragged = dragStart !== null && hasMoved;
    dragStart = null;
    cardButton.classList.remove("skill-card--dragging");
    cardButton.style.transform = "";
    callbacks.onHoverTarget(false);
    if (wasDragged && callbacks.isDropTarget(pointerEvent.clientX, pointerEvent.clientY)) {
      callbacks.onChoose(kind);
    }
  });
  // A drag the system cancels (the pointer leaves the window, a gesture takes
  // over) springs the card back without choosing it.
  cardButton.addEventListener("pointercancel", () => {
    dragStart = null;
    hasMoved = false;
    cardButton.classList.remove("skill-card--dragging");
    cardButton.style.transform = "";
    callbacks.onHoverTarget(false);
  });
  // A keyboard press or a pointer press that never moved arrives as a click.
  cardButton.addEventListener("click", () => {
    if (!hasMoved) {
      callbacks.onChoose(kind);
    }
    hasMoved = false;
  });
}
