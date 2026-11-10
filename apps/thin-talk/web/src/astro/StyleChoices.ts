import { makeButton, makeElement } from "./AstroDom";
import { AstroStyle } from "./AstroStyle";

/** One card in the style choice. A null style is the grown-up note, which the
 * first flight offers beside the two styles. */
interface StyleChoice {
  readonly name: string;
  readonly title: string;
  readonly detail: string;
  readonly style: AstroStyle | null;
}

const STYLE_CARDS: readonly StyleChoice[] = [
  { name: "explorer", title: "Explorer", detail: "Big buttons, lots of help, and Astro chats a little more. Feels right for about 8 to 12.", style: AstroStyle.EXPLORER },
  { name: "pilot", title: "Pilot", detail: "Sleek and quick. Astro gets to the point. Feels right for about 13 to 18.", style: AstroStyle.PILOT },
];

const GROWN_UP_CARD: StyleChoice = {
  name: "grown-up",
  title: "I'm a grown-up",
  detail: "Setting this up for someone? See what Astro does and what stays private.",
  style: null,
};

/** One line per style for Your Astro's summary row. */
export function styleSummary(style: AstroStyle): string {
  return style === AstroStyle.PILOT ? "Pilot · sleek and quick" : "Explorer · big buttons, lots of help";
}

export interface StyleChoiceOptions {
  /** Prefixes each card's id: "astro" gives "astro-style-pilot". */
  readonly idPrefix: string;
  /** The first flight offers the grown-up note here; Your Astro has its own row for it. */
  readonly includesGrownUpNote: boolean;
  /** The card shown as chosen, or null when none is yet. */
  readonly pickedStyle: AstroStyle | null;
  /** The style already saved, marked "Now" so a preview can be told apart from it. */
  readonly savedStyle: AstroStyle | null;
  onChoose(style: AstroStyle | null): void;
}

/** The Explorer / Pilot cards, shared by the first flight and Your Astro. */
export function buildStyleChoiceRow(document: Document, options: StyleChoiceOptions): HTMLElement {
  const choiceRow = makeElement(document, "div", "first-flight__choices");
  choiceRow.setAttribute("role", "group");
  choiceRow.setAttribute("aria-label", "Pick a style");
  const cards = options.includesGrownUpNote ? [...STYLE_CARDS, GROWN_UP_CARD] : STYLE_CARDS;
  choiceRow.classList.toggle("first-flight__choices--two", !options.includesGrownUpNote);
  for (const styleChoice of cards) {
    choiceRow.append(styleCardButton(document, styleChoice, options));
  }
  return choiceRow;
}

function styleCardButton(document: Document, styleChoice: StyleChoice, options: StyleChoiceOptions): HTMLButtonElement {
  const choiceButton = makeButton(document, `${options.idPrefix}-style-${styleChoice.name}`, "", true);
  choiceButton.className = "first-flight__choice";
  const heading = makeElement(document, "span", "first-flight__choice-heading");
  heading.append(makeElement(document, "strong", "", styleChoice.title));
  if (styleChoice.style !== null && styleChoice.style === options.savedStyle) {
    heading.append(makeElement(document, "span", "first-flight__choice-badge", "Now"));
  }
  choiceButton.append(heading, makeElement(document, "span", "", styleChoice.detail));
  if (styleChoice.style !== null && options.pickedStyle !== null) {
    choiceButton.setAttribute("aria-pressed", String(styleChoice.style === options.pickedStyle));
  }
  choiceButton.addEventListener("click", () => options.onChoose(styleChoice.style));
  return choiceButton;
}
