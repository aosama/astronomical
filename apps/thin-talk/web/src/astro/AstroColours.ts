import { makeButton, makeElement } from "./AstroDom";

/** The person's "Colours" choice: how the arms of Astro's galaxy are tinted,
 * and the download meter with them. Silver is the stylesheet's own look, the
 * stars in Astro's ink with nothing added; each other choice is a class that
 * defines four colours (see astro.css). Silver is the default because one
 * quiet colour reads as grown-up, where a rainbow by default read as made for
 * small children. */
export enum AstroColours {
  SILVER = "silver",
  RAINBOW = "rainbow",
  AURORA = "aurora",
  SUNSET = "sunset",
}

const COLOURS_CHOICES: readonly { colours: AstroColours; label: string }[] = [
  { colours: AstroColours.SILVER, label: "Silver" },
  { colours: AstroColours.RAINBOW, label: "Rainbow" },
  { colours: AstroColours.AURORA, label: "Aurora" },
  { colours: AstroColours.SUNSET, label: "Sunset" },
];

/** Reads stored colours; anything unknown is the default silver, so a damaged
 * profile only costs the colour choice. */
export function parseAstroColours(storedColours: unknown): AstroColours {
  const knownChoice = COLOURS_CHOICES.find((coloursChoice) => coloursChoice.colours === storedColours);
  return knownChoice !== undefined ? knownChoice.colours : AstroColours.SILVER;
}

/** Sets the colours class on one ancestor of every Astro figure (the window's
 * root), so the welcome, home, Your Astro and each answer draw the same
 * colours. Silver sets none: it is the look the stylesheet draws unaided. */
export function applyAstroColours(coloursOwner: HTMLElement, colours: AstroColours): void {
  for (const coloursChoice of COLOURS_CHOICES) {
    coloursOwner.classList.toggle(`astro-colours--${coloursChoice.colours}`, coloursChoice.colours === colours && colours !== AstroColours.SILVER);
  }
}

/** The colour chips, shared by the brain download's waiting moment and Your
 * Astro. The pressed chip follows the click at once; saving is the caller's. */
export function buildColoursChips(document: Document, idPrefix: string, pickedColours: AstroColours, onChoose: (colours: AstroColours) => void): HTMLElement {
  const chipRow = makeElement(document, "div", "astro-colours-chips");
  chipRow.setAttribute("role", "group");
  chipRow.setAttribute("aria-label", "Colours");
  for (const coloursChoice of COLOURS_CHOICES) {
    const coloursChip = makeButton(document, `${idPrefix}-colours-${coloursChoice.colours}`, coloursChoice.label, true);
    coloursChip.classList.add("astro-chip");
    coloursChip.setAttribute("aria-pressed", String(coloursChoice.colours === pickedColours));
    coloursChip.addEventListener("click", () => {
      for (const otherChip of chipRow.querySelectorAll<HTMLButtonElement>(".astro-chip")) {
        otherChip.setAttribute("aria-pressed", String(otherChip === coloursChip));
      }
      onChoose(coloursChoice.colours);
    });
    chipRow.append(coloursChip);
  }
  return chipRow;
}
