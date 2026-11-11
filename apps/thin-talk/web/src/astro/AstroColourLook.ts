import { makeButton, makeElement } from "./AstroDom";

/** The person's colour look, picked under "Colours": how the arms of Astro's galaxy are tinted,
 * and the download meter with them. Silver is the stylesheet's own look, the
 * stars in Astro's ink with nothing added; each other choice is a class that
 * defines four colours (see astro.css). Silver is the default because one
 * quiet colour reads as grown-up, where a rainbow by default read as made for
 * small children. */
export enum AstroColourLook {
  SILVER = "silver",
  RAINBOW = "rainbow",
  AURORA = "aurora",
  SUNSET = "sunset",
}

const COLOUR_LOOK_CHOICES: readonly { colourLook: AstroColourLook; label: string }[] = [
  { colourLook: AstroColourLook.SILVER, label: "Silver" },
  { colourLook: AstroColourLook.RAINBOW, label: "Rainbow" },
  { colourLook: AstroColourLook.AURORA, label: "Aurora" },
  { colourLook: AstroColourLook.SUNSET, label: "Sunset" },
];

/** Reads a stored colour look; anything unknown is the default silver, so a damaged
 * profile only costs the colour choice. */
export function parseAstroColourLook(storedColourLook: unknown): AstroColourLook {
  const knownChoice = COLOUR_LOOK_CHOICES.find((colourLookChoice) => colourLookChoice.colourLook === storedColourLook);
  return knownChoice !== undefined ? knownChoice.colourLook : AstroColourLook.SILVER;
}

/** Sets the colour look's class on one ancestor of every Astro figure (the window's
 * root), so the welcome, home, Your Astro and each answer draw the same
 * colours. Silver sets none: it is the look the stylesheet draws unaided. */
export function applyAstroColourLook(colourLookOwner: HTMLElement, colourLook: AstroColourLook): void {
  for (const colourLookChoice of COLOUR_LOOK_CHOICES) {
    colourLookOwner.classList.toggle(`astro-colour-look--${colourLookChoice.colourLook}`, colourLookChoice.colourLook === colourLook && colourLook !== AstroColourLook.SILVER);
  }
}

/** The colour chips, shared by the brain download's waiting moment and Your
 * Astro. The pressed chip follows the click at once; saving is the caller's. */
export function buildColourLookChips(document: Document, idPrefix: string, pickedColourLook: AstroColourLook, onChoose: (colourLook: AstroColourLook) => void): HTMLElement {
  const chipRow = makeElement(document, "div", "astro-colour-look-chips");
  chipRow.setAttribute("role", "group");
  chipRow.setAttribute("aria-label", "Colours");
  for (const colourLookChoice of COLOUR_LOOK_CHOICES) {
    const colourLookChip = makeButton(document, `${idPrefix}-colour-look-${colourLookChoice.colourLook}`, colourLookChoice.label, true);
    colourLookChip.classList.add("astro-chip");
    colourLookChip.setAttribute("aria-pressed", String(colourLookChoice.colourLook === pickedColourLook));
    colourLookChip.addEventListener("click", () => {
      for (const otherChip of chipRow.querySelectorAll<HTMLButtonElement>(".astro-chip")) {
        otherChip.setAttribute("aria-pressed", String(otherChip === colourLookChip));
      }
      onChoose(colourLookChoice.colourLook);
    });
    chipRow.append(colourLookChip);
  }
  return chipRow;
}
