import { makeButton, makeElement } from "./AstroDom";

/** The colours of the space trail behind Astro. Rainbow is the stylesheet's
 * own palette; the others are classes that override it (see astro.css). */
export enum AstroTrail {
  RAINBOW = "rainbow",
  AURORA = "aurora",
  SUNSET = "sunset",
}

const TRAIL_CHOICES: readonly { trail: AstroTrail; label: string }[] = [
  { trail: AstroTrail.RAINBOW, label: "Rainbow" },
  { trail: AstroTrail.AURORA, label: "Aurora" },
  { trail: AstroTrail.SUNSET, label: "Sunset" },
];

/** Reads a stored trail; anything unknown is the default rainbow, so a damaged
 * profile only costs the colour choice. */
export function parseAstroTrail(storedTrail: unknown): AstroTrail {
  const knownChoice = TRAIL_CHOICES.find((trailChoice) => trailChoice.trail === storedTrail);
  return knownChoice !== undefined ? knownChoice.trail : AstroTrail.RAINBOW;
}

/** Sets the trail class on one ancestor of every Astro figure (the window's
 * root), so the welcome, home and Your Astro always draw the same trail. */
export function applyAstroTrail(trailOwner: HTMLElement, trail: AstroTrail): void {
  for (const trailChoice of TRAIL_CHOICES) {
    trailOwner.classList.toggle(`astro-trail--${trailChoice.trail}`, trailChoice.trail === trail && trail !== AstroTrail.RAINBOW);
  }
}

/** The trail chips, shared by the brain download's waiting moment and Your
 * Astro. The pressed chip follows the click at once; saving is the caller's. */
export function buildTrailChips(document: Document, idPrefix: string, pickedTrail: AstroTrail, onChoose: (trail: AstroTrail) => void): HTMLElement {
  const chipRow = makeElement(document, "div", "astro-trail-chips");
  chipRow.setAttribute("role", "group");
  chipRow.setAttribute("aria-label", "Trail");
  for (const trailChoice of TRAIL_CHOICES) {
    const trailChip = makeButton(document, `${idPrefix}-trail-${trailChoice.trail}`, trailChoice.label, true);
    trailChip.classList.add("astro-chip");
    trailChip.setAttribute("aria-pressed", String(trailChoice.trail === pickedTrail));
    trailChip.addEventListener("click", () => {
      for (const otherChip of chipRow.querySelectorAll<HTMLButtonElement>(".astro-chip")) {
        otherChip.setAttribute("aria-pressed", String(otherChip === trailChip));
      }
      onChoose(trailChoice.trail);
    });
    chipRow.append(trailChip);
  }
  return chipRow;
}
