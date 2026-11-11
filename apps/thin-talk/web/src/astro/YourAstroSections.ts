import { makeElement } from "./AstroDom";
import type { AstroProfile } from "./AstroProfile";
import { buildColourLookChips } from "./AstroColourLook";
import type { AstroColourLook } from "./AstroColourLook";
import { makeRow, makeSection, rowButton } from "./YourAstroRows";
import { styleSummary } from "./StyleChoices";

/** Prefix for Your Astro's control ids. */
export const YOUR_ASTRO_ID_PREFIX = "your-astro";

/** What the page's rows do; YourAstroView turns each into a journey move or a save. */
export interface YourAstroRowActions {
  editName(): void;
  editStyle(): void;
  chooseColourLook(colourLook: AstroColourLook): void;
  toggleShown(): void;
  readGrownUpNote(): void;
  replay(): void;
}

export function buildAboutYouSection(document: Document, profile: AstroProfile, actions: YourAstroRowActions): HTMLElement {
  const section = makeSection(document, "About you", "your-astro-about-you");
  const hasNickname = profile.nickname.length > 0;
  const nameValue = makeElement(document, "span", hasNickname ? "" : "your-astro__row-value--empty", hasNickname ? profile.nickname : "Not set. Astro says “Hi!”");
  nameValue.dataset.testid = "your-astro-nickname-shown";
  section.append(
    makeRow(document, "Name", nameValue, rowButton(document, "your-astro-name-change", hasNickname ? "Change" : "Add a name", () => actions.editName())),
    makeRow(document, "Style", styleSummary(profile.style), rowButton(document, "your-astro-style-change", "Change", () => actions.editStyle())),
  );
  return section;
}

export function buildLookSection(document: Document, profile: AstroProfile, actions: YourAstroRowActions): HTMLElement {
  const section = makeSection(document, "Look", "your-astro-look");
  const colourLookChips = buildColourLookChips(document, YOUR_ASTRO_ID_PREFIX, profile.colourLook, (colourLook) => actions.chooseColourLook(colourLook));
  section.append(makeRow(document, "Colours", colourLookChips, null), makeRow(document, "Show Astro", shownDetail(document, profile.isAstroShown), shownSwitch(document, profile.isAstroShown, actions)));
  return section;
}

export function buildGrownUpSection(document: Document, actions: YourAstroRowActions): HTMLElement {
  const section = makeSection(document, "For grown-ups", "your-astro-grown-ups");
  section.append(
    makeRow(document, null, "What Astro does and what stays private", rowButton(document, "your-astro-grown-up-read", "Read", () => actions.readGrownUpNote())),
    makeRow(document, null, "Replay the welcome from the start", rowButton(document, "your-astro-replay", "Replay", () => actions.replay())),
  );
  return section;
}

function shownDetail(document: Document, isAstroShown: boolean): HTMLElement {
  const detailText = isAstroShown ? "On. Turn off for a plain chat window." : "Off. Chat looks plain until Astro is back.";
  return makeElement(document, "span", "your-astro__row-value--muted", detailText);
}

/** A real switch (role="switch" with aria-checked), so VoiceOver reads it as
 * on or off rather than as a button with a changing label. */
function shownSwitch(document: Document, isAstroShown: boolean, actions: YourAstroRowActions): HTMLButtonElement {
  const switchButton = makeElement(document, "button", "your-astro__switch");
  switchButton.type = "button";
  switchButton.id = switchButton.dataset.testid = "your-astro-shown";
  switchButton.setAttribute("role", "switch");
  switchButton.setAttribute("aria-checked", String(isAstroShown));
  switchButton.setAttribute("aria-label", "Show Astro");
  switchButton.append(makeElement(document, "span", "your-astro__switch-knob"));
  switchButton.addEventListener("click", () => actions.toggleShown());
  return switchButton;
}
