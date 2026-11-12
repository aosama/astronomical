import { makeElement } from "./AstroDom";
import type { AstroProfile } from "./AstroProfile";
import { makeRow, makeSection, rowButton } from "./YourAstroRows";

/** What the page's rows do; YourAstroView turns each into a journey move or a save. */
export interface YourAstroRowActions {
  editName(): void;
  toggleShown(): void;
  readPrivacyNote(): void;
  replay(): void;
}

export function buildAboutYouSection(document: Document, profile: AstroProfile, actions: YourAstroRowActions): HTMLElement {
  const section = makeSection(document, "About you", "your-astro-about-you");
  const hasNickname = profile.nickname.length > 0;
  const nameValue = makeElement(document, "span", hasNickname ? "" : "your-astro__row-value--empty", hasNickname ? profile.nickname : "Not set. Astro greets you without a name.");
  nameValue.dataset.testid = "your-astro-nickname-shown";
  section.append(makeRow(document, "Name", nameValue, rowButton(document, "your-astro-name-change", hasNickname ? "Change" : "Add a name", () => actions.editName())));
  return section;
}

export function buildAppearanceSection(document: Document, profile: AstroProfile, actions: YourAstroRowActions): HTMLElement {
  const section = makeSection(document, "Appearance", "your-astro-appearance");
  section.append(makeRow(document, "Show Astro", shownDetail(document, profile.isAstroShown), shownSwitch(document, profile.isAstroShown, actions)));
  return section;
}

export function buildPrivacySection(document: Document, actions: YourAstroRowActions): HTMLElement {
  const section = makeSection(document, "Privacy", "your-astro-privacy");
  section.append(
    makeRow(document, null, "What Astro does and what stays private", rowButton(document, "your-astro-privacy-read", "Read", () => actions.readPrivacyNote())),
    makeRow(document, null, "Show the welcome again", rowButton(document, "your-astro-replay", "Show", () => actions.replay())),
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
