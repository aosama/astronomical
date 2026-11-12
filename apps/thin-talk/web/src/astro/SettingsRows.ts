import { makeButton, makeElement } from "./AstroDom";

/* The building blocks every group in Settings is made of, kept apart from
   the groups themselves so the models group (SettingsModels) and the answer
   groups (SettingsSections) draw identical rows. */

/** A titled group of rows. Its rows go in the returned section, after the heading. */
export function makeSection(document: Document, title: string, testID: string): HTMLElement {
  const section = makeElement(document, "section", "settings__section");
  section.dataset.testid = testID;
  section.append(makeElement(document, "h2", "settings__section-title", title));
  return section;
}

/** One answer: what it is, what it is now, and the control that changes it.
 * A null label gives a full-width row for a sentence of its own. */
export function makeRow(document: Document, label: string | null, value: string | HTMLElement, control: HTMLElement | null): HTMLElement {
  const row = makeElement(document, "div", "settings__row");
  if (label !== null) {
    row.append(makeElement(document, "span", "settings__row-label", label));
  }
  const valueElement = typeof value === "string" ? makeElement(document, "span", "settings__row-value", value) : value;
  valueElement.classList.add("settings__row-value");
  row.append(valueElement);
  if (control !== null) {
    row.append(control);
  }
  return row;
}

/** A row's change button: quiet, so the page reads as a list and not a wall of calls to action. */
export function rowButton(document: Document, testID: string, label: string, onClick: () => void): HTMLButtonElement {
  const button = makeButton(document, testID, label, true);
  button.classList.add("settings__row-button");
  button.addEventListener("click", onClick);
  return button;
}
