import { makeElement } from "./AstroDom";
import { MAXIMUM_NICKNAME_CHARACTERS } from "./AstroProfile";

/** What a name form needs from the screen that shows it. The buttons come
 * from the caller because the first flight offers "That's me" and "Skip"
 * while Your Astro offers "Save" and "Cancel". */
export interface NameFormParts {
  /** The field's id and data-testid. */
  readonly fieldID: string;
  readonly nickname: string;
  /** Becomes the form's submit button, so Return in the field confirms. */
  readonly confirmButton: HTMLButtonElement;
  readonly secondaryButton: HTMLButtonElement;
  onSubmit(typedNickname: string): void;
}

/**
 * The "what should I call you?" form, shared by the first flight and Your
 * Astro so the field's limits and its privacy promise can never differ
 * between the two.
 */
export function buildNameForm(document: Document, parts: NameFormParts): HTMLFormElement {
  const nameForm = makeElement(document, "form", "first-flight__name");
  const nameLabel = makeElement(document, "label", "first-flight__note", "Your first name or a nickname");
  const nameField = makeElement(document, "input", "first-flight__name-field");
  nameField.id = nameLabel.htmlFor = parts.fieldID;
  nameField.dataset.testid = parts.fieldID;
  nameField.maxLength = MAXIMUM_NICKNAME_CHARACTERS;
  nameField.autocomplete = "off";
  nameField.value = parts.nickname;
  const privacyNote = makeElement(document, "p", "first-flight__note", "It stays on this Mac. Astro never sends it anywhere.");
  parts.confirmButton.type = "submit";
  nameForm.addEventListener("submit", (submitEvent) => {
    submitEvent.preventDefault();
    parts.onSubmit(nameField.value);
  });
  const actionRow = makeElement(document, "div", "first-flight__actions");
  actionRow.append(parts.confirmButton, parts.secondaryButton);
  nameForm.append(nameLabel, nameField, privacyNote, actionRow);
  return nameForm;
}
