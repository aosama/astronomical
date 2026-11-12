import { makeElement } from "./AstroDom";
import { MAXIMUM_NICKNAME_CHARACTERS } from "./AstroProfile";

/** What a name form needs from the screen that shows it. */
export interface NameFormParts {
  /** The field's id and data-testid. */
  readonly fieldID: string;
  readonly nickname: string;
  /** Becomes the form's submit button, so Return in the field confirms. */
  readonly confirmButton: HTMLButtonElement;
  readonly secondaryButton: HTMLButtonElement;
  onSubmit(typedNickname: string): void;
}

/** The optional name form in Settings, with the field's limit and its
 * privacy promise beside it. */
export function buildNameForm(document: Document, parts: NameFormParts): HTMLFormElement {
  const nameForm = makeElement(document, "form", "astro-screen__name");
  const nameLabel = makeElement(document, "label", "astro-screen__note", "Your first name or a nickname");
  const nameField = makeElement(document, "input", "astro-screen__name-field");
  nameField.id = nameLabel.htmlFor = parts.fieldID;
  nameField.dataset.testid = parts.fieldID;
  nameField.maxLength = MAXIMUM_NICKNAME_CHARACTERS;
  nameField.autocomplete = "off";
  nameField.value = parts.nickname;
  const privacyNote = makeElement(document, "p", "astro-screen__note", "It stays on this Mac. Astro never sends it anywhere.");
  parts.confirmButton.type = "submit";
  nameForm.addEventListener("submit", (submitEvent) => {
    submitEvent.preventDefault();
    parts.onSubmit(nameField.value);
  });
  const actionRow = makeElement(document, "div", "astro-screen__actions");
  actionRow.append(parts.confirmButton, parts.secondaryButton);
  nameForm.append(nameLabel, nameField, privacyNote, actionRow);
  return nameForm;
}
