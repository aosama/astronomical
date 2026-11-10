/** Creates an element with a class and, optionally, plain text. Text is always
 * set through textContent, so a nickname can never become markup. */
export function makeElement<K extends keyof HTMLElementTagNameMap>(
  document: Document,
  tagName: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const createdElement = document.createElement(tagName);
  createdElement.className = className;
  if (text !== undefined) {
    createdElement.textContent = text;
  }
  return createdElement;
}

/** A full-window Astro overlay is a modal dialog: while it is open the sidebar
 * and conversation behind it must not take keyboard focus or be read by
 * VoiceOver, so every sibling of the overlay is made inert. */
export function setSiblingsInert(overlay: HTMLElement, isInert: boolean): void {
  const windowRoot = overlay.parentElement;
  if (windowRoot === null) {
    return;
  }
  for (const sibling of Array.from(windowRoot.children)) {
    if (sibling !== overlay) {
      sibling.toggleAttribute("inert", isInert);
    }
  }
}

/** A button with a stable id equal to its data-testid, as every Thin Talk
 * control has, so automation and assistive technology share one name. */
export function makeButton(document: Document, testID: string, label: string, isQuiet: boolean): HTMLButtonElement {
  const createdButton = makeElement(document, "button", isQuiet ? "astro-button astro-button--quiet" : "astro-button", label);
  createdButton.type = "button";
  createdButton.id = testID;
  createdButton.dataset.testid = testID;
  return createdButton;
}
