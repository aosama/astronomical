import { makeElement } from "./AstroDom";

const PRIVACY_NOTE =
  "Astro runs entirely on this Mac. Nothing typed or shown to Astro leaves it, there is no account, and the name you give is kept only here. " +
  "Conversations are saved on this Mac and can be deleted from the sidebar. Astro can be hidden for a plain chat window from Settings or the View menu.";

/** What the app does and what stays on this Mac, shown from Settings. */
export function buildPrivacyNotePanel(document: Document): HTMLElement {
  const notePanel = makeElement(document, "div", "astro-screen__panel");
  notePanel.append(makeElement(document, "h2", "", "Privacy"), makeElement(document, "p", "", PRIVACY_NOTE));
  return notePanel;
}
