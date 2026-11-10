import { makeElement } from "./AstroDom";

const GROWN_UP_NOTE =
  "Astro runs entirely on this Mac. Nothing typed or shown to Astro leaves it, there is no account, and the name you give is kept only here. " +
  "Conversations are saved on this Mac and can be deleted from the sidebar. Astro can be hidden for a plain chat window from Your Astro or the View menu.";

/** What a parent or teacher usually wants to know, shown from the first
 * flight's style choice and from Your Astro's grown-up row. */
export function buildGrownUpNotePanel(document: Document): HTMLElement {
  const notePanel = makeElement(document, "div", "first-flight__panel");
  notePanel.append(makeElement(document, "h2", "", "For grown-ups"), makeElement(document, "p", "", GROWN_UP_NOTE));
  return notePanel;
}
