import { SessionDocument } from "./SessionDocument";

/** Characters of context shown on each side of a match. */
const SNIPPET_CONTEXT_CHARACTERS = 80;

/** One place a search phrase was found. */
export interface ConversationMatch {
  readonly title: string;
  readonly updatedAt: string;
  readonly snippet: string;
}

/**
 * Finding a phrase in saved conversations, kept apart from the tool that
 * loads them so the matching is tested on plain documents. Matching ignores
 * case; a snippet shows the match with some text on each side, marked with an
 * ellipsis where it was cut.
 */
export class ConversationSearch {
  /** The first place the phrase appears in a conversation's messages. */
  public static firstMessageMatch(conversation: SessionDocument, phrase: string): ConversationMatch | null {
    for (const message of conversation.messages) {
      const snippet = ConversationSearch.snippetAround(message.content, phrase);
      if (snippet != null) {
        return { title: conversation.title, updatedAt: conversation.updatedAt, snippet };
      }
    }
    return null;
  }

  /** The match with context, or null when the phrase does not appear. */
  public static snippetAround(searchedText: string, phrase: string): string | null {
    const matchIndex = searchedText.toLowerCase().indexOf(phrase.toLowerCase());
    if (matchIndex < 0) {
      return null;
    }
    const snippetStart = Math.max(0, matchIndex - SNIPPET_CONTEXT_CHARACTERS);
    const snippetEnd = Math.min(searchedText.length, matchIndex + phrase.length + SNIPPET_CONTEXT_CHARACTERS);
    const leadingMark = snippetStart > 0 ? "…" : "";
    const trailingMark = snippetEnd < searchedText.length ? "…" : "";
    return `${leadingMark}${searchedText.slice(snippetStart, snippetEnd).replace(/\s+/g, " ").trim()}${trailingMark}`;
  }
}
