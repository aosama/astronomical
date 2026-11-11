/**
 * A command the native host sends into the page: a menu item or toolbar button
 * that acts on the conversation. The host is a separate process boundary in
 * spirit (Swift evaluating JavaScript), so a payload is parsed and checked here
 * instead of trusted, and an unknown name is refused rather than guessed at.
 */
export enum HostCommand {
  NEW_CHAT = "newChat",
  TOGGLE_SIDEBAR = "toggleSidebar",
  STOP_GENERATING = "stopGenerating",
  COPY_LAST_ANSWER = "copyLastAnswer",
  DELETE_CONVERSATION = "deleteConversation",
  /** File ▸ Export… and the toolbar's Export button: the whole chat. */
  EXPORT_CONVERSATION = "exportConversation",
  FOCUS_COMPOSER = "focusComposer",
  /** Settings (⌘,): opens Your Astro, where each welcome answer changes on its own. */
  OPEN_YOUR_ASTRO = "openYourAstro",
  /** The Focus switch: hides or shows Astro. */
  TOGGLE_ASTRO = "toggleAstro",
  /** Carries `points`: the title bar height in CSS pixels at the current zoom. */
  SET_TITLEBAR_INSET = "setTitlebarInset",
}

const KNOWN_HOST_COMMANDS: ReadonlySet<string> = new Set<string>(Object.values(HostCommand));

/** Reads `{name: "<command>"}`; `null` for anything else, so a malformed or
 * future command from a newer host is a no-op instead of a page error. */
export function parseHostCommand(payload: unknown): HostCommand | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }
  const commandName = (payload as Record<string, unknown>)["name"];
  if (typeof commandName !== "string" || !KNOWN_HOST_COMMANDS.has(commandName)) {
    return null;
  }
  return commandName as HostCommand;
}
