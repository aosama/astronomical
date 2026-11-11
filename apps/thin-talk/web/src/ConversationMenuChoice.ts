/** What the user picked from a sidebar conversation's right-click menu. A
 * "delete" arrives only after the user confirmed it on the native sheet. */
export type ConversationMenuChoice = "rename" | "export" | "delete" | "dismissed";
