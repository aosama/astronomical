import type { ChatController } from "./ChatController";
import type { SessionBridge } from "./SessionBridge";
import type { SidebarView } from "./SidebarView";
import type { ExportLauncher } from "./ExportLauncher";

/**
 * Turns a choice from a conversation's right-click menu into its action. The
 * native host shows the menu and confirms a delete; the sidebar edits the
 * name; the controller deletes; the export launcher exports. This only connects them, so none of the three
 * needs to know about the others.
 */
export async function runConversationMenu(
  conversationID: string,
  targets: { bridge: SessionBridge; sidebar: SidebarView; controller: ChatController; exportLauncher: ExportLauncher },
): Promise<void> {
  // A host without the menu (the browser preview) or a timed-out wait simply
  // shows nothing; the conversation is left as it was.
  const menuChoice = await targets.bridge.showConversationMenu(conversationID).catch(() => "dismissed" as const);
  if (menuChoice === "rename") {
    targets.sidebar.beginRename(conversationID);
  } else if (menuChoice === "export") {
    // The sheet exports the chat on screen, so the chosen one is opened first.
    await targets.controller.openSession(conversationID);
    await targets.exportLauncher.open({ kind: "wholeChat" });
  } else if (menuChoice === "delete") {
    await targets.controller.deleteSession(conversationID);
  }
}
