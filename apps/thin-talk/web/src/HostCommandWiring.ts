import type { ChatController } from "./ChatController";
import type { ChatSurfaceState } from "./ChatSurfaceState";
import { ChatRole } from "./ChatRole";
import type { HostChannel } from "./HostChannel";
import type { HostCommandHandlers } from "./HostCommandRouter";
import type { TranscriptView } from "./TranscriptView";
import type { WindowChrome } from "./WindowChrome";
import type { AstroCompanion } from "./astro/AstroCompanion";
import type { ExportLauncher } from "./ExportLauncher";

/** Everything the host commands act on, supplied by boot.ts. */
export interface HostCommandTargets {
  controller: ChatController;
  transcript: TranscriptView;
  composerField: HTMLTextAreaElement;
  host: HostChannel;
  windowChrome: WindowChrome;
  astro: AstroCompanion;
  exportLauncher: ExportLauncher;
}

/** The id of the newest assistant message, or null before the first answer. */
export function lastAssistantMessageID(state: ChatSurfaceState | null): string | null {
  if (state === null) {
    return null;
  }
  for (let messageIndex = state.transcript.messages.length - 1; messageIndex >= 0; messageIndex -= 1) {
    const candidateMessage = state.transcript.messages[messageIndex];
    if (candidateMessage !== undefined && candidateMessage.role === ChatRole.ASSISTANT) {
      return candidateMessage.id;
    }
  }
  return null;
}

/**
 * Builds the menu and toolbar actions from the controller's published state.
 *
 * Each action does nothing when it has nothing to act on (no answer to copy, no
 * conversation to delete), so a menu item can never error on an empty window.
 */
export function hostCommandHandlers(targets: HostCommandTargets): HostCommandHandlers {
  let latestState: ChatSurfaceState | null = null;
  targets.controller.subscribe((publishedState) => {
    latestState = publishedState;
  });
  return {
    newChat: () => void targets.controller.newChat(),
    toggleSidebar: () => targets.windowChrome.toggleSidebar(),
    stopGenerating: () => void targets.controller.stop(),
    copyLastAnswer: () => {
      const answerMessageID = lastAssistantMessageID(latestState);
      if (answerMessageID !== null) {
        targets.host.postAction("copy", answerMessageID, targets.transcript.renderedText(answerMessageID));
      }
    },
    deleteConversation: () => {
      const activeSessionID = latestState?.activeSessionID ?? null;
      if (activeSessionID !== null) {
        void targets.controller.deleteSession(activeSessionID);
      }
    },
    exportConversation: () => void targets.exportLauncher.open({ kind: "wholeChat" }),
    focusComposer: () => targets.composerField.focus(),
    openYourAstro: () => targets.astro.openYourAstro(),
    toggleAstro: () => targets.astro.toggleShown(),
    setTitlebarInset: (titlebarInsetPixels) => targets.windowChrome.applyTitlebarInset(titlebarInsetPixels),
  };
}
