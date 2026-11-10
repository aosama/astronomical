import AppKit
import ThinTalkCore
import ThinTalkUI

/// Owns the one conversation window of the main app.
///
/// The window is made once and kept: closing it only takes it off screen, so a
/// reply still being generated carries on and reopening is instant. The main
/// app is a regular application for its whole life, so showing the window is
/// just activating the app and ordering the window front; nothing here touches
/// the activation policy or the menu bar.
@MainActor
final class ChatWindowController {
  private let conversationWindow: ThinTalkConversationWindow

  init(
    thinTalkIdentity: ThinTalkCore.ThinTalkApplicationIdentity,
    restartEngine: (@MainActor () -> Void)? = nil
  ) {
    conversationWindow = ThinTalkConversationWindow(
      applicationIdentity: thinTalkIdentity, restartEngine: restartEngine)
  }

  /// The commands the menu bar and the toolbar share for this window.
  var conversationActions: ThinTalkConversationActions { conversationWindow.conversationActions }

  var isShowingChat: Bool { conversationWindow.window.isVisible }

  func showChatWindow() {
    NSApp.activate(ignoringOtherApps: true)
    conversationWindow.window.makeKeyAndOrderFront(nil)
  }

  /// Opens the window on Your Astro, the page that holds every answer from
  /// Astro's welcome. The command waits for a newly made window's page to
  /// load (see ConversationCanvasView.runPageCommand).
  func showYourAstro() {
    showChatWindow()
    conversationActions.openYourAstro(nil)
  }
}
