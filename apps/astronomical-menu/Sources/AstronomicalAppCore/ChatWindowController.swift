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
    // Asked for by the person (the Dock, a link, the menu bar): a background
    // launch still waiting to go behind must not then hide what they opened.
    sendBehindAfterFirstDraw?.cancel()
    sendBehindAfterFirstDraw = nil
    conversationWindow.window.alphaValue = 1
    conversationWindow.window.ignoresMouseEvents = false
    NSApp.activate(ignoringOtherApps: true)
    conversationWindow.window.makeKeyAndOrderFront(nil)
  }

  /// Shows the window at launch. Behind the other windows it is on screen,
  /// so it can be looked at and driven, but it neither covers what the
  /// person is working in nor takes their keyboard; they bring it forward
  /// from the Dock when they want it.
  func showChatWindowAtLaunch(_ launchPlacement: ChatLaunchPlacement) {
    switch launchPlacement {
    case .inFront: showChatWindow()
    case .behindOtherWindows: showChatWindowBehindOthers()
    }
  }

  /// The wait before a background-launched window goes behind; cancelled
  /// when the person asks for the window first.
  private var sendBehindAfterFirstDraw: Task<Void, Never>?

  /// How long the window stays in front, invisible, once it is really on
  /// screen: long enough for the page to draw its first frame.
  static let firstDrawInterval: Duration = .seconds(3)
  /// How often a window waiting for a visible screen looks again.
  static let visibilityCheckInterval: Duration = .milliseconds(500)

  /// WebKit does not draw a page in a window that has never been visible, so
  /// a window ordered straight to the back stays blank however long it
  /// waits. The window is therefore put in front fully transparent and
  /// ignoring the pointer, which is visible to WebKit and invisible to the
  /// person, and goes behind the others once the page has drawn.
  ///
  /// The wait is for visibility, not for a fixed time: a rebuild that
  /// finishes while the person is on another Space, or with the display
  /// asleep, has no screen to draw on, and a window sent behind after three
  /// seconds there stayed blank. Until its screen is showing, the invisible
  /// window simply stays where it is.
  private func showChatWindowBehindOthers() {
    let window = conversationWindow.window
    window.alphaValue = 0
    window.ignoresMouseEvents = true
    window.orderFrontRegardless()
    sendBehindAfterFirstDraw = Task { @MainActor [weak self] in
      while !Task.isCancelled, !Self.isOnAVisibleScreen(window.occlusionState) {
        try? await Task.sleep(for: Self.visibilityCheckInterval)
      }
      try? await Task.sleep(for: Self.firstDrawInterval)
      guard !Task.isCancelled else { return }
      self?.sendBehindAfterFirstDraw = nil
      window.orderBack(nil)
      window.alphaValue = 1
      window.ignoresMouseEvents = false
    }
  }

  /// Whether macOS reports the window as showing on a screen, which is when
  /// WebKit draws into it.
  nonisolated static func isOnAVisibleScreen(_ occlusionState: NSWindow.OcclusionState) -> Bool {
    occlusionState.contains(.visible)
  }

  /// Opens the window on its Settings page. The command waits for a newly made window's page to
  /// load (see ConversationCanvasView.runPageCommand).
  func showSettings() {
    showChatWindow()
    conversationActions.openSettings(nil)
  }
}
