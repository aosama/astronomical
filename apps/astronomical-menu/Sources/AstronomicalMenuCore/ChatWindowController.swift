import AppKit
import ThinTalkCore
import ThinTalkUI

/// Owns the conversation window the menu application hosts, and the
/// accessory-application activation dance around it.
///
/// The menu application normally runs as `.accessory` (no Dock icon), and macOS
/// will not reliably key or focus a window for an accessory application.
/// Opening chat therefore switches the policy to `.regular`, activates the
/// application, and orders the window front; closing it restores `.accessory`
/// through the application-provided handler. This mirrors the first-run
/// welcome window so both hosted windows behave identically.
///
/// While chat is open the application is a regular app with a menu bar, so the
/// conversation's menu bar (`ThinTalkMainMenu`) is installed on show. Without
/// it the bar held only the Apple menu, and ⌘C/⌘V/⌘Z had no Edit menu to reach
/// the message field through.
@MainActor
final class ChatWindowController: NSObject, NSWindowDelegate {
  private var conversationWindow: ThinTalkConversationWindow?
  private let thinTalkIdentity: ThinTalkCore.ThinTalkApplicationIdentity
  private let hostMenuActions: ThinTalkHostMenuActions
  private var handleClose: (() -> Void)?
  /// The menu bar from before chat opened, put back on close so another window
  /// (the welcome window) never shows the conversation's menus.
  private var menuBeforeChat: NSMenu?

  init(
    thinTalkIdentity: ThinTalkCore.ThinTalkApplicationIdentity,
    hostMenuActions: ThinTalkHostMenuActions = ThinTalkHostMenuActions(applicationName: "Astronomical")
  ) {
    self.thinTalkIdentity = thinTalkIdentity
    self.hostMenuActions = hostMenuActions
    super.init()
  }

  func showChatWindow(handleClose: @escaping () -> Void) {
    self.handleClose = handleClose
    let shownConversationWindow: ThinTalkConversationWindow
    if let conversationWindow {
      shownConversationWindow = conversationWindow
    } else {
      shownConversationWindow = ThinTalkConversationWindow(applicationIdentity: thinTalkIdentity)
      shownConversationWindow.window.delegate = self
    }
    conversationWindow = shownConversationWindow
    NSApp.setActivationPolicy(.regular)
    if NSApp.mainMenu?.item(withTitle: "Conversation") == nil {
      menuBeforeChat = NSApp.mainMenu
    }
    ThinTalkMainMenu.install(
      ThinTalkMainMenu.make(
        hostActions: hostMenuActions,
        conversationActions: shownConversationWindow.conversationActions),
      in: NSApp)
    NSApp.activate(ignoringOtherApps: true)
    shownConversationWindow.window.makeKeyAndOrderFront(nil)
  }

  func windowWillClose(_ notification: Notification) {
    NSApp.mainMenu = menuBeforeChat
    menuBeforeChat = nil
    let closeHandler = handleClose
    handleClose = nil
    closeHandler?()
  }
}
