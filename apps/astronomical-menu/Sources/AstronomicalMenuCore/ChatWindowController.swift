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
/// through the application-provided handler, once AppKit has finished
/// handling any open menu (see `windowWillClose`).
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
  private let restartEngine: (@MainActor () -> Void)?
  private var handleClose: (() -> Void)?
  /// The menu bar from before chat opened, put back on close so the menu
  /// application never keeps the conversation's menus without its window.
  private var menuBeforeChat: NSMenu?
  /// True while a close's deferred work is queued. A close, reopen and close
  /// in quick succession share one run, so the menu from before chat is put
  /// back once and never replaced by an empty menu bar.
  private var isCloseWorkPending = false

  init(
    thinTalkIdentity: ThinTalkCore.ThinTalkApplicationIdentity,
    hostMenuActions: ThinTalkHostMenuActions = ThinTalkHostMenuActions(applicationName: "Astronomical"),
    restartEngine: (@MainActor () -> Void)? = nil
  ) {
    self.thinTalkIdentity = thinTalkIdentity
    self.hostMenuActions = hostMenuActions
    self.restartEngine = restartEngine
    super.init()
  }

  /// Whether the conversation window is on screen; while it is, the app must
  /// stay a regular app, because only a regular app owns the menu bar.
  var isShowingChat: Bool {
    conversationWindow?.window.isVisible == true
  }

  /// Opens the chat window on Your Astro, the page that holds every answer
  /// from Astro's welcome. The command waits for a newly made window's page
  /// to load (see ConversationCanvasView.runPageCommand).
  func showYourAstro(handleClose: @escaping () -> Void) {
    showChatWindow(handleClose: handleClose)
    conversationWindow?.conversationActions.openYourAstro(nil)
  }

  func showChatWindow(handleClose: @escaping () -> Void) {
    self.handleClose = handleClose
    let shownConversationWindow: ThinTalkConversationWindow
    if let conversationWindow {
      shownConversationWindow = conversationWindow
    } else {
      shownConversationWindow = ThinTalkConversationWindow(
        applicationIdentity: thinTalkIdentity, restartEngine: restartEngine)
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

  /// Closing from the menu bar (⌘W, File › Close Window) runs inside AppKit's
  /// menu handling. Swapping the menu bar or leaving `.regular` there orphans
  /// the open menu: it stays drawn on screen and answers no clicks until the
  /// app relaunches. So any open menu is cancelled first, and the swap and the
  /// close handler wait for the default run-loop mode, which macOS only
  /// returns to once menu tracking has ended.
  func windowWillClose(_ notification: Notification) {
    NSApp.mainMenu?.cancelTrackingWithoutAnimation()
    guard !isCloseWorkPending else { return }
    isCloseWorkPending = true
    RunLoop.main.perform(inModes: [.default]) { [weak self] in
      MainActor.assumeIsolated { self?.finishClosing() }
    }
  }

  /// Chat reopened before this ran keeps its menus and its Dock icon: the
  /// close is superseded, so neither the menu swap nor the handler applies.
  private func finishClosing() {
    isCloseWorkPending = false
    guard conversationWindow?.window.isVisible != true else { return }
    NSApp.mainMenu = menuBeforeChat
    menuBeforeChat = nil
    let closeHandler = handleClose
    handleClose = nil
    closeHandler?()
  }
}
