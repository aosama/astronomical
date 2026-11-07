import AppKit
import ThinTalkCore
import ThinTalkUI

/// Launches the app as a full-window application (visible in the Dock and
/// Spotlight), not a menu-bar item, matching the requirement that a thin chat
/// app is discoverable through first-class launch.
///
/// The window and menu bar are the same ones the menu application opens
/// (`ThinTalkConversationWindow`, `ThinTalkMainMenu`), so what developers run
/// here is what ships. Production always renders the real, client-backed
/// conversation; the UX preview and offscreen PNG exporter are development-only
/// and not reachable from this entry.
@MainActor
final class ThinTalkApplication: NSObject, NSApplicationDelegate {
  private var conversationWindow: ThinTalkConversationWindow?

  func applicationDidFinishLaunching(_ notification: Notification) {
    NSApp.setActivationPolicy(.regular)
    let applicationIdentity = ThinTalkApplicationIdentity.current()
    let launchedConversationWindow = ThinTalkConversationWindow(applicationIdentity: applicationIdentity)
    conversationWindow = launchedConversationWindow
    ThinTalkMainMenu.install(
      ThinTalkMainMenu.make(
        hostActions: ThinTalkHostMenuActions(applicationName: applicationIdentity.windowTitle),
        conversationActions: launchedConversationWindow.conversationActions),
      in: NSApp)
    launchedConversationWindow.window.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
  }

  /// A single-window app: with the conversation closed there is nothing left to
  /// use, so the app quits instead of lingering with an empty menu bar.
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    true
  }
}

@main
enum ThinTalkMain {
  static func main() {
    let application = NSApplication.shared
    let applicationDelegate = ThinTalkApplication()
    application.delegate = applicationDelegate
    // NSApplication holds its delegate weakly, and Swift may release a local
    // after its last use, so the delegate is kept alive for the whole run.
    withExtendedLifetime(applicationDelegate) {
      application.run()
    }
  }
}
