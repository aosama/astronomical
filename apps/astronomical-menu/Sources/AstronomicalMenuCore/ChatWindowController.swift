import AppKit
import SwiftUI
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
@MainActor
final class ChatWindowController: NSObject, NSWindowDelegate {
  private var chatWindow: NSWindow?
  private let thinTalkIdentity: ThinTalkCore.ThinTalkApplicationIdentity
  private var handleClose: (() -> Void)?

  init(thinTalkIdentity: ThinTalkCore.ThinTalkApplicationIdentity) {
    self.thinTalkIdentity = thinTalkIdentity
    super.init()
  }

  func showChatWindow(handleClose: @escaping () -> Void) {
    self.handleClose = handleClose
    let window = chatWindow ?? makeChatWindow()
    chatWindow = window
    NSApp.setActivationPolicy(.regular)
    NSApp.activate(ignoringOtherApps: true)
    window.makeKeyAndOrderFront(nil)
  }

  private func makeChatWindow() -> NSWindow {
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1080, height: 720),
      styleMask: [.titled, .closable, .miniaturizable, .resizable],
      backing: .buffered,
      defer: false)
    window.title = "Thin Talk"
    window.isReleasedWhenClosed = false
    window.delegate = self
    // The chat surface is the HTML client hosted by ThinTalkCanvas; the window
    // only supplies chrome and the channel identity the client configures
    // itself from.
    window.contentView = NSHostingView(
      rootView: ThinTalkUI.RootView(applicationIdentity: thinTalkIdentity))
    return window
  }

  func windowWillClose(_ notification: Notification) {
    let closeHandler = handleClose
    handleClose = nil
    closeHandler?()
  }
}
