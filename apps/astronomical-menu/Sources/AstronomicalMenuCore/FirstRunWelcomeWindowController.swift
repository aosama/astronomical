import AppKit
import SwiftUI

/// Owns the first-run welcome window and the accessory-app activation dance.
///
/// The menu application normally runs as an `.accessory` (no Dock icon), and
/// macOS will not reliably key or focus a window for an accessory application.
/// Showing the welcome therefore switches the policy to `.regular`, activates
/// the application, and orders the window front; closing it restores
/// `.accessory` through the application-provided handler. The window is
/// retained here so it can be reopened from the menu without recreating state.
@MainActor
final class FirstRunWelcomeWindowController: NSObject, NSWindowDelegate {
  private var welcomeWindow: NSWindow?
  private var handleClose: (() -> Void)?

  func showWelcomeWindow(
    telemetryStore: TelemetryStore,
    applicationIdentity: ApplicationIdentity,
    openObservatory: @escaping () -> Void,
    openLibrary: @escaping () -> Void,
    restartServer: @escaping () -> Void,
    handleClose: @escaping () -> Void
  ) {
    self.handleClose = handleClose
    let window = welcomeWindow ?? makeWelcomeWindow(
      telemetryStore: telemetryStore,
      applicationIdentity: applicationIdentity,
      openObservatory: openObservatory,
      openLibrary: openLibrary,
      restartServer: restartServer
    )
    welcomeWindow = window
    NSApp.setActivationPolicy(.regular)
    NSApp.activate(ignoringOtherApps: true)
    window.center()
    window.makeKeyAndOrderFront(nil)
  }

  private func makeWelcomeWindow(
    telemetryStore: TelemetryStore,
    applicationIdentity: ApplicationIdentity,
    openObservatory: @escaping () -> Void,
    openLibrary: @escaping () -> Void,
    restartServer: @escaping () -> Void
  ) -> NSWindow {
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 460, height: 380),
      styleMask: [.titled, .closable],
      backing: .buffered,
      defer: false)
    window.title = "Welcome to Astronomical"
    window.isReleasedWhenClosed = false
    window.delegate = self
    window.contentView = NSHostingView(
      rootView: FirstRunWelcomeView(
        telemetryStore: telemetryStore,
        applicationIdentity: applicationIdentity,
        openObservatory: openObservatory,
        openLibrary: openLibrary,
        restartServer: restartServer,
        dismissWelcome: { [weak window] in window?.close() }
      ))
    return window
  }

  func windowWillClose(_ notification: Notification) {
    let closeHandler = handleClose
    handleClose = nil
    closeHandler?()
  }
}
