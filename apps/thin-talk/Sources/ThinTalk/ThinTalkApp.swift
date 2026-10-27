import SwiftUI
import Foundation
import ThinTalkUI
import ThinTalkCanvas

/// Launches the app as a full-window application (visible in the Dock and
/// Spotlight), not a menu-bar item, matching the requirement that a thin chat
/// app is discoverable through first-class launch.
@MainActor
final class ThinTalkApplication: NSObject, NSApplicationDelegate {
  func applicationDidFinishLaunching(_ notification: Notification) {
    DispatchQueue.main.async { NSApp.setActivationPolicy(.regular) }
    // The window chrome itself must not be transparent: with `.hiddenTitleBar`
    // plus a clear backdrop the whole title-bar strip rendered as see-through
    // glass, which read as a rendering bug rather than a design choice. The
    // backdrop uses the app's own window background so the chrome blends in.
    for window in NSApp.windows where window.isVisible {
      window.isOpaque = true
      window.backgroundColor = NSColor(red: 0.06, green: 0.07, blue: 0.10, alpha: 1)
    }
  }
}

@main
struct ThinTalkApp: App {
  @NSApplicationDelegateAdaptor(ThinTalkApplication.self) private var applicationDelegate
  @State private var zoomController = CanvasZoomController()

  var body: some Scene {
    // Production always renders the real, client-backed conversation. The UX
    // preview and offscreen PNG exporter are development-only and no longer
    // reachable from the shipped app entry.
    WindowGroup {
      RootView(applicationIdentity: .current(), zoomController: zoomController)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
    .windowStyle(.hiddenTitleBar)
    .defaultSize(width: 1080, height: 720)
    .windowResizability(.contentMinSize)
    .commands {
      CanvasZoomCommands(controller: zoomController)
    }
  }
}

/// The keyboard zoom commands the web view has no menu of its own for. A web
/// canvas in a native window must speak the platform's View menu dialect, or
/// Cmd +/-/0 simply do nothing, which read as a broken app.
struct CanvasZoomCommands: Commands {
  let controller: CanvasZoomController

  var body: some Commands {
    // The .toolbar placement is the View menu, where zoom commands live in
    // every native Mac app.
    CommandGroup(after: .toolbar) {
      Button("Zoom In") { controller.zoomIn() }
        .keyboardShortcut("+", modifiers: .command)
      Button("Zoom Out") { controller.zoomOut() }
        .keyboardShortcut("-", modifiers: .command)
      Button("Actual Size") { controller.resetZoom() }
        .keyboardShortcut("0", modifiers: .command)
    }
  }
}
