import SwiftUI
import Foundation
import ThinTalkCore

/// Launches the app as a full-window application (visible in the Dock and
/// Spotlight), not a menu-bar item, matching the requirement that a thin chat
/// app is discoverable through first-class launch.
@MainActor
final class ThinTalkApplication: NSObject, NSApplicationDelegate {
  func applicationDidFinishLaunching(_ notification: Notification) {
    DispatchQueue.main.async { NSApp.setActivationPolicy(.regular) }
    /// The window chrome itself must not be transparent: with `.hiddenTitleBar`
  /// plus a clear backdrop the whole title-bar strip rendered as see-through
  /// glass, which read as a rendering bug rather than a design choice. The
  /// backdrop uses the app's own window background so the chrome blends in.
  for window in NSApp.windows where window.isVisible {
    window.isOpaque = true
    window.backgroundColor = NSColor(red: 0.06, green: 0.07, blue: 0.10, alpha: 1)
  }
  }
}

@main
struct ThinTalkApp: App {
  @NSApplicationDelegateAdaptor(ThinTalkApplication.self) private var applicationDelegate

  var body: some Scene {
    WindowGroup {
      Group {
        if ProcessInfo.processInfo.environment["THINTALK_RENDER_PNG"] != nil {
          // A display-independent PNG is produced (below) so the layout can be
          // reviewed on a machine whose screen cannot be captured live.
          Color.clear
        } else if ProcessInfo.processInfo.environment["THINTALK_PREVIEW"] != "0" {
          // The UX preview launches by default so the agreed layout can be reviewed
          // before the REST layer is wired. Set THINTALK_PREVIEW=0 to launch the
          // production path instead.
          PreviewRootView()
        } else {
          RootView()
        }
      }
      // Floating rounded-window look: the content clips to a large continuous
      // corner radius so the desktop shows through the corners. The window
      // backdrop is made transparent in the application delegate.
      .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
      // Rendering needs a live run loop on the main actor; fire it here and then
      // exit so the PNG is written and the process cleans up.
      .task {
        if let renderPath = ProcessInfo.processInfo.environment["THINTALK_RENDER_PNG"] {
          PNGExporter.exportPreviewPNG(to: URL(fileURLWithPath: renderPath))
          exit(0)
        }
      }
    }
    .windowStyle(.hiddenTitleBar)
    .defaultSize(width: 1080, height: 720)
  }
}
