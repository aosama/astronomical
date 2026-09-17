import AppKit
import Foundation
import SwiftUI

/// Renders the preview window off-screen through a real `NSWindow` hosting the
/// SwiftUI tree. `ImageRenderer` cannot be used here: it lays out `ScrollView`
/// content as empty because no live scroll geometry exists, which blanks the
/// whole conversation. An offscreen hosting view drives the same layout pass the
/// running app performs, so the captured PNG matches what a user would see. Used
/// when `THINTALK_RENDER_PNG` points at an output path, then the app exits — this
/// lets the surface be reviewed on a machine whose screen is asleep.
@MainActor
enum PNGExporter {
  static func exportPreviewPNG(to url: URL) {
    let size = NSSize(width: 1080, height: 720)
    let hosting = NSHostingView(
      rootView: PreviewRootView()
        .frame(width: 1080, height: 720)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
    )
    hosting.frame = NSRect(origin: .zero, size: size)

    // Offscreen, borderless and transparent: the window server participates in
    // layout, but nothing is ever shown on a display. The rounded clip mirrors
    // the floating rounded-window look of the real app.
    let window = NSWindow(contentRect: NSRect(x: -4000, y: -4000, width: 1080, height: 720),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isOpaque = false
    window.backgroundColor = .clear
    window.contentView = hosting

    // Give SwiftUI several run-loop ticks so scroll anchoring and the embedded
    // WebKit card settle before the snapshot.
    for _ in 0..<25 {
      hosting.layoutSubtreeIfNeeded()
      RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }

    // `bitmapImageRepForCachingDisplay` only allocates the bitmap; the actual
    // paint happens in `cacheDisplay`, so both calls are required.
    guard let rep = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else { return }
    hosting.cacheDisplay(in: hosting.bounds, to: rep)
    guard let data = rep.representation(using: .png, properties: [:]) else { return }
    try? data.write(to: url)
  }
}
