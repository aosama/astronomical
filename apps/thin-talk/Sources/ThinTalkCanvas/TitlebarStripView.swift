import AppKit

/// What a double-click on a title bar does, as the person set it in System
/// Settings ▸ Desktop & Dock ("Double-click a window's title bar to").
public enum TitlebarDoubleClickAction: Equatable, Sendable {
  case zoom
  case minimize
  case doNothing

  /// The action for the stored settings, read the way AppKit's own title bars
  /// do. The newer key holds "Minimize", "None", or a zoom choice
  /// ("Maximize", or "Fill" since macOS 15); without it, the older yes-or-no
  /// minimise key decides, and a Mac that never changed either zooms, which is
  /// the system default.
  public static func from(actionOnDoubleClick: String?, minimizesOnDoubleClick: Bool) -> TitlebarDoubleClickAction {
    switch actionOnDoubleClick {
    case "Minimize": return .minimize
    case "None": return .doNothing
    case .some: return .zoom
    case nil: return minimizesOnDoubleClick ? .minimize : .zoom
    }
  }

  /// The person's current choice.
  public static func current() -> TitlebarDoubleClickAction {
    let defaults = UserDefaults.standard
    return from(
      actionOnDoubleClick: defaults.string(forKey: "AppleActionOnDoubleClick"),
      minimizesOnDoubleClick: defaults.bool(forKey: "AppleMiniaturizeOnDoubleClick"))
  }
}

/// The window's title bar strip, laid over the top of the conversation page.
///
/// The window draws its content under a transparent title bar, and the
/// conversation scrolls up under the strip on purpose. But AppKit only sends
/// clicks in that strip to the title bar where nothing else is there; over
/// the page, the web view took them, so a double-click selected a word
/// instead of zooming the window and a drag selected text instead of moving
/// it. This view sits above the web view and does what a title bar does. The
/// toolbar's buttons live above the content view entirely and are unaffected.
final class TitlebarStripView: NSView {
  /// Read on every double-click, so a change in System Settings applies at once.
  var doubleClickAction: () -> TitlebarDoubleClickAction = { TitlebarDoubleClickAction.current() }

  override func mouseDown(with event: NSEvent) {
    guard let window else { return }
    if event.clickCount >= 2 {
      switch doubleClickAction() {
      case .zoom: window.performZoom(nil)
      case .minimize: window.performMiniaturize(nil)
      case .doNothing: break
      }
      return
    }
    window.performDrag(with: event)
  }

  /// A press here never starts a text selection or a page gesture underneath.
  override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}
