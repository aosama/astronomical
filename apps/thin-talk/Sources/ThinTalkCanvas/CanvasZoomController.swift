import Foundation
import WebKit

/// Routes the View menu's zoom commands to the live canvas web view. The app
/// scene owns one controller for its lifetime; the canvas registers itself when
/// the hosting view layer hands the controller over, and every zoom intent is a
/// no-op until the canvas has installed its web view.
@MainActor
public final class CanvasZoomController {
  /// Browser-like zoom bounds: half to triple size, the same bracket mainstream
  /// browsers offer, so the conversation stays readable but never absurd.
  public static let minimumPageZoom: CGFloat = 0.5
  public static let maximumPageZoom: CGFloat = 3.0
  /// The zoom the page ships with, so "Actual Size" is a real escape hatch.
  public static let defaultPageZoom: CGFloat = 1.0
  /// A multiplicative step, not additive: equal visual increments at every zoom
  /// level, the same reason browsers scale instead of adding points.
  public static let zoomInStep: CGFloat = 1.1

  private weak var canvas: ConversationCanvasView?

  public init() {}

  public func zoomIn() {
    canvas?.adjustPageZoom(by: Self.zoomInStep)
  }

  public func zoomOut() {
    canvas?.adjustPageZoom(by: 1 / Self.zoomInStep)
  }

  public func resetZoom() {
    canvas?.resetPageZoom()
  }

  func attach(_ canvas: ConversationCanvasView) {
    self.canvas = canvas
  }

  /// The clamped page zoom after applying the factor to the current zoom.
  /// Pure arithmetic so the bounds are unit-testable without a web view.
  static func clampedPageZoom(current: CGFloat, factor: CGFloat) -> CGFloat {
    return min(maximumPageZoom, max(minimumPageZoom, current * factor))
  }

  /// True when a key-down is one of the Command-only zoom-in spellings: bare
  /// ⌘= or the shifted ⌘+ (and the numpad plus) users actually press.
  ///
  /// The View menu registers Zoom In on "=", but SwiftUI resolves menu key
  /// equivalents against the event's charactersIgnoringModifiers — which for
  /// Shift+⌘+= is "=" and can never be "+" — so the advertised ⌘+ keystroke
  /// could only ever reach the canvas alias. The alias therefore owns both
  /// spellings; the View menu keeps "=" as the fallback for windows without an
  /// installed canvas. The canvas view tree answers performKeyEquivalent before
  /// the main menu is consulted, so exactly one zoom step fires per keystroke.
  /// Option and Control stay rejected: nothing zooms on those, and swallowing
  /// them would shadow keys the web page may want.
  static func isZoomInKeyAlias(
    keyCharacters: String?,
    modifierFlags: NSEvent.ModifierFlags
  ) -> Bool {
    guard keyCharacters == "=" || keyCharacters == "+" else { return false }
    guard modifierFlags.contains(.command) else { return false }
    let rejectedFlags: NSEvent.ModifierFlags = [.option, .control]
    return modifierFlags.isDisjoint(with: rejectedFlags)
  }
}
