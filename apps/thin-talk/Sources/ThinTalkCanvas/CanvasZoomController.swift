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
}
