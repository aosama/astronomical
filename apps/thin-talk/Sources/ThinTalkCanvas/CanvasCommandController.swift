import Foundation

/// Routes menu, toolbar and window commands to the live canvas, the same way
/// `CanvasZoomController` routes zoom: the canvas registers itself when the
/// hosting view hands the controller over, and every command is a no-op until
/// the canvas has a web view.
@MainActor
public final class CanvasCommandController {
  private weak var canvas: ConversationCanvasView?

  public init() {}

  public func run(_ pageCommand: CanvasPageCommand) {
    canvas?.runPageCommand(pageCommand)
  }

  /// Tells the page how far the native title bar reaches into it, in points.
  /// The window calls this after the title bar changes height (entering or
  /// leaving full screen); the canvas re-sends it after every zoom change.
  public func applyTitlebarInset(points insetPoints: Int) {
    canvas?.applyTitlebarInset(points: insetPoints)
  }

  /// Puts a draft in the page's message field without sending it.
  public func fillDraft(_ draftText: String) {
    canvas?.fillDraft(draftText)
  }

  func attach(_ canvas: ConversationCanvasView) {
    self.canvas = canvas
  }
}
