import Foundation

/// A command the native menu bar or toolbar sends into the page.
///
/// The raw values are the page's `HostCommand` names (web/src/HostCommand.ts);
/// `CanvasPageCommandContractTests` reads that file and requires both sides to
/// list the same commands, so a rename on either side fails a test instead of
/// leaving a dead menu item. The page's one further command, `setTitlebarInset`,
/// carries a number and is built by `titlebarInsetScript` instead.
public enum CanvasPageCommand: String, CaseIterable, Sendable {
  case newChat
  case toggleSidebar
  case stopGenerating
  case copyLastAnswer
  case deleteConversation
  case exportConversation
  case focusComposer
  case openYourAstro
  case toggleAstro

  /// The script that delivers the command to the page's command router.
  ///
  /// Optional chaining keeps a command sent before the page finished booting a
  /// no-op instead of a script error. The raw values are fixed identifiers, so
  /// no escaping is needed to embed them.
  public var scriptSource: String {
    "window.__thintalkCommands?.run({\"name\": \"\(rawValue)\"});"
  }
}

extension CanvasPageCommand {
  /// The page command that carries the title bar inset.
  public static let titlebarInsetCommandName = "setTitlebarInset"

  /// The script that tells the page how far the native title bar reaches into
  /// it, in CSS pixels at the current page zoom.
  ///
  /// Page zoom scales CSS pixels but not the native strip, so the inset is
  /// divided by the zoom: at 50% zoom the page needs twice the CSS pixels to
  /// clear the same strip. A non-positive zoom is treated as 100%.
  public static func titlebarInsetScript(insetPoints: Int, pageZoom: CGFloat) -> String {
    let effectivePageZoom = pageZoom > 0 ? pageZoom : 1
    let insetPixels = Double(insetPoints) / Double(effectivePageZoom)
    let roundedInsetPixels = (insetPixels * 100).rounded() / 100
    return "window.__thintalkCommands?.run({\"name\": \"\(titlebarInsetCommandName)\", \"points\": \(roundedInsetPixels)});"
  }
}
