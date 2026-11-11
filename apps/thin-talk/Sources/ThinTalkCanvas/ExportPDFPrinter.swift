import AppKit
import WebKit

/// Prints an exported page to a PDF file, page by page.
///
/// WebKit's own `createPDF` draws the whole page as one tall sheet, which is
/// no use to hand in or print, so the page goes through a print operation
/// instead, saved to the file rather than sent to a printer. The paper is the
/// Mac's default (Letter or A4 by region) with one-inch margins; the page's
/// own stylesheet keeps headings with what follows them and tables, code and
/// pictures whole.
@MainActor
public final class ExportPDFPrinter: NSObject, WKNavigationDelegate {
  static let pageMarginPoints: CGFloat = 72

  /// The sheet the PDF is cut into: the Mac's default unless a caller names
  /// one, which is how A4 is printed on a Mac set to Letter and the reverse.
  private let paperSize: NSSize
  private let webView: WKWebView
  /// A print operation runs as a sheet on a window; this one is never shown.
  private let offscreenWindow: NSWindow
  private var pageLoaded: CheckedContinuation<Void, Error>?
  private var printFinished: CheckedContinuation<Bool, Never>?

  public init(paperSize: NSSize = NSPrintInfo.shared.paperSize) {
    self.paperSize = paperSize
    let configuration = WKWebViewConfiguration()
    // The page is saved data, not the app: it keeps nothing and runs nothing.
    configuration.websiteDataStore = .nonPersistent()
    configuration.defaultWebpagePreferences.allowsContentJavaScript = false
    let pageFrame = NSRect(origin: .zero, size: paperSize)
    webView = WKWebView(frame: pageFrame, configuration: configuration)
    // A panel that does not activate its app: the print job runs as a sheet
    // on this window, and on an ordinary window that sheet brought the whole
    // app to the front, taking the keyboard from whatever the person had
    // moved on to while a long export was still printing.
    offscreenWindow = NSPanel(
      contentRect: NSRect(x: -20_000, y: -20_000, width: paperSize.width, height: paperSize.height),
      styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
    super.init()
    offscreenWindow.isReleasedWhenClosed = false
    offscreenWindow.contentView = webView
    webView.navigationDelegate = self
  }

  public enum PrintError: Error, Equatable {
    case pageDidNotLoad
    case printingFailed
  }

  /// Prints `html` to a PDF at `destinationURL`, replacing a file there.
  public func printPDF(html: String, to destinationURL: URL) async throws {
    // The window goes however the job ends, a page that never loaded included.
    defer { offscreenWindow.close() }
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
      pageLoaded = continuation
      webView.loadHTMLString(html, baseURL: nil)
    }
    let printOperation = webView.printOperation(with: Self.printInfo(savingTo: destinationURL, paperSize: paperSize))
    printOperation.showsPrintPanel = false
    printOperation.showsProgressPanel = false
    // WebKit's print view starts with no size of its own; without one the
    // operation lays out nothing.
    printOperation.view?.frame = webView.bounds
    let didPrint = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
      printFinished = continuation
      printOperation.runModal(
        for: offscreenWindow, delegate: self,
        didRun: #selector(printOperationDidRun(_:success:contextInfo:)), contextInfo: nil)
    }
    guard didPrint else { throw PrintError.printingFailed }
  }

  static func printInfo(savingTo destinationURL: URL, paperSize: NSSize) -> NSPrintInfo {
    let printInfo = (NSPrintInfo.shared.copy() as? NSPrintInfo) ?? NSPrintInfo()
    // Assigned only when it differs: writing back the size the Mac already
    // has still makes the print system re-derive the sheet, a hair under
    // 612 points for Letter, and WebKit then lays the text out one pixel
    // narrower (584 for 585), which printed a 22-point title at 22.04.
    if printInfo.paperSize != paperSize {
      printInfo.paperSize = paperSize
    }
    printInfo.jobDisposition = .save
    printInfo.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = destinationURL
    printInfo.topMargin = pageMarginPoints
    printInfo.bottomMargin = pageMarginPoints
    printInfo.leftMargin = pageMarginPoints
    printInfo.rightMargin = pageMarginPoints
    printInfo.horizontalPagination = .fit
    printInfo.verticalPagination = .automatic
    printInfo.isHorizontallyCentered = false
    printInfo.isVerticallyCentered = false
    return printInfo
  }

  /// AppKit finishes a web view's print job on a thread of its own and calls
  /// back there, so the result is carried to the main thread before it
  /// touches this printer's state.
  @objc nonisolated private func printOperationDidRun(
    _ printOperation: NSPrintOperation, success: Bool, contextInfo: UnsafeMutableRawPointer?
  ) {
    DispatchQueue.main.async {
      MainActor.assumeIsolated {
        self.finishPrinting(success: success)
      }
    }
  }

  private func finishPrinting(success: Bool) {
    printFinished?.resume(returning: success)
    printFinished = nil
  }

  public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    pageLoaded?.resume()
    pageLoaded = nil
  }

  public func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
    pageLoaded?.resume(throwing: PrintError.pageDidNotLoad)
    pageLoaded = nil
  }

  public func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error
  ) {
    pageLoaded?.resume(throwing: PrintError.pageDidNotLoad)
    pageLoaded = nil
  }
}
