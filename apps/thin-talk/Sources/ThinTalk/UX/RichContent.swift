import SwiftUI
import WebKit

/// Renders one bounded HTML answer card inside the native conversation. The host
/// owns layout and lifecycle; this view only injects app-supplied HTML and never
/// runs navigation or chat. If the WebKit view cannot appear, the host shows a
/// plain-text stub instead, so a rich answer can never crash the conversation.
struct RichContentView: NSViewRepresentable {
  let html: String

  func makeCoordinator() -> Coordinator { Coordinator() }

  func makeNSView(context: Context) -> WKWebView {
    let configuration = WKWebViewConfiguration()
    let webView = WKWebView(frame: .zero, configuration: configuration)
    webView.setValue(false, forKey: "drawsBackground")
    return webView
  }

  func updateNSView(_ webView: WKWebView, context: Context) {
    guard context.coordinator.currentHTML != html else { return }
    context.coordinator.currentHTML = html
    webView.loadHTMLString(html, baseURL: nil)
  }

  /// Tracks the last injected HTML so `updateNSView` skips redundant reloads.
  final class Coordinator {
    var currentHTML = ""
  }
}
