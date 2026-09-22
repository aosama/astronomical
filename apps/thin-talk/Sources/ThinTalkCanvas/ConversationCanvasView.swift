import AppKit
import SwiftUI
import ThinTalkCore
import WebKit

/// One interaction the canvas reported back to the application.
///
/// The canvas owns no behaviour: it renders and it reports. Copying text,
/// regenerating an answer, and opening a link all happen in Swift, where the same
/// code path runs whether the reader used the canvas or a keyboard shortcut.
public struct CanvasAction: Sendable, Equatable {
  public enum Kind: String, Sendable {
    case ready
    case copy
    case regenerate
    case openExternal
  }

  public let kind: Kind
  public let messageID: UUID?
  public let externalURL: URL?

  public init(kind: Kind, messageID: UUID? = nil, externalURL: URL? = nil) {
    self.kind = kind
    self.messageID = messageID
    self.externalURL = externalURL
  }
}

/// The conversation canvas: one web surface that renders the transcript while
/// SwiftUI keeps the surrounding chrome, the composer, and every behaviour.
public struct ConversationCanvasView: NSViewRepresentable {
  private let snapshot: TranscriptSnapshot
  private let isDarkAppearance: Bool
  private let webDirectory: URL
  private let assetRegistry: TranscriptAssetRegistry
  private let pageZoom: CGFloat
  private let onAction: (CanvasAction) -> Void

  public init(
    snapshot: TranscriptSnapshot,
    isDarkAppearance: Bool,
    webDirectory: URL,
    assetRegistry: TranscriptAssetRegistry,
    pageZoom: CGFloat = 1.0,
    onAction: @escaping (CanvasAction) -> Void
  ) {
    self.snapshot = snapshot
    self.isDarkAppearance = isDarkAppearance
    self.webDirectory = webDirectory
    self.assetRegistry = assetRegistry
    self.pageZoom = pageZoom
    self.onAction = onAction
  }

  public func makeCoordinator() -> Coordinator {
    Coordinator(onAction: onAction)
  }

  public func makeNSView(context: Context) -> WKWebView {
    let configuration = WKWebViewConfiguration()
    // Nothing the canvas renders should survive the session: no cookies, no
    // caches, no storage a later answer could read back.
    configuration.websiteDataStore = .nonPersistent()
    configuration.defaultWebpagePreferences.allowsContentJavaScript = true
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.userContentController.add(context.coordinator, name: Coordinator.messageHandlerName)
    configuration.setURLSchemeHandler(
      CanvasAssetSchemeHandler(webDirectory: webDirectory, assetRegistry: assetRegistry),
      forURLScheme: CanvasSecurityPolicy.shellScheme
    )

    let webView = WKWebView(frame: .zero, configuration: configuration)
    webView.navigationDelegate = context.coordinator
    webView.allowsMagnification = false
    // The GUI zoom control must resize the transcript too, so the web content
    // follows the same scale factor the SwiftUI surface uses.
    webView.pageZoom = pageZoom
    webView.setValue(false, forKey: "drawsBackground")
    // The window colour sits behind the page so the first paint does not flash
    // white between the SwiftUI background and the canvas background.
    webView.underPageBackgroundColor = NSColor(srgbRed: 0.06, green: 0.07, blue: 0.10, alpha: 1)
    #if DEBUG
      if #available(macOS 13.2, *) {
        webView.isInspectable = true
      }
    #endif

    context.coordinator.webView = webView
    context.coordinator.sendAppearance(isDarkAppearance)
    context.coordinator.send(snapshot: snapshot)
    if let documentURL = CanvasSecurityPolicy.shellDocumentURL() {
      webView.load(URLRequest(url: documentURL))
    }
    return webView
  }

  public func updateNSView(_ webView: WKWebView, context: Context) {
    context.coordinator.onAction = onAction
    webView.pageZoom = pageZoom
    context.coordinator.sendAppearance(isDarkAppearance)
    context.coordinator.send(snapshot: snapshot)
  }

  public static func dismantleNSView(_ webView: WKWebView, coordinator: Coordinator) {
    webView.configuration.userContentController.removeScriptMessageHandler(
      forName: Coordinator.messageHandlerName)
  }

  @MainActor
  public final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
    static let messageHandlerName = "thintalk"

    var onAction: (CanvasAction) -> Void
    weak var webView: WKWebView?

    private var lastSentSnapshot: TranscriptSnapshot?
    private var lastSentAppearance: Bool?
    private var queuedCommands: [TranscriptCommand] = []
    private var isReady = false

    init(onAction: @escaping (CanvasAction) -> Void) {
      self.onAction = onAction
    }

    // MARK: - Swift to canvas

    func send(snapshot: TranscriptSnapshot) {
      let commands = TranscriptCommandPlanner.plan(previous: lastSentSnapshot, next: snapshot)
      lastSentSnapshot = snapshot
      commands.forEach { push($0) }
    }

    func sendAppearance(_ isDark: Bool) {
      guard lastSentAppearance != isDark else { return }
      lastSentAppearance = isDark
      push(.appearance(dark: isDark))
    }

    private func push(_ command: TranscriptCommand) {
      guard isReady, let webView, let script = try? TranscriptCommandCodec.invocation(for: command)
      else {
        queuedCommands.append(command)
        return
      }
      webView.evaluateJavaScript(script)
    }

    private func flushQueuedCommands() {
      let queued = queuedCommands
      queuedCommands = []
      queued.forEach { push($0) }
    }

    // MARK: - Canvas to Swift

    public func userContentController(
      _ userContentController: WKUserContentController,
      didReceive message: WKScriptMessage
    ) {
      guard message.name == Self.messageHandlerName,
        let body = message.body as? [String: Any],
        let rawKind = body["kind"] as? String
      else { return }
      switch rawKind {
      case "action":
        handleReportedAction(body)
      case "ready":
        isReady = true
        flushQueuedCommands()
        onAction(CanvasAction(kind: .ready))
      default:
        return
      }
    }

    private func handleReportedAction(_ body: [String: Any]) {
      guard let rawAction = body["action"] as? String,
        let kind = CanvasAction.Kind(rawValue: rawAction)
      else { return }
      let rawMessageID = body["messageId"] as? String ?? ""
      let messageID = UUID(uuidString: rawMessageID)
      switch kind {
      case .openExternal:
        let rawURL = body["detail"] as? String ?? ""
        guard let externalURL = CanvasSecurityPolicy.externalURL(fromRawValue: rawURL) else { return }
        onAction(CanvasAction(kind: .openExternal, externalURL: externalURL))
      case .copy, .regenerate:
        onAction(CanvasAction(kind: kind, messageID: messageID))
      case .ready:
        return
      }
    }

    // MARK: - Navigation

    public func webView(
      _ webView: WKWebView,
      decidePolicyFor navigationAction: WKNavigationAction
    ) async -> WKNavigationActionPolicy {
      guard let url = navigationAction.request.url else { return .cancel }
      if CanvasSecurityPolicy.isShellDocument(url) {
        return .allow
      }
      // Links inside an answer may open in the reader's browser, but nothing may
      // navigate the canvas itself, and internal schemes never leave the app.
      if navigationAction.navigationType == .linkActivated,
        let externalURL = CanvasSecurityPolicy.externalURL(fromRawValue: url.absoluteString)
      {
        onAction(CanvasAction(kind: .openExternal, externalURL: externalURL))
      }
      return .cancel
    }
  }
}