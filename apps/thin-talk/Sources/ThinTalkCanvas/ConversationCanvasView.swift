import Foundation
import ThinTalkCore
import WebKit

/// The conversation canvas: one web view rendering the bundled HTML client.
///
/// This is a dumb host, deliberately. The page owns the conversation state, the
/// transcript rendering, and the supervisor conversation; Swift owns only what a
/// web page cannot do for itself: reading and writing session files, the system
/// pasteboard, opening external links, and injecting the runtime configuration.
/// Everything else crosses the bridge as data.
///
/// The bridge contract lives in the web client (`SessionBridge.ts`): the page
/// posts `{kind: "sessionCall", callId, op, payload}` envelopes and Swift answers
/// through `window.__thintalkBridge.resolve/reject`. A call the host fails to
/// answer rejects on the page after its own timeout, so the page never hangs on
/// a silent host.
@MainActor
public final class ConversationCanvasView: NSView {
  private let webDirectory: URL
  private let assetRegistry: TranscriptAssetRegistry
  private let sessionFileStore: SessionFileStore
  private let runtimeConfig: CanvasRuntimeConfig
  private var webView: WKWebView?
  // WebKit does not retain scheme handlers, so the view owns this one for as
  // long as the web view lives.
  private var assetSchemeHandler: CanvasAssetSchemeHandler?

  public init(
    webDirectory: URL,
    assetRegistry: TranscriptAssetRegistry,
    sessionFileStore: SessionFileStore,
    runtimeConfig: CanvasRuntimeConfig
  ) {
    self.webDirectory = webDirectory
    self.assetRegistry = assetRegistry
    self.sessionFileStore = sessionFileStore
    self.runtimeConfig = runtimeConfig
    super.init(frame: NSRect(x: 0, y: 0, width: 1080, height: 720))
  }

  @available(*, unavailable)
  required init?(coder: NSCoder) {
    fatalError("ConversationCanvasView is created in code, not from a nib.")
  }

  public override var acceptsFirstResponder: Bool { true }

  public override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    guard window != nil, webView == nil else { return }
    installWebView()
  }

  private func installWebView() {
    let configuration = WKWebViewConfiguration()
    // The page persists nothing itself: sessions and preferences cross the
    // bridge into SessionFileStore, so an ephemeral data store keeps the web
    // view from writing a second, divergent copy of the same state.
    configuration.websiteDataStore = .nonPersistent()
    // The shell loads from the private scheme; the handler serves it straight
    // from the bundled web directory. `loadFileURL` cannot be used here
    // because it throws on any URL that is not a `file://` URL.
    let assetSchemeHandler = CanvasAssetSchemeHandler(
      webDirectory: webDirectory, assetRegistry: assetRegistry)
    configuration.setURLSchemeHandler(
      assetSchemeHandler, forURLScheme: CanvasSecurityPolicy.shellScheme)
    let coordinator = CanvasCoordinator(
      sessionFileStore: sessionFileStore,
      assetRegistry: assetRegistry,
      webViewProvider: { [weak self] in self?.webView }
    )
    configuration.userContentController.add(coordinator, name: CanvasCoordinator.messageHandlerName)
    do {
      let configScript = try runtimeConfig.userScriptSource()
      configuration.userContentController.addUserScript(
        WKUserScript(source: configScript, injectionTime: .atDocumentStart, forMainFrameOnly: true))
    } catch {
      // Without the configuration the page refuses to boot by design; surface
      // the reason instead of a silent blank window.
      NSLog("ThinTalk canvas: runtime configuration could not be encoded: \(error)")
    }
    let pagePreferences = WKWebpagePreferences()
    pagePreferences.allowsContentJavaScript = true
    configuration.defaultWebpagePreferences = pagePreferences

    let webview = WKWebView(frame: bounds, configuration: configuration)
    webview.navigationDelegate = coordinator
    webview.autoresizingMask = [.width, .height]
    addSubview(webview)
    webView = webview

    let shellURL = CanvasSecurityPolicy.shellDocumentURL()
    guard let shellURL else {
      NSLog("ThinTalk canvas: the bundled shell document could not be located")
      return
    }
    self.assetSchemeHandler = assetSchemeHandler
    _ = webview.load(URLRequest(url: shellURL))
  }
}

/// The bridge and navigation policy for the canvas web view.
@MainActor
final class CanvasCoordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
  static let messageHandlerName = "thintalk"

  private let sessionFileStore: SessionFileStore
  private let assetRegistry: TranscriptAssetRegistry
  private let webViewProvider: () -> WKWebView?
  private let storeQueue = DispatchQueue(label: "astronomical.thintalk.canvas.store", qos: .userInitiated)

  init(
    sessionFileStore: SessionFileStore,
    assetRegistry: TranscriptAssetRegistry,
    webViewProvider: @escaping () -> WKWebView?
  ) {
    self.sessionFileStore = sessionFileStore
    self.assetRegistry = assetRegistry
    self.webViewProvider = webViewProvider
  }

  // MARK: - WKScriptMessageHandler

  func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
    guard let envelope = message.body as? [String: Any], let kind = envelope["kind"] as? String else {
      NSLog("ThinTalk canvas: ignoring message without a kind: \(String(describing: message.body))")
      return
    }
    switch kind {
    case "sessionCall":
      handleSessionCall(envelope)
    case "action":
      handleAction(envelope)
    case "error":
      NSLog("ThinTalk canvas page error: \(envelope["detail"] as? String ?? "unknown")")
    default:
      NSLog("ThinTalk canvas: ignoring unknown message kind: \(kind)")
    }
  }

  private func handleSessionCall(_ envelope: [String: Any]) {
    guard let callID = envelope["callId"] as? String, let op = envelope["op"] as? String else {
      NSLog("ThinTalk canvas: sessionCall missing callId or op")
      return
    }
    // The payload is serialized to JSON on the main thread so only Sendable
    // data crosses into the store queue.
    let payloadData: Data?
    if let payload = envelope["payload"], !(payload is NSNull) {
      do {
        payloadData = try JSONSerialization.data(withJSONObject: payload, options: [.fragmentsAllowed])
      } catch {
        settleSessionCall(
          callID: callID,
          outcome: .failure(.init(message: "\(op) payload is not JSON: \(error)")))
        return
      }
    } else {
      payloadData = nil
    }
    storeQueue.async { [weak self] in
      guard let self else { return }
      let outcome = self.performSessionCall(op: op, payloadData: payloadData)
      DispatchQueue.main.async { [weak self] in
        self?.settleSessionCall(callID: callID, outcome: outcome)
      }
    }
  }

  private nonisolated func performSessionCall(op: String, payloadData: Data?) -> Result<String?, SessionBridgeFailure> {
    let payloadObject: [String: Any]?
    if let payloadData {
      guard let parsed = try? JSONSerialization.jsonObject(with: payloadData) as? [String: Any] else {
        return .failure(.init(message: "\(op) payload is not an object"))
      }
      payloadObject = parsed
    } else {
      payloadObject = nil
    }
    do {
      switch op {
      case "list":
        let summaries = try sessionFileStore.list()
        let payloadJSON: [[String: String]] = summaries.map { summary in
          ["id": summary.id, "title": summary.title, "updatedAt": summary.updatedAt]
        }
        return .success(try encodedJSONString(payloadJSON))
      case "load":
        guard let documentID = payloadObject?["id"] as? String else {
          return .failure(.init(message: "load requires an id"))
        }
        guard let documentData = try sessionFileStore.load(id: documentID) else { return .success("null") }
        return .success(String(decoding: documentData, as: UTF8.self))
      case "save":
        guard let document = payloadObject, let documentID = document["id"] as? String else {
          return .failure(.init(message: "save requires a document with an id"))
        }
        let documentData = try JSONSerialization.data(withJSONObject: document)
        try sessionFileStore.save(id: documentID, documentData: documentData)
        return .success("null")
      case "delete":
        guard let documentID = payloadObject?["id"] as? String else {
          return .failure(.init(message: "delete requires an id"))
        }
        try sessionFileStore.delete(id: documentID)
        return .success("null")
      case "rename":
        guard let renameRequest = payloadObject, let documentID = renameRequest["id"] as? String,
              let newTitle = renameRequest["title"] as? String
        else {
          return .failure(.init(message: "rename requires an id and a title"))
        }
        try sessionFileStore.rename(id: documentID, title: newTitle)
        return .success("null")
      case "loadPrefs":
        guard let preferencesData = try sessionFileStore.loadPreferences() else { return .success("null") }
        return .success(String(decoding: preferencesData, as: UTF8.self))
      case "savePrefs":
        guard let preferences = payloadObject else {
          return .failure(.init(message: "savePrefs requires a preferences object"))
        }
        try sessionFileStore.savePreferences(JSONSerialization.data(withJSONObject: preferences))
        return .success("null")
      default:
        return .failure(.init(message: "unknown session op: \(op)"))
      }
    } catch {
      return .failure(.init(message: "\(op) failed: \(error)"))
    }
  }

  private func settleSessionCall(callID: String, outcome: Result<String?, SessionBridgeFailure>) {
    guard let webView = webViewProvider() else { return }
    let script: String
    switch outcome {
    case .success(let payloadJSON):
      script = "window.__thintalkBridge.resolve(\(javaScriptString(callID)), \(payloadJSON ?? "null"));"
    case .failure(let failure):
      script = "window.__thintalkBridge.reject(\(javaScriptString(callID)), \(javaScriptString(failure.message)));"
    }
    webView.evaluateJavaScript(script, completionHandler: nil)
  }

  private func handleAction(_ envelope: [String: Any]) {
    guard let action = envelope["action"] as? String else { return }
    switch action {
    case "copy":
      let detail = envelope["detail"] as? String ?? ""
      NSPasteboard.general.clearContents()
      NSPasteboard.general.setString(detail, forType: .string)
    case "openExternal":
      guard let rawURL = envelope["detail"] as? String,
            let externalURL = CanvasSecurityPolicy.externalURL(fromRawValue: rawURL)
      else {
        NSLog("ThinTalk canvas: refusing to open unapproved external URL")
        return
      }
      NSWorkspace.shared.open(externalURL)
    default:
      NSLog("ThinTalk canvas: ignoring unknown action: \(action)")
    }
  }

  // MARK: - WKNavigationDelegate

  func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction) async -> WKNavigationActionPolicy {
    guard let requestURL = navigationAction.request.url else { return .cancel }
    if CanvasSecurityPolicy.isShellDocument(requestURL) || CanvasSecurityPolicy.isInternalAsset(requestURL) {
      return .allow
    }
    NSLog("ThinTalk canvas: blocking navigation to \(requestURL.absoluteString)")
    return .cancel
  }

  // MARK: - JSON helpers

  private nonisolated func encodedJSONString(_ object: some Encodable) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: object)
    return String(decoding: data, as: UTF8.self)
  }

  /// A JSON string literal for embedding in generated JavaScript. JSON escaping
  /// is also valid JavaScript string escaping for every character this bridge
  /// can produce.
  private func javaScriptString(_ value: String) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]),
          let literal = String(data: data, encoding: .utf8)
    else {
      return "\"\""
    }
    return literal
  }
}

/// A bridge call the host could not complete.
struct SessionBridgeFailure: Error {
  let message: String
}
