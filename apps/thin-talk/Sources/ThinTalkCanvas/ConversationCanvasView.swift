import Foundation
import ThinTalkCore
import WebKit

/// The conversation canvas: one web view rendering the bundled HTML client.
///
/// This is a dumb host, deliberately. The page owns the conversation state, the
/// transcript rendering, and the daemon conversation; Swift owns only what a
/// web page cannot do for itself: reading and writing session files, keeping
/// generated images, asking the user's approval for tools and letting them pick
/// a file, saving exported files, the system pasteboard, opening external
/// links, asking its host to
/// restart the local engine, and injecting the runtime configuration.
/// Everything else crosses the bridge as data.
///
/// The bridge contract lives in the web client (`SessionBridge.ts`): the page
/// posts `{kind: "sessionCall", callId, op, payload}` envelopes and Swift answers
/// through `window.__thintalkBridge.resolve/reject`. A call the host fails to
/// answer rejects on the page after its own timeout, so the page never hangs on
/// a silent host.
@MainActor
public final class ConversationCanvasView: NSView {
  /// The accessibility identifier of the hosted web view, so automation and
  /// assistive technology can anchor on the canvas itself before locating the
  /// controls inside the web shell (issue 752).
  nonisolated public static let canvasAccessibilityIdentifier = "conversation-canvas"

  private let webDirectory: URL
  private let assetRegistry: TranscriptAssetRegistry
  private let sessionFileStore: SessionFileStore
  private let generatedImageStore: GeneratedImageStore
  private let runtimeConfig: CanvasRuntimeConfig
  /// Restarts the local engine when the page asks; nil where the host cannot.
  private let restartEngine: (@MainActor () -> Void)?
  private var webView: WKWebView?
  /// Takes the clicks in the title bar strip, which the page would otherwise get.
  private let titlebarStrip = TitlebarStripView()
  /// Commands sent before the page finished loading. A command for a page
  /// that has not booted is otherwise lost (the page's command router does
  /// not exist yet), which matters when a host opens the window and asks for
  /// Settings in the same breath.
  private var commandsAwaitingPageLoad: [CanvasPageCommand] = []
  /// A draft sent before the page finished loading; only the latest is kept.
  private var draftAwaitingPageLoad: String?
  /// Where the next export goes without a save panel, when a Development
  /// export-folder link set one. Cleared as soon as an export takes it.
  private var folderForNextExport: URL?
  private var isPageLoaded = false
  /// The native title bar's reach into the page, in points; nil for a plain
  /// title bar. Kept so every zoom change can re-send it in CSS pixels.
  private var titlebarInsetPoints: Int?
  // WebKit does not retain scheme handlers, so the view owns this one for as
  // long as the web view lives.
  private var assetSchemeHandler: CanvasAssetSchemeHandler?
  /// Receives the View menu's zoom intents. Set by the hosting view layer; the
  /// canvas registers itself on assignment so the commands reach the live web
  /// view once it exists.
  public var zoomController: CanvasZoomController? {
    didSet { zoomController?.attach(self) }
  }
  /// Receives the menu bar's and toolbar's conversation commands, registered
  /// the same way as the zoom controller.
  public var commandController: CanvasCommandController? {
    didSet { commandController?.attach(self) }
  }

  public init(
    webDirectory: URL,
    assetRegistry: TranscriptAssetRegistry,
    sessionFileStore: SessionFileStore,
    runtimeConfig: CanvasRuntimeConfig,
    restartEngine: (@MainActor () -> Void)? = nil
  ) {
    self.webDirectory = webDirectory
    self.assetRegistry = assetRegistry
    self.sessionFileStore = sessionFileStore
    self.generatedImageStore = GeneratedImageStore(directory: sessionFileStore.generatedImagesDirectory)
    self.runtimeConfig = runtimeConfig
    self.restartEngine = restartEngine
    self.titlebarInsetPoints = runtimeConfig.titlebarInset
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
      webDirectory: webDirectory, assetRegistry: assetRegistry, generatedImageStore: generatedImageStore)
    configuration.setURLSchemeHandler(
      assetSchemeHandler, forURLScheme: CanvasSecurityPolicy.shellScheme)
    let coordinator = CanvasCoordinator(
      sessionFileStore: sessionFileStore,
      generatedImageStore: generatedImageStore,
      assetRegistry: assetRegistry,
      restartEngine: restartEngine,
      webViewProvider: { [weak self] in self?.webView },
      takeFolderForNextExport: { [weak self] in self?.takeFolderForNextExport() },
      onPageLoaded: { [weak self] in self?.pageDidLoad() }
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
    // Trackpad pinch zooms the canvas; the View menu's Cmd +/-/0 commands carry
    // the keyboard half of the same behaviour.
    webview.allowsMagnification = true
    webview.setAccessibilityIdentifier(Self.canvasAccessibilityIdentifier)
    addSubview(webview)
    webView = webview
    // Above the web view, so clicks in the strip reach it first.
    addSubview(titlebarStrip, positioned: .above, relativeTo: webview)
    layoutTitlebarStrip()

    let shellURL = CanvasSecurityPolicy.shellDocumentURL()
    guard let shellURL else {
      NSLog("ThinTalk canvas: the bundled shell document could not be located")
      return
    }
    self.assetSchemeHandler = assetSchemeHandler
    _ = webview.load(URLRequest(url: shellURL))
  }
  // MARK: - Zoom

  public func adjustPageZoom(by factor: CGFloat) {
    guard let webView else { return }
    webView.pageZoom = CanvasZoomController.clampedPageZoom(current: webView.pageZoom, factor: factor)
    resendTitlebarInset()
  }

  public func resetPageZoom() {
    webView?.pageZoom = CanvasZoomController.defaultPageZoom
    resendTitlebarInset()
  }

  // MARK: - Title bar inset

  /// Records a new title bar height and passes it to the page.
  public func applyTitlebarInset(points insetPoints: Int) {
    titlebarInsetPoints = insetPoints
    layoutTitlebarStrip()
    resendTitlebarInset()
  }

  /// Pins the strip to the top edge at the title bar's height in points: the
  /// native strip is not scaled by page zoom, so neither is this. A plain
  /// title bar, or full screen without one, leaves it zero high.
  private func layoutTitlebarStrip() {
    let stripHeight = CGFloat(titlebarInsetPoints ?? 0)
    titlebarStrip.frame = Self.titlebarStripFrame(in: bounds, stripHeight: stripHeight, isFlipped: isFlipped)
    titlebarStrip.autoresizingMask = isFlipped ? [.width, .maxYMargin] : [.width, .minYMargin]
  }

  nonisolated static func titlebarStripFrame(in canvasBounds: NSRect, stripHeight: CGFloat, isFlipped: Bool) -> NSRect {
    let clampedHeight = min(max(0, stripHeight), canvasBounds.height)
    let originY = isFlipped ? canvasBounds.minY : canvasBounds.maxY - clampedHeight
    return NSRect(x: canvasBounds.minX, y: originY, width: canvasBounds.width, height: clampedHeight)
  }

  /// Zoom scales the page's CSS pixels but not the native strip, so the page's
  /// inset is restated at the new zoom; a plain title bar has nothing to send.
  private func resendTitlebarInset() {
    guard let webView, let titlebarInsetPoints else { return }
    webView.evaluateJavaScript(
      CanvasPageCommand.titlebarInsetScript(insetPoints: titlebarInsetPoints, pageZoom: webView.pageZoom),
      completionHandler: nil)
  }

  // MARK: - Page commands

  /// Delivers one menu or toolbar command to the page, holding it until the
  /// page has loaded when it has not yet.
  public func runPageCommand(_ pageCommand: CanvasPageCommand) {
    guard isPageLoaded else {
      commandsAwaitingPageLoad.append(pageCommand)
      return
    }
    webView?.evaluateJavaScript(pageCommand.scriptSource, completionHandler: nil)
  }

  /// Puts a draft in the page's message field. Kept until the page has loaded
  /// when it has not yet, so a link that launches the app still fills it.
  public func fillDraft(_ draftText: String) {
    guard isPageLoaded else {
      draftAwaitingPageLoad = draftText
      return
    }
    guard let draftScript = CanvasPageCommand.draftScript(text: draftText) else { return }
    webView?.evaluateJavaScript(draftScript, completionHandler: nil)
  }

  /// Has the next export written into `exportFolder` under its suggested
  /// name, with no save panel. It holds for one export only, so a later one
  /// asks again as usual.
  public func saveNextExportWithoutAsking(into exportFolder: URL) {
    folderForNextExport = exportFolder
  }

  private func takeFolderForNextExport() -> URL? {
    let unaskedFolder = folderForNextExport
    folderForNextExport = nil
    return unaskedFolder
  }

  private func pageDidLoad() {
    isPageLoaded = true
    let heldCommands = commandsAwaitingPageLoad
    commandsAwaitingPageLoad = []
    for heldCommand in heldCommands {
      runPageCommand(heldCommand)
    }
    if let heldDraft = draftAwaitingPageLoad {
      draftAwaitingPageLoad = nil
      fillDraft(heldDraft)
    }
  }

  /// Answers the zoom keys (⌘= and ⌘+, ⌘-, ⌘0) before the menu lookup, so each
  /// keystroke zooms exactly one step: the View menu never sees a key answered
  /// here. AppKit matches menu key equivalents against the unshifted character,
  /// so the View menu's "=" item can never answer ⌘+ itself; the canvas does.
  /// Every other key equivalent falls through so the web view and the menu keep
  /// their shortcuts.
  public override func performKeyEquivalent(with event: NSEvent) -> Bool {
    if let zoomController,
      zoomController.performZoomKey(
        keyCharacters: event.characters,
        modifierFlags: event.modifierFlags)
    {
      return true
    }
    return super.performKeyEquivalent(with: event)
  }
}

/// The bridge and navigation policy for the canvas web view.
@MainActor
final class CanvasCoordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
  static let messageHandlerName = "thintalk"

  private let sessionFileStore: SessionFileStore
  private let generatedImageStore: GeneratedImageStore
  private let assetRegistry: TranscriptAssetRegistry
  private let restartEngine: (@MainActor () -> Void)?
  private let webViewProvider: () -> WKWebView?
  /// Hands over, once, the folder a Development export-folder link named.
  private let takeFolderForNextExport: () -> URL?
  private let onPageLoaded: () -> Void
  private let storeQueue = DispatchQueue(label: "astronomical.thintalk.canvas.store", qos: .userInitiated)
  /// Session ops that show the user a sheet or a panel. They run on the main
  /// thread and may wait minutes for an answer, unlike file ops.
  private static let hostToolOps: Set<String> = ["useHostTool", "cancelHostToolCall"]
  private lazy var toolHost = ThinTalkToolHost(windowProvider: { [weak self] in self?.webViewProvider()?.window })
  private let conversationContextMenu = ConversationContextMenu()

  init(
    sessionFileStore: SessionFileStore,
    generatedImageStore: GeneratedImageStore,
    assetRegistry: TranscriptAssetRegistry,
    restartEngine: (@MainActor () -> Void)?,
    webViewProvider: @escaping () -> WKWebView?,
    takeFolderForNextExport: @escaping () -> URL?,
    onPageLoaded: @escaping () -> Void
  ) {
    self.sessionFileStore = sessionFileStore
    self.generatedImageStore = generatedImageStore
    self.assetRegistry = assetRegistry
    self.restartEngine = restartEngine
    self.webViewProvider = webViewProvider
    self.takeFolderForNextExport = takeFolderForNextExport
    self.onPageLoaded = onPageLoaded
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
      let errorDetail: String
      if let storedDetail = envelope["detail"] as? String {
        errorDetail = storedDetail
      } else {
        errorDetail = "unknown"
      }
      NSLog("ThinTalk canvas page error: \(errorDetail)")
    default:
      NSLog("ThinTalk canvas: ignoring unknown message kind: \(kind)")
    }
  }

  private func handleSessionCall(_ envelope: [String: Any]) {
    guard let callID = envelope["callId"] as? String, let op = envelope["op"] as? String else {
      NSLog("ThinTalk canvas: sessionCall missing callId or op")
      return
    }
    if Self.hostToolOps.contains(op) {
      handleHostToolCall(callID: callID, op: op, payloadObject: envelope["payload"] as? [String: Any])
      return
    }
    if op == "showConversationMenu" {
      handleConversationMenuCall(callID: callID)
      return
    }
    if op == "saveExport" {
      handleSaveExportCall(callID: callID, payloadObject: envelope["payload"] as? [String: Any])
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
          var summaryJSON = ["id": summary.id, "title": summary.title, "updatedAt": summary.updatedAt]
          if let skill = summary.skill {
            summaryJSON["skill"] = skill
          }
          return summaryJSON
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
      case "loadAstroProfile":
        guard let astroProfileData = try sessionFileStore.loadAstroProfile() else { return .success("null") }
        return .success(String(decoding: astroProfileData, as: UTF8.self))
      case "saveAstroProfile":
        guard let astroProfile = payloadObject else {
          return .failure(.init(message: "saveAstroProfile requires a profile object"))
        }
        try sessionFileStore.saveAstroProfile(JSONSerialization.data(withJSONObject: astroProfile))
        return .success("null")
      case "storeGeneratedImage", "deleteGeneratedImages", "loadGeneratedImage":
        return try performGeneratedImageCall(op: op, payloadObject: payloadObject)
      default:
        return .failure(.init(message: "unknown session op: \(op)"))
      }
    } catch {
      return .failure(.init(message: "\(op) failed: \(error)"))
    }
  }

  /// The generated-image ops, kept apart from the session ops: the store
  /// checks every image and every name, so the page never chooses a path.
  private nonisolated func performGeneratedImageCall(op: String, payloadObject: [String: Any]?) throws -> Result<String?, SessionBridgeFailure> {
    if op == "deleteGeneratedImages" {
      guard let imageNames = payloadObject?["imageNames"] as? [String] else {
        return .failure(.init(message: "deleteGeneratedImages requires a list of image names"))
      }
      try generatedImageStore.delete(imageNames: imageNames)
      return .success("null")
    }
    if op == "loadGeneratedImage" {
      guard let imageName = payloadObject?["imageName"] as? String else {
        return .failure(.init(message: "loadGeneratedImage requires an image name"))
      }
      guard let storedImage = try generatedImageStore.base64Image(named: imageName) else { return .success("null") }
      return .success(try encodedJSONString(["base64": storedImage.base64, "mimeType": storedImage.mimeType]))
    }
    guard let encodedImage = payloadObject?["base64"] as? String else {
      return .failure(.init(message: "storeGeneratedImage requires base64 image data"))
    }
    let imageName = try generatedImageStore.save(base64Image: encodedImage)
    return .success(try encodedJSONString(["imageName": imageName]))
  }

  /// `useHostTool` waits for the user, so it settles when they answer;
  /// `cancelHostToolCall` closes the named use's sheet or panel and settles at
  /// once. Encoding the outcome is the tool host's; this only routes.
  private func handleHostToolCall(callID: String, op: String, payloadObject: [String: Any]?) {
    let conversationID = payloadObject?["conversationID"] as? String ?? ""
    let toolName = payloadObject?["toolName"] as? String ?? ""
    if op == "cancelHostToolCall" {
      if let cancelledCallID = payloadObject?["callId"] as? String {
        toolHost.cancel(callID: cancelledCallID)
      }
      settleSessionCall(callID: callID, outcome: .success("null"))
      return
    }
    Task { @MainActor [weak self] in
      guard let self else { return }
      let toolOutcome = await self.toolHost.use(toolName: toolName, conversationID: conversationID, callID: callID)
      let encodedOutcome = Result { try self.encodedJSONString(toolOutcome.bridgePayload) }
        .mapError { SessionBridgeFailure(message: "useHostTool could not encode its outcome: \($0)") }
      self.settleSessionCall(callID: callID, outcome: encodedOutcome.map { Optional($0) })
    }
  }

  /// The sidebar's right-click menu runs on the main thread like the host
  /// tools, and settles when the user picks, dismisses, or answers the delete
  /// sheet.
  private func handleConversationMenuCall(callID: String) {
    guard let webView = webViewProvider() else {
      settleSessionCall(callID: callID, outcome: .success(#"{"choice":"dismissed"}"#))
      return
    }
    Task { @MainActor [weak self] in
      guard let self else { return }
      let menuChoice = await self.conversationContextMenu.choose(in: webView)
      let encodedChoice = Result { try self.encodedJSONString(["choice": menuChoice.rawValue]) }
        .mapError { SessionBridgeFailure(message: "showConversationMenu could not encode its choice: \($0)") }
      self.settleSessionCall(callID: callID, outcome: encodedChoice.map { Optional($0) })
    }
  }

  /// Saving an export shows a save panel and, for a PDF, prints the page, so
  /// it runs on the main thread like the menu and settles when it is done.
  private func handleSaveExportCall(callID: String, payloadObject: [String: Any]?) {
    guard let payloadObject, let exportRequest = ExportedFileRequest(payload: payloadObject) else {
      settleSessionCall(callID: callID, outcome: .failure(.init(message: "saveExport requires a format and its content")))
      return
    }
    let unaskedFolder = takeFolderForNextExport()
    Task { @MainActor [weak self] in
      guard let self else { return }
      let encodedOutcome: Result<String?, SessionBridgeFailure>
      do {
        let saveOutcome: ExportedFileSaveOutcome
        if let unaskedFolder {
          saveOutcome = try await ExportedFileSaver.save(exportRequest, into: unaskedFolder)
        } else {
          saveOutcome = try await ExportedFileSaver.save(exportRequest, from: self.webViewProvider()?.window)
        }
        encodedOutcome = .success(try self.encodedJSONString(saveOutcome.bridgePayload))
      } catch {
        NSLog("ThinTalk canvas: the export could not be saved: \(error)")
        encodedOutcome = .failure(.init(message: "saveExport failed: \(error)"))
      }
      self.settleSessionCall(callID: callID, outcome: encodedOutcome)
    }
  }

  private func settleSessionCall(callID: String, outcome: Result<String?, SessionBridgeFailure>) {
    guard let webView = webViewProvider() else { return }
    let script: String
    switch outcome {
    case .success(let payloadJSON):
      let resolvedPayloadJSON: String
      if let payloadJSON {
        resolvedPayloadJSON = payloadJSON
      } else {
        resolvedPayloadJSON = "null"
      }
      script = "window.__thintalkBridge.resolve(\(javaScriptString(callID)), \(resolvedPayloadJSON));"
    case .failure(let failure):
      script = "window.__thintalkBridge.reject(\(javaScriptString(callID)), \(javaScriptString(failure.message)));"
    }
    webView.evaluateJavaScript(script, completionHandler: nil)
  }

  private func handleAction(_ envelope: [String: Any]) {
    guard let action = envelope["action"] as? String else { return }
    switch action {
    case "copy":
      let copiedDetail: String
      if let storedDetail = envelope["detail"] as? String {
        copiedDetail = storedDetail
      } else {
        copiedDetail = ""
      }
      NSPasteboard.general.clearContents()
      NSPasteboard.general.setString(copiedDetail, forType: .string)
    case "openExternal":
      guard let rawURL = envelope["detail"] as? String,
            let externalURL = CanvasSecurityPolicy.externalURL(fromRawValue: rawURL)
      else {
        NSLog("ThinTalk canvas: refusing to open unapproved external URL")
        return
      }
      NSWorkspace.shared.open(externalURL)
    case "saveGeneratedImage", "copyGeneratedImage":
      // The store checks the name before any path is built, so a crafted
      // name cannot reach a file outside the generated-images folder.
      guard let imageName = envelope["detail"] as? String,
            let imageURL = generatedImageStore.fileURL(forImageName: imageName)
      else {
        NSLog("ThinTalk canvas: no stored picture to \(action)")
        return
      }
      if action == "saveGeneratedImage" {
        GeneratedImageSharing.saveCopy(of: imageURL, from: webViewProvider()?.window)
      } else {
        GeneratedImageSharing.copyToPasteboard(imageURL)
      }
    case "restartEngine":
      // Astro's welcome offers this when the engine never answered; a host
      // without the ability never advertises it, so nil only logs.
      guard let restartEngine else {
        NSLog("ThinTalk canvas: this host cannot restart the engine")
        return
      }
      restartEngine()
    default:
      NSLog("ThinTalk canvas: ignoring unknown action: \(action)")
    }
  }

  // MARK: - WKNavigationDelegate

  /// The shell's scripts have run by the time the load finishes, so its
  /// command router is installed and held commands can be delivered.
  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    onPageLoaded()
  }

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
