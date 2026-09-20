import AppKit
import Foundation
import ThinTalkCanvas
import ThinTalkCore
import WebKit
import XCTest

/// Drives the bundled conversation canvas in a real `WKWebView`.
///
/// The harness wires exactly what production wires: the canvas shell from the
/// target's resource bundle, the private asset scheme handler, and the script
/// message channel back to Swift. Every wait is bounded so a misbehaving page
/// fails a case instead of hanging it.
@MainActor
final class CanvasHarness: NSObject, WKScriptMessageHandler {
  static let defaultTimeout: TimeInterval = 10

  let webView: WKWebView
  let assetRegistry = TranscriptAssetRegistry()

  private(set) var isReady = false
  private var reportedMessages: [[String: Any]] = []
  private var reportedErrors: [String] = []

  /// Every action the canvas reported, in arrival order.
  var reportedActions: [[String: Any]] { reportedMessages }

  /// Every rendering failure the canvas reported. A non-empty list means the
  /// canvas could not do its job, even if pixels appeared.
  var renderingErrors: [String] { reportedErrors }

  /// - Parameter webDirectory: the bundled canvas shell directory, resolved by the
  ///   caller so a missing resource bundle fails an assertion instead of loading an
  ///   empty page.
  init(webDirectory: URL) {
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.defaultWebpagePreferences.allowsContentJavaScript = true
    configuration.setURLSchemeHandler(
      CanvasAssetSchemeHandler(webDirectory: webDirectory, assetRegistry: assetRegistry),
      forURLScheme: CanvasSecurityPolicy.shellScheme
    )
    webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 900, height: 700), configuration: configuration)
    super.init()
    configuration.userContentController.add(self, name: "thintalk")
    webView.navigationDelegate = self
  }

  /// Starts the shell and waits until it reports that it can receive commands.
  func start() async throws {
    guard let documentURL = CanvasSecurityPolicy.shellDocumentURL() else {
      throw CanvasHarnessError.missingShellDocumentURL
    }
    webView.load(URLRequest(url: documentURL))
    let ready = await wait(timeout: Self.defaultTimeout) { self.isReady }
    guard ready else { throw CanvasHarnessError.shellNeverBecameReady }
  }

  func send(_ command: TranscriptCommand) async throws {
    let invocation = try TranscriptCommandCodec.invocation(for: command)
    _ = try await webView.callAsyncJavaScript(invocation, arguments: [:], in: nil, contentWorld: .page)
  }

  // MARK: - Waiting

  func waitForElement(_ selector: String, timeout: TimeInterval = defaultTimeout) async throws {
    let appeared = await wait(timeout: timeout) { (try? await self.count(of: selector)) ?? 0 > 0 }
    if !appeared {
      XCTFail("timed out waiting for element \(selector)")
      throw CanvasHarnessError.timedOut("element \(selector) never appeared")
    }
  }

  func waitForMessageCount(_ expected: Int, timeout: TimeInterval = defaultTimeout) async throws {
    let reached = await wait(timeout: timeout) {
      ((try? await self.count(of: "article.message")) ?? -1) == expected
    }
    if !reached {
      XCTFail("timed out waiting for \(expected) rendered messages")
      throw CanvasHarnessError.timedOut("message count never reached \(expected)")
    }
  }

  func waitForParagraph(_ text: String, timeout: TimeInterval = defaultTimeout) async throws {
    let updated = await wait(timeout: timeout) {
      ((try? await self.text(of: "article .markdown-body p")) ?? "") == text
    }
    if !updated {
      XCTFail("timed out waiting for the paragraph to become \(text)")
      throw CanvasHarnessError.timedOut("paragraph never became \(text)")
    }
  }

  func waitForReportedActionCount(
    _ expected: Int, timeout: TimeInterval = defaultTimeout
  ) async throws {
    let reached = await wait(timeout: timeout) { self.reportedMessages.count >= expected }
    if !reached {
      XCTFail("timed out waiting for \(expected) reported actions")
      throw CanvasHarnessError.timedOut("action count never reached \(expected)")
    }
  }

  func waitForImageLoad(_ selector: String, timeout: TimeInterval = defaultTimeout) async throws -> Bool {
    let literal = Self.javascriptLiteral(selector)
    return await wait(
      timeout: timeout,
      {
        ((try? await self.bool(
          of:
            "document.querySelector(\(literal))?.complete && document.querySelector(\(literal))?.naturalWidth > 0"
        )) ?? false)
      }
    )
  }

  // MARK: - Reading the DOM

  func text(of selector: String) async throws -> String {
    let result = try await evaluate("return document.querySelector(selector)?.textContent ?? null", selector: selector)
    return result as? String ?? ""
  }

  func count(of selector: String) async throws -> Int {
    let result = try await evaluate("return document.querySelectorAll(selector).length", selector: selector)
    return (result as? NSNumber)?.intValue ?? 0
  }

  /// Evaluates a JavaScript expression that must produce a boolean.
  func bool(of expression: String) async throws -> Bool {
    let result = try await webView.callAsyncJavaScript(
      "return Boolean(\(expression))", arguments: [:], in: nil, contentWorld: .page)
    return (result as? NSNumber)?.boolValue ?? (result as? Bool) ?? false
  }

  func setJS(_ body: String) async throws {
    _ = try await webView.callAsyncJavaScript(
      "\(body)\nreturn true;", arguments: [:], in: nil, contentWorld: .page)
  }

  func evaluate(_ body: String, selector: String) async throws -> Any? {
    try await webView.callAsyncJavaScript(
      body, arguments: ["selector": selector], in: nil, contentWorld: .page)
  }

  /// A snapshot of what the page actually loaded. A canvas that renders nothing
  /// is diagnosed from here rather than guessed at.
  func pageDiagnostics() async throws -> String {
    let result = try await webView.callAsyncJavaScript(
      "return JSON.stringify(window.__thintalk ? window.__thintalk.diagnostics() : { entryPoint: 'missing' })",
      arguments: [:], in: nil, contentWorld: .page)
    return result as? String ?? "(no diagnostics)"
  }

  /// Captures the rendered canvas as a PNG.
  ///
  /// A canvas is judged by what a reader sees, and the DOM assertions cannot show
  /// typography, spacing, or whether a block actually painted. This makes the
  /// rendered result reviewable as an image.
  func snapshotPNG() async throws -> Data {
    let configuration = WKSnapshotConfiguration()
    configuration.rect = webView.bounds
    configuration.snapshotWidth = NSNumber(value: Double(webView.bounds.width))
    let image = try await webView.takeSnapshot(configuration: configuration)
    guard
      let tiff = image.tiffRepresentation,
      let representation = NSBitmapImageRep(data: tiff),
      let png = representation.representation(using: .png, properties: [:]),
      !png.isEmpty
    else {
      throw CanvasHarnessError.couldNotBuildFixture
    }
    return png
  }

  /// Drives the entry point with a known-good appearance command and returns what
  /// it saw. This separates "the shell loaded" from "the bridge actually works".
  func probeReceive() async throws -> String {
    let result = try await webView.callAsyncJavaScript(
      """
      var payload = 'eyJ2IjoxLCJraW5kIjoiYXBwZWFyYW5jZSIsImRhcmsiOnRydWV9';
      var entry = typeof window.__thintalk;
      var receiver = entry === 'object' ? typeof window.__thintalk.receive : 'none';
      var accepted = receiver === 'function' ? window.__thintalk.receive(payload) : null;
      return JSON.stringify({
        entry: entry,
        receiver: receiver,
        accepted: accepted,
        appearance: document.documentElement.dataset.appearance || ''
      });
      """,
      arguments: [:], in: nil, contentWorld: .page)
    return (result as? String) ?? "(no probe result)"
  }

  /// Whether the page can still reach the network. Sanitisation and the content
  /// security policy must both hold, so this is asserted rather than assumed.
  func canReachNetwork() async -> Bool {    do {
      let result = try await webView.callAsyncJavaScript(
        """
        try {
          await fetch('https://example.com/probe');
          return true;
        } catch (error) {
          return false;
        }
        """,
        arguments: [:], in: nil, contentWorld: .page)
      return (result as? NSNumber)?.boolValue ?? false
    } catch {
      return false
    }
  }

  private func wait(
    timeout: TimeInterval, _ condition: @escaping () async -> Bool
  ) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if await condition() { return true }
      try? await Task.sleep(nanoseconds: 25_000_000)
    }
    return await condition()
  }

  private static func javascriptLiteral(_ value: String) -> String {
    let escaped = value
      .replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "'", with: "\\'")
    return "'\(escaped)'"
  }

  // MARK: - Bridge

  nonisolated func userContentController(
    _ userContentController: WKUserContentController,
    didReceive message: WKScriptMessage
  ) {
    MainActor.assumeIsolated {
      guard let body = message.body as? [String: Any] else { return }
      if body["kind"] as? String == "ready" {
        isReady = true
        return
      }
      if body["kind"] as? String == "error" {
        reportedErrors.append(body["detail"] as? String ?? "unspecified canvas error")
        return
      }
      reportedMessages.append(body)
    }
  }
}

extension CanvasHarness: WKNavigationDelegate {
  func webView(
    _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction
  ) async -> WKNavigationActionPolicy {
    guard let url = navigationAction.request.url else { return .cancel }
    return CanvasSecurityPolicy.isShellDocument(url) ? .allow : .cancel
  }
}

enum CanvasHarnessError: Error {
  case missingShellDocumentURL
  case shellNeverBecameReady
  case timedOut(String)
  case couldNotBuildFixture
}