import Foundation
import ThinTalkCanvas
import XCTest

/// The URL decisions the canvas depends on. These run without a web view because
/// the policy is a value decision, not a rendering detail.
final class CanvasSecurityPolicyTests: XCTestCase {
  func test_should_recognise_only_the_shell_document_as_the_shell() {
    XCTAssertEqual(
      CanvasSecurityPolicy.shellDocumentURL()?.absoluteString,
      "thintalk-asset://shell/index.html")
    let shellDocument = URL(string: "thintalk-asset://shell/index.html")!
    XCTAssertTrue(CanvasSecurityPolicy.isShellDocument(shellDocument))
    XCTAssertFalse(
      CanvasSecurityPolicy.isShellDocument(URL(string: "thintalk-asset://shell/canvas.js")!))
    XCTAssertFalse(
      CanvasSecurityPolicy.isShellDocument(URL(string: "https://example.com/index.html")!))
    XCTAssertFalse(
      CanvasSecurityPolicy.isShellDocument(URL(string: "file:///etc/passwd")!))
  }

  func test_should_treat_both_private_hosts_as_internal_assets() {
    XCTAssertTrue(
      CanvasSecurityPolicy.isInternalAsset(URL(string: "thintalk-asset://shell/canvas.css")!))
    XCTAssertTrue(
      CanvasSecurityPolicy.isInternalAsset(
        URL(string: "thintalk-asset://attachment/2f1c")!))
    XCTAssertFalse(
      CanvasSecurityPolicy.isInternalAsset(URL(string: "https://cdn.example.com/marked.js")!))
    XCTAssertFalse(
      CanvasSecurityPolicy.isInternalAsset(URL(string: "file:///Users/somebody/notes.md")!))
  }

  func test_should_open_only_browser_openable_schemes() {
    XCTAssertEqual(
      CanvasSecurityPolicy.externalURL(fromRawValue: "https://ml-explore.github.io/mlx/")?
        .absoluteString,
      "https://ml-explore.github.io/mlx/")
    XCTAssertEqual(
      CanvasSecurityPolicy.externalURL(fromRawValue: "http://127.0.0.1:6732/v1/status")?
        .absoluteString,
      "http://127.0.0.1:6732/v1/status")
    XCTAssertEqual(
      CanvasSecurityPolicy.externalURL(fromRawValue: "mailto:someone@example.com")?.scheme,
      "mailto")
  }

  func test_should_refuse_every_scheme_that_could_run_code_or_read_files() {
    let refused = [
      "javascript:alert(1)",
      "JavaScript:alert(1)",
      "data:text/html,<script>alert(1)</script>",
      "file:///Users/somebody/.ssh/id_rsa",
      "thintalk-asset://attachment/token",
      "vbscript:msgbox(1)",
      "about:blank",
      "//example.com/protocol-relative",
      "",
      "   ",
    ]
    for rawValue in refused {
      XCTAssertNil(
        CanvasSecurityPolicy.externalURL(fromRawValue: rawValue),
        "\(rawValue) must never leave the application")
    }
  }

  func test_should_extract_attachment_tokens_and_nothing_else() {
    XCTAssertEqual(
      CanvasSecurityPolicy.attachmentToken(
        from: URL(string: "thintalk-asset://attachment/2f1c9a")!), "2f1c9a")
    XCTAssertNil(
      CanvasSecurityPolicy.attachmentToken(
        from: URL(string: "thintalk-asset://shell/index.html")!))
    XCTAssertNil(
      CanvasSecurityPolicy.attachmentToken(from: URL(string: "https://example.com/2f1c9a")!))
    XCTAssertNil(
      CanvasSecurityPolicy.attachmentToken(from: URL(string: "thintalk-asset://attachment/")!))
  }
}