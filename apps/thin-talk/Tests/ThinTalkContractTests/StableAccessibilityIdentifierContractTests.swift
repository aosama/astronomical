import Foundation
import ThinTalkCanvas
import XCTest

// Issue 752: automation agents and assistive technology must locate Thin Talk
// controls by stable, domain-meaningful identifiers instead of churned element
// ids or frame geometry. The interactive controls render inside the bundled
// web shell, so this contract reads the shipped shell document and pins the
// identifier inventory: every interactive control carries an id equal to its
// data-testid (WebKit exposes the DOM id to the accessibility tree, so one
// token serves both consumers), singleton controls resolve to exactly one
// element, and the canvas host names itself. The per-bubble identifiers
// (`message:<role>:<message id>`) are minted at render time from the message's
// own id; the live behaviour contract for them lives in
// apps/thin-talk/web/test/stableIdentifierContract.test.ts.
final class StableAccessibilityIdentifierContractTests: XCTestCase {
  private let packageDirectoryURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  /// The singleton interactive controls of the shell. Each token is both the
  /// element's id and its data-testid, is lowercase-hyphenated, and never
  /// derives from visible text or layout position.
  private static let singletonControlIdentifiers = [
    "composer-input",
    "composer-send",
    "composer-stop",
    "composer-retry",
    "composer-effort",
    "composer-effort-menu",
    "composer-model",
    "composer-model-menu",
    "sidebar-new-chat",
  ]

  func test_should_carry_a_matching_id_and_testid_for_every_interactive_shell_control() throws {
    let shellSource = try String(contentsOf: shellDocumentURL(), encoding: .utf8)
    for controlIdentifier in Self.singletonControlIdentifiers {
      // The lookbehind keeps `id="…"` from also matching inside the
      // `data-testid="…"` attribute, whose name ends with the same letters.
      let idPattern = #"(?<![A-Za-z-])id="\#(controlIdentifier)""#
      let testidPattern = #"data-testid="\#(controlIdentifier)""#
      XCTAssertEqual(
        matchCount(of: idPattern, in: shellSource), 1,
        "the shell must carry exactly one id for the control \(controlIdentifier)")
      XCTAssertEqual(
        matchCount(of: testidPattern, in: shellSource), 1,
        "the shell must carry exactly one data-testid for the control \(controlIdentifier)")
    }
  }

  func test_should_keep_shell_identifiers_unambiguous() throws {
    let shellSource = try String(contentsOf: shellDocumentURL(), encoding: .utf8)
    let shellIdentifierValues = matches(of: #"(?<![A-Za-z-])id="([^"]*)""#, in: shellSource)
    XCTAssertFalse(shellIdentifierValues.isEmpty, "the shell document carries identifiers")
    let identifiersByCount = Dictionary(grouping: shellIdentifierValues, by: { $0 })
    let duplicatedIdentifiers = identifiersByCount
      .filter { _, occurrences in occurrences.count > 1 }
      .map { identifier, _ in identifier }
    XCTAssertTrue(
      duplicatedIdentifiers.isEmpty,
      "identifiers must resolve to exactly one element each; duplicated: \(duplicatedIdentifiers)")
  }

  func test_should_name_the_canvas_host_with_a_stable_identifier() {
    XCTAssertEqual(
      ConversationCanvasView.canvasAccessibilityIdentifier, "conversation-canvas",
      "the hosted web view must expose one stable identifier so automation can anchor on the canvas")
  }

  // MARK: - Helpers

  private func shellDocumentURL() -> URL {
    packageDirectoryURL
      .appendingPathComponent("Sources/ThinTalkCanvas/Resources/web/index.html")
  }

  private func matchCount(of pattern: String, in source: String) -> Int {
    matches(of: pattern, in: source).count
  }

  private func matches(of pattern: String, in source: String) -> [String] {
    guard let regularExpression = try? NSRegularExpression(pattern: pattern) else {
      return []
    }
    let wholeRange = NSRange(source.startIndex..., in: source)
    return regularExpression
      .matches(in: source, range: wholeRange)
      .map { match in
        // Patterns with one capture group (attribute extraction) report the
        // captured value; group-less patterns report the whole match.
        let reportedRange = regularExpression.numberOfCaptureGroups > 0
          ? match.range(at: 1)
          : match.range
        guard let matchedRange = Range(reportedRange, in: source) else {
          return ""
        }
        return String(source[matchedRange])
      }
  }
}
