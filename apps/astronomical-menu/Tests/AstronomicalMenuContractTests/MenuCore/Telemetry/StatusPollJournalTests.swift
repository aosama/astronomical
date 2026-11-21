import XCTest

@testable import AstronomicalMenuCore

final class StatusPollJournalTests: XCTestCase {
  private func readyDocument() throws -> DaemonStatusDocument {
    try JSONDecoder().decode(
      DaemonStatusDocument.self, from: Data(#"{"status":"ready","activity":"idle"}"#.utf8))
  }

  func test_should_log_the_first_failed_poll_with_its_reason() {
    var journal = StatusPollJournal()
    let entry = journal.entry(
      statusDocument: .unavailable, pollErrorMessage: "Could not connect to the server.",
      engineIsDown: false)
    XCTAssertEqual(entry, "status poll failed: Could not connect to the server.")
  }

  func test_should_stay_silent_while_nothing_changes() {
    var journal = StatusPollJournal()
    _ = journal.entry(
      statusDocument: .unavailable, pollErrorMessage: "timed out", engineIsDown: false)
    for _ in 0..<5 {
      XCTAssertNil(
        journal.entry(
          statusDocument: .unavailable, pollErrorMessage: "timed out", engineIsDown: false))
    }
  }

  func test_should_log_when_the_reason_changes_and_when_the_menu_bar_mark_appears() {
    var journal = StatusPollJournal()
    _ = journal.entry(
      statusDocument: .unavailable, pollErrorMessage: "timed out", engineIsDown: false)
    XCTAssertEqual(
      journal.entry(
        statusDocument: .unavailable, pollErrorMessage: "timed out", engineIsDown: true),
      "status poll failed: timed out; menu bar shows Not running")
    XCTAssertEqual(
      journal.entry(
        statusDocument: .unavailable, pollErrorMessage: "wrong instance", engineIsDown: true),
      "status poll failed: wrong instance; menu bar shows Not running")
  }

  func test_should_log_the_recovery_once() throws {
    var journal = StatusPollJournal()
    _ = journal.entry(
      statusDocument: .unavailable, pollErrorMessage: "timed out", engineIsDown: true)
    let readyDocument = try readyDocument()
    XCTAssertEqual(
      journal.entry(statusDocument: readyDocument, pollErrorMessage: nil, engineIsDown: false),
      "engine answered with status ready")
    XCTAssertNil(
      journal.entry(statusDocument: readyDocument, pollErrorMessage: nil, engineIsDown: false))
  }

  /// An engine that answers "unavailable" is reachable; the log must not call
  /// that a failed poll, or the two causes cannot be told apart afterwards.
  func test_should_tell_an_engine_reported_unavailable_from_a_failed_poll() {
    var journal = StatusPollJournal()
    XCTAssertEqual(
      journal.entry(statusDocument: .unavailable, pollErrorMessage: nil, engineIsDown: false),
      "engine answered with status unavailable")
  }
}
