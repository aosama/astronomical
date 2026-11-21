import Foundation

/// Turns the once-a-second status poll into a few log lines: one each time
/// what the agent learned changes, and none while it stays the same.
///
/// The menu bar shows "Not running" when polls go unanswered, but the reason
/// (the engine refused the connection, answered for another instance, sent a
/// reply the agent could not read, or itself reported its worker unavailable)
/// was kept only in memory, so a mark that came and went left nothing to
/// investigate. A line per change keeps the log small enough to leave on.
struct StatusPollJournal {
  private var previousEntry: String?

  /// The line to log for this poll, or nil when nothing changed since the
  /// previous one.
  mutating func entry(
    statusDocument: DaemonStatusDocument, pollErrorMessage: String?, engineIsDown: Bool
  ) -> String? {
    let pollOutcome: String
    if let pollErrorMessage {
      pollOutcome = "status poll failed: \(pollErrorMessage)"
    } else if statusDocument.status == "unavailable" {
      // The engine answered, and said its worker is unavailable: a different
      // situation from an engine that cannot be reached at all.
      pollOutcome = "engine answered with status unavailable"
    } else {
      pollOutcome = "engine answered with status \(statusDocument.status)"
    }
    let menuBarMark = engineIsDown ? "; menu bar shows Not running" : ""
    let currentEntry = pollOutcome + menuBarMark
    guard currentEntry != previousEntry else { return nil }
    previousEntry = currentEntry
    return currentEntry
  }
}
