// Decides when a silent engine counts as a problem worth showing in the menu bar.

import Foundation

/// The engine answers nothing for a few seconds on every launch, restart and update, so a single
/// unanswered poll is ordinary. Ten seconds is longer than any of those take on a working install
/// and short enough that a person who just sent a message still connects the mark to it.
let engineDowntimeGraceSeconds: TimeInterval = 10

struct EngineDowntimePolicy {
  private var unavailableSince: Date?

  /// Whether the engine has gone unanswered for longer than the grace period.
  mutating func engineIsDown(for statusDocument: DaemonStatusDocument, at pollTime: Date) -> Bool {
    guard statusDocument.status == "unavailable" else {
      unavailableSince = nil
      return false
    }
    guard let unavailableSince else {
      self.unavailableSince = pollTime
      return false
    }
    return pollTime.timeIntervalSince(unavailableSince) >= engineDowntimeGraceSeconds
  }
}
