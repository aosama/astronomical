// What the popover's two speed tiles say: how fast the model reads prompts and how fast it writes
// replies. These are the two figures people compare between local AI apps (prefill and decode
// tokens per second), so they carry plain names with the usual names in a tooltip.
//
// Both tiles always show the whole session's average, the running request included: the daemon
// adds a running request's tokens and time to the session's as they happen, so the figures move
// with a long reply instead of waiting for it to end. SessionSpeedTilesHold refreshes them once a
// second.

import Foundation

struct SessionSpeedTile: Equatable {
  let title: String
  let speedTitle: String
  let explanation: String
}

struct SessionSpeedTilesPresentation: Equatable {
  let readingTile: SessionSpeedTile
  let writingTile: SessionSpeedTile
  let caption: String

  init(statusDocument: DaemonStatusDocument) {
    let servingSession = statusDocument.servingSession
    readingTile = SessionSpeedTile(
      title: "Reading speed",
      sessionTokensPerSecond: servingSession.averagePrefillTokensPerSecond,
      explanation:
        "Prefill: how fast the model reads a prompt, in tokens per second, averaged over this session. Text restored from the prompt cache is not counted."
    )
    writingTile = SessionSpeedTile(
      title: "Writing speed",
      sessionTokensPerSecond: servingSession.averageGenerationTokensPerSecond,
      explanation:
        "Decode: how fast the model writes a reply, in tokens per second, averaged over this session."
    )
    caption = sessionSpeedCaption(
      completedRequestCount: servingSession.completedRequestCount,
      isRequestRunning: statusDocument.isChatRequestRunning)
  }
}

extension SessionSpeedTile {
  init(title: String, sessionTokensPerSecond: Double, explanation: String) {
    self.title = title
    self.explanation = explanation
    // A session with nothing measured for this phase has no speed; zero would read as "stalled".
    speedTitle =
      sessionTokensPerSecond > 0 ? sessionSpeedTitle(tokensPerSecond: sessionTokensPerSecond) : "—"
  }
}

extension DaemonStatusDocument {
  /// Whether a chat request is reading its prompt or writing its reply, the work the two speeds
  /// measure.
  var isChatRequestRunning: Bool {
    activity == "prompt_processing" || activity == "generating"
  }
}

/// "Average this session · 12 requests, 1 running": the requests the figures cover, the running
/// one named apart because its work is counted so far, not finished.
func sessionSpeedCaption(completedRequestCount: UInt64, isRequestRunning: Bool) -> String {
  let runningPart = isRequestRunning ? "1 running" : nil
  let completedPart =
    completedRequestCount == 0
    ? nil : "\(completedRequestCount) \(completedRequestCount == 1 ? "request" : "requests")"
  let coveredParts = [completedPart, runningPart].compactMap { $0 }
  if coveredParts.isEmpty { return "Averages appear after the first request" }
  return "Average this session · " + coveredParts.joined(separator: ", ")
}

/// Whole tokens per second with a thousands separator: reading speeds run to four digits, and a
/// decimal place on either figure changes from request to request without telling anyone anything.
func sessionSpeedTitle(tokensPerSecond: Double) -> String {
  let wholeTokensPerSecond = Int(tokensPerSecond.rounded())
  let groupedDigits = wholeTokensPerSecond.formatted(
    .number.grouping(.automatic).locale(Locale(identifier: "en_US")))
  return "\(groupedDigits) tok/s"
}

/// Shows the tiles at most once a second. The status is polled four times a second while a request
/// runs, for the progress bar; figures that changed four times a second would be hard to read.
/// The first figures, and the figures when the popover opens, show at once.
struct SessionSpeedTilesHold {
  static let refreshInterval = Duration.seconds(1)

  private(set) var shownPresentation: SessionSpeedTilesPresentation?
  private var shownAt: ContinuousClock.Instant?

  /// The tiles to show at `now` for the latest status: the latest figures when a second has passed
  /// since the shown ones (or nothing is shown yet, or `showsAtOnce`), otherwise the shown ones.
  mutating func presentation(
    for latestPresentation: SessionSpeedTilesPresentation, at now: ContinuousClock.Instant,
    showsAtOnce: Bool = false
  ) -> SessionSpeedTilesPresentation {
    let isRefreshDue =
      shownAt.map { $0.duration(to: now) >= Self.refreshInterval } ?? true
    if let shownPresentation, !showsAtOnce, !isRefreshDue {
      return shownPresentation
    }
    shownPresentation = latestPresentation
    shownAt = now
    return latestPresentation
  }
}
