// What the popover's two speed tiles say: how fast the model reads prompts and how fast it writes
// replies. These are the two figures people compare between local AI apps (prefill and decode
// tokens per second), so they carry plain names with the usual names in a tooltip.

import Foundation

struct SessionSpeedTile: Equatable {
  let title: String
  let speedTitle: String
  /// True while the tile shows the running request's own speed instead of the session average.
  let isLive: Bool
  let explanation: String
}

struct SessionSpeedTilesPresentation: Equatable {
  let readingTile: SessionSpeedTile
  let writingTile: SessionSpeedTile
  let caption: String

  init(statusDocument: DaemonStatusDocument) {
    let servingSession = statusDocument.servingSession
    let liveTokensPerSecond = statusDocument.currentPhaseTokensPerSecond
    readingTile = SessionSpeedTile(
      title: "Reading speed",
      sessionTokensPerSecond: servingSession.averagePrefillTokensPerSecond,
      liveTokensPerSecond: statusDocument.activity == "prompt_processing" ? liveTokensPerSecond : nil,
      explanation:
        "Prefill: how fast the model reads a prompt, in tokens per second. Text restored from the prompt cache is not counted."
    )
    writingTile = SessionSpeedTile(
      title: "Writing speed",
      sessionTokensPerSecond: servingSession.averageGenerationTokensPerSecond,
      liveTokensPerSecond: statusDocument.activity == "generating" ? liveTokensPerSecond : nil,
      explanation: "Decode: how fast the model writes a reply, in tokens per second."
    )
    let requestCount = servingSession.completedRequestCount
    caption =
      requestCount == 0
      ? "Averages appear after the first request"
      : "Average this session · \(requestCount) \(requestCount == 1 ? "request" : "requests")"
  }
}

extension SessionSpeedTile {
  init(
    title: String, sessionTokensPerSecond: Double, liveTokensPerSecond: Double?,
    explanation: String
  ) {
    self.title = title
    self.explanation = explanation
    if let liveTokensPerSecond {
      speedTitle = sessionSpeedTitle(tokensPerSecond: liveTokensPerSecond)
      isLive = true
    } else {
      // A session with nothing measured for this phase has no speed; zero would read as "stalled".
      speedTitle =
        sessionTokensPerSecond > 0 ? sessionSpeedTitle(tokensPerSecond: sessionTokensPerSecond) : "—"
      isLive = false
    }
  }
}

/// Whole tokens per second with a thousands separator: reading speeds run to four digits, and a
/// decimal place on either figure changes from request to request without telling anyone anything.
func sessionSpeedTitle(tokensPerSecond: Double) -> String {
  let wholeTokensPerSecond = Int(tokensPerSecond.rounded())
  let groupedDigits = wholeTokensPerSecond.formatted(
    .number.grouping(.automatic).locale(Locale(identifier: "en_US")))
  return "\(groupedDigits) tok/s"
}
