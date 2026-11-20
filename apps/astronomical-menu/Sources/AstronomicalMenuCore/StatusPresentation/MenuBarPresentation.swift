// What the menu bar item shows for one status poll: the icon, the optional text beside it and the
// tooltip. The popover carries the full numbers; the menu bar stays quiet unless a person asks for
// the text, because a title that changes several times a second pulls the eye from their work.

import Foundation

struct MenuBarPresentation: Equatable {
  let glyph: MenuBarGlyph
  /// Progress and speed in words; shown only when the person switched it on.
  let activityTitle: String
  let showsProgressAndSpeed: Bool
  let engineIsDown: Bool
  let stateSentence: String

  init(statusDocument: DaemonStatusDocument, engineIsDown: Bool) {
    glyph = statusDocument.menuBarGlyph(engineIsDown: engineIsDown)
    activityTitle = statusDocument.menuBarActivityTitle
    showsProgressAndSpeed = statusDocument.menuBarShowsProgressAndSpeed
    self.engineIsDown = engineIsDown
    stateSentence = statusDocument.menuBarStateSentence(engineIsDown: engineIsDown)
  }

  /// Before the first poll answers: nothing is known, so nothing is claimed.
  static let launching = MenuBarPresentation(statusDocument: .unavailable, engineIsDown: false)

  func toolTip(buildTitle: String) -> String { "\(stateSentence)\n\(buildTitle)" }
}

let menuBarEngineDownTitle = "Not running"

/// The text beside the icon. A stopped engine is always named, because it is a problem only the
/// person can act on; the Development label always shows so the two instances are never confused.
func menuBarTitleToShow(presentation: MenuBarPresentation, isDevelopmentChannel: Bool) -> String {
  var titleParts: [String] = []
  if isDevelopmentChannel { titleParts.append("DEV") }
  if presentation.engineIsDown {
    titleParts.append(menuBarEngineDownTitle)
  } else if presentation.showsProgressAndSpeed, !presentation.activityTitle.isEmpty {
    titleParts.append(presentation.activityTitle)
  }
  // The leading space is the gap between the icon and its text.
  return titleParts.isEmpty ? "" : " " + titleParts.joined(separator: " · ")
}

/// Pads a count with figure spaces, which are as wide as a digit, so the menu bar item keeps one
/// width while the number changes and its neighbours do not shift.
func steadyWidthCount(_ count: Int, minimumDigitCount: Int) -> String {
  let digits = String(count)
  let paddingCount = max(0, minimumDigitCount - digits.count)
  return String(repeating: "\u{2007}", count: paddingCount) + digits
}

extension DaemonStatusDocument {
  var menuBarActivityTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.menuBarTitle }
    guard status == "ready" else { return status == "loading" ? "Loading" : "" }
    if activity == "generating", let currentPhaseTokensPerSecond {
      // Whole numbers: a tenth of a token per second changes on every poll and reads as noise.
      let wholeTokensPerSecond = Int(currentPhaseTokensPerSecond.rounded())
      return "\(steadyWidthCount(wholeTokensPerSecond, minimumDigitCount: 2)) tok/s"
    }
    if activity == "image_generation", let progress {
      if progress.hasCompletedDenoising { return "Image · Denoising" }
      return progress.phase == "denoising"
        ? "Image \(progress.steadyWidthPercentageTitle)"
        : "Image · \(progress.shortImagePhaseTitle)"
    }
    if activity == "generation_preparation" { return "Preparing…" }
    if activity == "speaking" { return "Speaking…" }
    if isRestoringAstroSsdPromptCache { return "Restoring…" }
    guard activity == "prompt_processing", let progress else {
      // Only when idle: a generation's own title always says more than a background count.
      return activity == "idle" ? neuralEnginePrefill?.menuBarTitle ?? "" : ""
    }
    let promptTitle = "Prompt \(progress.steadyWidthPercentageTitle)"
    guard let currentPhaseTokensPerSecond else { return promptTitle }
    let wholeTokensPerSecond = Int(currentPhaseTokensPerSecond.rounded())
    return "\(promptTitle) · \(steadyWidthCount(wholeTokensPerSecond, minimumDigitCount: 3)) tok/s"
  }

  /// One plain sentence for the tooltip, so hovering answers "what is it doing" without a click.
  func menuBarStateSentence(engineIsDown: Bool) -> String {
    if engineIsDown { return "The engine is not running. Click, then choose Restart server." }
    if status == "unavailable" { return "Starting" }
    if isLoadingModel || isActive { return flightTitle }
    if status == "loading" { return "Loading" }
    guard let readyModelIdentifier else { return "No model loaded" }
    return "\(readyModelIdentifier) · \(modelFootprintTitle)"
  }
}

extension DaemonStatusDocument.Progress {
  var steadyWidthPercentageTitle: String {
    "\(steadyWidthCount(Int((completionFraction * 100).rounded(.down)), minimumDigitCount: 2))%"
  }
}
