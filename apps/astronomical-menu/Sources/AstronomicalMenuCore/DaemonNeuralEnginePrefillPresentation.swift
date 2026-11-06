// Owns the words the menu uses for Neural Engine programs. The model already serves on the GPU while
// they are prepared, so these say how far along the speedup is and never suggest the model is busy.

import Foundation

extension DaemonStatusDocument.NeuralEnginePrefill {
  var isPreparing: Bool { state == "preparing" }
  var isEngaged: Bool { state == "active" }
  var isUnavailable: Bool { state == "unavailable" }

  private var programCountTitle: String? {
    guard let completedProgramCount, let totalProgramCount else { return nil }
    return "\(completedProgramCount)/\(totalProgramCount)"
  }

  /// Short enough to sit beside the menu bar icon.
  var menuBarTitle: String? {
    guard isPreparing, let programCountTitle else { return nil }
    return "Neural Engine \(programCountTitle)"
  }

  /// The popover's row; `nil` hides it for a user who never enabled the setting. No status arrives
  /// during a request, so an estimate would sit frozen on screen for as long as a reply takes.
  func metricTitle(showsEstimate: Bool) -> String? {
    if isEngaged { return "Engaged" }
    if isUnavailable { return "On the GPU" }
    guard isPreparing, let programCountTitle else { return nil }
    guard showsEstimate, let estimatedRemainingMilliseconds, estimatedRemainingMilliseconds > 0
    else {
      return "Preparing \(programCountTitle)"
    }
    // Rounded up, so the last moments never claim nothing is left.
    let remainingSeconds = (estimatedRemainingMilliseconds + 999) / 1_000
    return "Preparing \(programCountTitle) · about \(remainingSeconds) s"
  }
}
