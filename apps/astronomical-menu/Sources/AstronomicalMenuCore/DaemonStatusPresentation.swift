// Owns user-facing activity and progress interpretation for the status menu. Keeping presentation
// outside the wire document makes new server states reviewable without enlarging its decoder.

import Foundation

extension DaemonStatusDocument {
  // The worker drops the previous model before loading the next while the daemon still names it,
  // so every fact about "the model" below defers to the load rather than describe one that is gone.
  var isLoadingModel: Bool { modelLoadInFlight != nil }

  var isActive: Bool {
    activity == "prompt_processing" || activity == "generation_preparation"
      || activity == "generating" || activity == "image_generation"
  }

  // Prompt-processing activity is published as soon as the request starts, but
  // the first PrefillProgress event arrives only after prompt-cache restore
  // returns. Missing progress in that window is restore work, not idle and not
  // expert SSD streaming.
  private var isRestoringPromptCache: Bool {
    activity == "prompt_processing" && progress == nil
  }

  // The status endpoint supplies request-elapsed time during prompt processing and phase-elapsed
  // time during generation, so only token progress can produce a meaningful token rate.
  var currentPhaseTokensPerSecond: Double? {
    guard let progress, progress.unit == .tokens, progress.completedUnitCount > 0,
      progress.elapsedMilliseconds > 0
    else {
      return nil
    }
    return Double(progress.completedUnitCount) / (Double(progress.elapsedMilliseconds) / 1_000)
  }

  var menuBarTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.menuBarTitle }
    guard status == "ready" else { return status == "loading" ? " Loading" : "" }
    if activity == "generating", let currentPhaseTokensPerSecond {
      return String(format: "GEN %.1f tok/s", currentPhaseTokensPerSecond)
    }
    if activity == "image_generation", let progress {
      if progress.hasCompletedDenoising { return "Image · Denoising" }
      return progress.phase == "denoising" && !progress.hasCompletedDenoising
        ? "Image \(progress.completionPercentageTitle)"
        : "Image · \(progress.shortImagePhaseTitle)"
    }
    if activity == "generation_preparation" { return "Preparing…" }
    if isRestoringPromptCache { return "Restoring…" }
    guard activity == "prompt_processing", let progress else {
      // Only when idle: a generation's own title always says more than a background count.
      return activity == "idle" ? neuralEnginePrefill?.menuBarTitle ?? "" : ""
    }
    let completionPercentageTitle = progress.completionPercentageTitle
    guard let currentPhaseTokensPerSecond else { return "Prompt \(completionPercentageTitle)" }
    return "Prompt \(completionPercentageTitle) · \(Int(currentPhaseTokensPerSecond.rounded())) avg tok/s"
  }

  var flightTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.flightTitle }
    guard isActive else { return "Standing by" }
    if activity == "generating" {
      if let currentPhaseTokensPerSecond {
        return String(format: "Generating · %.1f tok/s", currentPhaseTokensPerSecond)
      }
      return "Generating"
    }
    if activity == "image_generation", let progress {
      if progress.hasCompletedDenoising { return "Denoising complete" }
      return progress.phase == "denoising"
        ? "\(progress.imagePhaseTitle) · \(progress.completionPercentageTitle)"
        : progress.imagePhaseTitle
    }
    if activity == "generation_preparation" { return "Preparing generation…" }
    if isRestoringPromptCache { return "Restoring prompt cache" }
    guard let progress else { return phaseTitle }
    guard let currentPhaseTokensPerSecond else {
      return "Prompt processing · \(progress.completionPercentageTitle)"
    }
    return "Prompt processing · \(progress.completionPercentageTitle) · \(Int(currentPhaseTokensPerSecond.rounded())) avg tok/s"
  }

  var phaseTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.phaseTitle }
    switch activity {
    case "generating": return "Generating"
    case "generation_preparation": return "Preparing generation"
    case "prompt_processing" where isRestoringPromptCache: return "Restoring prompt cache"
    case "prompt_processing": return "Prompt processing"
    case "image_generation":
      if let progress {
        return progress.imagePhaseTitle
      }
      return "Generating image"
    default:
      switch status {
      case "ready": return "Ready"
      case "loading": return "Loading"
      default: return "Unavailable"
      }
    }
  }

  var modelFootprintTitle: String {
    if isLoadingModel { return "Being prepared" }
    // The daemon owns this label so the menu and Observatory cannot drift.
    // An older daemon omits expert_placement. A loaded model then stays fully
    // in memory, and an unloaded one says no model is loaded.
    if let stockSentence = expertPlacement?.stockSentence, !stockSentence.isEmpty {
      return stockSentence
    }
    if readyModelIdentifier == nil {
      return "No model loaded"
    }
    return "All in RAM"
  }

  var modelDiskSizeTitle: String {
    if isLoadingModel { return "Being prepared" }
    if let readyModelSizeBytes {
      return gigabyteText(byteCount: readyModelSizeBytes)
    }
    return "Not measured"
  }

  var progressTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.progressTitle }
    guard let progress else {
      return isRestoringPromptCache ? "Restoring prompt cache" : "Standing by"
    }
    if progress.unit == .steps {
      guard progress.phase == "denoising" else { return progress.imagePhaseTitle }
      if progress.hasCompletedDenoising { return "Denoising complete" }
      return "\(progress.completionPercentageTitle) · \(progress.completedUnitCount) / \(progress.totalUnitCount) steps"
    }
    if progress.phase == "generation_preparation" { return "Preparing the first output" }
    if progress.phase == "generation" {
      let tokenCountTitle = "\(progress.completedUnitCount) / \(progress.totalUnitCount) tokens"
      // A request without an explicit output cap reports the endpoint's
      // default budget (65,535), which makes "x / y tokens" read as a
      // permanently near-zero bar; large budgets present as a plain count
      // and no bar.
      if progress.totalUnitCount >= 8_192 {
        return "\(progress.completedUnitCount) tokens"
      }
      return tokenCountTitle
    }
    let tokenCountTitle = "\(progress.completedUnitCount) / \(progress.totalUnitCount) tokens"
    return "\(progress.completionPercentageTitle) · \(tokenCountTitle)"
  }

  var elapsedTimeMetricTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.elapsedTimeMetricTitle }
    return progress?.unit == .steps || progress?.phase == "generation"
      || progress?.phase == "generation_preparation"
      ? "Elapsed" : "Elapsed / ETA"
  }

  var elapsedTimeTitle: String {
    if let modelLoadInFlight { return modelLoadInFlight.elapsedTimeTitle }
    if isRestoringPromptCache { return "Calculating" }
    guard let progress else { return "Not active" }
    let elapsedSeconds = Double(progress.elapsedMilliseconds) / 1_000
    guard progress.unit != .steps && progress.phase != "generation"
      && progress.phase != "generation_preparation"
    else {
      return String(format: "%.1f s", elapsedSeconds)
    }
    guard progress.completedUnitCount > 0 else {
      return String(format: "%.1f s / Calculating", elapsedSeconds)
    }
    let remainingTokenCount =
      progress.totalUnitCount > progress.completedUnitCount
      ? progress.totalUnitCount - progress.completedUnitCount : 0
    let estimatedRemainingSeconds =
      elapsedSeconds * Double(remainingTokenCount) / Double(progress.completedUnitCount)
    return String(format: "%.1f s / %.1f s", elapsedSeconds, estimatedRemainingSeconds)
  }

  /// Whether the popover draws its progress bar. It does when there is a
  /// completion fraction, and while idle, where an empty bar keeps the section
  /// readable. It does not during a model load, a prompt-cache restore, a
  /// generation against the default budget, or an image phase other than
  /// denoising: the Flight and Progress rows describe those instead.
  var showsProgressBar: Bool {
    if isLoadingModel { return false }
    if isRestoringPromptCache { return false }
    guard let progress else { return true }
    // A generation against the endpoint's default budget has no meaningful
    // completion fraction; the popover shows no bar instead of ~0%.
    if progress.phase == "generation" && progress.totalUnitCount >= 8_192 {
      return false
    }
    return progress.unit != .steps || progress.phase == "denoising"
  }

  var progressCompletedUnitCount: UInt32 {
    guard let progress else { return 0 }
    return progress.completedUnitCount
  }
  var progressTotalUnitCount: UInt32 {
    guard let progress else { return 1 }
    return max(1, progress.totalUnitCount)
  }
}
