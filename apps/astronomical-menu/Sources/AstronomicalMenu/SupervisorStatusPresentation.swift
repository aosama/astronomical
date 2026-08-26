// Owns user-facing activity and progress interpretation for the status menu. Keeping presentation
// outside the wire document makes new server states reviewable without enlarging its decoder.

import Foundation

extension SupervisorStatusDocument {
  var isActive: Bool {
    activity == "prompt_processing" || activity == "generation_preparation"
      || activity == "generating" || activity == "image_generation"
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
    guard activity == "prompt_processing", let progress else { return "" }
    let completionPercentageTitle = progress.completionPercentageTitle
    if progress.phase == "drafter" { return "Drafting…" }
    guard let currentPhaseTokensPerSecond else { return "Prompt \(completionPercentageTitle)" }
    return "Prompt \(completionPercentageTitle) · \(Int(currentPhaseTokensPerSecond.rounded())) avg tok/s"
  }

  var flightTitle: String {
    guard isActive else { return "Standing by" }
    if activity == "generating" {
      return currentPhaseTokensPerSecond.map { String(format: "Generating · %.1f tok/s", $0) }
        ?? "Generating"
    }
    if activity == "image_generation", let progress {
      if progress.hasCompletedDenoising { return "Denoising complete" }
      return progress.phase == "denoising"
        ? "\(progress.imagePhaseTitle) · \(progress.completionPercentageTitle)"
        : progress.imagePhaseTitle
    }
    if activity == "generation_preparation" { return "Preparing generation…" }
    guard let progress else { return phaseTitle }
    if progress.phase == "drafter" { return "Drafting…" }
    guard let currentPhaseTokensPerSecond else {
      return "Prompt processing · \(progress.completionPercentageTitle)"
    }
    return "Prompt processing · \(progress.completionPercentageTitle) · \(Int(currentPhaseTokensPerSecond.rounded())) avg tok/s"
  }

  var phaseTitle: String {
    switch activity {
    case "generating": "Generating"
    case "generation_preparation": "Preparing generation"
    case "prompt_processing": progress?.phase == "drafter" ? "Drafting…" : "Prompt processing"
    case "image_generation": progress?.imagePhaseTitle ?? "Generating image"
    default:
      switch status {
      case "ready": "Ready"
      case "loading": "Loading"
      default: "Unavailable"
      }
    }
  }

  var modelFootprintTitle: String {
    if expertMemoryMode == "resident" {
      return "Fully in memory"
    }
    if readyModelIdentifier != nil && expertMemoryMode == nil { return "Fully in memory" }
    return expertMemoryMode == "paged" || expertMemoryMode == "hybrid"
      ? "RAM + SSD streaming" : "Not loaded"
  }

  var modelDiskSizeTitle: String {
    readyModelSizeBytes.map(decimalGigabyteText) ?? "Not measured"
  }

  var mtpRuntimeStateTitle: String {
    guard readyModelIdentifier != nil else { return "Not loaded" }
    switch mtpRuntimeState {
    case "active": return "Active"
    case "target_only": return "Standard generation"
    case "unavailable": return "Unavailable"
    default: return "Disabled"
    }
  }

  var progressTitle: String {
    guard let progress else { return "Standing by" }
    if progress.unit == .steps {
      guard progress.phase == "denoising" else { return progress.imagePhaseTitle }
      if progress.hasCompletedDenoising { return "Denoising complete" }
      return "\(progress.completionPercentageTitle) · \(progress.completedUnitCount) / \(progress.totalUnitCount) steps"
    }
    if progress.phase == "generation_preparation" { return "Preparing the first output" }
    if progress.phase == "drafter" { return "Drafting…" }
    let tokenCountTitle = "\(progress.completedUnitCount) / \(progress.totalUnitCount) tokens"
    return progress.phase == "generation"
      ? tokenCountTitle
      : "\(progress.completionPercentageTitle) · \(tokenCountTitle)"
  }

  var elapsedTimeMetricTitle: String {
    progress?.unit == .steps || progress?.phase == "generation"
      || progress?.phase == "generation_preparation"
      ? "Elapsed" : "Elapsed / ETA"
  }

  var elapsedTimeTitle: String {
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

  var hasDeterminateProgress: Bool {
    guard let progress else { return true }
    return progress.unit != .steps || progress.phase == "denoising"
  }

  var progressCompletedUnitCount: UInt32 { progress?.completedUnitCount ?? 0 }
  var progressTotalUnitCount: UInt32 { max(1, progress?.totalUnitCount ?? 1) }
}
