// Decides which state the menu bar glyph shows. The glyph uses the product's
// state texture (solid = in memory, hatched = streaming from the SSD) so the
// menu bar answers "where does the model live right now" without opening the
// popover. MenuBarGlyphDrawing.swift draws it.

import Foundation

enum MenuBarGlyphState: Equatable {
  /// No model is ready: an empty plate.
  case idle
  /// The ready model's experts are held in memory.
  case resident
  /// The ready model pages experts from the SSD.
  case streaming
  /// A request is live; the reticle fills, so the state never depends on colour.
  case generating
  /// A model is loading: the disc is hatched like streaming, but the filled reticle
  /// marks work in progress.
  case loadingModel
  /// The engine has stopped answering: the plate holds an exclamation mark.
  case engineNotRunning

  var accessibilityTitle: String {
    switch self {
    case .idle: return "idle"
    case .resident: return "model in memory"
    case .streaming: return "streaming experts from SSD"
    case .generating: return "generating"
    case .loadingModel: return "loading a model"
    case .engineNotRunning: return "engine not running"
    }
  }
}

/// A pie six points across cannot show finer steps than these, and redrawing
/// only when a step is crossed keeps the status item from re-laying out on
/// every poll.
let menuBarGlyphProgressStepCount = 16

struct MenuBarGlyph: Equatable {
  let state: MenuBarGlyphState
  /// How much of a measured piece of work is finished, in steps out of
  /// `menuBarGlyphProgressStepCount`; `nil` when the work has no measured end.
  let completedProgressStepCount: Int?

  init(state: MenuBarGlyphState, completedProgressStepCount: Int? = nil) {
    self.state = state
    self.completedProgressStepCount = completedProgressStepCount
  }

  var accessibilityTitle: String {
    guard let completedProgressStepCount else { return state.accessibilityTitle }
    let completedPercentage = completedProgressStepCount * 100 / menuBarGlyphProgressStepCount
    return "\(state.accessibilityTitle), \(completedPercentage) percent"
  }
}

extension DaemonStatusDocument {
  var menuBarGlyphState: MenuBarGlyphState {
    // Checked first because the status can read "ready" throughout a model swap.
    if modelLoadInFlight != nil { return .loadingModel }
    guard status == "ready", readyModelIdentifier != nil else { return .idle }
    if isActive { return .generating }
    // A dense model that keeps only some decoder layers reads the rest from
    // the SSD for every token, whatever its expert mode reads.
    if expertResidency?.denseLayers?.streamsFromTheSSD == true { return .streaming }
    // "paged" and "hybrid" both stream some experts from the SSD; a dense
    // model with every layer in memory reports no mode.
    switch expertMemoryMode {
    case "paged", "hybrid": return .streaming
    default: return .resident
    }
  }

  /// Progress the glyph can draw: reading a prompt and denoising an image
  /// both have a measured end. Writing a reply does not (its total is only a
  /// budget), and a model load reports no progress at all.
  var menuBarGlyphCompletedProgressStepCount: Int? {
    guard menuBarGlyphState == .generating, let progress, showsProgressBar,
      progress.phase != "generation", progress.phase != "generation_preparation"
    else { return nil }
    return Int((progressBarFillFraction * Double(menuBarGlyphProgressStepCount)).rounded(.down))
  }

  func menuBarGlyph(engineIsDown: Bool) -> MenuBarGlyph {
    if engineIsDown { return MenuBarGlyph(state: .engineNotRunning) }
    return MenuBarGlyph(
      state: menuBarGlyphState,
      completedProgressStepCount: menuBarGlyphCompletedProgressStepCount)
  }
}
