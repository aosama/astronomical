import Foundation

/// How much thinking the model may spend before it must answer.
///
/// The levels are the user-facing promise and the token counts are what the
/// supervisor enforces, so the number shown next to a level is the budget the
/// server applies to that turn. Quick is the default because a first answer
/// should feel immediate; a reader who wants depth asks for it explicitly.
public enum ThinkingEffort: String, CaseIterable, Identifiable, Sendable {
  case quick
  case balanced
  case high

  /// The effort a fresh install starts on.
  public static let `default` = ThinkingEffort.quick

  public var id: String { rawValue }

  /// The thinking-token budget the supervisor enforces for this level.
  public var thinkingBudgetTokens: Int {
    switch self {
    case .quick: 256
    case .balanced: 512
    case .high: 1024
    }
  }

  public var displayName: String {
    switch self {
    case .quick: "Quick"
    case .balanced: "Balanced"
    case .high: "High"
    }
  }

  /// One line for the control itself, so the cost of depth is never hidden.
  public var budgetSummary: String {
    "\(displayName) · \(thinkingBudgetTokens) tokens"
  }
}

/// Remembers the last effort a reader chose.
///
/// Session-only state would silently reset a deliberate choice on every launch,
/// while the first run must still start on Quick — so the stored value wins only
/// once the reader has actually picked one.
public enum ThinkingEffortPreference {
  /// Internal rather than private because a stored value outlives the release that
  /// wrote it: a test has to be able to place a legacy or unrecognised string under
  /// this exact key and observe that the reader still gets a usable level.
  static let storageKey = "thin-talk.thinking-effort"

  public static func load(from defaults: UserDefaults = .standard) -> ThinkingEffort {
    guard let stored = defaults.string(forKey: storageKey) else {
      return ThinkingEffort.default
    }
    return ThinkingEffort(rawValue: stored) ?? ThinkingEffort.default
  }

  public static func save(_ effort: ThinkingEffort, to defaults: UserDefaults = .standard) {
    defaults.set(effort.rawValue, forKey: storageKey)
  }
}