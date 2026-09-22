import Foundation

/// Manages the composer's font size with keyboard shortcuts.
///
/// The size persists across launches so a deliberate choice isn't lost,
/// while a fresh install starts on the theme default.
public enum ComposerFontSize {
  /// Default size in points, matching the theme's composerInput.
  public static let defaultSize: CGFloat = 14

  /// Minimum and maximum sizes in points.
  public static let minSize: CGFloat = 10
  public static let maxSize: CGFloat = 24

  /// Step size in points for each zoom action.
  public static let step: CGFloat = 1.0

  private static let storageKey = "thin-talk.composer-font-size"

  public static func load(from defaults: UserDefaults = .standard) -> CGFloat {
    guard let stored = defaults.object(forKey: storageKey) as? CGFloat else {
      return defaultSize
    }
    return max(minSize, min(maxSize, stored))
  }

  public static func save(_ size: CGFloat, to defaults: UserDefaults = .standard) {
    let clamped = max(minSize, min(maxSize, size))
    defaults.set(clamped, forKey: storageKey)
  }
}
