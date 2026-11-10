import AppKit
import SwiftUI

/// The one colour that belongs to Astronomical: the violet of the app icon.
///
/// The panel's primary buttons, live-work capsules and progress fills use it
/// so the menu bar panel, Observatory and the chat window share one accent; a
/// system tint here made the panel read as a different product. The values are the same pair as
/// `--accent` in Observatory's console.css and the chat window's tokens.css.
enum AstronomicalBrandPalette {
  /// The icon's exact violet; white text on it holds 6.7:1.
  static let lightAppearanceAccent = NSColor(
    srgbRed: 106 / 255, green: 63 / 255, blue: 196 / 255, alpha: 1)
  /// The icon violet is too dark to read as a fill on a dark panel (2.7:1
  /// against graphite), so dark appearances use a lifted violet.
  static let darkAppearanceAccent = NSColor(
    srgbRed: 166 / 255, green: 140 / 255, blue: 255 / 255, alpha: 1)

  /// Resolves per appearance when drawn, so the panel follows the Mac between
  /// light and dark without the views reading the colour scheme themselves.
  static let accentColor = Color(
    nsColor: NSColor(name: nil, dynamicProvider: resolveAccent(for:)))

  static func resolveAccent(for appearance: NSAppearance) -> NSColor {
    let isDarkAppearance =
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
    return isDarkAppearance ? darkAppearanceAccent : lightAppearanceAccent
  }
}
