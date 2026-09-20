import SwiftUI

/// Design-token palette — matched to Copilot's bold, spacious dark style.
public enum PreviewTheme {
  // Colors
  public static let windowBackground = Color(red: 0.06, green: 0.07, blue: 0.10)
  public static let panel = Color(red: 0.10, green: 0.12, blue: 0.16)
  public static let panelAlt = Color(red: 0.14, green: 0.16, blue: 0.20)
  public static let accent = Color(red: 0.35, green: 0.40, blue: 0.95)
  public static let border = Color(red: 0.22, green: 0.25, blue: 0.32)
  public static let textPrimary = Color(red: 0.95, green: 0.95, blue: 0.96)
  public static let textSecondary = Color(red: 0.72, green: 0.74, blue: 0.80)
  public static let danger = Color(red: 0.95, green: 0.45, blue: 0.45)
  public static let linkBlue = Color(red: 0.45, green: 0.58, blue: 1.0)

  // Typography — desktop-chat scale (Copilot's actual body text is ~14–16px,
  // not the oversized 22–28pt a previous pass assumed)
  public static let h1 = Font.system(size: 22, weight: .bold, design: .default)
  public static let h2 = Font.system(size: 17, weight: .semibold, design: .default)
  public static let body = Font.system(size: 14, weight: .regular, design: .default)
  public static let code = Font.system(size: 13, weight: .regular, design: .monospaced)
  public static let reasoning = Font.system(size: 13, design: .default).italic()
  public static let sidebarLabel = Font.system(size: 14, weight: .medium, design: .default)
  public static let sidebarHeading = Font.system(size: 15, weight: .semibold, design: .default)
  public static let sidebarConversation = Font.system(size: 13.5, weight: .medium, design: .default)
  public static let composerInput = Font.system(size: 14, weight: .regular, design: .default)

  // Spacing
  public static let messagePadding = CGFloat(16)
  public static let sidebarNavHeight = CGFloat(36)
}
