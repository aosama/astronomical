import Foundation

/// What the user picked from a sidebar conversation's right-click menu, as the
/// page receives it.
public enum ConversationMenuChoice: String, Sendable {
  case rename
  /// Chosen and confirmed on the delete sheet.
  case delete
  /// The menu closed without a choice, or the delete sheet was cancelled.
  case dismissed
}
