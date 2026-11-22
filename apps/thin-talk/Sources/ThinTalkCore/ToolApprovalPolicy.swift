import Foundation

/// Which host tools the model may ask for, which of them need the person's
/// approval first, and which the person has already allowed for a
/// conversation.
///
/// The allowlist is fixed here, in native code: the page names the tool it
/// wants, but only these names can ever reach a sheet or an open panel, so a
/// page that renders model output cannot invent a new capability. "Allow for
/// This Conversation" is remembered only while the app runs, so every launch
/// asks again.
public struct ToolApprovalPolicy: Sendable {
  /// The name the approval sheet and the file picker use for the app: the
  /// one the person installed and sees in the menu bar.
  public static let applicationName = "Astronomical"

  /// The tools whose open panel is the person's approval. Nothing is read
  /// until they pick a file there, and Cancel ends the use, so a sheet before
  /// the panel would ask the same question twice for work that never leaves
  /// this Mac.
  public static let toolNamesChosenInOpenPanel: Set<String> = ["read_file", "describe_image"]

  /// What each tool that asks first does, completing "Allow Astronomical to
  /// …?". Conversation search has no panel, so this sheet is the person's
  /// only say before other conversations are read.
  public static let actionDescriptionsByToolName: [String: String] = [
    "search_conversations": "search your saved conversations",
  ]

  /// What happens to what each asking tool reads, shown under the sheet's
  /// question. Results are saved with the conversation and sent with later
  /// questions, so the sheet says so rather than only "stays on this Mac".
  public static let consequencesByToolName: [String: String] = [
    "search_conversations": "Excerpts from your other conversations are sent to the model on this Mac and saved in this conversation.",
  ]

  /// Every tool the page may use: the open-panel tools and the asking ones.
  public static var allowedToolNames: Set<String> {
    toolNamesChosenInOpenPanel.union(actionDescriptionsByToolName.keys)
  }

  /// How a request for a tool should be answered before any sheet is shown.
  public enum Decision: Equatable, Sendable {
    /// Not an allowed tool, or no conversation open: refuse.
    case refuse
    /// The person approves by picking a file in the open panel; no sheet.
    case chooseInOpenPanel
    /// The user already allowed this tool for this conversation.
    case alreadyAllowed
    /// Ask the user, completing the sheet's question with this description.
    case ask(actionDescription: String)
  }

  private var allowedConversationTools = Set<String>()

  public init() {}

  public func decision(toolName: String, conversationID: String) -> Decision {
    guard !conversationID.isEmpty else { return .refuse }
    if Self.toolNamesChosenInOpenPanel.contains(toolName) {
      return .chooseInOpenPanel
    }
    guard let actionDescription = Self.actionDescriptionsByToolName[toolName] else {
      return .refuse
    }
    if allowedConversationTools.contains(Self.approvalKey(toolName: toolName, conversationID: conversationID)) {
      return .alreadyAllowed
    }
    return .ask(actionDescription: actionDescription)
  }

  /// Remembers "Allow for This Conversation" until the app quits.
  public mutating func allowForConversation(toolName: String, conversationID: String) {
    guard Self.actionDescriptionsByToolName[toolName] != nil, !conversationID.isEmpty else { return }
    allowedConversationTools.insert(Self.approvalKey(toolName: toolName, conversationID: conversationID))
  }

  private static func approvalKey(toolName: String, conversationID: String) -> String {
    "\(conversationID)\u{0}\(toolName)"
  }
}
// Test impact selection probe: a Swift-only change.
