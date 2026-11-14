import Foundation

/// Which host tools the model may ask for, and which of them the user has
/// already allowed for a conversation.
///
/// The allowlist is fixed here, in native code: the page names the tool it
/// wants, but only these names can ever reach an approval sheet, so a page that
/// renders model output cannot invent a new capability. "Allow for This
/// Conversation" is remembered only while the app runs, so every launch asks
/// again.
public struct ToolApprovalPolicy: Sendable {
  /// The name the approval sheet and the file picker use for the app: the
  /// one the person installed and sees in the menu bar.
  public static let applicationName = "Astronomical"

  /// What each allowed tool does, completing "Allow Astronomical to …?".
  public static let actionDescriptionsByToolName: [String: String] = [
    "read_file": "read a file you choose",
    "describe_image": "look at an image you choose",
    "search_conversations": "search your saved conversations",
  ]

  /// What happens to what each tool reads, shown under the sheet's question.
  /// Results are saved with the conversation and sent with later questions,
  /// so the sheet says so rather than only "stays on this Mac".
  public static let consequencesByToolName: [String: String] = [
    "read_file": "The file's text is sent to the model on this Mac and saved in this conversation.",
    "describe_image": "The image is sent to a model on this Mac, and its description is saved in this conversation.",
    "search_conversations": "Excerpts from your other conversations are sent to the model on this Mac and saved in this conversation.",
  ]

  /// How a request for a tool should be answered before any sheet is shown.
  public enum Decision: Equatable, Sendable {
    /// Not an allowed tool, or no conversation to remember a choice for: refuse.
    case refuse
    /// The user already allowed this tool for this conversation.
    case alreadyAllowed
    /// Ask the user, completing the sheet's question with this description.
    case ask(actionDescription: String)
  }

  private var allowedConversationTools = Set<String>()

  public init() {}

  public func decision(toolName: String, conversationID: String) -> Decision {
    guard let actionDescription = Self.actionDescriptionsByToolName[toolName], !conversationID.isEmpty else {
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
