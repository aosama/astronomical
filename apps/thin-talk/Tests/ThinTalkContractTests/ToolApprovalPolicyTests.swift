import ThinTalkCore
import XCTest

/// The approval policy decides which host tools can ever reach a sheet and
/// remembers "Allow for This Conversation". The page names the tool, so the
/// allowlist is the boundary these tests pin.
final class ToolApprovalPolicyTests: XCTestCase {
  func test_should_ask_for_each_allowed_tool_with_its_description() {
    let policy = ToolApprovalPolicy()
    XCTAssertEqual(policy.decision(toolName: "read_file", conversationID: "conversation-1"), .ask(actionDescription: "read a file you choose"))
    XCTAssertEqual(
      policy.decision(toolName: "search_conversations", conversationID: "conversation-1"),
      .ask(actionDescription: "search your saved conversations"))
  }

  /// The sheets ask on behalf of the app the person installed. An internal
  /// name there reads as some other program asking for their files.
  func test_should_name_the_installed_app_in_what_the_sheets_say() {
    XCTAssertEqual(ToolApprovalPolicy.applicationName, "Astronomical")
    XCTAssertFalse(ToolApprovalPolicy.applicationName.contains("Thin Talk"))
    XCTAssertFalse(ToolApprovalPolicy.applicationName.isEmpty)
  }

  func test_should_refuse_a_tool_outside_the_allowlist_or_without_a_conversation() {
    let policy = ToolApprovalPolicy()
    for toolName in ["run_shell", "", "READ_FILE", "read_file "] {
      XCTAssertEqual(policy.decision(toolName: toolName, conversationID: "conversation-1"), .refuse, toolName)
    }
    XCTAssertEqual(policy.decision(toolName: "read_file", conversationID: ""), .refuse)
  }

  func test_should_remember_allow_for_this_conversation_only_for_that_tool_and_conversation() {
    var policy = ToolApprovalPolicy()
    policy.allowForConversation(toolName: "read_file", conversationID: "conversation-1")
    XCTAssertEqual(policy.decision(toolName: "read_file", conversationID: "conversation-1"), .alreadyAllowed)
    XCTAssertNotEqual(policy.decision(toolName: "read_file", conversationID: "conversation-2"), .alreadyAllowed)
    XCTAssertNotEqual(policy.decision(toolName: "describe_image", conversationID: "conversation-1"), .alreadyAllowed)
  }

  func test_should_not_remember_a_tool_outside_the_allowlist() {
    var policy = ToolApprovalPolicy()
    policy.allowForConversation(toolName: "run_shell", conversationID: "conversation-1")
    XCTAssertEqual(policy.decision(toolName: "run_shell", conversationID: "conversation-1"), .refuse)
  }
}
