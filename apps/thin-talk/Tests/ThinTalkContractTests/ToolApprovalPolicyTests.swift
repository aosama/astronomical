import ThinTalkCore
import XCTest

/// The approval policy decides which host tools can ever reach a sheet and
/// remembers "Allow for This Conversation". The page names the tool, so the
/// allowlist is the boundary these tests pin.
final class ToolApprovalPolicyTests: XCTestCase {
  func test_should_ask_before_searching_conversations() {
    let policy = ToolApprovalPolicy()
    XCTAssertEqual(
      policy.decision(toolName: "search_conversations", conversationID: "conversation-1"),
      .ask(actionDescription: "search your saved conversations"))
  }

  /// Picking a file in the open panel is the approval, so the file tools
  /// never show a sheet first.
  func test_should_send_the_file_tools_straight_to_the_open_panel() {
    let policy = ToolApprovalPolicy()
    for toolName in ["read_file", "describe_image"] {
      XCTAssertEqual(policy.decision(toolName: toolName, conversationID: "conversation-1"), .chooseInOpenPanel, toolName)
      XCTAssertNil(ToolApprovalPolicy.actionDescriptionsByToolName[toolName], toolName)
    }
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
    XCTAssertEqual(policy.decision(toolName: "describe_image", conversationID: ""), .refuse)
  }

  func test_should_remember_allow_for_this_conversation_only_for_that_tool_and_conversation() {
    var policy = ToolApprovalPolicy()
    policy.allowForConversation(toolName: "search_conversations", conversationID: "conversation-1")
    XCTAssertEqual(policy.decision(toolName: "search_conversations", conversationID: "conversation-1"), .alreadyAllowed)
    XCTAssertNotEqual(policy.decision(toolName: "search_conversations", conversationID: "conversation-2"), .alreadyAllowed)
  }

  /// An open panel tool has nothing to remember: allowing it leaves it going
  /// straight to the panel rather than becoming "already allowed".
  func test_should_keep_an_open_panel_tool_going_to_the_panel_after_allow_for_this_conversation() {
    var policy = ToolApprovalPolicy()
    policy.allowForConversation(toolName: "describe_image", conversationID: "conversation-1")
    XCTAssertEqual(policy.decision(toolName: "describe_image", conversationID: "conversation-1"), .chooseInOpenPanel)
  }

  func test_should_not_remember_a_tool_outside_the_allowlist() {
    var policy = ToolApprovalPolicy()
    policy.allowForConversation(toolName: "run_shell", conversationID: "conversation-1")
    XCTAssertEqual(policy.decision(toolName: "run_shell", conversationID: "conversation-1"), .refuse)
  }
}
