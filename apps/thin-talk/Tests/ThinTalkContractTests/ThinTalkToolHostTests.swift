import AppKit
import Foundation
import ThinTalkCore
import XCTest

@testable import ThinTalkCanvas

/// The host-tool paths that must end without showing anything: a tool outside
/// the allowlist, a conversation window that is not on screen, and a use the
/// page cancelled before its turn. The sheet and panel themselves need a
/// person, so they are exercised live, not here.
@MainActor
final class ThinTalkToolHostTests: XCTestCase {
  func test_should_refuse_a_tool_outside_the_allowlist_without_asking() async {
    let toolHost = ThinTalkToolHost(windowProvider: { nil })
    let toolOutcome = await toolHost.use(toolName: "run_shell", conversationID: "conversation-1", callID: "call-1")
    XCTAssertEqual(toolOutcome, .unavailable(reason: "this tool is not available in this conversation"))
  }

  func test_should_not_ask_on_a_missing_or_hidden_window() async {
    let missingWindowHost = ThinTalkToolHost(windowProvider: { nil })
    let missingWindowOutcome = await missingWindowHost.use(toolName: "read_file", conversationID: "conversation-1", callID: "call-1")
    XCTAssertEqual(missingWindowOutcome, .unavailable(reason: "the conversation window is closed"))

    let hiddenWindow = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 100), styleMask: [.titled], backing: .buffered, defer: true)
    hiddenWindow.isReleasedWhenClosed = false
    let hiddenWindowHost = ThinTalkToolHost(windowProvider: { hiddenWindow })
    let hiddenWindowOutcome = await hiddenWindowHost.use(toolName: "search_conversations", conversationID: "conversation-1", callID: "call-2")
    XCTAssertEqual(hiddenWindowOutcome, .unavailable(reason: "the conversation window is closed"))
  }

  func test_should_skip_a_use_cancelled_before_its_turn() async {
    let toolHost = ThinTalkToolHost(windowProvider: { nil })
    toolHost.cancel(callID: "call-1")
    let toolOutcome = await toolHost.use(toolName: "read_file", conversationID: "conversation-1", callID: "call-1")
    XCTAssertEqual(toolOutcome, .cancelled)
    let laterOutcome = await toolHost.use(toolName: "read_file", conversationID: "conversation-1", callID: "call-1")
    XCTAssertEqual(laterOutcome, .unavailable(reason: "the conversation window is closed"), "a cancellation applies to one use only")
  }

  func test_should_send_each_outcome_to_the_page_in_its_bridge_shape() {
    let textFile = ChosenFileReader.TextFile(fileName: "a.txt", text: "hi")
    XCTAssertEqual(HostToolOutcome.chosenTextFile(textFile).bridgePayload, ["outcome": "chosen", "fileName": "a.txt", "text": "hi"])
    XCTAssertEqual(HostToolOutcome.refused.bridgePayload, ["outcome": "refused"])
    XCTAssertEqual(HostToolOutcome.unavailable(reason: "closed").bridgePayload, ["outcome": "unavailable", "reason": "closed"])
    XCTAssertEqual(HostToolOutcome.nothingChosen.bridgePayload, ["outcome": "nothingChosen"])
  }
}

/// The page names the host tool it wants; the native allowlist decides which
/// names can ever reach a sheet. A name present on one side only is a tool
/// that is always refused, so both sides are read and must match.
final class HostToolNameContractTests: XCTestCase {
  private let webSourceDirectory = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("web/src")

  private func declaredToolName(inWebSource fileName: String) throws -> String {
    let toolSource = try String(contentsOf: webSourceDirectory.appendingPathComponent(fileName), encoding: .utf8)
    let namePattern = try NSRegularExpression(pattern: #"public readonly name = "([a-z_]+)";"#)
    let sourceRange = NSRange(toolSource.startIndex..., in: toolSource)
    let nameMatch = try XCTUnwrap(namePattern.firstMatch(in: toolSource, range: sourceRange), fileName)
    return String(toolSource[try XCTUnwrap(Range(nameMatch.range(at: 1), in: toolSource))])
  }

  func test_should_allow_exactly_the_host_tools_the_page_declares() throws {
    let pageHostToolNames = try Set(
      ["ReadChosenFileTool.ts", "DescribeChosenImageTool.ts", "SearchConversationsTool.ts"].map(declaredToolName(inWebSource:)))
    XCTAssertEqual(pageHostToolNames, Set(ToolApprovalPolicy.actionDescriptionsByToolName.keys))
    XCTAssertEqual(Set(ToolApprovalPolicy.consequencesByToolName.keys), pageHostToolNames, "every tool explains what happens to what it reads")
  }
}
