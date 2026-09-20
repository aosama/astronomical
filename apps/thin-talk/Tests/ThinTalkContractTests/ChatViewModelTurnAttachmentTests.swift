import Foundation
import ThinTalkCore
@testable import ThinTalkUI
import XCTest

/// Contract for the real bug surfaced in the live UX: when the user sends a second
/// turn, the leaked `assistantMessageID` from turn 1 caused turn 2's stream to keep
/// appending inside turn 1's bubble. After the fix each turn gets its own assistant
/// message and the transcript stays in chronological order.
@MainActor
final class ChatViewModelTurnAttachmentTests: XCTestCase {

  private var client: ThinTalkClient!

  override func setUp() async throws {
    try await super.setUp()
    StubSupervisorURLProtocol.statusResponse = .init(
      statusCode: 200,
      responseBody: Data(#"{"application":{"version":"0.1.0","build_number":1,"commit":"test","is_dirty":false,"channel":"development","state_directory":"~/.astronomical-dev"},"status":"ready"}"#.utf8)
    )
    StubSupervisorURLProtocol.modelsResponse = .init(
      statusCode: 200,
      responseBody: Data(#"{"data":[{"id":"test-model","input_modalities":["text"],"supported_endpoints":["chat"]}]}"#.utf8)
    )
    // Every turn returns the same single-delta SSE body; the test verifies that
    // the two responses do not merge into one bubble.
    let sse = """
    data: {"choices":[{"delta":{"content":"first answer"},"finish_reason":"stop"}]}

    data: [DONE]

    """
    StubSupervisorURLProtocol.chatResponse = .init(
      statusCode: 200,
      responseBody: Data(sse.utf8)
    )
    client = ThinTalkClient(
      applicationIdentity: ThinTalkApplicationIdentity(channel: .development, supervisorPort: 6733),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration()),
      stallTimeout: 60
    )
  }

  override func tearDown() async throws {
    StubSupervisorURLProtocol.statusResponse = nil
    StubSupervisorURLProtocol.modelsResponse = nil
    StubSupervisorURLProtocol.chatResponse = nil
    client = nil
    try await super.tearDown()
  }

  func test_two_turns_remain_as_separate_assistant_messages() async throws {
    let vm = ChatViewModel(client: client)
    await vm.load()
    XCTAssertEqual(vm.state, .ready)

    // Turn 1
    vm.draft = "hello"
    vm.sendMessage()
    try await waitForStreamingToFinish(vm)

    XCTAssertEqual(vm.messages.count, 2, "Expected 2 messages after turn 1, got \(vm.messages.count)")
    XCTAssertEqual(vm.messages[0].role, .user)
    XCTAssertEqual(vm.messages[1].role, .assistant)
    XCTAssertEqual(vm.messages[1].content, "first answer")

    // Turn 2
    vm.draft = "world"
    vm.sendMessage()
    try await waitForStreamingToFinish(vm)

    XCTAssertEqual(vm.messages.count, 4, "Expected 4 messages after turn 2, got \(vm.messages.count)")
    XCTAssertEqual(vm.messages[0].role, .user)
    XCTAssertEqual(vm.messages[1].role, .assistant)
    XCTAssertEqual(vm.messages[2].role, .user)
    XCTAssertEqual(vm.messages[3].role, .assistant)

    // The critical assertion: turn 1's answer must not have been appended to.
    XCTAssertEqual(
      vm.messages[1].content, "first answer",
      "First assistant turn must not absorb turn 2's stream (bug: assistantMessageID leaked across turns)"
    )
    XCTAssertEqual(vm.messages[3].content, "first answer")
  }

  @MainActor
  private func waitForStreamingToFinish(_ vm: ChatViewModel, timeout: TimeInterval = 10) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while vm.isStreaming {
      if Date() > deadline {
        XCTFail("Timed out waiting for streaming to finish")
        return
      }
      try await Task.sleep(nanoseconds: 50_000_000)
    }
  }
}