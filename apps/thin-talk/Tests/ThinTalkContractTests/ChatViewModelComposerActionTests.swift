import Foundation
import ThinTalkCanvas
import ThinTalkCore
@testable import ThinTalkUI
import XCTest

/// Contracts for the composer actions the canvas page reports. The page owns the
/// ask field, the effort pill, and the failure banner's Retry; the view model
/// must turn each reported action into exactly the behaviour the native controls
/// used to perform, so switching the surface to HTML cannot change what a turn
/// does.
@MainActor
final class ChatViewModelComposerActionTests: XCTestCase {

  private var client: ThinTalkClient!

  override func setUp() async throws {
    try await super.setUp()
    StubSupervisorURLProtocol.statusResponse = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.1.0","build_number":1,"commit":"test","is_dirty":false,"channel":"development","state_directory":"~/.astronomical-dev"},"status":"ready"}"#
          .utf8)
    )
    StubSupervisorURLProtocol.modelsResponse = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"data":[{"id":"test-model","input_modalities":["text"],"supported_endpoints":["chat"]}]}"#
          .utf8)
    )
    let sse = """
    data: {"choices":[{"delta":{"content":"first answer"},"finish_reason":"stop"}]}

    data: [DONE]

    """
    StubSupervisorURLProtocol.chatResponse = .init(
      statusCode: 200,
      responseBody: Data(sse.utf8)
    )
    client = ThinTalkClient(
      applicationIdentity: ThinTalkApplicationIdentity(
        channel: .development, supervisorPort: 6733),
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

  func test_a_reported_send_streams_the_reported_text_and_leaves_no_draft() async throws {
    let viewModel = ChatViewModel(client: client)
    await viewModel.load()

    viewModel.handle(CanvasAction(kind: .send, detail: "what is expert paging?"))
    XCTAssertEqual(viewModel.messages.first?.role, .user)
    XCTAssertEqual(viewModel.messages.first?.content, "what is expert paging?")
    try await waitForStreamingToFinish(viewModel)

    XCTAssertEqual(viewModel.messages.last?.content, "first answer")
    XCTAssertTrue(
      viewModel.draft.isEmpty,
      "a reported send must leave no draft behind")
  }

  func test_a_reported_stop_ends_the_stream_without_duplicating_turns() async throws {
    let viewModel = ChatViewModel(client: client)
    await viewModel.load()
    viewModel.handle(CanvasAction(kind: .send, detail: "hello"))
    try await waitForStreamingToFinish(viewModel)

    viewModel.handle(CanvasAction(kind: .stop))

    XCTAssertFalse(viewModel.isStreaming)
    XCTAssertEqual(viewModel.messages.count, 2, "stopping must not add or remove turns")
  }

  func test_a_reported_effort_choice_switches_the_level() async throws {
    let viewModel = ChatViewModel(client: client)
    let previous = ThinkingEffortPreference.load()
    defer { ThinkingEffortPreference.save(previous) }

    viewModel.handle(CanvasAction(kind: .setEffort, detail: "high"))
    XCTAssertEqual(viewModel.thinkingEffort, .high)

    viewModel.handle(CanvasAction(kind: .setEffort, detail: "turbo"))
    XCTAssertEqual(
      viewModel.thinkingEffort, .high,
      "an unrecognised level must be ignored, not crash or reset the choice")
  }

  func test_a_reported_retry_restreams_the_last_ask_without_duplicating_it() async throws {
    let viewModel = ChatViewModel(client: client)
    await viewModel.load()
    viewModel.handle(CanvasAction(kind: .send, detail: "hello"))
    try await waitForStreamingToFinish(viewModel)

    viewModel.handle(CanvasAction(kind: .retry))
    try await waitForStreamingToFinish(viewModel)

    XCTAssertEqual(
      viewModel.messages.count, 2,
      "retrying must not add a second copy of the user's ask")
    XCTAssertEqual(viewModel.messages.last?.content, "first answer")
  }

  func test_composer_state_mirrors_the_surface_for_the_page() async throws {
    let viewModel = ChatViewModel(client: client)
    XCTAssertFalse(
      viewModel.composerState.isReady,
      "before the handshake the page must not offer send")

    await viewModel.load()
    let ready = viewModel.composerState
    XCTAssertTrue(ready.isReady)
    XCTAssertTrue(ready.acceptsInput)
    XCTAssertFalse(ready.isStreaming)
    XCTAssertNil(ready.failure)
    XCTAssertEqual(ready.effort.value, ThinkingEffort.default.rawValue)
    XCTAssertEqual(
      ready.effortOptions.map(\.value), ThinkingEffort.allCases.map(\.rawValue),
      "the page must offer exactly the levels the core defines")
  }

  private func waitForStreamingToFinish(
    _ viewModel: ChatViewModel, timeout: TimeInterval = 10
  ) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while viewModel.isStreaming {
      if Date() > deadline {
        XCTFail("Timed out waiting for streaming to finish")
        return
      }
      try await Task.sleep(nanoseconds: 50_000_000)
    }
  }
}
