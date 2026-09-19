import XCTest

@testable import ThinTalkCore

/// Builds a client wired to the stubbed wire for the given channel identity.
final class StubbedClientFactory {
  static func client(channel: ThinTalkChannel, stallTimeout: TimeInterval = 60) -> ThinTalkClient {
    ThinTalkClient(
      applicationIdentity: ThinTalkApplicationIdentity(
        channel: channel, supervisorPort: channel.defaultSupervisorPort),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration()),
      stallTimeout: stallTimeout
    )
  }

  static func stableClient(stallTimeout: TimeInterval = 60) -> ThinTalkClient {
    client(channel: .stable, stallTimeout: stallTimeout)
  }
}

final class ThinTalkClientContractTests: XCTestCase {
  override func tearDown() {
    StubSupervisorURLProtocol.statusResponse = nil
    StubSupervisorURLProtocol.modelsResponse = nil
    StubSupervisorURLProtocol.chatResponse = nil
    StubSupervisorURLProtocol.chatHoldOpen = false
    StubSupervisorURLProtocol.receivedRequestPaths = []
    StubSupervisorURLProtocol.receivedRequestMethods = []
    super.tearDown()
  }

  // MARK: - Readiness handshake

  func test_should_return_ready_status_for_the_matching_channel() async throws {
    StubSupervisorURLProtocol.statusResponse = StubSupervisorURLProtocol.matchingStableStatusResponse
    let client = StubbedClientFactory.stableClient()
    let status = try await client.handshake()
    XCTAssertEqual(status, "ready")
    XCTAssertEqual(StubSupervisorURLProtocol.receivedRequestPaths, ["/v1/status"])
  }

  func test_should_reject_status_from_the_opposite_runtime_channel() async {
    StubSupervisorURLProtocol.statusResponse = StubSupervisorURLProtocol.ResponseConfiguration(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"channel":"development","state_directory":"~/.astronomical-dev"},"status":"ready"}"#.utf8)
    )
    do {
      _ = try await StubbedClientFactory.stableClient().handshake()
      XCTFail("A stable client must not adopt a development supervisor")
    } catch let error as ThinTalkClientError {
      XCTAssertEqual(error, .wrongInstance(
        "This app must connect to its own runtime channel and state directory."))
    } catch {
      XCTFail("Expected a wrongInstance error, got \(error)")
    }
  }

  func test_should_reject_an_unidentified_server() async {
    StubSupervisorURLProtocol.statusResponse = StubSupervisorURLProtocol.ResponseConfiguration(
      statusCode: 200, responseBody: Data(#"{"status":"ready"}"#.utf8))
    await XCTAssertThrowsErrorAsync(
      try await StubbedClientFactory.stableClient().handshake()
    ) { error in
      XCTAssertEqual(error as? ThinTalkClientError, .wrongInstance(
        "This app must connect to its own runtime channel and state directory."))
    }
  }

  // MARK: - Model listing

  func test_should_keep_only_chat_capable_models_and_report_vision() async throws {
    StubSupervisorURLProtocol.modelsResponse = StubSupervisorURLProtocol.ResponseConfiguration(
      statusCode: 200,
      responseBody: Data(
        #"{"data":[{"id":"fictional/chat-7b","input_modalities":["text"],"supported_endpoints":["/v1/chat/completions"]},{"id":"fictional/embed-7b","input_modalities":["text"],"supported_endpoints":["/v1/embeddings"]},{"id":"fictional/vision-7b","input_modalities":["text","image"],"supported_endpoints":["/v1/chat/completions"]}]}"#.utf8)
    )
    let models = try await StubbedClientFactory.stableClient().models()
    XCTAssertEqual(models.map { $0.id }, ["fictional/chat-7b", "fictional/vision-7b"])
    XCTAssertTrue(models.first(where: { $0.id == "fictional/vision-7b" })?.supportsVision ?? false)
  }

  // MARK: - Streaming chat

  func test_should_stream_text_deltas_into_sequential_events() async throws {
    let sseBody = "data: {"
      + #""choices":[{"delta":{"role":"assistant","content":"Hello"}}]}"# + "\n\n"
      + "data: {" + #""choices":[{"delta":{"content":" world"}}]}"# + "\n\n"
      + "data: {" + #""choices":[{"delta":{},"finish_reason":"stop"}]}"# + "\n\n"
      + "data: [DONE]\n\n"
    StubSupervisorURLProtocol.chatResponse = StubSupervisorURLProtocol.stubbedChat(
      sseBody: Data(sseBody.utf8))
    let client = StubbedClientFactory.stableClient()
    var events: [ChatEvent] = []
    for await event in client.chatStream(
      modelID: "fictional/chat-7b", messages: [ChatMessage(role: .user, content: "Hi")]
    ) {
      events.append(event)
    }
    let texts = events.compactMap { event -> String? in
      guard case .text(let text) = event else { return nil }
      return text
    }
    XCTAssertEqual(texts, ["Hello", " world"])
  }

  func test_should_accumulate_reasoning_and_text_into_one_assistant_message() {
    var messages = [ChatMessage(id: UUID(), role: .user, content: "Why is the sky blue?")]
    var assistantID: UUID?

    // Reasoning can arrive before any visible text and opens the message.
    _ = ChatConversationReducer.apply(
      event: .reasoning("Rayleigh scattering"), to: &messages, assistantID: &assistantID)
    let createdID = messages.last?.id
    XCTAssertNotNil(assistantID)
    XCTAssertEqual(messages.count, 2)
    XCTAssertEqual(messages.last?.reasoning, "Rayleigh scattering")
    XCTAssertEqual(messages.last?.content, "")

    // Later reasoning and visible text both accumulate into that same message.
    _ = ChatConversationReducer.apply(
      event: .reasoning(" scatters blue light"), to: &messages, assistantID: &assistantID)
    _ = ChatConversationReducer.apply(
      event: .text("The sky is blue because of Rayleigh scattering."),
      to: &messages, assistantID: &assistantID)
    XCTAssertEqual(messages.count, 2)
    XCTAssertEqual(messages.last?.id, createdID)
    XCTAssertEqual(messages.last?.reasoning, "Rayleigh scattering scatters blue light")
    XCTAssertEqual(messages.last?.content, "The sky is blue because of Rayleigh scattering.")
  }

  func test_should_surface_a_stream_failure_event() async throws {
    let sseBody = "data: {"
      + #""error":{"message":"The model 'a/b' is not loaded by the local worker."}}"# + "\n\n"
    StubSupervisorURLProtocol.chatResponse = StubSupervisorURLProtocol.stubbedChat(
      sseBody: Data(sseBody.utf8))
    let client = StubbedClientFactory.stableClient()
    var failure: ChatFailure?
    for await event in client.chatStream(
      modelID: "a/b", messages: [ChatMessage(role: .user, content: "Hi")]
    ) {
      if case .failure(let captured) = event { failure = captured }
    }
    XCTAssertEqual(failure?.kind, .unknown)
    XCTAssertEqual(
      failure?.message, "The model 'a/b' is not loaded by the local worker.")
  }

  func test_should_report_a_stall_when_no_token_arrives() async throws {
    StubSupervisorURLProtocol.stubbedChatHoldOpen()
    let client = StubbedClientFactory.stableClient(stallTimeout: 1.5)
    var stallFailure: ChatFailure?
    for await captured in client.chatStream(
      modelID: "fictional/chat-7b", messages: [ChatMessage(role: .user, content: "Hi")]
    ) {
      if case .failure(let failure) = captured { stallFailure = failure }
      break
    }
    XCTAssertEqual(stallFailure?.kind, .stall)
  }

  // MARK: - Failure classification

  func test_should_classify_the_public_chat_codes() {
    let client = StubbedClientFactory.stableClient()
    let cases: [(String, ChatFailureKind)] = [
      ("model_not_found: no such model", .noUsableModel),
      ("model_load_failed: out of memory, reduce image size or history", .modelLoadFailed),
      ("chat_engine_busy: another request in flight", .engineBusy),
      ("chat_worker_unavailable: worker down", .workerUnavailable),
      ("chat_malformed_model_output", .malformedModelOutput),
      ("context_length exceeded: shorten the request", .invalidRequest),
    ]
    for (reason, expected) in cases {
      XCTAssertEqual(client.failure(reason: reason).kind, expected)
    }
  }

  func test_should_classify_by_status_when_reason_is_unknown() {
    let client = StubbedClientFactory.stableClient()
    XCTAssertEqual(client.mapFailure(status: 404, body: Data()).kind, .noUsableModel)
    XCTAssertEqual(client.mapFailure(status: 429, body: Data()).kind, .engineBusy)
    XCTAssertEqual(client.mapFailure(status: 503, body: Data()).kind, .workerUnavailable)
    XCTAssertEqual(client.mapFailure(status: 400, body: Data()).kind, .invalidRequest)
    XCTAssertEqual(client.mapFailure(status: 500, body: Data()).kind, .malformedModelOutput)
  }

  func test_should_offer_a_specific_next_action_for_each_failure() {
    XCTAssertEqual(
      ChatFailure(kind: .noUsableModel).nextAction, "Get a model from the Library, then try again.")
    XCTAssertEqual(
      ChatFailure(kind: .modelLoadFailed).nextAction,
      "Pick a smaller model, or raise the memory ceiling, then try again.")
    XCTAssertEqual(
      ChatFailure(kind: .engineBusy).nextAction, "The runner is busy. Wait a moment and try again.")
    XCTAssertEqual(ChatFailure(kind: .stall).nextAction, "The reply stalled. Try again.")
  }
}

extension XCTestCase {
  func XCTAssertThrowsErrorAsync(
    _ expression: @autoclosure () async throws -> Any,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line,
    _ errorHandler: (Error) -> Void
  ) async {
    do {
      _ = try await expression()
      XCTFail(message(), file: file, line: line)
    } catch {
      errorHandler(error)
    }
  }
}
