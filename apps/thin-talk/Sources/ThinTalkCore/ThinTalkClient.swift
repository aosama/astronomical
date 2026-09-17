import Foundation

/// One decoded raw model entry from `/v1/models`. The supervisor publishes input
/// output modalities and the endpoints each model supports, so a chat app can
/// keep only the models that can receive chat requests.
struct RawModel: Decodable {
  let id: String
  let inputModalities: [String]
  let supportedEndpoints: [String]?

  enum CodingKeys: String, CodingKey {
    case id
    case inputModalities = "input_modalities"
    case supportedEndpoints = "supported_endpoints"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(String.self, forKey: .id)
    inputModalities = try container.decodeIfPresent([String].self, forKey: .inputModalities) ?? []
    supportedEndpoints = try container.decodeIfPresent([String].self, forKey: .supportedEndpoints)
  }

  var supportsChat: Bool {
    let keepsText = inputModalities.isEmpty || inputModalities.contains("text")
    let keepsChat = supportedEndpoints?.contains { $0.contains("chat") } ?? true
    return keepsText && keepsChat
  }
}

/// A discovered chat-capable Library model the user can send messages to.
/// The supervisor models an OpenAI-compatible model list; vision models
/// advertise an "image" input modality (macOS-visible via the model's advertised
/// capabilities), so attachment can be offered only when the model supports it.
public struct ThinTalkModel: Identifiable, Sendable, Equatable {
  public let id: String
  public let name: String
  public let inputModalities: [String]

  public init(id: String, name: String, inputModalities: [String]) {
    self.id = id
    self.name = name
    self.inputModalities = inputModalities
  }

  /// Whether the loaded model can accept images.
  public var supportsVision: Bool { inputModalities.contains("image") }
}

/// A decoded OpenAI-compatible chat completion chunk from the supervisor stream.
/// `choices` is optional because an error-only chunk carries no choices.
struct ChatCompletionChunk: Decodable {
  var error: ChatError?
  var choices: [ChatCompletionChoice]?

  struct ChatError: Decodable {
    var message: String?
  }

  struct ChatCompletionChoice: Decodable {
    var delta: ChatCompletionDelta
    var finishReason: String?

    enum CodingKeys: String, CodingKey {
      case delta
      case finishReason = "finish_reason"
    }
  }

  struct ChatCompletionDelta: Decodable {
    var content: String?
    var reasoningContent: String?

    enum CodingKeys: String, CodingKey {
      case content
      case reasoningContent = "reasoning_content"
    }
  }
}

/// One non-streaming chat completion request. Kept minimal for the text-only
/// surface; richer modality content is composed at the request boundary later.
struct ChatCompletionRequest: Encodable {
  var model: String
  var stream: Bool
  var messages: [ChatCompletionMessagePayload]
}

/// One request message payload. For text-only messages, content is a string.
struct ChatCompletionMessagePayload: Encodable {
  var role: String
  var content: String
}

/// A lenient error document from the supervisor. Both `{status,message}` and
/// `{error:{message}}` shapes are accepted so the client can classify failures.
struct SupervisorErrorDocument: Decodable {
  var status: String?
  var message: String?
  var error: ChatCompletionChunk.ChatError?

  var reason: String? {
    if let status, !status.isEmpty { return status }
    if let message, !message.isEmpty { return message }
    return error?.message
  }
}

/// One decoded status document used for the readiness handshake.
struct SupervisorStatusDocument: Decodable {
  struct Application: Decodable {
    var channel: String?
    var stateDirectory: String?

    enum CodingKeys: String, CodingKey {
      case channel
      case stateDirectory = "state_directory"
    }
  }
  var application: Application?
  var status: String?
}

/// Signals emitted while consuming one chat stream. Failures are events (not
/// thrown) so the view can render them inline and keep the conversation recoverable.
public enum ChatEvent: Sendable {
  /// One assistant text delta to append to the current message.
  case text(String)
  /// One reasoning delta to append to the current message.
  case reasoning(String)
  /// The stream stopped with a classified failure and a specific next action.
  case failure(ChatFailure)
}

/// Errors that prevent any request from reaching the supervisor.
public enum ThinTalkClientError: LocalizedError, Equatable {
  case invalidEndpoint
  case responseTooLarge
  case unexpectedResponse
  case wrongInstance(String)

  public var errorDescription: String? {
    switch self {
    case .invalidEndpoint: "The local server address is invalid."
    case .responseTooLarge: "The server response was too large."
    case .unexpectedResponse: "The server returned an unexpected response."
    case .wrongInstance(let message): message
    }
  }
}

/// A minimal thread-safe FIFO event channel shared between the producer task and
/// the stall-watchdog task. `next(timeout:)` blocks until an event arrives or the
/// timeout elapses, returning `nil` in both cases so the watchdog can distinguish
/// a slow stream from a finished one via `isFinished`.
final class EventChannel<T: Sendable>: @unchecked Sendable {
  // One condition guards both the queue and the finished flag; a second lock
  // alongside the condition would leave `next` reading the queue unlocked.
  private let condition = NSCondition()
  private var queue: [T] = []
  private var finished = false

  func enqueue(_ element: T) {
    condition.lock()
    queue.append(element)
    condition.broadcast()
    condition.unlock()
  }

  func finish() {
    condition.lock()
    finished = true
    condition.broadcast()
    condition.unlock()
  }

  var isFinished: Bool {
    condition.lock()
    defer { condition.unlock() }
    return finished
  }

  func next(timeout: TimeInterval) -> T? {
    condition.lock()
    let deadline = Date().addingTimeInterval(timeout)
    while queue.isEmpty && !finished {
      if !condition.wait(until: deadline) { condition.unlock(); return nil }
    }
    if !queue.isEmpty {
      let element = queue.removeFirst()
      condition.unlock()
      return element
    }
    condition.unlock()
    return nil
  }
}

/// Talks to a single supervisor instance over its public REST endpoints: the
/// readiness handshake, the model list, and streaming chat completion. All
/// network and decoding logic is isolated here so it can be tested against a
/// stubbed wire.
public struct ThinTalkClient: Sendable {
  public let applicationIdentity: ThinTalkApplicationIdentity
  private let urlSession: URLSession
  private let maximumResponseByteCount: Int
  private let maximumRetryBackoff: TimeInterval
  /// No stream token within this window counts as a stalled generation. A stall
  /// has no backend marker, so the chat surface owns this watchdog.
  public let stallTimeout: TimeInterval

  public init(
    applicationIdentity: ThinTalkApplicationIdentity,
    urlSession: URLSession = .shared,
    maximumResponseByteCount: Int = 1_048_576,
    maximumRetryBackoff: TimeInterval = 4,
    stallTimeout: TimeInterval = 60
  ) {
    self.applicationIdentity = applicationIdentity
    self.urlSession = urlSession
    self.maximumResponseByteCount = maximumResponseByteCount
    self.maximumRetryBackoff = maximumRetryBackoff
    self.stallTimeout = stallTimeout
  }

  /// Shared decoder for supervisor responses. Keys are matched through explicit
  /// CodingKeys, so no automatic case conversion is applied.
  private static let decoder = JSONDecoder()

  // MARK: - Readiness handshake

  /// Verifies this identity is talking to the matching supervisor instance,
  /// returning its status text. Throws when the endpoint answers with a
  /// different runtime channel or state directory.
  public func handshake() async throws -> String {
    let responseBody = try await requestRaw(
      path: "/v1/status", method: "GET", acceptedStatusCodes: [200])
    let statusDocument = try ThinTalkClient.decoder.decode(SupervisorStatusDocument.self, from: responseBody)
    guard let connectedChannel = statusDocument.application?.channel else {
      throw ThinTalkClientError.wrongInstance(
        "This app must connect to its own runtime channel and state directory.")
    }
    guard connectedChannel == applicationIdentity.channel.rawValue else {
      throw ThinTalkClientError.wrongInstance(
        "This app must connect to its own runtime channel and state directory.")
    }
    guard
      let connectedState = statusDocument.application?.stateDirectory,
      connectedState == applicationIdentity.expectedServerStateDirectory
    else {
      throw ThinTalkClientError.wrongInstance(
        "This app must connect to its own runtime channel and state directory.")
    }
    return statusDocument.status ?? "unknown"
  }

  /// Returns the chat-capable models discovered on this channel.
  public func models() async throws -> [ThinTalkModel] {
    let responseBody = try await requestRaw(
      path: "/v1/models", method: "GET", acceptedStatusCodes: [200])
    let modelsDocument = try ThinTalkClient.decoder.decode(ModelsDocument.self, from: responseBody)
    return modelsDocument.data.filter { $0.supportsChat }.map { model in
      ThinTalkModel(
        id: model.id, name: model.id, inputModalities: model.inputModalities
      )
    }
  }

  // MARK: - Streaming chat

  /// Consumes one streaming chat completion and yields deltas until the stream
  /// finishes, a failure is classified, or the watchdog reports a stall.
  public func chatStream(modelID: String, messages: [ChatMessage]) -> AsyncStream<ChatEvent> {
    AsyncStream { externalContinuation in
      let channel = EventChannel<ChatEvent>()
      let producer = Task { await self.consumeInto(modelID: modelID, messages: messages, channel: channel) }
      let watchdog = Task {
        let pollInterval: TimeInterval = 1.0
        var lastActivity = Date()
        while !channel.isFinished {
          let activityDate = lastActivity
          guard let event = channel.next(timeout: pollInterval) else {
            let elapsed = Date().timeIntervalSince(activityDate)
            if channel.isFinished { break }
            if elapsed >= self.stallTimeout {
              externalContinuation.yield(.failure(ChatFailure(kind: .stall)))
              channel.finish()
              producer.cancel()
              break
            }
            continue
          }
          lastActivity = Date()
          externalContinuation.yield(event)
          if case .failure = event { channel.finish(); break }
        }
        // Every watchdog exit path must complete the stream, otherwise the
        // consumer waits forever after the last event.
        externalContinuation.finish()
      }
      externalContinuation.onTermination = { _ in producer.cancel(); watchdog.cancel() }
    }
  }

  private func consumeInto(
    modelID: String, messages: [ChatMessage], channel: EventChannel<ChatEvent>
  ) async {
    do {
      try await sendChat(modelID: modelID, messages: messages) { channel.enqueue($0) }
      channel.finish()
    } catch let failure as ChatFailure {
      channel.enqueue(.failure(failure))
      channel.finish()
    } catch {
      channel.enqueue(.failure(ChatFailure(kind: .workerUnavailable)))
      channel.finish()
    }
  }

  private func sendChat(
    modelID: String, messages: [ChatMessage], yield: (ChatEvent) -> Void
  ) async throws {
    let request = ChatCompletionRequest(
      model: modelID,
      stream: true,
      messages: messages.map { message in
        ChatCompletionMessagePayload(role: message.role.rawValue, content: message.content)
      }
    )
    let requestBody = try JSONEncoder().encode(request)
    var requestURL = URLRequest(url: try applicationIdentity.endpointURL(path: "/v1/chat/completions"))
    requestURL.httpMethod = "POST"
    requestURL.setValue("application/json", forHTTPHeaderField: "Content-Type")
    requestURL.httpBody = requestBody
    requestURL.timeoutInterval = 120

    let stream: URLSession.AsyncBytes
    do {
      let (streamData, _) = try await urlSession.bytes(for: requestURL)
      stream = streamData
    } catch {
      // Pre-stream failure (e.g. a model that will not load). Re-fetch to capture
      // the bounded failure reason the supervisor reports.
      let (bodyData, response) = try await urlSession.data(for: requestURL)
      if let httpResponse = response as? HTTPURLResponse {
        throw mapFailure(status: httpResponse.statusCode, body: bodyData)
      }
      throw ChatFailure(kind: .workerUnavailable)
    }

    for try await line in stream.lines {
      let trimmedLine = line.trimmingCharacters(in: .whitespaces)
      guard trimmedLine.hasPrefix("data:") else { continue }
      let payload = String(trimmedLine.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces)
      guard !payload.isEmpty, payload != "[DONE]" else { continue }
      guard let chunk = try? ThinTalkClient.decoder.decode(ChatCompletionChunk.self, from: Data(payload.utf8)) else {
        continue
      }
      if let chunkError = chunk.error {
        throw failure(reason: chunkError.message)
      }
      for choice in chunk.choices ?? [] {
        if let deltaText = choice.delta.content, !deltaText.isEmpty { yield(.text(deltaText)) }
        if let deltaReasoning = choice.delta.reasoningContent, !deltaReasoning.isEmpty {
          yield(.reasoning(deltaReasoning))
        }
      }
    }
  }

  // MARK: - HTTP

  private func requestRaw(
    path: String, method: String, acceptedStatusCodes: Set<Int>
  ) async throws -> Data {
    let endpointURL = try applicationIdentity.endpointURL(path: path)
    var request = URLRequest(url: endpointURL)
    request.httpMethod = method
    request.timeoutInterval = 2
    let (responseBody, response) = try await urlSession.data(for: request)
    guard responseBody.count <= maximumResponseByteCount else {
      throw ThinTalkClientError.responseTooLarge
    }
    guard let httpResponse = response as? HTTPURLResponse else {
      throw ThinTalkClientError.unexpectedResponse
    }
    guard acceptedStatusCodes.contains(httpResponse.statusCode) else {
      throw mapFailure(status: httpResponse.statusCode, body: responseBody)
    }
    return responseBody
  }

  // MARK: - Failure classification

  /// Maps a supervisor HTTP status and raw body to a classified failure with a
  /// specific next action. Recognises the public chat codes first, then falls
  /// back to status-code heuristics so no failure is left unclassified.
  public func mapFailure(status: Int, body: Data) -> ChatFailure {
    let document = (try? ThinTalkClient.decoder.decode(SupervisorErrorDocument.self, from: body))
      ?? SupervisorErrorDocument(status: nil, message: nil, error: nil)
    let reason = document.reason ?? ""
    let kind = classify(status: status, reason: reason)
    return ChatFailure(kind: kind, message: document.reason)
  }

  /// Classifies a human-readable reason (from an HTTP body or an error chunk) into
  /// a failure kind. Public so the chat surface can classify a streaming error
  /// chunk with the same logic as an HTTP failure.
  public func failure(reason: String?) -> ChatFailure {
    ChatFailure(kind: classifyReason(reason: reason ?? ""), message: reason)
  }

  private func classifyReason(reason: String) -> ChatFailureKind {
    let reason = reason.lowercased()
    if reason.contains("model_not_found") { return .noUsableModel }
    if reason.contains("model_load_failed") { return .modelLoadFailed }
    if reason.contains("chat_malformed_model_output") { return .malformedModelOutput }
    if reason.contains("chat_worker_unavailable") { return .workerUnavailable }
    if reason.contains("chat_engine_busy") || reason.contains("server_capacity") || reason.contains("busy") {
      return .engineBusy
    }
    if reason.contains("context_length") || reason.contains("context") || reason.contains("invalid_request") {
      return .invalidRequest
    }
    return .unknown
  }

  private func classify(status: Int, reason: String) -> ChatFailureKind {
    let reasonKind = classifyReason(reason: reason)
    if reasonKind != .unknown { return reasonKind }
    switch status {
    case 404: return .noUsableModel
    case 429: return .engineBusy
    case 503: return .workerUnavailable
    case 400: return .invalidRequest
    case 500: return .malformedModelOutput
    default: return .unknown
    }
  }
}

private extension ChatFailureKind {
  var defaultReason: String {
    switch self {
    case .noUsableModel: return "No usable model is available."
    case .modelLoadFailed: return "The model could not be loaded."
    case .workerUnavailable: return "The runner is not available."
    case .engineBusy: return "The runner is busy."
    case .invalidRequest: return "The request was invalid."
    case .malformedModelOutput: return "The reply could not be parsed."
    case .stall: return "The reply stalled."
    case .unknown: return "Something went wrong."
    }
  }
}

struct ModelsDocument: Decodable {
  let data: [RawModel]
}
