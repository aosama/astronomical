import Foundation
import Observation
import ThinTalkCore

/// How the chat surface discovered its models at startup.
enum LoadState: Equatable, Sendable {
  case idle
  case loading
  case ready
  case empty
  case failed(String)
}

/// Owns the conversation: drives the streaming client, keeps the message history,
/// and preserves the user's ask so a failure always offers a recoverable next
/// action. It only renders state; all REST and failure logic lives in the client.
@MainActor
@Observable
final class ChatViewModel {
  var messages: [ChatMessage] = []
  var availableModels: [ThinTalkModel] = []
  var selectedModelID: String?
  var draft = ""
  private(set) var state: LoadState = .idle
  private(set) var currentFailure: ChatFailure?
  private(set) var isStreaming = false
  private(set) var assistantMessageID: UUID?

  private let client: ThinTalkClient
  private var streamingTask: Task<Void, Never>?

  // Derived markdown is memoised so streaming deltas do not re-parse the whole
  // conversation on every body pass. Cache writes must not notify observers,
  // hence @ObservationIgnored.
  @ObservationIgnored private var markdownCache: [String: AttributedString] = [:]

  init(client: ThinTalkClient) {
    self.client = client
  }

  static func `default`() -> ChatViewModel {
    ChatViewModel(client: ThinTalkClient(applicationIdentity: ThinTalkApplicationIdentity.current()))
  }

  // MARK: - Startup

  func load() async {
    state = .loading
    do {
      _ = try await client.handshake()
      availableModels = try await client.models()
      if selectedModelID == nil { selectedModelID = availableModels.first?.id }
      state = availableModels.isEmpty ? .empty : .ready
    } catch {
      state = .failed(error.localizedDescription)
    }
  }

  // MARK: - Sending

  func sendMessage() {
    let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, selectedModelID != nil else { return }
    draft = ""
    let userMessage = ChatMessage(role: .user, content: trimmed)
    messages.append(userMessage)
    currentFailure = nil
    beginStreaming(requestMessages: messages)
  }

  /// Retries after a failure, keeping the exact ask the user typed.
  func retry() {
    guard let userMessage = messages.last, userMessage.role == .user else { return }
    let requestMessages = Array(messages.dropLast()) + [userMessage]
    currentFailure = nil
    beginStreaming(requestMessages: requestMessages)
  }

  func stopStreaming() {
    streamingTask?.cancel()
    streamingTask = nil
    isStreaming = false
  }

  /// Renders assistant markdown inline, memoised per raw string. Falls back to
  /// the raw string when the text is not valid markdown, because a partially
  /// streamed reply is often mid-syntax and must never crash the conversation.
  func attributedText(_ raw: String) -> AttributedString {
    if let cached = markdownCache[raw] { return cached }
    let attributed: AttributedString =
      (try? AttributedString(
        markdown: raw,
        options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
      )) ?? AttributedString(raw)
    markdownCache[raw] = attributed
    return attributed
  }

  private func beginStreaming(requestMessages: [ChatMessage]) {
    streamingTask?.cancel()
    streamingTask = nil
    isStreaming = true
    streamingTask = Task {
      guard let modelID = selectedModelID else { isStreaming = false; return }
      for await event in client.chatStream(modelID: modelID, messages: requestMessages) {
        if let failure = ChatConversationReducer.apply(event: event, to: &messages, assistantID: &assistantMessageID) {
          currentFailure = failure
          isStreaming = false
        }
      }
      isStreaming = false
    }
  }
}
