import AppKit
import Foundation
import Observation
import ThinTalkCanvas
import ThinTalkCore

/// How the chat surface discovered its models at startup.
public enum LoadState: Equatable, Sendable {
  case idle
  case loading
  case ready
  case empty
  case failed(String)
}

/// Owns the conversation: drives the streaming client, keeps the message history,
/// and preserves the user's ask so a failure always offers a recoverable next
/// action. It only renders state; all REST and failure logic lives in the client.
///
/// The canvas is a pure function of `canvasSnapshot`, so this type decides what the
/// conversation looks like and the web surface decides only how it is drawn.
@MainActor
@Observable
public final class ChatViewModel {
  public var messages: [ChatMessage] = []
  public var availableModels: [ThinTalkModel] = []
  public var selectedModelID: String?
  public var draft = ""
  /// How much thinking the next turn may spend.
  ///
  /// Persisted so a deliberate choice survives a relaunch, while a fresh install
  /// still starts on Quick. The choice applies to the next request, which is why
  /// it is read at send time rather than captured when the window opens.
  public var thinkingEffort: ThinkingEffort {
    didSet {
      guard thinkingEffort != oldValue else { return }
      ThinkingEffortPreference.save(thinkingEffort)
    }
  }
  private(set) var state: LoadState = .idle
  private(set) var currentFailure: ChatFailure?
  private(set) var isStreaming = false
  private(set) var assistantMessageID: UUID?
  private(set) var canvasSnapshot = TranscriptSnapshot()

  /// Files the canvas may display, addressed by opaque token rather than path.
  let assetRegistry = TranscriptAssetRegistry()

  private let client: ThinTalkClient
  private var streamingTask: Task<Void, Never>?

  public init(
    client: ThinTalkClient,
    thinkingEffort: ThinkingEffort = ThinkingEffortPreference.load()
  ) {
    self.client = client
    self.thinkingEffort = thinkingEffort
  }

  public static func `default`() -> ChatViewModel {
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
    refreshCanvasSnapshot()
  }

  // MARK: - Sending

  func sendMessage() {
    let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, selectedModelID != nil else { return }
    draft = ""
    let userMessage = ChatMessage(role: .user, content: trimmed)
    messages.append(userMessage)
    currentFailure = nil
    refreshCanvasSnapshot()
    beginStreaming(requestMessages: messages)
  }

  /// Retries after a failure, keeping the exact ask the user typed.
  func retry() {
    guard let userMessage = messages.last, userMessage.role == .user else { return }
    let requestMessages = Array(messages.dropLast()) + [userMessage]
    currentFailure = nil
    refreshCanvasSnapshot()
    beginStreaming(requestMessages: requestMessages)
  }

  func stopStreaming() {
    streamingTask?.cancel()
    streamingTask = nil
    isStreaming = false
    removeEmptyAssistantTurn()
    assistantMessageID = nil
    refreshCanvasSnapshot()
  }

  // MARK: - Canvas interaction

  /// Performs the one interaction the canvas reported. Copying, regenerating, and
  /// opening a link all run here so the canvas never owns a behaviour.
  func handle(_ action: CanvasAction) {
    switch action.kind {
    case .ready:
      return
    case .copy:
      copyToPasteboard(messageID: action.messageID)
    case .regenerate:
      retry()
    case .openExternal:
      guard let externalURL = action.externalURL else { return }
      NSWorkspace.shared.open(externalURL)
    }
  }

  private func copyToPasteboard(messageID: UUID?) {
    let resolvedID = messageID ?? messages.last?.id
    guard let message = messages.first(where: { $0.id == resolvedID }) else { return }
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(message.content, forType: .string)
  }

  // MARK: - Canvas state

  private func beginStreaming(requestMessages: [ChatMessage]) {
    streamingTask?.cancel()
    // Open the assistant turn up-front and hand its id to the reducer. Without
    // this, assistantMessageID leaks from the previous turn and a second answer
    // keeps appending inside the first assistant bubble; a visible "thinking"
    // placeholder also prevents stacked user turns from looking unanswered.
    let placeholderID = UUID()
    assistantMessageID = placeholderID
    messages.append(ChatMessage(id: placeholderID, role: .assistant, content: ""))
    isStreaming = true
    refreshCanvasSnapshot()
    streamingTask = Task {
      guard let modelID = selectedModelID else {
        isStreaming = false
        refreshCanvasSnapshot()
        return
      }
      for await event in client.chatStream(
        modelID: modelID, messages: requestMessages, thinkingEffort: thinkingEffort
      ) {
        if let failure = ChatConversationReducer.apply(
          event: event, to: &messages, assistantID: &assistantMessageID)
        {
          currentFailure = failure
          isStreaming = false
          removeEmptyAssistantTurn()
          assistantMessageID = nil
        }
        refreshCanvasSnapshot()
      }
      isStreaming = false
      removeEmptyAssistantTurn()
      assistantMessageID = nil
      refreshCanvasSnapshot()
    }
  }

  /// Drops a turn the model never filled, so a stopped or failed request leaves
  /// no empty failed-completed bubble behind the user's question.
  private func removeEmptyAssistantTurn() {
    guard let id = assistantMessageID,
          let index = messages.firstIndex(where: { $0.id == id }),
          messages[index].content.isEmpty, messages[index].reasoning.isEmpty
    else { return }
    messages.remove(at: index)
  }

  /// Rebuilds the snapshot the canvas renders. Called after every mutation so the
  /// planner can send the smallest update that reflects what changed.
  private func refreshCanvasSnapshot() {
    canvasSnapshot = TranscriptSnapshot(
      messages: messages.map(transcriptMessage(from:)),
      model: selectedModelID,
      channel: client.applicationIdentity.channel.displayName,
      notice: noticeText
    )
  }

  private func transcriptMessage(from message: ChatMessage) -> TranscriptMessage {
    TranscriptMessage(
      id: message.id,
      role: message.role == .user ? .user : .assistant,
      markdown: message.content,
      reasoning: message.reasoning,
      attachments: message.attachments.compactMap { attachment in
        guard let assetURL = attachment.assetURL else { return nil }
        return TranscriptAttachment(
          id: attachment.id, assetURL: assetURL, label: attachment.label)
      },
      state: messageState(for: message)
    )
  }

  private func messageState(for message: ChatMessage) -> TranscriptMessageState {
    if isStreaming, message.id == messages.last?.id, message.role == .assistant {
      return .streaming
    }
    if message.role == .assistant, currentFailure != nil, message.id == messages.last?.id {
      return .failed
    }
    return .complete
  }

  private var noticeText: String? {
    switch state {
    case .empty:
      return "No chat model is available yet. Add one from the Library."
    case .idle, .loading:
      return "Looking for a chat model on this Mac."
    case .failed(let message):
      return message
    case .ready:
      return nil
    }
  }
}