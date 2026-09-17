import Foundation
import SwiftUI
import ThinTalkCore

/// Owns the conversation area state: messages, streaming, copy, regenerate,
/// and failure handling. Mirrors `PreviewViewModel` but is scoped to a single
/// conversation so it can be instantiated independently by the preview root.
@MainActor
final class PreviewConversationViewModel: ObservableObject {
  @Published var messages: [PreviewMessage]
  @Published var isStreaming = false
  @Published var currentFailure: ChatFailure?

  private var streamingTask: Task<Void, Never>?
  private var conversation: PreviewConversationItem

  init(conversation: PreviewConversationItem) {
    self.conversation = conversation
    self.messages = PreviewData.messages(for: conversation.id)
  }

  var hasMessages: Bool { !messages.isEmpty }

  // MARK: - Actions

  func copyMessage(_ message: PreviewMessage) {
    let board = NSPasteboard.general
    board.setString(message.content, forType: .string)
  }

  func copyAll() {
    let text = messages.map { $0.displayText }
      .joined(separator: "\n\n")
    NSPasteboard.general.setString(text, forType: .string)
  }

  func regenerate() {
    if let last = messages.last, last.role == .assistant {
      messages.removeLast()
    }
    streamAssistant()
  }

  func stopStreaming() {
    streamingTask?.cancel()
    streamingTask = nil
    isStreaming = false
  }

  func fail(_ kind: ChatFailureKind, message: String? = nil) {
    messages.append(PreviewMessage(role: .user,
      content: "Tell me how matrix multiplication works in a local model."))
    currentFailure = ChatFailure(kind: kind, message: message)
  }

  // MARK: - Simulated streaming

  private func streamAssistant() {
    guard !isStreaming else { return }
    streamingTask?.cancel()
    streamingTask = nil
    isStreaming = true
    streamingTask = Task {
      let id = UUID()
      messages.append(PreviewMessage(id: id, role: .assistant, content: ""))
      guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
      messages[index].reasoning = PreviewData.scripted.reasoning
      for token in PreviewData.scripted.chunked {
        try? await Task.sleep(for: .milliseconds(30))
        if Task.isCancelled { return }
        messages[index].content.append(token)
      }
      messages[index].richHTML = PreviewData.scripted.richHTML
      currentFailure = nil
      isStreaming = false
    }
  }
}

extension PreviewMessage {
  /// Plain-text representation used for copy-all.
  var displayText: String {
    var text = content
    if let richHTML, !richHTML.isEmpty {
      text += "\n\n" + richHTML
    }
    return text
  }
}
