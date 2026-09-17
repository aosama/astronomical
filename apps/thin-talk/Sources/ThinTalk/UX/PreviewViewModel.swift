import Foundation
import SwiftUI
import ThinTalkCore

/// Owns the UX-preview state: local session history, selected model, the
/// conversation, and simulation of streaming, failure, copy and regenerate.
/// This is a demonstration of the agreed layout and interactions. All data is
/// local mock content; there is no network call. Production streaming reuses the
/// tested `ThinTalkCore.ChatConversationReducer`, but here updates drive the
/// message list directly so the surface can be reviewed without a supervisor.
@MainActor
final class PreviewViewModel: ObservableObject {
  @Published var sessions: [PreviewSession] = []
  @Published var activeSessionID: UUID?
  @Published var availableModels: [ThinTalkModel] = []
  @Published var selectedModelID: String?
  @Published var draft = ""
  @Published var messages: [PreviewMessage] = []
  @Published var isStreaming = false
  @Published var currentFailure: ChatFailure?

  private var streamingTask: Task<Void, Never>?

  init() {
    self.availableModels = PreviewData.models
    self.activeSessionID = PreviewData.sampleSessionID
    self.sessions = PreviewData.sessions
    self.selectedModelID = PreviewData.models.first { !$0.supportsVision }?.id
    loadActiveSession()
  }

  // MARK: - Sessions

  var activeSession: PreviewSession? {
    guard let id = activeSessionID else { return nil }
    return sessions.first(where: { $0.id == id })
  }

  func selectSession(_ session: PreviewSession) {
    activeSessionID = session.id
    loadActiveSession()
  }

  func newConversation() {
    let session = PreviewSession(title: "New conversation", lastText: "", hasFailure: false)
    sessions.insert(session, at: 0)
    activeSessionID = session.id
    messages = []
  }

  private func loadActiveSession() {
    messages = PreviewData.messages(for: activeSessionID)
    currentFailure = nil
  }

  // MARK: - Sending

  var hasModel: Bool { !availableModels.isEmpty }

  func sendMessage() {
    let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, !isStreaming else { return }
    draft = ""
    messages.append(PreviewMessage(role: .user, content: trimmed))
    currentFailure = nil
    updateActiveTitle(fromFirstMessage: trimmed)
    streamAssistant()
  }

  func retry() {
    guard let userMessage = messages.last, userMessage.role == .user else { return }
    messages.removeLast()
    currentFailure = nil
    updateActiveTitle(fromFirstMessage: userMessage.content)
    streamAssistant()
    messages.append(userMessage)
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

  func copyMessage(_ message: PreviewMessage) {
    let board = NSPasteboard.general
    board.setString(message.content, forType: .string)
  }

  /// Drives the failure banner with a specific next action, matching the
  /// failure→action contract: no dead-end dialog, the typed ask is preserved.
  func fail(_ kind: ChatFailureKind, message: String? = nil) {
    if messages.last?.role != .user {
      messages.append(PreviewMessage(role: .user, content: scriptedAsk))
    }
    currentFailure = ChatFailure(kind: kind, message: message)
  }

  private var scriptedAsk: String {
    "Tell me how matrix multiplication works in a local model."
  }

  private func updateActiveTitle(fromFirstMessage first: String) {
    guard let index = sessions.firstIndex(where: { $0.id == activeSessionID }) else { return }
    if sessions[index].title == "New conversation" {
      let word = first.split(whereSeparator: { $0 == " " || $0 == "\n" }).first.map(String.init) ?? "Conversation"
      sessions[index].title = String(word.prefix(24))
    }
    sessions[index].lastText = first
    sessions[index].updatedAt = Date()
  }

  // MARK: - Simulated streaming

  private func streamAssistant() {
    streamingTask?.cancel()
    streamingTask = nil
    isStreaming = true
    streamingTask = Task {
      let id = UUID()
      messages.append(PreviewMessage(id: id, role: .assistant, content: ""))
      guard let index = messages.firstIndex(where: { $0.id == id }) else { return }

      // Reasoning appears quietly before the visible answer builds up.
      messages[index].reasoning = PreviewData.scripted.reasoning

      // Stream the visible answer, one token at a time, for realism.
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
