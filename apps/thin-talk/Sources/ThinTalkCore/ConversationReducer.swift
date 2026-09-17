import Foundation

/// Pure reducer that folds chat stream events into the message history. Keeping
/// it free of SwiftUI makes the conversation behaviour unit-testable. On a
/// failure it returns the failure so the caller can record it; on a delta it
/// accumulates the delta into the current assistant message or starts one.
public enum ChatConversationReducer {
  public static func apply(
    event: ChatEvent,
    to messages: inout [ChatMessage],
    assistantID: inout UUID?
  ) -> ChatFailure? {
    switch event {
    case .text(let text):
      appendAssistant(text: text, to: &messages, assistantID: &assistantID)
      return nil
    case .reasoning(let text):
      appendAssistantReasoning(text: text, to: &messages, assistantID: &assistantID)
      return nil
    case .failure(let failure):
      return failure
    }
  }

  private static func appendAssistant(
    text: String, to messages: inout [ChatMessage], assistantID: inout UUID?
  ) {
    if let id = assistantID, let index = messages.firstIndex(where: { $0.id == id }) {
      messages[index].content += text
    } else {
      let newID = UUID()
      assistantID = newID
      messages.append(ChatMessage(id: newID, role: .assistant, content: text))
    }
  }

  private static func appendAssistantReasoning(
    text: String, to messages: inout [ChatMessage], assistantID: inout UUID?
  ) {
    if let id = assistantID, let index = messages.firstIndex(where: { $0.id == id }) {
      messages[index].reasoning += text
    } else {
      // Reasoning can arrive before any visible text: open the message now.
      let newID = UUID()
      assistantID = newID
      messages.append(ChatMessage(id: newID, role: .assistant, content: "", reasoning: text))
    }
  }
}
