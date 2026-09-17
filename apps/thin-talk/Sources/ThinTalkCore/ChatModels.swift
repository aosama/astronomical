import Foundation

/// The sender of one stored message in the conversation.
public enum ChatRole: String, Sendable, Codable {
  case user
  case assistant
}

/// One complete message the user typed or the assistant produced. Storage stays
/// simple (plain text plus separate attachment metadata) so richer modalities can
/// be added later without rewriting the core model.
public struct ChatMessage: Identifiable, Sendable, Equatable {
  public let id: UUID
  public let role: ChatRole
  public var content: String
  /// Model reasoning kept separate from the assistant-visible answer, so the
  /// surface can render the thinking quietly instead of mixing it into the reply.
  public var reasoning: String
  /// Attachment metadata is stored separately from content. A message always
  /// carries this field so future features can compose richer request content.
  public var attachments: [ChatAttachment]

  public init(
    id: UUID = UUID(), role: ChatRole, content: String, reasoning: String = "",
    attachments: [ChatAttachment] = []
  ) {
    self.id = id
    self.role = role
    self.content = content
    self.reasoning = reasoning
    self.attachments = attachments
  }
}

/// Metadata for one attachment the user may attach to a message. Kept minimal so
/// extensible features can grow the shape instead of replacing it.
public struct ChatAttachment: Identifiable, Sendable, Equatable {
  public let id: UUID
  public let type: String
  public let label: String

  public init(id: UUID = UUID(), type: String, label: String) {
    self.id = id
    self.type = type
    self.label = label
  }
}

/// Every failure the conversation can hit, derived from the supervisor's public
/// REST error signals. This is the client-facing view of the backend code, so the
/// pure layer can map each failure to a concrete next action without knowing how
/// the response was decoded.
public enum ChatFailureKind: String, Sendable, Equatable {
  /// No usable chat-capable Library model is discoverable.
  case noUsableModel
  /// The selected model will not load, including the memory-ceiling refusal.
  case modelLoadFailed
  /// The supervisor/daemon is unavailable or not ready to serve requests.
  case workerUnavailable
  /// Another request already owns the worker's bounded capacity.
  case engineBusy
  /// The request was malformed, including an over-length ask.
  case invalidRequest
  /// The model's output could not be parsed into the declared output contract.
  case malformedModelOutput
  /// No stream token arrived within the client watchdog timeout (a stall).
  case stall
  /// Anything the client could not classify into the categories above.
  case unknown
}

/// One failure the conversation can hit, plus the human-readable reason the
/// supervisor reported. The reason is shown verbatim where it is safe; the next
/// action is derived from the kind.
public struct ChatFailure: Identifiable, Sendable, Equatable, Error {
  public var id: UUID
  public let kind: ChatFailureKind
  public let message: String?

  public init(kind: ChatFailureKind, message: String? = nil) {
    self.id = UUID()
    self.kind = kind
    self.message = message
  }

  /// The one specific, plain-language action the user can take after this
  /// failure. This is the failure→action contract kept in one place so every
  /// surface stays consistent.
  public var nextAction: String {
    switch kind {
    case .noUsableModel:
      return "Get a model from the Library, then try again."
    case .modelLoadFailed:
      return "Pick a smaller model, or raise the memory ceiling, then try again."
    case .workerUnavailable:
      return "The runner is not running. Start it and try again."
    case .engineBusy:
      return "The runner is busy. Wait a moment and try again."
    case .invalidRequest:
      return "Shorten the message and try again."
    case .malformedModelOutput:
      return "Something went wrong producing that reply. Try again."
    case .stall:
      return "The reply stalled. Try again."
    case .unknown:
      return "Try again."
    }
  }
}
