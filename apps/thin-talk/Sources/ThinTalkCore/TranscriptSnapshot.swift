import Foundation

/// Who produced one transcript entry.
public enum TranscriptRole: String, Sendable, Equatable {
  case user
  case assistant
}

/// The lifecycle state of one transcript entry, mirrored in the canvas.
public enum TranscriptMessageState: String, Sendable, Equatable {
  case complete
  case streaming
  case stopped
  case failed
}

/// One image or file the canvas may render inside a message. The canvas never
/// receives a file path: `assetURL` carries an opaque token that only the asset
/// scheme handler can resolve.
public struct TranscriptAttachment: Sendable, Equatable, Identifiable {
  public let id: UUID
  public let assetURL: String
  public let label: String

  public init(id: UUID = UUID(), assetURL: String, label: String) {
    self.id = id
    self.assetURL = assetURL
    self.label = label
  }
}

/// An app-authored rich HTML block. The application builds this from typed data
/// it already controls, and the canvas sanitises it like any other markup.
public struct TranscriptCard: Sendable, Equatable, Identifiable {
  public let id: UUID
  public let html: String

  public init(id: UUID = UUID(), html: String) {
    self.id = id
    self.html = html
  }
}

/// One entry in the conversation canvas.
public struct TranscriptMessage: Sendable, Equatable, Identifiable {
  public let id: UUID
  public let role: TranscriptRole
  public var markdown: String
  public var reasoning: String
  public var attachments: [TranscriptAttachment]
  public var cards: [TranscriptCard]
  public var state: TranscriptMessageState

  public init(
    id: UUID = UUID(),
    role: TranscriptRole,
    markdown: String,
    reasoning: String = "",
    attachments: [TranscriptAttachment] = [],
    cards: [TranscriptCard] = [],
    state: TranscriptMessageState = .complete
  ) {
    self.id = id
    self.role = role
    self.markdown = markdown
    self.reasoning = reasoning
    self.attachments = attachments
    self.cards = cards
    self.state = state
  }

  public static func == (lhs: TranscriptMessage, rhs: TranscriptMessage) -> Bool {
    lhs.id == rhs.id
      && lhs.role == rhs.role
      && lhs.markdown == rhs.markdown
      && lhs.reasoning == rhs.reasoning
      && lhs.attachments == rhs.attachments
      && lhs.cards == rhs.cards
      && lhs.state == rhs.state
  }
}

/// Everything the canvas needs to draw the conversation. It is a value type so
/// the bridge can compare two snapshots and send the smallest possible update.
public struct TranscriptSnapshot: Sendable, Equatable {
  public var messages: [TranscriptMessage]
  public var model: String?
  public var channel: String
  public var notice: String?

  public init(
    messages: [TranscriptMessage] = [],
    model: String? = nil,
    channel: String = "",
    notice: String? = nil
  ) {
    self.messages = messages
    self.model = model
    self.channel = channel
    self.notice = notice
  }

  /// The single message differing from `other`, when exactly one does.
  func solelyChangedMessage(comparedTo other: TranscriptSnapshot) -> TranscriptMessage? {
    guard messages.count == other.messages.count else { return nil }
    var changed: TranscriptMessage?
    for (candidate, previous) in zip(messages, other.messages) {
      guard candidate.id == previous.id else { return nil }
      if candidate != previous {
        guard changed == nil else { return nil }
        changed = candidate
      }
    }
    return changed
  }
}