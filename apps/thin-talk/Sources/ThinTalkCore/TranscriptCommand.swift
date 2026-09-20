import Foundation

/// One instruction the canvas can execute.
public enum TranscriptCommand: Sendable, Equatable {
  case snapshot(TranscriptSnapshot)
  case message(TranscriptMessage)
  case appearance(dark: Bool)
}

/// Chooses the smallest command that turns one snapshot into the next.
///
/// Streaming answers a token at a time, and re-sending and re-rendering the whole
/// conversation for each token is what makes chat canvases slow. When only the
/// growing answer changed, the canvas receives that one message and patches it in
/// place, so finished blocks keep their rendered form.
public enum TranscriptCommandPlanner {
  public static func plan(
    previous: TranscriptSnapshot?,
    next: TranscriptSnapshot
  ) -> [TranscriptCommand] {
    guard let previous else { return [.snapshot(next)] }
    guard previous != next else { return [] }

    let metadataChanged =
      previous.model != next.model
      || previous.channel != next.channel
      || previous.notice != next.notice
    guard !metadataChanged else { return [.snapshot(next)] }

    if let changedMessage = next.solelyChangedMessage(comparedTo: previous) {
      return [.message(changedMessage)]
    }
    return [.snapshot(next)]
  }
}

/// Encodes a command for the canvas as an invocation of its single entry point.
///
/// The payload travels as Base64 of UTF-8 JSON rather than interpolated source
/// text. A prompt or a model answer can contain any character, including a quote
/// or a closing script tag, and Base64 has no character that can terminate the
/// call, so no escaping mistake can turn content into executable code.
public enum TranscriptCommandCodec {
  public static let entryPoint = "__thintalk.receive"

  public static func invocation(for command: TranscriptCommand) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let payload = try encoder.encode(EncodedCommand(command: command))
    return "window.\(entryPoint)(\"\(payload.base64EncodedString())\")"
  }
}

/// The wire form the canvas expects. Keys are spelled out rather than derived
/// from Swift property names so the two sides stay independently readable.
private struct EncodedCommand: Encodable {
  let command: TranscriptCommand

  private enum CodingKeys: String, CodingKey {
    case version = "v"
    case kind
    case messages
    case message
    case model
    case channel
    case notice
    case dark
  }

  private enum MessageKeys: String, CodingKey {
    case id
    case role
    case markdown
    case reasoning
    case attachments
    case cards
    case state
  }

  private enum AttachmentKeys: String, CodingKey {
    case id
    case assetURL
    case label
  }

  private enum CardKeys: String, CodingKey {
    case id
    case html
  }

  private struct EncodedMessage: Encodable {
    let message: TranscriptMessage

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: MessageKeys.self)
      try container.encode(message.id.uuidString, forKey: .id)
      try container.encode(message.role.rawValue, forKey: .role)
      try container.encode(message.markdown, forKey: .markdown)
      try container.encode(message.reasoning, forKey: .reasoning)
      try container.encode(message.state.rawValue, forKey: .state)
      try container.encode(
        message.attachments.map { attachment in
          EncodedAttachment(attachment: attachment)
        }, forKey: .attachments)
      try container.encode(
        message.cards.map { card in EncodedCard(card: card) }, forKey: .cards)
    }
  }

  private struct EncodedAttachment: Encodable {
    let attachment: TranscriptAttachment

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: AttachmentKeys.self)
      try container.encode(attachment.id.uuidString, forKey: .id)
      try container.encode(attachment.assetURL, forKey: .assetURL)
      try container.encode(attachment.label, forKey: .label)
    }
  }

  private struct EncodedCard: Encodable {
    let card: TranscriptCard

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CardKeys.self)
      try container.encode(card.id.uuidString, forKey: .id)
      try container.encode(card.html, forKey: .html)
    }
  }

  func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(1, forKey: .version)
    switch command {
    case .snapshot(let snapshot):
      try container.encode("snapshot", forKey: .kind)
      try container.encode(
        snapshot.messages.map { EncodedMessage(message: $0) }, forKey: .messages)
      try container.encodeIfPresent(snapshot.model, forKey: .model)
      try container.encode(snapshot.channel, forKey: .channel)
      try container.encodeIfPresent(snapshot.notice, forKey: .notice)
    case .message(let message):
      try container.encode("message", forKey: .kind)
      try container.encode(EncodedMessage(message: message), forKey: .message)
    case .appearance(let dark):
      try container.encode("appearance", forKey: .kind)
      try container.encode(dark, forKey: .dark)
    }
  }
}