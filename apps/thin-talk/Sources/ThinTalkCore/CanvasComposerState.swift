import Foundation

/// Everything the canvas page needs to draw the composer, the thinking-effort
/// control, and the failure banner.
///
/// The page owns how these controls look and what a click or keypress feels like,
/// while Swift owns what they mean: this state is the whole of what Swift pushes,
/// and every behaviour the controls trigger arrives back as a `CanvasAction`.
/// The draft text is deliberately absent: the page owns the field's in-progress
/// text while the reader types, and echoing a Swift-side copy back on every state
/// push would clobber that typing mid-sentence.
public struct CanvasComposerState: Sendable, Equatable {
  /// One selectable thinking level as the page should present it.
  public struct EffortLevel: Sendable, Equatable {
    public let value: String
    public let displayName: String
    public let budgetSummary: String

    public init(effort: ThinkingEffort) {
      value = effort.rawValue
      displayName = effort.displayName
      budgetSummary = effort.budgetSummary
    }
  }

  public var isStreaming: Bool
  public var acceptsInput: Bool
  public var isReady: Bool
  public var effort: EffortLevel
  public var failure: CanvasComposerFailure?

  public init(
    isStreaming: Bool,
    acceptsInput: Bool,
    isReady: Bool,
    effort: ThinkingEffort,
    failure: CanvasComposerFailure?
  ) {
    self.isStreaming = isStreaming
    self.acceptsInput = acceptsInput
    self.isReady = isReady
    self.effort = EffortLevel(effort: effort)
    self.failure = failure
  }

  /// The levels the pill must offer, in the order the core defines them.
  public var effortOptions: [EffortLevel] {
    ThinkingEffort.allCases.map(EffortLevel.init)
  }
}

/// A recoverable failure as the banner should present it: what went wrong and
/// what the reader can do about it.
public struct CanvasComposerFailure: Sendable, Equatable {
  public let message: String?
  public let nextAction: String

  public init(message: String?, nextAction: String) {
    self.message = message
    self.nextAction = nextAction
  }
}
