import Foundation;

/// Qwen-owned hard reasoning-budget state. The controller advances only from
/// tokens committed to the decoder; keeping forcing here prevents the public
/// token stream from diverging from the model's autoregressive history when a
/// reasoning allowance is exhausted.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/thinking_budget.rs.
public struct Qwen35ThinkingBudgetState: Sendable {

    private enum ThinkingBudgetPhase: Sendable {
        case visibleAnswer;
        case thinking;
        case forcingTransition;
    }

    /// The template-authored pivot text the model receives when its reasoning
    /// allowance is exhausted; the closing think token id is appended after
    /// these ids, so the complete forced transition ends at a real boundary.
    public static let thinkingBudgetTransitionText =
        "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

    private let thinkingBudget: UInt16?;
    private var thinkingTokenCountValue: UInt16 = 0;
    private var phase: ThinkingBudgetPhase;
    private let forcedTransitionTokenIds: Array<UInt32>;
    private let naturalReasoningEndTokenIds: Array<UInt32>;
    private var nextForcedTransitionTokenIndex: Int = 0;
    private var forcedTokenAwaitingCommit: UInt32?;

    /// Creates a state whose model-owned transition must end at a recognized
    /// reasoning boundary whenever a positive hard allowance is active.
    public init(
        startsInsideThinking: Bool,
        thinkingBudget: UInt16?,
        forcedTransitionTokenIds: Array<UInt32>,
        naturalReasoningEndTokenIds: Array<UInt32>
    ) throws {
        let hasPositiveThinkingBudget = thinkingBudget.map { $0 > 0 } ?? false;
        let startsInsideReasoning = startsInsideThinking && thinkingBudget != 0;
        if startsInsideThinking && hasPositiveThinkingBudget {
            guard let finalForcedTokenId = forcedTransitionTokenIds.last else {
                throw Qwen35ThinkingBudgetError.missingForcedTransition;
            }
            if naturalReasoningEndTokenIds.contains(finalForcedTokenId) == false {
                throw Qwen35ThinkingBudgetError.transitionDoesNotEndReasoning;
            }
            let leadingForcedTokenIds = forcedTransitionTokenIds.dropLast();
            if leadingForcedTokenIds.contains(where: { (tokenIds: UInt32) -> Bool in
                naturalReasoningEndTokenIds.contains(tokenIds)
            }) {
                throw Qwen35ThinkingBudgetError.transitionEndsReasoningEarly;
            }
        }
        self.thinkingBudget = startsInsideReasoning ? thinkingBudget : nil;
        self.phase = startsInsideReasoning ? .thinking : .visibleAnswer;
        self.forcedTransitionTokenIds = forcedTransitionTokenIds;
        self.naturalReasoningEndTokenIds = naturalReasoningEndTokenIds;
    }

    /// Selects the next forced token before ordinary sampling. The caller must
    /// feed this exact token through the decoder before observing its commit.
    public mutating func nextForcedTransitionTokenId() throws -> UInt32? {
        if self.phase != .forcingTransition {
            return nil;
        }
        if self.forcedTokenAwaitingCommit != nil {
            throw Qwen35ThinkingBudgetError.forcedTokenNotCommitted;
        }
        guard self.nextForcedTransitionTokenIndex < self.forcedTransitionTokenIds.count else {
            throw Qwen35ThinkingBudgetError.forcedTransitionExhausted;
        }
        let forcedTokenId = self.forcedTransitionTokenIds[self.nextForcedTransitionTokenIndex];
        self.forcedTokenAwaitingCommit = forcedTokenId;
        return forcedTokenId;
    }

    /// Records the exact token committed to decoder history and returns
    /// whether its decoded text belongs to reasoning rather than the visible
    /// answer.
    public mutating func observeCommittedToken(_ committedTokenId: UInt32) throws -> Bool {
        switch self.phase {
        case .visibleAnswer:
            return false;
        case .thinking:
            if self.isNaturalReasoningEnd(committedTokenId) {
                self.phase = .visibleAnswer;
                return false;
            }
            if let thinkingBudget = self.thinkingBudget {
                let (incrementedCount, didOverflow) = self.thinkingTokenCountValue
                    .addingReportingOverflow(1);
                if didOverflow {
                    throw Qwen35ThinkingBudgetError.tokenCountOverflow;
                }
                self.thinkingTokenCountValue = incrementedCount;
                if self.thinkingTokenCountValue >= thinkingBudget {
                    self.phase = .forcingTransition;
                }
            }
            return true;
        case .forcingTransition:
            guard let expectedForcedTokenId = self.forcedTokenAwaitingCommit else {
                throw Qwen35ThinkingBudgetError.forcedTokenWasNotSelected;
            }
            self.forcedTokenAwaitingCommit = nil;
            if committedTokenId != expectedForcedTokenId {
                throw Qwen35ThinkingBudgetError.forcedTokenMismatch(
                    expectedTokenId: expectedForcedTokenId, actualTokenId: committedTokenId);
            }
            self.nextForcedTransitionTokenIndex += 1;
            let endsReasoning = self.isNaturalReasoningEnd(committedTokenId);
            if endsReasoning {
                self.phase = .visibleAnswer;
            } else if self.nextForcedTransitionTokenIndex >= self.forcedTransitionTokenIds.count {
                throw Qwen35ThinkingBudgetError.transitionDoesNotEndReasoning;
            }
            return endsReasoning == false;
        }
    }

    public var isInsideThinking: Bool {
        return self.phase != .visibleAnswer;
    }

    public var isForcingTransition: Bool {
        return self.phase == .forcingTransition;
    }

    public var activeThinkingBudget: UInt16? {
        return self.thinkingBudget;
    }

    public var thinkingTokenCount: UInt16 {
        return self.thinkingTokenCountValue;
    }

    private func isNaturalReasoningEnd(_ tokenIds: UInt32) -> Bool {
        return self.naturalReasoningEndTokenIds.contains(tokenIds);
    }
}

/// A violated model-owned hard-budget state transition.
public enum Qwen35ThinkingBudgetError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingForcedTransition;
    case transitionDoesNotEndReasoning;
    case transitionEndsReasoningEarly;
    case forcedTokenNotCommitted;
    case forcedTransitionExhausted;
    case forcedTokenWasNotSelected;
    case forcedTokenMismatch(expectedTokenId: UInt32, actualTokenId: UInt32);
    case tokenCountOverflow;

    public var description: String {
        switch self {
        case .missingForcedTransition:
            return "positive thinking budget requires a forced reasoning transition";
        case .transitionDoesNotEndReasoning:
            return "forced reasoning transition does not end at a recognized reasoning boundary";
        case .transitionEndsReasoningEarly:
            return "forced reasoning transition contains a reasoning boundary before its final token";
        case .forcedTokenNotCommitted:
            return "the previous forced reasoning token was not committed";
        case .forcedTransitionExhausted:
            return "forced reasoning transition ended before its reasoning boundary";
        case .forcedTokenWasNotSelected:
            return "a model-selected token was committed while a forced reasoning token was required";
        case let .forcedTokenMismatch(expectedTokenId, actualTokenId):
            return "forced reasoning token mismatch: expected \(expectedTokenId), received \(actualTokenId)";
        case .tokenCountOverflow:
            return "thinking-budget token counter overflowed";
        }
    }
}

/// Thinking-allowance resolution against the caller's output budget. The
/// thinking allowance is a ceiling on reasoning, not a floor: the model may
/// finish reasoning earlier through the natural transition. When the caller's
/// own output cap cannot hold the requested allowance plus the transition
/// reservation plus one visible answer token, the allowance shrinks to fit
/// instead of rejecting a satisfiable request (#652).
public enum Qwen35ThinkingAllowance {

    /// Resolves the effective thinking allowance for one request: `nil` when
    /// no positive allowance is active, the requested allowance when it fits,
    /// and the shrunken allowance otherwise.
    public static func resolveEffectiveThinkingAllowance(
        requestedThinkingBudget: UInt16?,
        enableThinking: Bool,
        maximumOutputTokens: UInt16,
        transitionTokenCount: Int
    ) -> UInt16? {
        guard let thinkingBudget = requestedThinkingBudget else {
            return nil;
        }
        if enableThinking == false || thinkingBudget == 0 {
            return nil;
        }
        if let minimumBoundedOutputTokens = minimumBoundedOutputTokenCount(
            thinkingBudget: thinkingBudget, forcedTransitionTokenCount: transitionTokenCount),
            Int(maximumOutputTokens) >= minimumBoundedOutputTokens
        {
            return thinkingBudget;
        }
        let allowanceRoom = Int(maximumOutputTokens) - transitionTokenCount;
        let shrunkenAllowance = max(allowanceRoom - 1, 0);
        return UInt16(clamping: shrunkenAllowance);
    }

    /// The reservation the effective allowance demands from the output budget.
    public static func minimumBoundedOutputTokenCount(
        thinkingBudget: UInt16, forcedTransitionTokenCount: Int
    ) -> Int? {
        let boundedTokenCount = Int(thinkingBudget)
            .addingReportingOverflow(forcedTransitionTokenCount);
        if boundedTokenCount.overflow {
            return nil;
        }
        let reservedTokenCount = boundedTokenCount.partialValue.addingReportingOverflow(1);
        if reservedTokenCount.overflow {
            return nil;
        }
        return reservedTokenCount.partialValue;
    }
}
