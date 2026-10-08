import Foundation

/**
 * Owns fixed Qwen3.5 chunk sizing and deterministic memory-capacity
 * reduction, port of the Rust `Qwen3_5PromptProcessingChunkSizer`.
 */
public struct Qwen35PromptProcessingChunkSizer: Equatable, Sendable {

    private let fixedPromptProcessingChunkSizeTokens: Int

    private let ssdStreamingPromptProcessingChunkSizeTokens: Int

    public init(fixedPromptProcessingChunkSizeTokens: UInt32) throws {
        try self.init(
            fixedPromptProcessingChunkSizeTokens: fixedPromptProcessingChunkSizeTokens,
            fixedSsdStreamingPromptProcessingChunkSizeTokens: fixedPromptProcessingChunkSizeTokens)
    }

    /// Paged experts use a separate chunk so complete-layer SSD reads can
    /// be amortized.
    public init(
        fixedPromptProcessingChunkSizeTokens: UInt32,
        fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32
    ) throws {
        self.fixedPromptProcessingChunkSizeTokens = try Self
            .promptProcessingChunkSizeTokens(from: fixedPromptProcessingChunkSizeTokens)
        self.ssdStreamingPromptProcessingChunkSizeTokens = try Self
            .promptProcessingChunkSizeTokens(from: fixedSsdStreamingPromptProcessingChunkSizeTokens)
    }

    public func nextPromptProcessingChunkEnd(
        chunkStartTokenPosition: Int,
        finalPromptEndTokenPositionExclusive: Int
    ) -> Int {
        nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: chunkStartTokenPosition,
            finalPromptEndTokenPositionExclusive: finalPromptEndTokenPositionExclusive,
            sparseExpertsArePaged: false)
    }

    /// The largest token count any single prompt-processing forward can span.
    ///
    /// Paged experts stream with their own chunk size, so the operation bound
    /// for memory planning is the larger of the two configured sizes (issue
    /// #644: activation reserves are operation-scoped and need the operation
    /// bound, not the prompt length).
    public var maximumPromptProcessingChunkSizeTokens: Int {
        max(fixedPromptProcessingChunkSizeTokens, ssdStreamingPromptProcessingChunkSizeTokens)
    }

    public func nextPromptProcessingChunkEndForExpertResidency(
        chunkStartTokenPosition: Int,
        finalPromptEndTokenPositionExclusive: Int,
        sparseExpertsArePaged: Bool
    ) -> Int {
        nextPromptProcessingChunkEndWithMaximumExecutableCapacity(
            chunkStartTokenPosition: chunkStartTokenPosition,
            finalPromptEndTokenPositionExclusive: finalPromptEndTokenPositionExclusive,
            sparseExpertsArePaged: sparseExpertsArePaged,
            maximumExecutableChunkSizeTokens: Int.max)
    }

    public func nextPromptProcessingChunkEndWithMaximumExecutableCapacity(
        chunkStartTokenPosition: Int,
        finalPromptEndTokenPositionExclusive: Int,
        sparseExpertsArePaged: Bool,
        maximumExecutableChunkSizeTokens: Int
    ) -> Int {
        let configuredChunkSizeTokens = sparseExpertsArePaged
            ? ssdStreamingPromptProcessingChunkSizeTokens
            : fixedPromptProcessingChunkSizeTokens
        let executableChunkSizeTokens = max(
            min(configuredChunkSizeTokens, maximumExecutableChunkSizeTokens),
            1)
        if sparseExpertsArePaged {
            return Self.pagedChunkEndAfterFoldingShortRemainder(
                chunkStartTokenPosition: chunkStartTokenPosition,
                finalPromptEndTokenPositionExclusive: finalPromptEndTokenPositionExclusive,
                executableChunkSizeTokens: executableChunkSizeTokens)
        }
        return min(
            Self.saturatingAdd(chunkStartTokenPosition, executableChunkSizeTokens),
            finalPromptEndTokenPositionExclusive)
    }

    /// Plans every remaining activation chunk without durable-cache clamping.
    public func remainingPromptProcessingChunkRanges(
        chunkStartTokenPosition: Int,
        finalPromptEndTokenPositionExclusive: Int,
        sparseExpertsArePaged: Bool,
        maximumExecutableChunkSizeTokens: Int
    ) -> [(chunkStartTokenPosition: Int, chunkEndTokenPositionExclusive: Int)] {
        var remainingChunkRanges:
            [(chunkStartTokenPosition: Int, chunkEndTokenPositionExclusive: Int)] = []
        var currentChunkStartTokenPosition = chunkStartTokenPosition
        while currentChunkStartTokenPosition < finalPromptEndTokenPositionExclusive {
            let currentChunkEndTokenPosition =
                nextPromptProcessingChunkEndWithMaximumExecutableCapacity(
                    chunkStartTokenPosition: currentChunkStartTokenPosition,
                    finalPromptEndTokenPositionExclusive: finalPromptEndTokenPositionExclusive,
                    sparseExpertsArePaged: sparseExpertsArePaged,
                    maximumExecutableChunkSizeTokens: maximumExecutableChunkSizeTokens)
            if currentChunkEndTokenPosition <= currentChunkStartTokenPosition {
                break
            }
            remainingChunkRanges.append(
                (currentChunkStartTokenPosition, currentChunkEndTokenPosition))
            currentChunkStartTokenPosition = currentChunkEndTokenPosition
        }
        return remainingChunkRanges
    }

    /// Halving provides bounded deterministic recovery without runtime learning state.
    public static func nextSmallerExecutableChunkSizeTokens(
        attemptedChunkSizeTokens: Int
    ) -> Int? {
        let smallerChunkSizeTokens = attemptedChunkSizeTokens / 2
        if smallerChunkSizeTokens == 0 {
            return nil
        }
        return smallerChunkSizeTokens
    }

    private static func promptProcessingChunkSizeTokens(
        from configuredChunkSizeTokens: UInt32
    ) throws -> Int {
        let chunkSizeTokens = Int(configuredChunkSizeTokens)
        if chunkSizeTokens == 0 {
            throw Qwen35PromptProcessingChunkSizerError.mustBePositive
        }
        return chunkSizeTokens
    }

    /// Each paged prefill forward streams every unseated complete MoE layer.
    /// Fold a trailing stub smaller than one configured chunk into this
    /// forward so that stub does not pay a second leftover-layer SSD sweep.
    private static func pagedChunkEndAfterFoldingShortRemainder(
        chunkStartTokenPosition: Int,
        finalPromptEndTokenPositionExclusive: Int,
        executableChunkSizeTokens: Int
    ) -> Int {
        let remainingPromptTokenCount = max(
            finalPromptEndTokenPositionExclusive - chunkStartTokenPosition,
            0)
        if remainingPromptTokenCount <= executableChunkSizeTokens {
            return saturatingAdd(chunkStartTokenPosition, remainingPromptTokenCount)
        }
        let remainderAfterFullChunk = remainingPromptTokenCount - executableChunkSizeTokens
        if remainderAfterFullChunk < executableChunkSizeTokens {
            return saturatingAdd(chunkStartTokenPosition, remainingPromptTokenCount)
        }
        return saturatingAdd(chunkStartTokenPosition, executableChunkSizeTokens)
    }

    /// Token positions arrive from untrusted request state, so the mirror of
    /// the Rust `saturating_add` keeps the sizer total instead of trapping.
    private static func saturatingAdd(_ baseTokenPosition: Int, _ tokenCount: Int) -> Int {
        let (summedTokenPosition, additionOverflowed) =
            baseTokenPosition.addingReportingOverflow(tokenCount)
        if additionOverflowed {
            return Int.max
        }
        return summedTokenPosition
    }
}
