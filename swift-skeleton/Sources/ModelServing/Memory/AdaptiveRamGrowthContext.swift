import Foundation

/**
 * Exact recurrent execution shape whose temporary allocation evidence may
 * recur (port of `budget/adaptive_growth.rs`).
 *
 * Observations are deliberately not transferable between chunk sizes, prompt
 * positions, visual requests, or sparse-expert residency modes.
 */
public struct AdaptiveRamGrowthContext: Equatable, Hashable, Sendable {

    private let contextMemoryPhase: MemoryPhase
    private let contextForwardTokenCount: Int
    private let contextPromptPositionBucket: Int
    private let contextHasVisualEmbeddings: Bool
    private let contextSparseExpertsArePaged: Bool

    internal init(
        memoryPhase: MemoryPhase,
        forwardTokenCount: Int,
        promptPositionContextBucket: Int,
        hasVisualEmbeddings: Bool,
        sparseExpertsArePaged: Bool
    ) {
        self.contextMemoryPhase = memoryPhase
        self.contextForwardTokenCount = forwardTokenCount
        self.contextPromptPositionBucket = promptPositionContextBucket
        self.contextHasVisualEmbeddings = hasVisualEmbeddings
        self.contextSparseExpertsArePaged = sparseExpertsArePaged
    }

    /**
     * Builds a prefill context using the prompt sizer's position bucket.
     *
     * - Parameters:
     *   - forwardTokenCount: Exact number of tokens this operation forwards.
     *   - promptPositionContextBucket: Prompt-position bucket the sizer assigned.
     *   - hasVisualEmbeddings: Whether the forward carries visual tokens.
     *   - sparseExpertsArePaged: Whether sparse experts are paged for the forward.
     * - Returns: A prefill-phase execution context.
     */
    public static func prefill(
        forwardTokenCount: Int,
        promptPositionContextBucket: Int,
        hasVisualEmbeddings: Bool,
        sparseExpertsArePaged: Bool
    ) -> AdaptiveRamGrowthContext {
        return AdaptiveRamGrowthContext(
            memoryPhase: MemoryPhase.prefill,
            forwardTokenCount: forwardTokenCount,
            promptPositionContextBucket: promptPositionContextBucket,
            hasVisualEmbeddings: hasVisualEmbeddings,
            sparseExpertsArePaged: sparseExpertsArePaged)
    }

    /**
     * Builds a decode context. Decode has no prompt-chunk position bucket.
     *
     * - Parameters:
     *   - forwardTokenCount: Exact number of tokens this operation forwards.
     *   - sparseExpertsArePaged: Whether sparse experts are paged for the forward.
     * - Returns: A decode-phase execution context.
     */
    public static func decode(
        forwardTokenCount: Int,
        sparseExpertsArePaged: Bool
    ) -> AdaptiveRamGrowthContext {
        return AdaptiveRamGrowthContext(
            memoryPhase: MemoryPhase.decode,
            forwardTokenCount: forwardTokenCount,
            promptPositionContextBucket: 0,
            hasVisualEmbeddings: false,
            sparseExpertsArePaged: sparseExpertsArePaged)
    }

    /// Request lifecycle position the evidence was learned in.
    public var memoryPhase: MemoryPhase {
        return self.contextMemoryPhase
    }

    /// The exact number of tokens forwarded by this operation.
    public var forwardTokenCount: Int {
        return self.contextForwardTokenCount
    }

    /// Prompt-position bucket the prompt sizer assigned to this forward.
    public var promptPositionContextBucket: Int {
        return self.contextPromptPositionBucket
    }

    /// Whether the forward carries visual embeddings.
    public var hasVisualEmbeddings: Bool {
        return self.contextHasVisualEmbeddings
    }

    /// Whether sparse experts are paged for this forward.
    public var sparseExpertsArePaged: Bool {
        return self.contextSparseExpertsArePaged
    }

    /**
     * Replaces only the observed sparse-expert residency dimension after a forward.
     *
     * - Parameter sparseExpertsArePaged: The residency mode actually observed.
     * - Returns: A copy of this context with the residency dimension replaced.
     */
    public func withSparseExpertsArePaged(
        _ sparseExpertsArePaged: Bool
    ) -> AdaptiveRamGrowthContext {
        return AdaptiveRamGrowthContext(
            memoryPhase: self.contextMemoryPhase,
            forwardTokenCount: self.contextForwardTokenCount,
            promptPositionContextBucket: self.contextPromptPositionBucket,
            hasVisualEmbeddings: self.contextHasVisualEmbeddings,
            sparseExpertsArePaged: sparseExpertsArePaged)
    }
}
