import Foundation

/// Rescales observed transient windows to different forward token counts
/// (port of `budget/adaptive_growth_projection.rs::scale_transient_to_token_count`).
enum AdaptiveRamGrowthTransientReserveScaling {

    /**
     * Rescales an observed transient window to a different forward token count.
     *
     * Activation workspace grows with the number of tokens in the forward, so
     * the estimate is proportional with ceiling division (conservative).
     * Returns `nil` when the observation carries a zero token count, which
     * cannot bound a proportional estimate; such observations still participate
     * through the global-maximum fallback. The multiply runs in wide
     * arithmetic because the intermediate product may exceed the word size
     * even when the final scaled bytes fit.
     *
     * - Parameters:
     *   - observedHighWaterBytes: Transient high-water bytes the observation completed at.
     *   - observedForwardTokenCount: Token count of the completed observation.
     *   - targetForwardTokenCount: Token count of the forward being admitted.
     * - Returns: Ceiling-divided proportional bytes, or `nil` when either token
     *   count is zero or the scaled bytes exceed the word size.
     */
    internal static func scaleToTokenCount(
        observedHighWaterBytes: Int,
        observedForwardTokenCount: Int,
        targetForwardTokenCount: Int
    ) -> Int? {
        if (observedForwardTokenCount == 0) || (targetForwardTokenCount == 0) {
            return nil
        }
        let observedBytesWide: Int128 = Int128(observedHighWaterBytes)
        let targetTokenCountWide: Int128 = Int128(targetForwardTokenCount)
        let observedTokenCountWide: Int128 = Int128(observedForwardTokenCount)
        let scaledBytesWide: Int128 = (
            observedBytesWide * targetTokenCountWide
                + observedTokenCountWide - 1
        ) / observedTokenCountWide
        return Int(exactly: scaledBytesWide)
    }
}
