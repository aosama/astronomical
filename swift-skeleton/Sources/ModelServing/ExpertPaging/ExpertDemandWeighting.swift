import Foundation

/// Demand-evidence multipliers for expert paging decisions.
public enum ExpertDemandWeighting {

    /**
     * Scales last-chunk assignments so their token density matches earlier
     * prefill. Decode continues from the prompt tail, so one last-chunk
     * token counts as `earlier / last_chunk` assignments, floored at one.
     * This keeps prompt-tail coverage useful at the transition to decode
     * without turning demand into speculative I/O.
     *
     * - Parameters:
     *   - earlierPrefillTokenCount: Tokens prefilled before the last chunk.
     *   - lastPrefillChunkTokenCount: Tokens in the final prefill chunk.
     * - Returns: The per-assignment demand weight, at least one.
     */
    public static func lastPrefillChunkDemandWeight(
        earlierPrefillTokenCount: UInt64,
        lastPrefillChunkTokenCount: UInt64
    ) -> UInt64 {
        if earlierPrefillTokenCount == 0 || lastPrefillChunkTokenCount == 0 {
            return 1
        }
        let tokenDensityRatio: UInt64 = earlierPrefillTokenCount / lastPrefillChunkTokenCount
        return max(tokenDensityRatio, 1)
    }
}
