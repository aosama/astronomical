import Foundation;

/// Arithmetic for aligning prompt processing with durable cache boundaries,
/// port of the Rust `prefill_boundary` module. Cache-enabled prefill is
/// clamped to one boundary per forward: that keeps the captured decoder
/// state and token slice at the same exact point, and ensures a required
/// synchronous publication succeeds before processing can advance.
public enum PersistentPromptCachePrefillBoundary {

    /// Returns local completed-token counts for every persistent
    /// prompt-cache boundary crossed by one attempted prefill forward.
    public static func completedPrefillChunkTokens(
        prefillChunkStart: Int,
        prefillChunkEnd: Int,
        persistentPromptCacheBlockTokenCount: Int
    ) -> [Int] {
        if prefillChunkEnd <= prefillChunkStart || persistentPromptCacheBlockTokenCount == 0 {
            return [];
        }
        let completedPersistentPromptCacheBlockCount: Int =
            prefillChunkStart / persistentPromptCacheBlockTokenCount;
        let (nextBlockCount, blockCountOverflow) = completedPersistentPromptCacheBlockCount
            .addingReportingOverflow(1);
        if blockCountOverflow {
            return [];
        }
        let (firstBoundary, boundaryOverflow) = nextBlockCount
            .multipliedReportingOverflow(by: persistentPromptCacheBlockTokenCount);
        if boundaryOverflow {
            return [];
        }
        var absolutePersistentPromptCacheBoundary: Int = firstBoundary;
        var completedPrefillChunkTokens: [Int] = [];
        while absolutePersistentPromptCacheBoundary <= prefillChunkEnd {
            completedPrefillChunkTokens.append(
                absolutePersistentPromptCacheBoundary - prefillChunkStart);
            let (nextBoundary, nextOverflow) = absolutePersistentPromptCacheBoundary
                .addingReportingOverflow(persistentPromptCacheBlockTokenCount);
            if nextOverflow {
                break;
            }
            absolutePersistentPromptCacheBoundary = nextBoundary;
        }
        return completedPrefillChunkTokens;
    }

    /// Clamps an attempted cache-enabled prefill chunk so it can publish at
    /// most one mandatory persistent prompt-cache boundary.
    public static func clampedPrefillChunkEnd(
        prefillChunkStart: Int,
        requestedPrefillChunkEnd: Int,
        persistentPromptCacheBlockTokenCount: Int
    ) -> Int {
        if requestedPrefillChunkEnd <= prefillChunkStart
            || persistentPromptCacheBlockTokenCount == 0 {
            return requestedPrefillChunkEnd;
        }
        // Integer division identifies the block containing the current
        // cursor; the next multiple is the earliest boundary this forward is
        // allowed to cross.
        let completedBlockCount: Int =
            prefillChunkStart / persistentPromptCacheBlockTokenCount;
        let (nextBoundaryBlockCount, blockCountOverflow) = completedBlockCount
            .addingReportingOverflow(1);
        if blockCountOverflow {
            return requestedPrefillChunkEnd;
        }
        let (nextPersistentPromptCacheBoundary, boundaryOverflow) = nextBoundaryBlockCount
            .multipliedReportingOverflow(by: persistentPromptCacheBlockTokenCount);
        if boundaryOverflow {
            return requestedPrefillChunkEnd;
        }
        return min(requestedPrefillChunkEnd, nextPersistentPromptCacheBoundary);
    }
}
