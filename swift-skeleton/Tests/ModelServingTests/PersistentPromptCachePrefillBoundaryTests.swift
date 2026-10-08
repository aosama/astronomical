import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the prefill/cache boundary arithmetic: every
/// boundary one attempted chunk crosses is reported in local token counts,
/// and a cache-enabled chunk is clamped so a forward can publish at most one
/// mandatory boundary.
final class PersistentPromptCachePrefillBoundaryTests {

    @Test
    func should_report_every_persistent_prompt_cache_boundary_crossed_by_one_prefill_chunk() {
        let boundaryCases: Array<(
            prefillChunkStart: Int,
            prefillChunkEnd: Int,
            persistentPromptCacheBlockTokenCount: Int,
            expectedCompletedPrefillChunkTokens: [Int])> = [
                (0, 128, 2_048, []),
                (0, 2_048, 2_048, [2_048]),
                (0, 4_096, 2_048, [2_048, 4_096]),
                (128, 4_096, 2_048, [1_920, 3_968]),
                (22_528, 26_624, 2_048, [2_048, 4_096]),
                (22_528, 25_000, 2_048, [2_048]),
                (0, 1_536, 512, [512, 1_024, 1_536]),
                (2_048, 2_048, 2_048, []),
                (4_096, 2_048, 2_048, []),
                (0, 4_096, 0, []),
                (Int.max - 1_024, Int.max, 2_048, []),
            ];
        for boundaryCase in boundaryCases {
            #expect(
                PersistentPromptCachePrefillBoundary.completedPrefillChunkTokens(
                    prefillChunkStart: boundaryCase.prefillChunkStart,
                    prefillChunkEnd: boundaryCase.prefillChunkEnd,
                    persistentPromptCacheBlockTokenCount:
                        boundaryCase.persistentPromptCacheBlockTokenCount)
                    == boundaryCase.expectedCompletedPrefillChunkTokens);
        }
    }

    @Test
    func should_clamp_cache_enabled_prefill_to_the_next_persistent_boundary() {
        let clampCases: Array<(
            prefillChunkStart: Int,
            requestedPrefillChunkEnd: Int,
            persistentPromptCacheBlockTokenCount: Int,
            expectedPrefillChunkEnd: Int)> = [
                (0, 128, 2_048, 128),
                (0, 4_096, 2_048, 2_048),
                (128, 4_096, 2_048, 2_048),
                (2_048, 8_192, 2_048, 4_096),
                (2_048, 2_048, 2_048, 2_048),
                (4_096, 2_048, 2_048, 2_048),
                (0, 4_096, 0, 4_096),
                (Int.max - 1_024, Int.max, 2_048, Int.max),
            ];
        for clampCase in clampCases {
            #expect(
                PersistentPromptCachePrefillBoundary.clampedPrefillChunkEnd(
                    prefillChunkStart: clampCase.prefillChunkStart,
                    requestedPrefillChunkEnd: clampCase.requestedPrefillChunkEnd,
                    persistentPromptCacheBlockTokenCount:
                        clampCase.persistentPromptCacheBlockTokenCount)
                    == clampCase.expectedPrefillChunkEnd);
        }
    }
}
