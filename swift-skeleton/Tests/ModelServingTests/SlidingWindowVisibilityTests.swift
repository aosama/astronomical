import Foundation

import Testing

import ModelServing

/// Behavior coverage for architecture-neutral sliding-window causal
/// visibility, port of
/// crates/model-serving/tests/hermetic/attention/sliding_window_visibility.rs.
@Suite
final class SlidingWindowVisibilityTests {

    @Test
    func shouldHideFutureKeysAndKeysOutsideTheAbsoluteWindow() throws {
        #expect(try SlidingWindowVisibility.slidingWindowPositionIsVisible(
            queryAbsolutePosition: 10, keyAbsolutePosition: 10, windowSize: 4))
        #expect(try SlidingWindowVisibility.slidingWindowPositionIsVisible(
            queryAbsolutePosition: 10, keyAbsolutePosition: 7, windowSize: 4))
        #expect(try SlidingWindowVisibility.slidingWindowPositionIsVisible(
            queryAbsolutePosition: 10, keyAbsolutePosition: 6, windowSize: 4) == false)
        #expect(try SlidingWindowVisibility.slidingWindowPositionIsVisible(
            queryAbsolutePosition: 10, keyAbsolutePosition: 11, windowSize: 4) == false)
    }

    @Test
    func shouldBuildAPrefixPlusChunkVisibilityTable() throws {
        let visibilityTable = try SlidingWindowVisibility.slidingWindowVisibilityTable(
            firstQueryAbsolutePosition: 6,
            queryTokenCount: 4,
            firstKeyAbsolutePosition: 0,
            keyTokenCount: 10,
            windowSize: 4)

        #expect(
            visibilityTable[0]
                == [false, false, false, true, true, true, true, false, false, false])
        #expect(
            visibilityTable[3]
                == [false, false, false, false, false, false, true, true, true, true])
    }

    @Test
    func shouldRejectZeroWindowOrTokenCounts() {
        #expect(throws: SlidingWindowVisibilityError.zeroWindowSize) {
            try SlidingWindowVisibility.slidingWindowPositionIsVisible(
                queryAbsolutePosition: 1, keyAbsolutePosition: 0, windowSize: 0)
        }
        #expect(throws: SlidingWindowVisibilityError.zeroTokenCount(
            description: "query token count must be positive")) {
            try SlidingWindowVisibility.slidingWindowVisibilityTable(
                firstQueryAbsolutePosition: 0,
                queryTokenCount: 0,
                firstKeyAbsolutePosition: 0,
                keyTokenCount: 1,
                windowSize: 4)
        }
        #expect(throws: SlidingWindowVisibilityError.zeroTokenCount(
            description: "key token count must be positive")) {
            try SlidingWindowVisibility.slidingWindowVisibilityTable(
                firstQueryAbsolutePosition: 0,
                queryTokenCount: 1,
                firstKeyAbsolutePosition: 0,
                keyTokenCount: 0,
                windowSize: 4)
        }
    }
}
