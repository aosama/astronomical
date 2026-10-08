import Foundation

/// Architecture-neutral sliding-window causal visibility, port of the
/// Rust `attention::sliding_window_visibility` module.
///
/// A query at absolute position `queryAbsolutePosition` may attend a key
/// at absolute position `keyAbsolutePosition` only when
/// `queryAbsolutePosition >= keyAbsolutePosition` and the key stays inside
/// the trailing window. Positions are absolute token indices, never local
/// query ranks.
public enum SlidingWindowVisibility {

    /// Returns whether one absolute query position may attend one absolute
    /// key position.
    public static func slidingWindowPositionIsVisible(
        queryAbsolutePosition: UInt32,
        keyAbsolutePosition: UInt32,
        windowSize: UInt32
    ) throws -> Bool {
        if windowSize == 0 {
            throw SlidingWindowVisibilityError.zeroWindowSize
        }
        let (windowUpperBound, additionOverflowed) =
            keyAbsolutePosition.addingReportingOverflow(windowSize)
        let saturatedWindowUpperBound = additionOverflowed
            ? UInt32.max
            : windowUpperBound
        return queryAbsolutePosition >= keyAbsolutePosition
            && queryAbsolutePosition < saturatedWindowUpperBound
    }

    /// Builds a row-major visibility table from absolute query and key ranges.
    public static func slidingWindowVisibilityTable(
        firstQueryAbsolutePosition: UInt32,
        queryTokenCount: UInt32,
        firstKeyAbsolutePosition: UInt32,
        keyTokenCount: UInt32,
        windowSize: UInt32
    ) throws -> [[Bool]] {
        if windowSize == 0 {
            throw SlidingWindowVisibilityError.zeroWindowSize
        }
        if queryTokenCount == 0 {
            throw SlidingWindowVisibilityError.zeroTokenCount(
                description: "query token count must be positive")
        }
        if keyTokenCount == 0 {
            throw SlidingWindowVisibilityError.zeroTokenCount(
                description: "key token count must be positive")
        }
        var visibilityRows: [[Bool]] = []
        visibilityRows.reserveCapacity(Int(queryTokenCount))
        for queryOffset in 0..<queryTokenCount {
            let (queryAbsolutePosition, queryOverflowed) =
                firstQueryAbsolutePosition.addingReportingOverflow(queryOffset)
            var visibilityColumns: [Bool] = []
            visibilityColumns.reserveCapacity(Int(keyTokenCount))
            for keyOffset in 0..<keyTokenCount {
                let (keyAbsolutePosition, keyOverflowed) =
                    firstKeyAbsolutePosition.addingReportingOverflow(keyOffset)
                visibilityColumns.append(try slidingWindowPositionIsVisible(
                    queryAbsolutePosition: queryOverflowed ? UInt32.max : queryAbsolutePosition,
                    keyAbsolutePosition: keyOverflowed ? UInt32.max : keyAbsolutePosition,
                    windowSize: windowSize))
            }
            visibilityRows.append(visibilityColumns)
        }
        return visibilityRows
    }
}
