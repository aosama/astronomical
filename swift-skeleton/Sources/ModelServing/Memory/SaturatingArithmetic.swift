import Foundation

/// Saturating and checked 64-bit arithmetic shared by memory accounting.
///
/// Cache policies must never wrap byte counters: saturating at
/// `UInt64.max` keeps telemetry finite while overflow checks on commit
/// paths reject impossible projections instead of silently shrinking them.
internal enum SaturatingArithmetic {

    /// Adds two byte counts, saturating at `UInt64.max`.
    internal static func add(_ augend: UInt64, _ addend: UInt64) -> UInt64 {
        let (summedTotal, addOverflowed) = augend.addingReportingOverflow(addend)
        if addOverflowed {
            return UInt64.max
        }
        return summedTotal
    }

    /// Multiplies two byte counts, saturating at `UInt64.max`.
    internal static func multiply(_ multiplicand: UInt64, _ multiplier: UInt64) -> UInt64 {
        let (product, multiplyOverflowed) = multiplicand.multipliedReportingOverflow(by: multiplier)
        if multiplyOverflowed {
            return UInt64.max
        }
        return product
    }

    /// Subtracts a byte count, saturating at zero.
    internal static func subtract(_ minuend: UInt64, _ subtrahend: UInt64) -> UInt64 {
        let (difference, subtractOverflowed) = minuend.subtractingReportingOverflow(subtrahend)
        if subtractOverflowed {
            return 0
        }
        return difference
    }

    /// Subtracts a signed count, saturating at zero (word-size byte math).
    internal static func subtractInt(_ minuend: Int, _ subtrahend: Int) -> Int {
        let (difference, subtractOverflowed) = minuend.subtractingReportingOverflow(subtrahend)
        if subtractOverflowed || difference < 0 {
            return 0
        }
        return difference
    }
}
