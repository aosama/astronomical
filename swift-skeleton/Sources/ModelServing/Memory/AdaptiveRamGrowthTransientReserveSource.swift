import Foundation

/// Which evidence level supplied a forward admission's transient reserve
/// (port of `budget/adaptive_growth_projection.rs`).
///
/// Recorded on the projection and in admission-decision logs so a demotion can
/// be attributed to the reserve that caused it (issue #623).
public enum AdaptiveRamGrowthTransientReserveSource: Equatable, Hashable, Sendable {

    /// The same phase, token count, position bucket, and ownership mode
    /// completed a forward before.
    case exactContext

    /// Same-phase observations rescaled to this forward's token count.
    case phaseScaled

    /// The phase had no evidence; the largest window ever observed in any
    /// phase was reserved.
    case globalMaximum
}
