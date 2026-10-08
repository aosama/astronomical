import Foundation

/// Typed rejection from adaptive RAM growth admission
/// (port of `budget/adaptive_growth.rs::AdaptiveRamGrowthGuardError`).
public enum AdaptiveRamGrowthGuardError: Error, Equatable, Sendable {

    /// Adaptive RAM growth requires a positive active-memory limit.
    case invalidActiveMemoryCeiling

    /// Adaptive RAM growth memory projection overflowed.
    case memoryProjectionOverflow
}
