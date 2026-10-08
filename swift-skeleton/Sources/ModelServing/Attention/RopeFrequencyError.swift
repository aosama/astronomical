import Foundation

/// Errors from rotary frequency-denominator construction, port of the
/// Rust `RopeFrequencyError`.
public enum RopeFrequencyError: Error, Equatable, Sendable {

    /// Rotary width was zero or odd.
    case invalidRotaryDimension(rotaryDimension: UInt32, description: String)

    /// Theta was not a finite value greater than one.
    case invalidTheta(theta: Double, description: String)

    /// The context-extension factor was not a positive finite value.
    case invalidFactor(factor: Double, description: String)

    /// The original training context was zero.
    case invalidOriginalMaximumPositionCount(
        originalMaximumPositionCount: UInt32, description: String)

    /// A YaRN rotation-count threshold was invalid.
    case invalidBeta(description: String)
}
