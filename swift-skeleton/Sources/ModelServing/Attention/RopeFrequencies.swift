import Foundation

/// Architecture-neutral rotary frequency denominators for MLX
/// `fast.rope`, port of the Rust `attention::yarn_frequencies` module.
///
/// MLX takes the reciprocal of the supplied `freqs` array. This owner
/// therefore emits denominators `theta^(2i/d)`, never inverse frequencies.
/// YaRN blends these denominators while family owners retain their own
/// attention factor.
public enum RopeFrequencies {

    /// Builds default RoPE denominators `theta^(2i/d)` for `i` in `[0, d/2)`.
    public static func computeDefaultRopeFrequencyDenominators(
        theta: Double,
        rotaryDimension: UInt32
    ) throws -> [Float] {
        try validateRotaryDimension(rotaryDimension)
        try validateTheta(theta)
        let pairCount = rotaryDimension / 2
        var frequencyDenominators: [Float] = []
        frequencyDenominators.reserveCapacity(Int(pairCount))
        for pairIndex in 0..<pairCount {
            frequencyDenominators.append(Float(defaultFrequencyDenominator(
                theta: theta,
                rotaryDimension: rotaryDimension,
                pairIndex: pairIndex)))
        }
        return frequencyDenominators
    }

    /// Builds YaRN-blended frequency denominators for MLX `fast.rope`.
    public static func computeYarnRopeFrequencyDenominators(
        theta: Double,
        rotaryDimension: UInt32,
        originalMaximumPositionCount: UInt32,
        factor: Double,
        betaFast: Double,
        betaSlow: Double
    ) throws -> YarnRopeFrequencyDenominators {
        try validateRotaryDimension(rotaryDimension)
        try validateTheta(theta)
        if factor.isFinite == false || factor <= 0.0 {
            throw RopeFrequencyError.invalidFactor(
                factor: factor,
                description: "YaRN factor must be a positive finite value")
        }
        if originalMaximumPositionCount == 0 {
            throw RopeFrequencyError.invalidOriginalMaximumPositionCount(
                originalMaximumPositionCount: originalMaximumPositionCount,
                description: "original maximum position count must be positive")
        }
        if betaFast.isFinite == false || betaFast <= 0.0
            || betaSlow.isFinite == false || betaSlow <= 0.0 {
            throw RopeFrequencyError.invalidBeta(
                description: "beta_fast and beta_slow must be positive finite rotation counts")
        }
        if betaFast < betaSlow {
            throw RopeFrequencyError.invalidBeta(
                description: "beta_fast must be greater than or equal to beta_slow")
        }

        let (rampLow, rampHigh) = yarnCorrectionRange(
            rotaryDimension: rotaryDimension,
            originalMaximumPositionCount: originalMaximumPositionCount,
            theta: theta,
            betaFast: betaFast,
            betaSlow: betaSlow)
        let pairCount = rotaryDimension / 2
        var frequencyDenominators: [Float] = []
        frequencyDenominators.reserveCapacity(Int(pairCount))
        for pairIndex in 0..<pairCount {
            let extraDenominator = defaultFrequencyDenominator(
                theta: theta,
                rotaryDimension: rotaryDimension,
                pairIndex: pairIndex)
            let interpolatedDenominator = factor * extraDenominator
            let keepUnscaledWeight = 1.0 - yarnLinearRamp(
                pairIndex: Double(pairIndex),
                rampLow: rampLow,
                rampHigh: rampHigh)
            // Harmonic blending preserves the published inverse-frequency ramp.
            let blendedDenominator = interpolatedDenominator * keepUnscaledWeight
                + extraDenominator * (1.0 - keepUnscaledWeight)
            frequencyDenominators.append(Float(
                (interpolatedDenominator * extraDenominator) / blendedDenominator))
        }
        return YarnRopeFrequencyDenominators(frequencyDenominators: frequencyDenominators)
    }

    private static func defaultFrequencyDenominator(
        theta: Double,
        rotaryDimension: UInt32,
        pairIndex: UInt32
    ) -> Double {
        return pow(theta, 2.0 * Double(pairIndex) / Double(rotaryDimension))
    }

    private static func yarnCorrectionRange(
        rotaryDimension: UInt32,
        originalMaximumPositionCount: UInt32,
        theta: Double,
        betaFast: Double,
        betaSlow: Double
    ) -> (rampLow: Double, rampHigh: Double) {
        let rampLow = max(floor(yarnCorrectionDimension(
            rotationCount: betaFast,
            rotaryDimension: rotaryDimension,
            originalMaximumPositionCount: originalMaximumPositionCount,
            theta: theta)), 0.0)
        let rampHigh = min(ceil(yarnCorrectionDimension(
            rotationCount: betaSlow,
            rotaryDimension: rotaryDimension,
            originalMaximumPositionCount: originalMaximumPositionCount,
            theta: theta)), Double(rotaryDimension - 1))
        return (rampLow, rampHigh)
    }

    private static func yarnCorrectionDimension(
        rotationCount: Double,
        rotaryDimension: UInt32,
        originalMaximumPositionCount: UInt32,
        theta: Double
    ) -> Double {
        return Double(rotaryDimension)
            * log(Double(originalMaximumPositionCount) / (rotationCount * 2.0 * Double.pi))
            / (2.0 * log(theta))
    }

    private static func yarnLinearRamp(
        pairIndex: Double,
        rampLow: Double,
        rampHigh: Double
    ) -> Double {
        var effectiveHigh = rampHigh
        if abs(effectiveHigh - rampLow) < Double.ulpOfOne {
            effectiveHigh += 0.001
        }
        return min(max((pairIndex - rampLow) / (effectiveHigh - rampLow), 0.0), 1.0)
    }

    private static func validateRotaryDimension(_ rotaryDimension: UInt32) throws {
        if rotaryDimension == 0 {
            throw RopeFrequencyError.invalidRotaryDimension(
                rotaryDimension: rotaryDimension,
                description: "rotary dimension must be positive")
        }
        if rotaryDimension % 2 != 0 {
            throw RopeFrequencyError.invalidRotaryDimension(
                rotaryDimension: rotaryDimension,
                description: "rotary dimension must be even")
        }
    }

    private static func validateTheta(_ theta: Double) throws {
        if theta.isFinite == false || theta <= 1.0 {
            throw RopeFrequencyError.invalidTheta(
                theta: theta,
                description: "theta must be a finite value greater than one")
        }
    }
}
