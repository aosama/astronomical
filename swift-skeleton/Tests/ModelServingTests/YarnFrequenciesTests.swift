import Foundation

import Testing

import ModelServing

/// Behavior coverage for default and YaRN rotary frequency denominators,
/// port of
/// crates/model-serving/tests/hermetic/attention/yarn_frequencies.rs.
@Suite
final class YarnFrequenciesTests {

    @Test
    func shouldKeepHighFrequencyYarnPairsAndScaleLowFrequencyPairs() throws {
        let defaultDenominators = try RopeFrequencies.computeDefaultRopeFrequencyDenominators(
            theta: 500_000.0,
            rotaryDimension: 64)
        let yarnDenominators = try RopeFrequencies.computeYarnRopeFrequencyDenominators(
            theta: 500_000.0,
            rotaryDimension: 64,
            originalMaximumPositionCount: 8_192,
            factor: 32.0,
            betaFast: 32.0,
            betaSlow: 1.0)
        let blended = yarnDenominators.frequencyDenominators

        #expect(abs(blended[0] - defaultDenominators[0]) < 1e-5)
        let lastIndex = blended.count - 1
        let expectedScaled = defaultDenominators[lastIndex] * 32.0
        #expect(abs(blended[lastIndex] - expectedScaled) <= expectedScaled * 1e-5)
    }

    @Test
    func shouldRejectInvalidRotaryAndYarnGeometry() {
        #expect(throws: (any Error).self) {
            try RopeFrequencies.computeDefaultRopeFrequencyDenominators(
                theta: 10_000.0, rotaryDimension: 0)
        }
        #expect(throws: (any Error).self) {
            try RopeFrequencies.computeDefaultRopeFrequencyDenominators(
                theta: 10_000.0, rotaryDimension: 7)
        }
        #expect(throws: (any Error).self) {
            try RopeFrequencies.computeYarnRopeFrequencyDenominators(
                theta: 500_000.0,
                rotaryDimension: 64,
                originalMaximumPositionCount: 0,
                factor: 32.0,
                betaFast: 32.0,
                betaSlow: 1.0)
        }
        #expect(throws: (any Error).self) {
            try RopeFrequencies.computeYarnRopeFrequencyDenominators(
                theta: 500_000.0,
                rotaryDimension: 64,
                originalMaximumPositionCount: 8_192,
                factor: 32.0,
                betaFast: 1.0,
                betaSlow: 32.0)
        }
    }
}
