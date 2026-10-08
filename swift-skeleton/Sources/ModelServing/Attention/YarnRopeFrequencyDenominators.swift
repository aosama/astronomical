import Foundation

/// Validated YaRN frequency denominators ready for MLX `fast.rope`, port
/// of the Rust `YarnRopeFrequencyDenominators`.
public struct YarnRopeFrequencyDenominators: Equatable, Sendable {

    private let frequencyDenominatorValues: [Float]

    init(frequencyDenominators: [Float]) {
        self.frequencyDenominatorValues = frequencyDenominators
    }

    /// The rank-one denominator vector of length `rotaryDimension / 2`.
    public var frequencyDenominators: [Float] {
        return frequencyDenominatorValues
    }
}
