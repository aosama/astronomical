import Foundation;

/// Validates probe output values against fixed expected values with the same
/// relative tolerance the direct-MLX contracts use, port of the Rust
/// `validate_probe_outputs`. Deliberately rejects the all-zeros signature of
/// a silently dropped dispatch.
public func validateProbeOutputs(
    _ actualOutputValues: [Float],
    _ expectedOutputValues: [Float]
) -> Result<Void, KernelCapabilityError> {
    if actualOutputValues.count != expectedOutputValues.count {
        return .failure(.outputMismatch(description:
            "probe produced \(actualOutputValues.count) output values"
            + " but \(expectedOutputValues.count) were expected"));
    }
    for (valueIndex, expectedValue) in expectedOutputValues.enumerated() {
        let actualValue = actualOutputValues[valueIndex];
        let comparisonScale = max(abs(expectedValue), 1.0);
        if abs(actualValue - expectedValue) > 1e-5 * comparisonScale {
            return .failure(.outputMismatch(description:
                String(
                    format: "probe output value %ld read %.6f but expected %.6f",
                    valueIndex,
                    actualValue,
                    expectedValue)));
        }
    }
    return .success(());
}
