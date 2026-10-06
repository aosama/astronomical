import Foundation;

/// Bounded chat-processor output failures.
///
/// Mirrors crates/model-serving/src/model_generation_processor.rs
/// `ModelGenerationOutputError`. Malformed output fails the request while the
/// worker stays responsive; fatal output trouble kills the worker process.
public enum ModelGenerationOutputError: Error, Equatable {

    /// Generated tokens could not be decoded or parsed into the declared
    /// output contract; the wire reason is the fixed malformed-model-output.
    case malformedOutput;

    /// A fatal processor failure the worker reports before exiting.
    case fatalExecution(reason: String);
}
