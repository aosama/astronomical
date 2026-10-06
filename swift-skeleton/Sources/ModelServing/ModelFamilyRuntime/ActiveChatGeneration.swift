import Foundation;

import IpcProtocol;

/// One accepted chat request carrying its prepared engine input and the
/// request-local translator that decodes generated tokens.
///
/// Mirrors crates/model-serving/src/model_generation_processor.rs
/// `PreparedModelGeneration` plus the request-state half of
/// `ModelGenerationProcessor`: the Rust trait kept request state on the
/// shared processor; the Swift worker holds one translator per active
/// request, so concurrent-looking fields stay request-local by construction.
public protocol ActiveChatGeneration: AnyObject {

    /// The prepared input the paired engine consumes.
    var inferenceRequest: any PreparedInferenceRequest { get };

    /// The token count the protocol reports as the prompt size.
    var promptTokenCount: Int { get };

    /// Whether one generated token is a model end-of-sequence marker.
    func isEndOfSequenceToken(_ generatedTokenId: UInt32) -> Bool;

    /// Translates one generated token through request-local output state.
    func translateGeneratedToken(_ generatedTokenId: UInt32) throws -> ModelGeneratedTokenTranslation;

    /// Flushes bounded state after generation stops.
    func finishOutputs() throws -> Array<ChatGenerationOutput>;
}
