import Foundation;

import IpcProtocol;

/// Public outputs plus optional tokenized feedback that must be injected
/// back into the active model before decoding continues.
///
/// Mirrors crates/model-serving/src/model_generation_processor.rs
/// `ModelGeneratedTokenTranslation`.
public struct ModelGeneratedTokenTranslation: Equatable {

    public let publicOutputs: Array<ChatGenerationOutput>;
    public let modelFeedbackTokenIds: Array<UInt32>;

    public init(
        publicOutputs: Array<ChatGenerationOutput>,
        modelFeedbackTokenIds: Array<UInt32> = []
    ) {
        self.publicOutputs = publicOutputs;
        self.modelFeedbackTokenIds = modelFeedbackTokenIds;
    }
}
