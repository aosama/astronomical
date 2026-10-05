import Foundation;

/** Embedding-serving limits carried by a discovered model, in tokens. */
internal struct DiscoveryEmbeddingModelCapabilities: Equatable, Sendable {
    internal let vectorWidth: UInt32;
    internal let maximumInputTokens: UInt32;

    internal init(vectorWidth: UInt32, maximumInputTokens: UInt32) {
        self.vectorWidth = vectorWidth;
        self.maximumInputTokens = maximumInputTokens;
    }
}
