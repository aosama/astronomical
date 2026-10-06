import Foundation;

/** Embedding-serving limits carried by a discovered model, in tokens. */
public struct DiscoveryEmbeddingModelCapabilities: Equatable, Sendable {
    public let vectorWidth: UInt32;
    public let maximumInputTokens: UInt32;

    public init(vectorWidth: UInt32, maximumInputTokens: UInt32) {
        self.vectorWidth = vectorWidth;
        self.maximumInputTokens = maximumInputTokens;
    }
}
