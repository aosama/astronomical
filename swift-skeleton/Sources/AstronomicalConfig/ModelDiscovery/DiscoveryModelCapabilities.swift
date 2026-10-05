import Foundation;

/** The single serving mode a discovered model supports, mirroring `ModelCapabilities`. */
internal enum DiscoveryModelCapabilities: Equatable, Sendable {
    case chat(DiscoveryChatModelCapabilities);
    case imageGeneration(DiscoveryImageGenerationCapabilities);
    case embeddings(DiscoveryEmbeddingModelCapabilities);
}
